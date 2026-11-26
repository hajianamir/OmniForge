"""Enterprise External Data Connectors for Excel Workbooks (.xlsx/.xls), SQL Databases (SQLite/PostgreSQL/MySQL/MSSQL), and Microsoft Access (.accdb/.mdb).

Security Hardening (CWE-89 / CWE-22 / CWE-918):
- SQL queries must be single read-only ``SELECT`` / ``WITH`` statements (comments stripped, stacked
  statements and DDL/DML/PRAGMA/ATTACH rejected).
- SQLite connections enforce three independent read-only layers:
  1. Path confinement to allowed OmniForge workspace / ``OMNIFORGE_ALLOWED_DB_DIRS`` directories,
     existing-file verification, suffix check, and SQLite header magic verification.
  2. OS-level read-only URI open (``file:...?mode=ro``) plus ``PRAGMA query_only = ON``.
  3. C-level ``sqlite3.Connection.set_authorizer(...)`` blocking all writes, DDL, ``ATTACH``,
     ``DETACH``, ``load_extension``, and write/arbitrary ``PRAGMA`` statements.
- Remote SQL database URLs enforce an explicit dialect allowlist, block cloud metadata / SSRF
  targets (``169.254.0.0/16``, link-local, unspecified, metadata hostnames, and private/loopback
  hosts unless explicitly allowed), and block dangerous query options such as ``local_infile``.
- Microsoft Access files enforce allowed-directory confinement, ``.accdb``/``.mdb`` suffix checks,
  read-only ODBC flags, and table-name verification against inspected tables.
"""

from __future__ import annotations

import io
import ipaddress
import os
import re
import shutil
import socket
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, urlparse

import numpy as np
import pandas as pd

from omniforge.config.settings import get_settings
from omniforge.core.exceptions import DataIngestionError, SecurityError
from omniforge.core.logging import get_logger
from omniforge.core.utils import to_json_safe

logger = get_logger(__name__)

_ALLOWED_SQL_BACKENDS: frozenset[str] = frozenset(
    {"sqlite", "postgresql", "mysql", "mariadb", "mssql", "duckdb"}
)

_ALLOWED_SQLITE_SUFFIXES: frozenset[str] = frozenset({".db", ".sqlite", ".sqlite3", ".s3db"})
_ALLOWED_DUCKDB_SUFFIXES: frozenset[str] = frozenset({".duckdb", ".ddb"})
_ALLOWED_ACCESS_SUFFIXES: frozenset[str] = frozenset({".accdb", ".mdb"})
_SQLITE_MAGIC_HEADER: bytes = b"SQLite format 3\x00"

_BLOCKED_METADATA_HOSTS: frozenset[str] = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "metadata.azure.com",
        "instance-data",
        "169.254.169.254",
        "fd00:ec2::254",
    }
)

_BLOCKED_SQL_QUERY_OPTIONS: frozenset[str] = frozenset(
    {
        "local_infile",
        "allow_local_infile",
        "load_data_local_infile",
        "allowloadlocalinfile",
        "allowurlinlocalinfile",
        "host",
        "hostaddr",
        "port",
        "server",
        "unix_socket",
        "dsn",
        "odbc_connect",
    }
)

_BLOCKED_SQLITE_FUNCTIONS: frozenset[str] = frozenset(
    {
        "load_extension",
        "fts3_tokenizer",
        "edit",
        "readfile",
        "writefile",
        "sqlar_compress",
        "sqlar_uncompress",
    }
)

_READONLY_INTROSPECTION_PRAGMAS: frozenset[str] = frozenset(
    {
        "table_info",
        "table_xinfo",
        "index_list",
        "index_info",
        "foreign_key_list",
        "table_list",
    }
)

_FORBIDDEN_SQL_RE = re.compile(
    r"\b("
    r"DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|"
    r"VACUUM|REINDEX|EXPLAIN|EXEC|EXECUTE|MERGE|GRANT|REVOKE|CALL|DO|COPY|INSTALL|LOAD|"
    r"BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE|LOAD_EXTENSION|LOAD_FILE|OUTFILE|DUMPFILE|"
    r"READ_CSV|READ_CSV_AUTO|READ_PARQUET|READ_JSON|READ_JSON_AUTO|READ_TEXT|READ_BLOB|"
    r"PARQUET_SCAN|CSV_SCAN|SQLITE_SCAN|POSTGRES_SCAN|MYSQL_SCAN|ICEBERG_SCAN|DELTA_SCAN|"
    r"INTO|OPENROWSET|SLEEP|BENCHMARK|PG_READ_FILE|GLOB"
    r")\b",
    re.IGNORECASE,
)


def _quote_dialect_identifier(ident: str, dialect: str) -> str:
    """Quote a verified table/view identifier using dialect-exact escaping (CWE-89)."""
    if not ident or "\x00" in ident:
        raise SecurityError("Invalid empty or NUL-containing SQL identifier.")
    d = dialect.lower()
    if d in {"mysql", "mariadb"}:
        return "`" + ident.replace("`", "``") + "`"
    if d == "mssql":
        return "[" + ident.replace("]", "]]") + "]"
    return '"' + ident.replace('"', '""') + '"'


def _strip_sql_comments_and_scan(query: str) -> tuple[str, str]:
    """Strip SQL comments and return ``(cleaned_sql, code_without_string_literals)``.

    Raises ``SecurityError`` on unterminated comments/quotes or stacked multi-statement semicolons.
    """
    n = len(query)
    i = 0
    cleaned_parts: list[str] = []
    code_parts: list[str] = []
    semicolon_positions: list[int] = []

    while i < n:
        ch = query[i]
        nxt = query[i + 1] if i + 1 < n else ""

        # Line comment: -- ... \n
        if ch == "-" and nxt == "-":
            i += 2
            while i < n and query[i] != "\n":
                i += 1
            cleaned_parts.append(" ")
            code_parts.append(" ")
            continue

        # Block comment: /* ... */
        if ch == "/" and nxt == "*":
            end_idx = query.find("*/", i + 2)
            if end_idx == -1:
                raise SecurityError("Unterminated SQL block comment is not permitted.")
            i = end_idx + 2
            cleaned_parts.append(" ")
            code_parts.append(" ")
            continue

        # Single-quoted string literal ('...''...')
        if ch == "'":
            start = i
            i += 1
            closed = False
            while i < n:
                if query[i] == "'":
                    if i + 1 < n and query[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    closed = True
                    break
                i += 1
            if not closed:
                raise SecurityError("Unterminated SQL string literal is not permitted.")
            cleaned_parts.append(query[start:i])
            code_parts.append(" '' ")
            continue

        # Double-quoted identifier ("...""...")
        if ch == '"':
            start = i
            i += 1
            closed = False
            while i < n:
                if query[i] == '"':
                    if i + 1 < n and query[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    closed = True
                    break
                i += 1
            if not closed:
                raise SecurityError("Unterminated SQL quoted identifier is not permitted.")
            ident = query[start:i]
            cleaned_parts.append(ident)
            code_parts.append(ident)
            continue

        if ch == ";":
            semicolon_positions.append(len(cleaned_parts))
            cleaned_parts.append(";")
            code_parts.append(";")
            i += 1
            continue

        cleaned_parts.append(ch)
        code_parts.append(ch)
        i += 1

    cleaned = "".join(cleaned_parts).strip()
    code_only = "".join(code_parts).strip()

    # Allow at most a single trailing semicolon; reject any embedded statement separator.
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if code_only.endswith(";"):
        code_only = code_only[:-1].strip()

    if ";" in code_only:
        raise SecurityError("Stacked multi-statement SQL queries are blocked for security.")

    return cleaned, code_only


_FILE_PATH_LITERAL_RE = re.compile(
    r"['\"](?:[^'\"]*[\\/][^'\"]*|[^'\"]*\.\w{2,4})['\"]",
)


def _validate_readonly_sql(query: str) -> str:
    """Ensure user-supplied SQL queries are strictly single read-only ``SELECT`` / ``WITH`` statements."""
    if not isinstance(query, str) or not query.strip():
        raise DataIngestionError("SQL query cannot be empty.")
    if "\x00" in query:
        raise SecurityError("NUL bytes are forbidden in SQL queries.")

    cleaned, code_only = _strip_sql_comments_and_scan(query)
    if not cleaned or not code_only:
        raise DataIngestionError("SQL query cannot be empty.")

    tokens = code_only.split()
    first_tok = tokens[0].upper() if tokens else ""
    if first_tok not in {"SELECT", "WITH"}:
        raise SecurityError("Only single read-only SELECT / WITH SQL queries are permitted.")
    if _FORBIDDEN_SQL_RE.search(code_only):
        raise SecurityError(
            "Write, DDL, PRAGMA, ATTACH, or file-access SQL statements are blocked for security. "
            "Provide a read-only SELECT query."
        )
    # Keep path-shaped tokens out of relation names and function arguments.
    # An ordinary string on the right of a comparison is a scalar filter value.
    for match in _FILE_PATH_LITERAL_RE.finditer(cleaned):
        literal = match.group()
        if literal.startswith("'") and literal.endswith("'") and cleaned[:match.start()].rstrip().endswith(("=", "<>", "!=")):
            continue
        raise SecurityError(
            "File path string literals are forbidden outside scalar SQL comparisons. "
            "Provide a read-only SELECT query referencing tables, not files."
        )
    return cleaned


def validate_readonly_sql(query: str) -> str:
    """Public wrapper over the internal read-only SQL validator (single SELECT/WITH only)."""
    return _validate_readonly_sql(query)


def make_sqlite_authorizer(
    *,
    allow_introspection_pragmas: bool = False,
    allow_read_uncommitted: bool = False,
) -> Any:
    """Public wrapper exposing the C-level read-only SQLite authorizer."""
    return _make_sqlite_authorizer(
        allow_introspection_pragmas=allow_introspection_pragmas,
        allow_read_uncommitted=allow_read_uncommitted,
    )


def _allowed_local_db_roots() -> list[Path]:
    """Return canonical directories from which local SQLite / Access files may be opened."""
    settings = get_settings()
    roots: list[Path] = [
        settings.paths.workspace_dir.expanduser().resolve(),
        (Path.cwd() / ".omniforge_workspace").resolve(),
    ]
    extra = os.getenv("OMNIFORGE_ALLOWED_DB_DIRS", "").strip()
    if extra:
        for raw_dir in extra.split(os.pathsep):
            item = raw_dir.strip()
            if item:
                roots.append(Path(item).expanduser().resolve())

    deduped: list[Path] = []
    for r in roots:
        if r not in deduped:
            deduped.append(r)
    return deduped


def _resolve_allowed_local_db_path(
    raw_path: str | Path,
    *,
    allowed_suffixes: frozenset[str],
    verify_sqlite_magic: bool = False,
) -> Path:
    """Resolve and validate a local database path against workspace directory confinement."""
    raw_str = str(raw_path).strip()
    if not raw_str or "\x00" in raw_str:
        raise SecurityError("Invalid local database file path.")
    if "?" in raw_str or "#" in raw_str:
        raise SecurityError("Query/fragment parameters are forbidden in local database paths.")

    resolved = Path(raw_str).expanduser().resolve()
    allowed_roots = _allowed_local_db_roots()
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise SecurityError(
            "Local database path is outside the allowed OmniForge workspace directories. "
            "Place database files inside '.omniforge_workspace/' or explicitly configure "
            "OMNIFORGE_ALLOWED_DB_DIRS."
        )
    if resolved.suffix.lower() not in allowed_suffixes:
        raise SecurityError(
            f"Unsupported local database file extension '{resolved.suffix or '<none>'}'. "
            f"Allowed: {sorted(allowed_suffixes)}."
        )
    if not resolved.is_file():
        raise DataIngestionError(f"Database file not found: {resolved.name}")

    if verify_sqlite_magic:
        with resolved.open("rb") as fh:
            header = fh.read(16)
        if header != _SQLITE_MAGIC_HEADER:
            raise SecurityError(
                f"File '{resolved.name}' does not have a valid SQLite 3 database header."
            )
    return resolved


def _make_sqlite_authorizer(
    *,
    allow_introspection_pragmas: bool = False,
    allow_read_uncommitted: bool = True,
) -> Any:
    """Build a C-level SQLite authorizer callback enforcing strict read-only execution."""
    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
    }
    recursive_action = getattr(sqlite3, "SQLITE_RECURSIVE", None)
    if recursive_action is not None:
        allowed_actions.add(recursive_action)

    def _authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        dbname: str | None,
        source: str | None,
    ) -> int:
        if action in allowed_actions:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            fn_name = (arg2 or arg1 or "").strip().lower()
            if fn_name in _BLOCKED_SQLITE_FUNCTIONS:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_PRAGMA:
            pragma_name = (arg1 or "").strip().lower()
            if allow_read_uncommitted and pragma_name == "read_uncommitted" and arg2 is None:
                return sqlite3.SQLITE_OK
            if allow_introspection_pragmas and pragma_name in _READONLY_INTROSPECTION_PRAGMAS:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_DENY

    return _authorizer


def _expand_ip_candidates(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Expand an IP address with any embedded IPv4 transition targets (v4-mapped, 6to4, Teredo)."""
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            candidates.append(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            candidates.append(ip.sixtofour)
        if ip.teredo is not None:
            candidates.extend(ip.teredo)
    return candidates


def _assert_ip_not_ssrf(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    norm_host: str,
    allowed_hosts: set[str],
    allow_private: bool,
    kind_label: str,
    allowlist_env: str,
    private_env: str,
) -> None:
    """Enforce link-local/metadata unconditional block and private/loopback policy across IP + v6 transition forms."""
    for cand in _expand_ip_candidates(ip):
        if (
            cand.is_link_local
            or cand.is_multicast
            or cand.is_unspecified
            or str(cand) in _BLOCKED_METADATA_HOSTS
        ):
            raise SecurityError(f"Blocked link-local/metadata {kind_label} target '{ip}'.")
        if (cand.is_private or cand.is_loopback or cand.is_reserved) and (
            norm_host not in allowed_hosts and not allow_private
        ):
            raise SecurityError(
                f"Blocked private/loopback {kind_label} host '{norm_host}' ({ip}) to prevent SSRF. "
                f"Add the host to {allowlist_env} or set {private_env}=true."
            )


def _validate_remote_db_host(host: str | None, port: int | None) -> None:
    """Block SSRF against cloud metadata services and unauthorized private/loopback endpoints."""
    if not host or not host.strip():
        raise SecurityError("Remote SQL connection URL must specify a valid hostname.")

    norm_host = host.strip().lower().rstrip(".")
    if norm_host in _BLOCKED_METADATA_HOSTS:
        raise SecurityError(f"Blocked SSRF database host '{norm_host}'.")

    allowed_hosts = {
        h.strip().lower().rstrip(".")
        for h in os.getenv("OMNIFORGE_ALLOWED_DB_HOSTS", "").split(",")
        if h.strip()
    }
    allow_private = os.getenv("OMNIFORGE_ALLOW_PRIVATE_DB_HOSTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    resolved_ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    try:
        resolved_ips.append(ipaddress.ip_address(norm_host))
    except ValueError:
        try:
            addr_info = socket.getaddrinfo(norm_host, port or 5432, type=socket.SOCK_STREAM)
            for info in addr_info:
                ip_str = str(info[4][0]).split("%", 1)[0]
                resolved_ips.append(ipaddress.ip_address(ip_str))
        except OSError as exc:
            raise DataIngestionError(f"Unable to resolve SQL database host '{norm_host}': {exc}") from exc

    for ip in resolved_ips:
        _assert_ip_not_ssrf(
            ip,
            norm_host=norm_host,
            allowed_hosts=allowed_hosts,
            allow_private=allow_private,
            kind_label="database",
            allowlist_env="OMNIFORGE_ALLOWED_DB_HOSTS",
            private_env="OMNIFORGE_ALLOW_PRIVATE_DB_HOSTS",
        )


@dataclass(frozen=True)
class ResolvedOutboundHTTPTarget:
    """Validated outbound HTTP(S) target with vetted, DNS-pinned IP addresses."""

    url: str
    scheme: str
    hostname: str
    port: int | None
    host_header: str
    pinned_ips: tuple[str, ...]
    is_ip_literal: bool

    @property
    def effective_port(self) -> int:
        if self.port is not None:
            return self.port
        return 443 if self.scheme == "https" else 80


def resolve_and_validate_outbound_http_url(url: str) -> ResolvedOutboundHTTPTarget:
    """Validate an outbound HTTP(S) URL and return vetted, DNS-pinned IPs against DNS rebinding."""
    if not isinstance(url, str) or not url.strip():
        raise DataIngestionError("HTTP URL cannot be empty.")
    cleaned = url.strip()
    if any(ch in cleaned for ch in ("\x00", "\r", "\n")):
        raise SecurityError("Control characters (NUL/CR/LF) are forbidden in outbound HTTP URLs.")
    authority_and_path = cleaned.split("?", 1)[0].split("#", 1)[0]
    if "\\" in authority_and_path:
        raise SecurityError("Backslashes are forbidden in outbound HTTP URL authority/path.")

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise SecurityError(f"Unsupported outbound HTTP URL scheme '{scheme or '<none>'}'.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise SecurityError(f"Invalid port in outbound HTTP URL: {exc}") from exc

    host = parsed.hostname
    if not host or not host.strip():
        raise SecurityError("Outbound HTTP URL must specify a valid hostname.")

    norm_host = host.strip().lower().rstrip(".")
    if norm_host in _BLOCKED_METADATA_HOSTS:
        raise SecurityError(f"Blocked SSRF HTTP host '{norm_host}'.")

    allowed_hosts = {
        h.strip().lower().rstrip(".")
        for h in os.getenv("OMNIFORGE_ALLOWED_HTTP_HOSTS", "").split(",")
        if h.strip()
    }
    allow_private = os.getenv("OMNIFORGE_ALLOW_PRIVATE_HTTP_HOSTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    default_port = 443 if scheme == "https" else 80
    resolved_ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    is_ip_literal = False
    wire_host = norm_host
    try:
        literal_ip = ipaddress.ip_address(norm_host)
        resolved_ips.append(literal_ip)
        is_ip_literal = True
    except ValueError:
        try:
            wire_host = norm_host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise SecurityError(f"Invalid internationalized HTTP hostname '{norm_host}': {exc}") from exc
        try:
            addr_info = socket.getaddrinfo(wire_host, port or default_port, type=socket.SOCK_STREAM)
            for info in addr_info:
                ip_str = str(info[4][0]).split("%", 1)[0]
                resolved_ips.append(ipaddress.ip_address(ip_str))
        except OSError as exc:
            raise DataIngestionError(f"Unable to resolve HTTP host '{norm_host}': {exc}") from exc

    if not resolved_ips:
        raise DataIngestionError(f"Unable to resolve HTTP host '{norm_host}': no addresses returned.")

    for ip in resolved_ips:
        _assert_ip_not_ssrf(
            ip,
            norm_host=norm_host,
            allowed_hosts=allowed_hosts,
            allow_private=allow_private,
            kind_label="HTTP",
            allowlist_env="OMNIFORGE_ALLOWED_HTTP_HOSTS",
            private_env="OMNIFORGE_ALLOW_PRIVATE_HTTP_HOSTS",
        )

    deduped_ips = tuple(dict.fromkeys(str(ip) for ip in resolved_ips))
    formatted_host = f"[{wire_host}]" if (is_ip_literal and ":" in wire_host) else wire_host
    host_header = (
        f"{formatted_host}:{port}"
        if (port is not None and port != default_port)
        else formatted_host
    )
    return ResolvedOutboundHTTPTarget(
        url=cleaned,
        scheme=scheme,
        hostname=wire_host,
        port=port,
        host_header=host_header,
        pinned_ips=deduped_ips,
        is_ip_literal=is_ip_literal,
    )


def validate_outbound_http_url(url: str) -> str:
    """Validate an outbound HTTP(S) URL against SSRF, cloud metadata, and control-char injection."""
    return resolve_and_validate_outbound_http_url(url).url


def _normalize_sql_url(connection_url: str) -> str:
    """Normalize a raw SQLite/DuckDB path or SQLAlchemy URL into a validated connection URL."""
    url = connection_url.strip()
    if not url:
        raise DataIngestionError("Database connection string cannot be empty.")
    if "\x00" in url:
        raise SecurityError("NUL bytes are forbidden in database connection strings.")
    if "://" not in url:
        p = Path(url).expanduser()
        if p.suffix.lower() in _ALLOWED_DUCKDB_SUFFIXES:
            return f"duckdb:///{p.as_posix()}"
        return f"sqlite:///{p.as_posix()}"
    return url


def _resolve_duckdb_url_path(connection_url: str) -> Path | None:
    """If ``connection_url`` targets DuckDB, enforce workspace confinement and return the resolved Path."""
    norm = _normalize_sql_url(connection_url)
    if not norm.lower().startswith("duckdb://"):
        return None
    raw_target = norm[len("duckdb://") :]
    if raw_target.startswith("/"):
        # duckdb:///abs/path or duckdb://rel -> preserve leading slash for absolute POSIX paths
        db_target = raw_target[1:] if (len(raw_target) >= 3 and raw_target[2] == ":") else raw_target
        if not Path(db_target).is_absolute() and raw_target.startswith("/") and not raw_target.startswith("//"):
            db_target = raw_target[1:]
    else:
        db_target = raw_target
    db_target = db_target.strip()
    if not db_target or db_target == ":memory:":
        raise SecurityError("In-memory or empty DuckDB URLs are not permitted on connector endpoints.")
    return _resolve_allowed_local_db_path(
        db_target,
        allowed_suffixes=_ALLOWED_DUCKDB_SUFFIXES,
        verify_sqlite_magic=False,
    )


def _validated_sql_url(connection_url: str) -> tuple[Any, str, Path | None]:
    """Validate connection options and endpoints for read-only and explicitly writable SDK operations."""
    url = _normalize_sql_url(connection_url)
    try:
        from sqlalchemy.engine.url import make_url
    except ImportError as exc:
        raise DataIngestionError(
            "SQLAlchemy is required for SQL database connections. Install via: pip install SQLAlchemy"
        ) from exc

    try:
        parsed = make_url(url)
    except Exception as exc:
        raise DataIngestionError("Invalid SQL connection URL.") from exc

    backend = parsed.get_backend_name().lower()
    if backend not in _ALLOWED_SQL_BACKENDS:
        raise SecurityError(
            f"Unsupported SQL dialect '{backend}'. Permitted dialects: {sorted(_ALLOWED_SQL_BACKENDS)}."
        )

    for qkey in parsed.query:
        if str(qkey).strip().lower() in _BLOCKED_SQL_QUERY_OPTIONS:
            raise SecurityError(f"Forbidden SQL connection query option '{qkey}'.")

    if backend == "sqlite":
        db_target = (parsed.database or "").strip()
        if not db_target or db_target == ":memory:":
            raise SecurityError("In-memory or empty SQLite URLs are not permitted on connector endpoints.")
        if db_target.lower().startswith("file:"):
            raise SecurityError("Raw SQLite URI parameters are forbidden; supply a workspace file path.")
        return parsed, backend, _resolve_allowed_local_db_path(
            db_target,
            allowed_suffixes=_ALLOWED_SQLITE_SUFFIXES,
            verify_sqlite_magic=True,
        )
    _validate_remote_db_host(parsed.host, parsed.port)
    return parsed, backend, None


def _create_readonly_sql_engine(
    connection_url: str,
    *,
    allow_introspection_pragmas: bool = False,
) -> tuple[Any, str]:
    """Create a hardened read-only SQLAlchemy engine for SQLite or remote SQL databases."""
    parsed, backend, local_path = _validated_sql_url(connection_url)
    from sqlalchemy import create_engine

    if local_path is not None:
        readonly_uri = f"file:{quote(str(local_path))}?mode=ro"
        authorizer = _make_sqlite_authorizer(allow_introspection_pragmas=allow_introspection_pragmas)

        def _sqlite_creator() -> sqlite3.Connection:
            conn = sqlite3.connect(readonly_uri, uri=True, check_same_thread=False)
            conn.execute("PRAGMA query_only = ON")
            conn.set_authorizer(authorizer)
            return conn

        engine = create_engine("sqlite://", creator=_sqlite_creator)
        return engine, "sqlite"

    engine = create_engine(parsed, pool_pre_ping=True)
    return engine, backend


def list_excel_sheets(raw_bytes: bytes) -> list[str]:
    """Return all worksheet names inside an uploaded Excel workbook."""
    try:
        xl = pd.ExcelFile(io.BytesIO(raw_bytes))
        return [str(s) for s in xl.sheet_names]
    except Exception as exc:
        raise DataIngestionError(f"Unable to inspect Excel workbook sheets: {exc}") from exc


_EXCEL_MISSING_SENTINELS: frozenset[str] = frozenset(
    {
        "",
        "-",
        "--",
        "—",
        "–",
        "n/a",
        "na",
        "#n/a",
        "#na",
        "#div/0!",
        "#value!",
        "#ref!",
        "#name?",
        "#num!",
        "null",
        "none",
        "nan",
    }
)


def _normalize_excel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean common Excel layout artifacts (title banners, blank rows/cols, string-formatted numbers)."""
    if df.empty:
        return df

    work = df.dropna(axis=0, how="all").copy()
    if work.empty:
        return df

    # 1. If ≥50% of columns are pandas 'Unnamed: ...', check whether an early row (0..8)
    #    actually contains the real worksheet column header row beneath a title banner.
    col_strs = [str(c).strip() for c in work.columns]
    unnamed_ratio = sum(1 for c in col_strs if not c or c.lower().startswith("unnamed:")) / max(len(col_strs), 1)
    if unnamed_ratio >= 0.5 and len(work) >= 2:
        max_probe = min(8, len(work) - 1)
        for row_pos in range(max_probe):
            row_vals = work.iloc[row_pos]
            non_null = [str(v).strip() for v in row_vals if pd.notna(v) and str(v).strip()]
            if len(non_null) < max(2, int(np.ceil(work.shape[1] * 0.5))):
                continue
            # Prefer a row with distinct text-like header labels rather than pure numbers
            alpha_like = sum(1 for v in non_null if any(ch.isalpha() for ch in v))
            if alpha_like >= max(1, len(non_null) // 2) and len(set(non_null)) >= max(2, int(len(non_null) * 0.7)):
                new_headers = [
                    str(v).strip() if (pd.notna(v) and str(v).strip()) else f"column_{idx + 1}"
                    for idx, v in enumerate(row_vals)
                ]
                work = work.iloc[row_pos + 1 :].reset_index(drop=True)
                work.columns = pd.Index(new_headers)
                break

    # 2. Drop completely empty 'Unnamed:' trailing/spacer columns and all-NaN rows
    keep_cols = [
        i
        for i, col in enumerate(work.columns)
        if not (str(col).strip().lower().startswith("unnamed:") and bool(work.iloc[:, i].isna().all()))
    ]
    if keep_cols and len(keep_cols) < work.shape[1]:
        work = work.iloc[:, keep_cols]
    work = work.dropna(axis=0, how="all").reset_index(drop=True)

    # 3. Ensure unique non-empty string column names
    seen: dict[str, int] = {}
    norm_cols: list[str] = []
    for idx, raw_col in enumerate(work.columns):
        base = str(raw_col).strip() if pd.notna(raw_col) else ""
        if not base or base.lower().startswith("unnamed:"):
            base = f"column_{idx + 1}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        norm_cols.append(base if count == 1 else f"{base}_{count}")
    work.columns = pd.Index(norm_cols)

    # 4. Coerce object columns that are genuinely numeric (e.g. formatted numbers or Excel '-'/'#N/A' sentinels)
    for col in work.columns:
        ser = work[col]
        if pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_datetime64_any_dtype(ser) or pd.api.types.is_bool_dtype(ser):
            continue
        non_na = ser.dropna()
        if len(non_na) == 0:
            continue
        str_vals = non_na.astype(str).str.strip()
        non_sentinel_mask = ~str_vals.str.lower().isin(_EXCEL_MISSING_SENTINELS)
        candidate = str_vals[non_sentinel_mask]
        if len(candidate) < 3:
            continue
        cleaned = (
            candidate.str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False)
        )
        has_pct = bool(cleaned.str.endswith("%").all())
        if has_pct:
            cleaned = cleaned.str[:-1].str.strip()
        coerced_cand = pd.to_numeric(cleaned, errors="coerce")
        valid_ratio = float(coerced_cand.notna().mean())
        if valid_ratio >= 0.8:
            full_str = ser.astype("string").str.strip()
            full_clean = full_str.mask(full_str.str.lower().isin(_EXCEL_MISSING_SENTINELS), pd.NA)
            full_clean = (
                full_clean.str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("£", "", regex=False)
            )
            if has_pct:
                full_clean = full_clean.str.rstrip("%").str.strip()
            work[col] = pd.to_numeric(full_clean, errors="coerce")

    return work


def read_excel_sheet(raw_bytes: bytes, sheet_name: str | int = 0) -> pd.DataFrame:
    """Load a specific worksheet from an Excel workbook into a normalized DataFrame."""
    try:
        raw_df = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet_name)
        return _normalize_excel_dataframe(raw_df)
    except Exception as exc:
        raise DataIngestionError(f"Failed to read Excel worksheet '{sheet_name}': {exc}") from exc


def ensure_demo_sqlite_database(workspace_dir: Path | None = None) -> Path:
    """Create (or refresh) a realistic built-in SQLite database for live SQL & Supervisory Dashboard testing."""
    target_dir = workspace_dir or (Path.cwd() / ".omniforge_workspace" / "demo_databases")
    target_dir.mkdir(parents=True, exist_ok=True)
    db_path = target_dir / "omniforge_enterprise_demo.sqlite"
    if db_path.exists() and db_path.stat().st_size > 10_000:
        return db_path

    rng = np.random.default_rng(2026)
    n = 360
    ts = pd.date_range("2026-01-01", periods=n, freq="h").astype(str)
    temp = rng.normal(72.0, 6.5, n)
    pressure = rng.normal(115.0, 9.0, n)
    vibration = rng.gamma(2.5, 1.2, n)
    load_pct = rng.uniform(45.0, 96.0, n)
    efficiency = (
        94.0
        - 0.18 * (temp - 70.0)
        - 0.09 * (pressure - 110.0)
        - 1.35 * vibration
        + 0.08 * load_pct
        + rng.normal(0.0, 1.1, n)
    )
    fault_idx = np.arange(n - 45, n, 4)
    efficiency[fault_idx] -= rng.uniform(9.0, 18.0, len(fault_idx))
    status = np.where(efficiency < 78.0, "ALERT", "OK")

    telemetry = pd.DataFrame(
        {
            "timestamp": ts,
            "asset_id": [f"TURBINE_{(i % 4) + 1:02d}" for i in range(n)],
            "temperature_c": np.round(temp, 3),
            "pressure_bar": np.round(pressure, 3),
            "vibration_mms": np.round(vibration, 3),
            "load_pct": np.round(load_pct, 2),
            "sensor_status": status,
            "thermal_efficiency": np.round(efficiency, 3),
        }
    )

    incoming_ops = telemetry.tail(80).reset_index(drop=True)

    m = 250
    income = rng.lognormal(10.8, 0.45, m)
    debt_ratio = rng.uniform(0.05, 0.72, m)
    credit_score = np.clip(rng.normal(690, 65, m), 420, 850)
    loan_amt = rng.uniform(5_000, 95_000, m)
    logit = -3.5 + 4.2 * debt_ratio - 0.008 * (credit_score - 650) + 0.00002 * loan_amt
    default_prob = 1.0 / (1.0 + np.exp(-logit))
    defaulted = (rng.uniform(0, 1, m) < default_prob).astype(int)
    credit_df = pd.DataFrame(
        {
            "application_id": [f"APP-{1000 + i}" for i in range(m)],
            "annual_income": np.round(income, 2),
            "debt_to_income": np.round(debt_ratio, 4),
            "credit_score": np.round(credit_score, 1),
            "loan_amount": np.round(loan_amt, 2),
            "default_flag": defaulted,
        }
    )

    conn = sqlite3.connect(db_path)
    try:
        with conn:
            telemetry.to_sql("industrial_turbine_telemetry", conn, if_exists="replace", index=False)
            incoming_ops.to_sql("live_turbine_monitoring_batch", conn, if_exists="replace", index=False)
            credit_df.to_sql("finance_credit_applications", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    logger.info("demo_sqlite_db_created", path=str(db_path))
    return db_path


def sql_inspect_database(connection_url: str) -> dict[str, Any]:
    """Connect read-only to a validated SQL database and list tables/views with row counts."""
    duck_path = _resolve_duckdb_url_path(connection_url)
    if duck_path is not None:
        try:
            import duckdb
        except ImportError as exc:
            raise DataIngestionError("Install 'duckdb' to inspect DuckDB databases.") from exc
        try:
            with duckdb.connect(
                str(duck_path), read_only=True, config={"enable_external_access": False}
            ) as dconn:
                rows = dconn.execute("SHOW TABLES").fetchall()
                details_duck: list[dict[str, Any]] = []
                for (tbl_raw,) in rows[:80]:
                    tbl_s = str(tbl_raw)
                    q_tbl = _quote_dialect_identifier(tbl_s, "duckdb")
                    desc = dconn.execute(f"DESCRIBE {q_tbl}").fetchall()
                    cols = [str(r[0]) for r in desc]
                    cnt_row = dconn.execute(f"SELECT COUNT(*) FROM {q_tbl}").fetchone()
                    cnt_v = int(cnt_row[0]) if cnt_row is not None else 0
                    details_duck.append({"name": tbl_s, "columns": cols, "n_columns": len(cols), "row_count": cnt_v})
            return cast(
                dict[str, Any],
                to_json_safe(
                    {
                        "connected": True,
                        "dialect": "duckdb",
                        "tables": details_duck,
                        "table_names": [d["name"] for d in details_duck],
                    }
                ),
            )
        except (SecurityError, DataIngestionError):
            raise
        except Exception as exc:
            raise DataIngestionError(f"DuckDB inspection failed ({type(exc).__name__}): {exc}") from exc

    from sqlalchemy import inspect, text

    engine, dialect_name = _create_readonly_sql_engine(
        connection_url,
        allow_introspection_pragmas=True,
    )
    try:
        with engine.connect() as conn:
            insp = inspect(conn)
            tables = list(insp.get_table_names())
            views = list(insp.get_view_names())
            details: list[dict[str, Any]] = []
            for tbl in (tables + views)[:80]:
                cols = [c["name"] for c in insp.get_columns(tbl)]
                row_cnt = None
                try:
                    q_tbl = _quote_dialect_identifier(tbl, dialect_name)
                    row_cnt = int(conn.execute(text(f"SELECT COUNT(*) FROM {q_tbl}")).scalar() or 0)
                except Exception:
                    row_cnt = None
                details.append(
                    {
                        "name": tbl,
                        "kind": "view" if tbl in views else "table",
                        "columns": cols,
                        "n_columns": len(cols),
                        "row_count": row_cnt,
                    }
                )
        return cast(
            dict[str, Any],
            to_json_safe(
                {
                    "connected": True,
                    "dialect": dialect_name,
                    "tables": details,
                    "table_names": [d["name"] for d in details],
                }
            ),
        )
    except (SecurityError, DataIngestionError):
        raise
    except Exception as exc:
        raise DataIngestionError(f"SQL connection failed ({type(exc).__name__}): {exc}") from exc
    finally:
        engine.dispose()


def sql_load_dataframe(
    connection_url: str,
    *,
    table_name: str | None = None,
    query: str | None = None,
    limit: int = 100_000,
) -> pd.DataFrame:
    """Load a verified table or read-only SQL query from a validated database."""
    duck_path = _resolve_duckdb_url_path(connection_url)
    if duck_path is not None:
        try:
            import duckdb
        except ImportError as exc:
            raise DataIngestionError("Install 'duckdb' to load DuckDB databases.") from exc
        row_limit = int(max(1, min(limit, 500_000)))
        if query and query.strip():
            sql_q = _validate_readonly_sql(query)
            try:
                with duckdb.connect(
                    str(duck_path), read_only=True, config={"enable_external_access": False}
                ) as dconn:
                    return dconn.execute(sql_q).df().head(row_limit)
            except (SecurityError, DataIngestionError):
                raise
            except Exception as exc:
                raise DataIngestionError(f"DuckDB query execution failed: {exc}") from exc
        if not table_name or not table_name.strip():
            raise DataIngestionError("Provide either a table name or a read-only SQL SELECT query.")
        req_tbl = table_name.strip()
        try:
            with duckdb.connect(
                str(duck_path), read_only=True, config={"enable_external_access": False}
            ) as dconn:
                valid_names = [str(r[0]) for r in dconn.execute("SHOW TABLES").fetchall()]
                canonical = next((n for n in valid_names if n == req_tbl), None)
                if canonical is None:
                    raise DataIngestionError(f"Table or view '{req_tbl}' was not found in the target database.")
                q_tbl = _quote_dialect_identifier(canonical, "duckdb")
                return dconn.execute(f"SELECT * FROM {q_tbl} LIMIT {row_limit}").df()
        except (SecurityError, DataIngestionError):
            raise
        except Exception as exc:
            raise DataIngestionError(f"DuckDB query execution failed: {exc}") from exc

    from sqlalchemy import inspect, text

    if query and query.strip():
        sql = _validate_readonly_sql(query)
        engine, dialect_name = _create_readonly_sql_engine(
            connection_url,
            allow_introspection_pragmas=False,
        )
        try:
            with engine.connect() as conn:
                if dialect_name == "sqlite":
                    raw_dbapi = getattr(getattr(conn, "connection", None), "dbapi_connection", None)
                    if raw_dbapi is not None and hasattr(raw_dbapi, "set_authorizer"):
                        raw_dbapi.set_authorizer(
                            _make_sqlite_authorizer(
                                allow_introspection_pragmas=False,
                                allow_read_uncommitted=False,
                            )
                        )
                elif dialect_name == "postgresql":
                    conn = conn.execution_options(postgresql_readonly=True)
                df = pd.read_sql_query(text(sql), conn)
            logger.info("sql_data_loaded", dialect=dialect_name, rows=len(df), cols=df.shape[1])
            return df
        except (SecurityError, DataIngestionError):
            raise
        except Exception as exc:
            raise DataIngestionError(f"SQL query execution failed: {exc}") from exc
        finally:
            engine.dispose()

    if not table_name or not table_name.strip():
        raise DataIngestionError("Provide either a table name or a read-only SQL SELECT query.")

    requested_table = table_name.strip()
    engine, dialect_name = _create_readonly_sql_engine(
        connection_url,
        allow_introspection_pragmas=True,
    )
    try:
        with engine.connect() as conn:
            insp = inspect(conn)
            valid_objects = set(insp.get_table_names()) | set(insp.get_view_names())
            if requested_table not in valid_objects:
                raise DataIngestionError(
                    f"Table or view '{requested_table}' was not found in the target database."
                )
            q_tbl = _quote_dialect_identifier(requested_table, dialect_name)
            row_limit = int(max(1, min(limit, 500_000)))
            sql = (
                f"SELECT TOP {row_limit} * FROM {q_tbl}"
                if dialect_name == "mssql"
                else f"SELECT * FROM {q_tbl} LIMIT {row_limit}"
            )
            if dialect_name == "postgresql":
                conn = conn.execution_options(postgresql_readonly=True)
            df = pd.read_sql_query(text(sql), conn)
        logger.info("sql_data_loaded", dialect=dialect_name, rows=len(df), cols=df.shape[1])
        return df
    except (SecurityError, DataIngestionError):
        raise
    except Exception as exc:
        raise DataIngestionError(f"SQL query execution failed: {exc}") from exc
    finally:
        engine.dispose()


def access_inspect_database(file_path: str | Path) -> dict[str, Any]:
    """Inspect tables inside an allowed Microsoft Access (``.accdb`` / ``.mdb``) database."""
    p = _resolve_allowed_local_db_path(
        file_path,
        allowed_suffixes=_ALLOWED_ACCESS_SUFFIXES,
        verify_sqlite_magic=False,
    )

    try:
        import pyodbc

        drivers = [d for d in pyodbc.drivers() if "access" in d.lower()]
        if drivers:
            conn_str = f"DRIVER={{{drivers[0]}}};DBQ={p};ReadOnly=1;"
            with pyodbc.connect(conn_str) as conn:
                cur = conn.cursor()
                tables = [
                    row.table_name
                    for row in cur.tables(tableType="TABLE")
                    if not str(row.table_name).startswith("MSys")
                ]
            return cast(
                dict[str, Any],
                to_json_safe({"connected": True, "driver": f"pyodbc:{drivers[0]}", "table_names": tables}),
            )
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("access_pyodbc_inspect_failed", error=str(exc))

    mdb_tables_bin = shutil.which("mdb-tables")
    if mdb_tables_bin:
        try:
            proc = subprocess.run(
                [mdb_tables_bin, "-1", str(p)],
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
            tables = [line.strip() for line in proc.stdout.splitlines() if line.strip() and not line.startswith("MSys")]
            return cast(
                dict[str, Any],
                to_json_safe({"connected": True, "driver": "mdbtools", "table_names": tables}),
            )
        except Exception as exc:
            raise DataIngestionError(f"mdb-tables failed on Access file '{p.name}': {exc}") from exc

    raise DataIngestionError(
        "Microsoft Access driver not detected on this host. On Windows, install 'Microsoft Access Database Engine "
        "Redistributable (64-bit)' + 'pip install pyodbc'; on Linux/macOS, install 'mdbtools' (apt install mdbtools)."
    )


def access_load_table(
    file_path: str | Path,
    table_name: str,
    *,
    limit: int = 100_000,
) -> pd.DataFrame:
    """Load a verified table from an allowed Microsoft Access (``.accdb`` / ``.mdb``) database."""
    p = _resolve_allowed_local_db_path(
        file_path,
        allowed_suffixes=_ALLOWED_ACCESS_SUFFIXES,
        verify_sqlite_magic=False,
    )
    requested_table = table_name.strip()
    if not requested_table or requested_table.startswith("-") or "\x00" in requested_table:
        raise SecurityError("Invalid Microsoft Access table name.")

    inspected = access_inspect_database(p)
    valid_tables = set(inspected.get("table_names", []))
    if requested_table not in valid_tables:
        raise DataIngestionError(f"Table '{requested_table}' not found in Access database '{p.name}'.")

    try:
        import pyodbc

        drivers = [d for d in pyodbc.drivers() if "access" in d.lower()]
        if drivers:
            conn_str = f"DRIVER={{{drivers[0]}}};DBQ={p};ReadOnly=1;"
            safe_tbl = requested_table.replace("]", "]]")
            with pyodbc.connect(conn_str) as conn:
                df = pd.read_sql(f"SELECT TOP {int(max(1, min(limit, 500_000)))} * FROM [{safe_tbl}]", conn)
            logger.info("access_table_loaded", driver=drivers[0], table=requested_table, rows=len(df))
            return df
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("access_pyodbc_load_failed", error=str(exc))

    mdb_export_bin = shutil.which("mdb-export")
    if mdb_export_bin:
        try:
            proc = subprocess.run(
                [mdb_export_bin, str(p), requested_table],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            df = pd.read_csv(io.StringIO(proc.stdout), nrows=int(max(1, min(limit, 500_000))))
            logger.info("access_table_loaded", driver="mdbtools", table=requested_table, rows=len(df))
            return df
        except Exception as exc:
            raise DataIngestionError(f"mdb-export failed for table '{requested_table}': {exc}") from exc

    raise DataIngestionError(
        "Microsoft Access driver not detected. Install 'Microsoft Access Database Engine (64-bit)' + 'pyodbc' on Windows, "
        "or 'mdbtools' on Linux/macOS."
    )
