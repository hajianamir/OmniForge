"""Enterprise External Data Connectors for Excel Workbooks (.xlsx/.xls), SQL Databases (SQLite/PostgreSQL/MySQL/MSSQL), and Microsoft Access (.accdb/.mdb)."""

from __future__ import annotations

import io
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omniforge.core.exceptions import DataIngestionError
from omniforge.core.logging import get_logger
from omniforge.core.utils import to_json_safe

logger = get_logger(__name__)

_FORBIDDEN_SQL_RE = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|EXEC|EXECUTE|MERGE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def _validate_readonly_sql(query: str) -> str:
    """Ensure user-supplied SQL queries are strictly read-only SELECT/WITH statements."""
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned:
        raise DataIngestionError("SQL query cannot be empty.")
    first_tok = cleaned.split()[0].upper()
    if first_tok not in {"SELECT", "WITH", "PRAGMA"}:
        raise DataIngestionError("Only read-only SELECT / WITH SQL queries are permitted.")
    if _FORBIDDEN_SQL_RE.search(cleaned):
        raise DataIngestionError("Write or DDL SQL statements are blocked for safety. Provide a read-only SELECT query.")
    return cleaned


def list_excel_sheets(raw_bytes: bytes) -> list[str]:
    """Return all worksheet names inside an uploaded Excel workbook."""
    try:
        xl = pd.ExcelFile(io.BytesIO(raw_bytes))
        return [str(s) for s in xl.sheet_names]
    except Exception as exc:
        raise DataIngestionError(f"Unable to inspect Excel workbook sheets: {exc}") from exc


def read_excel_sheet(raw_bytes: bytes, sheet_name: str | int = 0) -> pd.DataFrame:
    """Load a specific worksheet from an Excel workbook into a DataFrame."""
    try:
        return pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet_name)
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
    # Physical equation + small sensor noise + 18 realistic anomaly/fault events in the latest 60 hours
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

    # Second table: batch incoming operational measurements (latest 80 readings)
    incoming_ops = telemetry.tail(80).reset_index(drop=True)

    # Third table: credit/finance risk transactions
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

    with sqlite3.connect(db_path) as conn:
        telemetry.to_sql("industrial_turbine_telemetry", conn, if_exists="replace", index=False)
        incoming_ops.to_sql("live_turbine_monitoring_batch", conn, if_exists="replace", index=False)
        credit_df.to_sql("finance_credit_applications", conn, if_exists="replace", index=False)

    logger.info("demo_sqlite_db_created", path=str(db_path))
    return db_path


def _normalize_sql_url(connection_url: str) -> str:
    """Normalize a raw SQLite path or SQLAlchemy URL into a valid SQLAlchemy URL."""
    url = connection_url.strip()
    if not url:
        raise DataIngestionError("Database connection string cannot be empty.")
    if "://" not in url:
        p = Path(url).expanduser()
        return f"sqlite:///{p.as_posix()}"
    return url


def sql_inspect_database(connection_url: str) -> dict[str, Any]:
    """Connect to any SQLAlchemy-supported SQL database and list tables/views with row counts."""
    url = _normalize_sql_url(connection_url)
    try:
        from sqlalchemy import create_engine, inspect, text
    except ImportError as exc:
        raise DataIngestionError(
            "SQLAlchemy is required for SQL database connections. Install via: pip install SQLAlchemy"
        ) from exc

    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            insp = inspect(conn)
            tables = list(insp.get_table_names())
            views = list(insp.get_view_names())
            details: list[dict[str, Any]] = []
            for tbl in (tables + views)[:80]:
                cols = [c["name"] for c in insp.get_columns(tbl)]
                row_cnt = None
                try:
                    safe_tbl = tbl.replace('"', '""')
                    row_cnt = int(conn.execute(text(f'SELECT COUNT(*) FROM "{safe_tbl}"')).scalar() or 0)
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
        engine.dispose()
        return to_json_safe(
            {
                "connected": True,
                "dialect": engine.dialect.name,
                "tables": details,
                "table_names": [d["name"] for d in details],
            }
        )
    except Exception as exc:
        raise DataIngestionError(f"SQL connection failed ({type(exc).__name__}): {exc}") from exc


def sql_load_dataframe(
    connection_url: str,
    *,
    table_name: str | None = None,
    query: str | None = None,
    limit: int = 100_000,
) -> pd.DataFrame:
    """Load a table or read-only SQL query from any SQLAlchemy-supported database."""
    url = _normalize_sql_url(connection_url)
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise DataIngestionError("SQLAlchemy is required for SQL database connections.") from exc

    if query and query.strip():
        sql = _validate_readonly_sql(query)
    elif table_name and table_name.strip():
        safe_tbl = table_name.strip().replace('"', '""')
        sql = f'SELECT * FROM "{safe_tbl}" LIMIT {int(max(1, min(limit, 500_000)))}'
    else:
        raise DataIngestionError("Provide either a table name or a read-only SQL SELECT query.")

    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn)
        engine.dispose()
        logger.info("sql_data_loaded", dialect=engine.dialect.name, rows=len(df), cols=df.shape[1])
        return df
    except Exception as exc:
        raise DataIngestionError(f"SQL query execution failed: {exc}") from exc


def access_inspect_database(file_path: str | Path) -> dict[str, Any]:
    """Inspect tables inside a Microsoft Access (.accdb / .mdb) database via pyodbc or mdbtools."""
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise DataIngestionError(f"Microsoft Access file not found: {p}")

    # 1. Try Windows / ODBC pyodbc Access driver first
    try:
        import pyodbc  # type: ignore[import-not-found]

        drivers = [d for d in pyodbc.drivers() if "access" in d.lower()]
        if drivers:
            conn_str = f"DRIVER={{{drivers[0]}}};DBQ={p};"
            with pyodbc.connect(conn_str) as conn:
                cur = conn.cursor()
                tables = [
                    row.table_name
                    for row in cur.tables(tableType="TABLE")
                    if not str(row.table_name).startswith("MSys")
                ]
            return to_json_safe({"connected": True, "driver": f"pyodbc:{drivers[0]}", "table_names": tables})
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("access_pyodbc_inspect_failed", error=str(exc))

    # 2. Cross-platform mdbtools fallback (Linux / macOS)
    mdb_tables_bin = shutil.which("mdb-tables")
    if mdb_tables_bin:
        try:
            proc = subprocess.run([mdb_tables_bin, "-1", str(p)], capture_output=True, text=True, check=True, timeout=20)
            tables = [line.strip() for line in proc.stdout.splitlines() if line.strip() and not line.startswith("MSys")]
            return to_json_safe({"connected": True, "driver": "mdbtools", "table_names": tables})
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
    """Load a table from a Microsoft Access (.accdb / .mdb) database via pyodbc or mdb-export."""
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise DataIngestionError(f"Microsoft Access file not found: {p}")
    if not table_name.strip():
        raise DataIngestionError("Provide an Access table name to load.")

    # 1. pyodbc path
    try:
        import pyodbc  # type: ignore[import-not-found]

        drivers = [d for d in pyodbc.drivers() if "access" in d.lower()]
        if drivers:
            conn_str = f"DRIVER={{{drivers[0]}}};DBQ={p};"
            safe_tbl = table_name.strip().replace("]", "]]")
            with pyodbc.connect(conn_str) as conn:
                df = pd.read_sql(f"SELECT TOP {int(max(1, limit))} * FROM [{safe_tbl}]", conn)
            logger.info("access_table_loaded", driver=drivers[0], table=table_name, rows=len(df))
            return df
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("access_pyodbc_load_failed", error=str(exc))

    # 2. mdb-export fallback
    mdb_export_bin = shutil.which("mdb-export")
    if mdb_export_bin:
        try:
            proc = subprocess.run([mdb_export_bin, str(p), table_name.strip()], capture_output=True, text=True, check=True, timeout=60)
            df = pd.read_csv(io.StringIO(proc.stdout), nrows=int(max(1, limit)))
            logger.info("access_table_loaded", driver="mdbtools", table=table_name, rows=len(df))
            return df
        except Exception as exc:
            raise DataIngestionError(f"mdb-export failed for table '{table_name}': {exc}") from exc

    raise DataIngestionError(
        "Microsoft Access driver not detected. Install 'Microsoft Access Database Engine (64-bit)' + 'pyodbc' on Windows, "
        "or 'mdbtools' on Linux/macOS."
    )
