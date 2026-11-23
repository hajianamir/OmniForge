"""Multi-source enterprise data ingestion engine supporting 30+ formats, databases, cloud, APIs, and Kafka."""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx
import numpy as np
import pandas as pd
import yaml

from omniforge.core.constants import SUPPORTED_CLOUD_SCHEMES, SUPPORTED_DB_SCHEMES, SUPPORTED_FILE_FORMATS
from omniforge.core.exceptions import DataIngestionError, DependencyMissingError, SecurityError
from omniforge.core.logging import get_logger
from omniforge.core.types import ColumnSchema, DataModality, DatasetMetadata
from omniforge.core.utils import (
    compute_dataframe_checksum,
    infer_semantic_column_type,
    is_package_available,
    optimize_dataframe_dtypes,
)

logger = get_logger(__name__)

_MAX_HTTP_RESPONSE_BYTES: int = 256 * 1024 * 1024  # 256 MiB ceiling per HTTP response
_MAX_HTTP_REDIRECTS: int = 5
_SENSITIVE_QUERY_KEYS: frozenset[str] = frozenset(
    {"password", "passwd", "pwd", "token", "api_key", "apikey", "access_token", "secret", "auth"}
)
_CROSS_ORIGIN_SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {"authorization", "cookie", "proxy-authorization", "x-api-key"}
)

def _quote_sql_identifier(identifier: str) -> str:
    """Quote a SQL identifier (ANSI double-quoting with embedded-quote doubling).

    Rejects empty/NUL-containing/absurdly long identifiers. Doubling makes any
    identifier value safe to interpolate into a double-quoted position, so the
    caller still chooses the canonical table name explicitly.
    """
    if not isinstance(identifier, str):
        raise SecurityError("SQL identifier must be a string.")
    name = identifier.strip()
    if not name:
        raise SecurityError("SQL identifier cannot be empty.")
    if "\x00" in name:
        raise SecurityError("NUL bytes are forbidden in SQL identifiers.")
    if len(name) > 128:
        raise SecurityError("SQL identifier exceeds 128 characters.")
    if "." in name:
        parts = [p for p in name.split(".")]
        return ".".join(_quote_sql_identifier(part) for part in parts)
    return '"' + name.replace('"', '""') + '"'


def _sqlite_canonical_table_name(conn: sqlite3.Connection, requested: str) -> str:
    """Allowlist check: return the canonical sqlite_master table/view name matching ``requested``."""
    wanted = requested.strip()
    if not wanted or "\x00" in wanted:
        raise DataIngestionError("Table name cannot be empty.")
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name = ? COLLATE NOCASE",
        (wanted,),
    ).fetchone()
    if row is None:
        raise DataIngestionError(f"Table or view '{wanted}' was not found in the SQLite database.")
    return str(row[0])


def _duckdb_canonical_table_name(conn: Any, requested: str) -> str:
    """Allowlist check: return the canonical table/view name reported by DuckDB matching ``requested``."""
    wanted = requested.strip()
    if not wanted or "\x00" in wanted:
        raise DataIngestionError("Table name cannot be empty.")
    rows = conn.execute("SHOW TABLES").fetchall()
    names = [str(r[0]) for r in rows]
    match = next((n for n in names if n.lower() == wanted.lower()), None)
    if match is None:
        raise DataIngestionError(f"Table or view '{wanted}' was not found in the DuckDB database.")
    return match



@dataclass
class IngestionResult:
    """Container pairing an ingested DataFrame with rich lineage metadata."""

    data: pd.DataFrame
    metadata: DatasetMetadata
    warnings: list[str] = field(default_factory=list)


class DataIngestionEngine:
    """Unified multi-source ingestion engine with schema inference, chunking, optimization, and checksums.

    Supported sources:
        - Local & cloud files (via ``fsspec``): CSV, TSV, Excel, Parquet, Feather, Arrow/IPC,
          JSON, JSONL, ORC, Avro, HDF5, Pickle, XML, YAML, TOML, SAS, SPSS, Stata.
        - Databases: PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, BigQuery, Redshift,
          ClickHouse, DuckDB, MongoDB.
        - In-memory objects: ``pd.DataFrame``, ``dict``, ``list[dict]``, ``np.ndarray``, Polars DataFrame.
        - HTTP REST APIs (JSON/CSV with optional pagination).
        - Streaming: Kafka topic batch consumer and JSONL stream iterator.
    """

    def __init__(
        self,
        *,
        optimize_dtypes: bool = True,
        compute_checksum: bool = True,
        default_encoding: str = "utf-8",
        tenant_id: str = "default-tenant",
        allow_unsafe_pickle: bool = False,
    ) -> None:
        self.optimize_dtypes = optimize_dtypes
        self.compute_checksum = compute_checksum
        self.default_encoding = default_encoding
        self.tenant_id = tenant_id
        self.allow_unsafe_pickle = allow_unsafe_pickle

    def load_file(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Convenience wrapper around :meth:`ingest` returning the loaded DataFrame directly."""
        return self.ingest(path, **kwargs).data

    def ingest(
        self,
        source: Any,
        *,
        source_format: str | None = None,
        name: str | None = None,
        modality: DataModality = DataModality.TABULAR,
        query: str | None = None,
        storage_options: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> IngestionResult:
        """Ingest data from any supported source into a DataFrame with full metadata.

        Args:
            source: File path, URL, database URI, Kafka URI (``kafka://host:port/topic``),
                or in-memory Python object.
            source_format: Optional explicit format override (e.g. ``"csv"``, ``"parquet"``).
            name: Optional logical dataset name.
            modality: Semantic data modality.
            query: SQL query or MongoDB collection/filter when ingesting from a database.
            storage_options: Credentials/options forwarded to ``fsspec`` for S3/GCS/Azure.
            headers: HTTP headers for REST API ingestion.
            params: Query parameters for REST API ingestion.
            tags: Arbitrary key-value lineage tags.
            **kwargs: Format-specific reader options forwarded to underlying loaders.

        Returns:
            :class:`IngestionResult` containing DataFrame, :class:`DatasetMetadata`, and warnings.
        """
        warnings: list[str] = []

        if not isinstance(source, (str, Path)):
            df, resolved_fmt, source_label = self._ingest_in_memory(source)
        else:
            source_str = str(source)
            parsed = urlparse(source_str)
            scheme = parsed.scheme.lower()

            if scheme == "kafka":
                df = self._ingest_kafka(source_str, **kwargs)
                resolved_fmt = "kafka"
                source_label = source_str
            elif scheme in ("http", "https") and (
                source_format == "api" or self._looks_like_api_endpoint(source_str, source_format)
            ):
                df = self._ingest_api(
                    source_str, headers=headers, params=params, source_format=source_format, **kwargs
                )
                resolved_fmt = source_format or "api_json"
                source_label = self._sanitize_uri(source_str)
            elif any(scheme.startswith(db_s) for db_s in SUPPORTED_DB_SCHEMES) or (
                scheme == "" and source_str.lower().endswith((".db", ".sqlite", ".sqlite3", ".duckdb"))
            ):
                df, resolved_fmt = self._ingest_database(source_str, query=query, **kwargs)
                source_label = self._sanitize_uri(source_str)
            elif scheme in SUPPORTED_CLOUD_SCHEMES:
                fmt = source_format or self._detect_format_from_path(parsed.path)
                if scheme in ("http", "https"):
                    df = self._ingest_http_file(
                        source_str,
                        fmt=fmt,
                        headers=headers,
                        params=params,
                        **kwargs,
                    )
                    source_label = self._sanitize_uri(source_str)
                else:
                    df = self._ingest_file(
                        source_str,
                        fmt=fmt,
                        storage_options=storage_options,
                        **kwargs,
                    )
                    source_label = self._sanitize_uri(source_str)
                resolved_fmt = f"{scheme}:{fmt}"
            else:
                path = Path(source_str)
                if not path.exists():
                    raise DataIngestionError(
                        f"Data source file not found: {source_str}",
                        suggestions=[
                            "Verify the file path exists and is readable.",
                            "For remote files, include the scheme (s3://, gs://, https://).",
                        ],
                        context={"source": source_str},
                    )
                fmt = source_format or self._detect_format_from_path(str(path))
                df = self._ingest_file(str(path), fmt=fmt, storage_options=storage_options, **kwargs)
                resolved_fmt = fmt
                source_label = str(path.resolve())

        if self.optimize_dtypes and len(df) > 0:
            before_mem = int(df.memory_usage(deep=True).sum())
            df = optimize_dataframe_dtypes(df)
            after_mem = int(df.memory_usage(deep=True).sum())
            if after_mem < before_mem:
                logger.info(
                    "dtypes_optimized",
                    saved_bytes=before_mem - after_mem,
                    reduction_pct=round((1 - after_mem / max(before_mem, 1)) * 100.0, 2),
                )

        metadata = self.build_metadata(
            df=df,
            name=name or self._derive_name(source_label),
            source=source_label,
            source_format=resolved_fmt,
            modality=modality,
            tags=tags or {},
        )
        if len(df) == 0:
            warnings.append("Ingested dataset contains zero rows.")

        logger.info(
            "data_ingested",
            dataset_id=metadata.dataset_id,
            source_format=resolved_fmt,
            rows=metadata.n_rows,
            columns=metadata.n_columns,
        )
        return IngestionResult(data=df, metadata=metadata, warnings=warnings)

    @staticmethod
    def list_excel_sheets(source: str | Path | bytes | io.BytesIO) -> list[str]:
        """Inspect an Excel workbook and return all available worksheet names in order."""
        try:
            excel_input: Any = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
            with pd.ExcelFile(excel_input) as workbook:
                return [str(name) for name in workbook.sheet_names]
        except Exception as exc:
            raise DataIngestionError(
                f"Failed to inspect Excel worksheets: {exc}",
                suggestions=[
                    "Verify the file is a valid .xlsx / .xlsm / .xls Excel workbook.",
                    "Ensure openpyxl is installed for modern Excel workbooks.",
                ],
            ) from exc

    def ingest_excel_sheets(
        self,
        source: str | Path | bytes | io.BytesIO,
        *,
        sheets: list[str] | str | None = None,
        combine_mode: str = "single",
        merge_on: str | None = None,
        merge_how: str = "inner",
        add_sheet_column: bool = True,
        header_row: int = 0,
        name: str | None = None,
    ) -> IngestionResult:
        """Ingest one or multiple sheets from an Excel workbook with explicit user control.

        Args:
            source: File path, raw bytes, or BytesIO stream of an Excel workbook.
            sheets: Single sheet name, list of sheet names, or ``None`` (defaults to first sheet
                for ``single`` mode or all sheets for ``concat``/``merge`` modes).
            combine_mode: ``'single'`` (one sheet), ``'concat'`` (stack rows), or ``'merge'`` (join columns).
            merge_on: Key column name used when ``combine_mode == 'merge'``.
            merge_how: Pandas merge join type (``'inner'``, ``'left'``, ``'outer'``, ``'right'``).
            add_sheet_column: When stacking multiple sheets, add a ``_source_sheet`` origin column.
            header_row: Zero-based row index to treat as column headers.
            name: Optional logical dataset name.

        Returns:
            An ``IngestionResult`` containing the resulting DataFrame and lineage metadata.
        """
        available = self.list_excel_sheets(source)
        if not available:
            raise DataIngestionError("Excel workbook contains no readable worksheets.")

        if isinstance(sheets, str):
            chosen = [sheets]
        elif sheets:
            chosen = list(sheets)
        else:
            chosen = [available[0]] if combine_mode == "single" else list(available)

        invalid = [s for s in chosen if s not in available]
        if invalid:
            raise DataIngestionError(
                f"Requested Excel sheet(s) not found: {invalid}",
                suggestions=[f"Available sheets: {', '.join(available)}"],
            )

        excel_input: Any = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        loaded: dict[str, pd.DataFrame] = {}
        try:
            with pd.ExcelFile(excel_input) as workbook:
                for sheet_name in chosen:
                    loaded[sheet_name] = pd.read_excel(workbook, sheet_name=sheet_name, header=header_row)
        except Exception as exc:
            raise DataIngestionError(f"Failed to read Excel sheet(s) {chosen}: {exc}") from exc

        warnings: list[str] = []
        if combine_mode == "single" or len(chosen) == 1:
            df = loaded[chosen[0]].copy()
            label_suffix = chosen[0]
        elif combine_mode == "concat":
            frames: list[pd.DataFrame] = []
            for s_name in chosen:
                part = loaded[s_name].copy()
                if add_sheet_column:
                    part["_source_sheet"] = s_name
                frames.append(part)
            df = pd.concat(frames, ignore_index=True)
            label_suffix = f"stacked_{len(chosen)}_sheets"
        elif combine_mode == "merge":
            if not merge_on:
                common_cols = set(loaded[chosen[0]].columns)
                for s_name in chosen[1:]:
                    common_cols &= set(loaded[s_name].columns)
                if not common_cols:
                    raise DataIngestionError(
                        "Cannot merge selected Excel sheets because they share no common key column.",
                        suggestions=[
                            "Choose 'Combine / Stack Rows' instead, or verify your sheets share an ID column."
                        ],
                    )
                merge_on = sorted(common_cols)[0]
                warnings.append(f"Auto-selected common join key '{merge_on}'.")
            df = loaded[chosen[0]].copy()
            for idx, s_name in enumerate(chosen[1:], start=2):
                right = loaded[s_name].copy()
                if merge_on not in df.columns or merge_on not in right.columns:
                    raise DataIngestionError(
                        f"Join key column '{merge_on}' not present in sheet '{s_name}'.",
                        suggestions=["Pick a key column shared across all selected sheets."],
                    )
                df = df.merge(right, on=merge_on, how=merge_how, suffixes=("", f"_{s_name or idx}"))
            label_suffix = f"merged_{len(chosen)}_sheets"
        else:
            raise DataIngestionError(f"Unsupported Excel combine_mode: '{combine_mode}'.")

        if self.optimize_dtypes and len(df) > 0:
            df = optimize_dataframe_dtypes(df)

        src_label = str(source) if isinstance(source, (str, Path)) else f"excel_upload:{label_suffix}"
        ds_name = name or (f"{Path(source).stem}_{label_suffix}" if isinstance(source, (str, Path)) else label_suffix)
        metadata = self.build_metadata(
            df=df,
            name=ds_name,
            source=src_label,
            source_format="excel",
            modality=DataModality.TABULAR,
            tags={"sheets": ",".join(chosen), "combine_mode": combine_mode},
        )
        return IngestionResult(data=df, metadata=metadata, warnings=warnings)

    def ingest_chunks(
        self,
        source: str | Path,
        *,
        chunk_size: int = 50_000,
        source_format: str | None = None,
        **kwargs: Any,
    ) -> Iterator[pd.DataFrame]:
        """Yield DataFrame chunks from large CSV, TSV, JSONL, or Parquet files."""
        source_str = str(source)
        fmt = source_format or self._detect_format_from_path(source_str)
        try:
            if fmt in ("csv", "tsv"):
                sep = "\t" if fmt == "tsv" else kwargs.pop("sep", ",")
                for chunk in pd.read_csv(
                    source_str, sep=sep, chunksize=chunk_size, encoding=self.default_encoding, **kwargs
                ):
                    yield chunk
            elif fmt == "jsonl":
                for chunk in pd.read_json(source_str, lines=True, chunksize=chunk_size, **kwargs):
                    yield chunk
            elif fmt == "parquet":
                if is_package_available("pyarrow"):
                    import pyarrow.parquet as pq

                    pf = pq.ParquetFile(source_str)
                    for batch in pf.iter_batches(batch_size=chunk_size):
                        yield batch.to_pandas()
                else:
                    full_df = pd.read_parquet(source_str, **kwargs)
                    for start in range(0, len(full_df), chunk_size):
                        yield full_df.iloc[start : start + chunk_size].copy()
            else:
                full_df = self.ingest(source_str, source_format=fmt, **kwargs).data
                for start in range(0, len(full_df), chunk_size):
                    yield full_df.iloc[start : start + chunk_size].copy()
        except Exception as exc:
            raise DataIngestionError(
                f"Chunked ingestion failed for '{source_str}' (format={fmt}).",
                cause=exc,
            ) from exc

    def build_metadata(
        self,
        *,
        df: pd.DataFrame,
        name: str,
        source: str,
        source_format: str,
        modality: DataModality = DataModality.TABULAR,
        tags: dict[str, str] | None = None,
        parent_dataset_id: str | None = None,
    ) -> DatasetMetadata:
        """Build complete :class:`DatasetMetadata` including inferred schema and checksum."""
        n_rows = int(len(df))
        schema_cols: list[ColumnSchema] = []
        for col in df.columns:
            series = df[col]
            null_cnt = int(series.isna().sum())
            non_null = series.dropna()
            samples = [self._safe_scalar(v) for v in non_null.head(3).tolist()]
            schema_cols.append(
                ColumnSchema(
                    name=str(col),
                    dtype=str(series.dtype),
                    inferred_semantic_type=infer_semantic_column_type(series),
                    nullable=null_cnt > 0,
                    null_count=null_cnt,
                    null_ratio=round(null_cnt / max(n_rows, 1), 6),
                    unique_count=int(series.nunique(dropna=True))
                    if self._is_hashable_series(series)
                    else 0,
                    sample_values=samples,
                )
            )

        checksum = compute_dataframe_checksum(df) if self.compute_checksum else ""
        return DatasetMetadata(
            name=name,
            source=source,
            source_format=source_format,
            modality=modality,
            n_rows=n_rows,
            n_columns=int(len(df.columns)),
            memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
            checksum_sha256=checksum,
            schema_columns=schema_cols,
            tenant_id=self.tenant_id,
            parent_dataset_id=parent_dataset_id,
            tags=tags or {},
            lineage=[
                {
                    "operation": "ingest",
                    "source": source,
                    "format": source_format,
                    "rows": n_rows,
                    "columns": int(len(df.columns)),
                }
            ],
        )

    def _ingest_in_memory(self, obj: Any) -> tuple[pd.DataFrame, str, str]:
        if isinstance(obj, pd.DataFrame):
            return obj.copy(), "dataframe", "in_memory:DataFrame"
        if isinstance(obj, pd.Series):
            return obj.to_frame(), "series", "in_memory:Series"
        if isinstance(obj, np.ndarray):
            arr = obj if obj.ndim == 2 else obj.reshape(-1, 1)
            cols = [f"feature_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols), "ndarray", "in_memory:ndarray"
        if isinstance(obj, dict):
            try:
                return pd.DataFrame(obj), "dict", "in_memory:dict"
            except ValueError:
                return pd.DataFrame([obj]), "dict", "in_memory:dict"
        if isinstance(obj, (list, tuple)):
            return pd.DataFrame(obj), "records", "in_memory:records"
        if hasattr(obj, "to_pandas") and callable(obj.to_pandas):
            return obj.to_pandas(), "external_dataframe", f"in_memory:{type(obj).__name__}"
        raise DataIngestionError(
            f"Unsupported in-memory data type: {type(obj).__name__}.",
            suggestions=["Pass a pandas DataFrame, dict, list of dicts, numpy ndarray, or file path."],
        )

    def _ingest_file(
        self,
        path_or_uri: str,
        *,
        fmt: str,
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        so = {"storage_options": storage_options} if storage_options else {}
        try:
            if fmt == "csv":
                enc = kwargs.pop("encoding", self.default_encoding)
                if not so and not kwargs and is_package_available("pyarrow"):
                    try:
                        return pd.read_csv(path_or_uri, encoding=enc, engine="pyarrow")
                    except Exception:
                        pass
                return pd.read_csv(path_or_uri, encoding=enc, **so, **kwargs)
            if fmt == "tsv":
                enc = kwargs.pop("encoding", self.default_encoding)
                if not so and not kwargs and is_package_available("pyarrow"):
                    try:
                        return pd.read_csv(path_or_uri, sep="\t", encoding=enc, engine="pyarrow")
                    except Exception:
                        pass
                return pd.read_csv(
                    path_or_uri,
                    sep="\t",
                    encoding=enc,
                    **so,
                    **kwargs,
                )
            if fmt == "excel":
                excel_obj = pd.read_excel(path_or_uri, **so, **kwargs)
                if isinstance(excel_obj, dict):
                    frames: list[pd.DataFrame] = []
                    for s_name, s_df in excel_obj.items():
                        part = s_df.copy()
                        part["_source_sheet"] = str(s_name)
                        frames.append(part)
                    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                return excel_obj
            if fmt == "parquet":
                return pd.read_parquet(path_or_uri, **so, **kwargs)
            if fmt == "feather":
                return pd.read_feather(path_or_uri, **so, **kwargs)
            if fmt == "arrow":
                if not is_package_available("pyarrow"):
                    raise DependencyMissingError("Install 'pyarrow' to read Arrow IPC files.")
                import pyarrow.ipc as pa_ipc

                with open(path_or_uri, "rb") as handle:
                    try:
                        return pa_ipc.open_file(handle).read_all().to_pandas()
                    except Exception:
                        handle.seek(0)
                        return pa_ipc.open_stream(handle).read_all().to_pandas()
            if fmt == "json":
                try:
                    return pd.read_json(path_or_uri, **so, **kwargs)
                except ValueError:
                    raw = json.loads(Path(path_or_uri).read_text(encoding=self.default_encoding))
                    return pd.json_normalize(raw)
            if fmt == "jsonl":
                return pd.read_json(path_or_uri, lines=True, **so, **kwargs)
            if fmt == "orc":
                return pd.read_orc(path_or_uri, **kwargs)
            if fmt == "avro":
                if not is_package_available("fastavro"):
                    raise DependencyMissingError(
                        "Install 'fastavro' to read Apache Avro files.",
                        suggestions=["Run: pip install fastavro"],
                    )
                import fastavro

                with open(path_or_uri, "rb") as handle:
                    records = list(fastavro.reader(handle))
                return pd.DataFrame(records)
            if fmt == "hdf5":
                return pd.read_hdf(path_or_uri, **kwargs)
            if fmt == "pickle":
                if not self.allow_unsafe_pickle:
                    raise DataIngestionError(
                        "Pickle file ingestion (.pkl/.pickle) is disabled by default for security (CWE-502: unsafe deserialization).",
                        context={"path": path_or_uri, "format": fmt},
                        suggestions=[
                            "Convert your dataset to a safe tabular format such as Parquet (.parquet), CSV (.csv), JSON (.json), or Excel (.xlsx)."
                        ],
                    )
                obj = pd.read_pickle(path_or_uri, **so, **kwargs)
                return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
            if fmt == "xml":
                raw_xml = Path(path_or_uri).read_text(encoding=self.default_encoding, errors="replace")
                xml_upper = raw_xml.upper()
                if "<!DOCTYPE" in xml_upper or "<!ENTITY" in xml_upper:
                    raise DataIngestionError(
                        "Unsafe XML document rejected: DOCTYPE and ENTITY declarations are forbidden (XXE protection).",
                        context={"path": path_or_uri, "format": fmt},
                        suggestions=["Remove DTD/DOCTYPE/ENTITY declarations from the XML file before ingesting."],
                    )
                try:
                    return pd.read_xml(io.StringIO(raw_xml), **kwargs)
                except Exception:
                    # Audit B314: never fall back to stdlib ET.fromstring (unsafe). Defusedxml
                    # is a declared core dependency; if it is absent, fail closed.
                    try:
                        import defusedxml.ElementTree as safe_et
                    except ImportError as exc:
                        raise DependencyMissingError(
                            "Install 'defusedxml' to parse XML without pandas' XML reader.",
                            suggestions=["pip install defusedxml  (or reinstall omniforge: pip install omniforge)"],
                        ) from exc
                    root = safe_et.fromstring(raw_xml)
                    rows = [{child.tag: child.text for child in elem} for elem in root]
                    return pd.DataFrame(rows)
            if fmt == "yaml":
                loaded = yaml.safe_load(Path(path_or_uri).read_text(encoding=self.default_encoding))
                return pd.json_normalize(loaded) if isinstance(loaded, (dict, list)) else pd.DataFrame([{"value": loaded}])
            if fmt == "toml":
                with open(path_or_uri, "rb") as handle:
                    loaded = tomllib.load(handle)
                if "records" in loaded and isinstance(loaded["records"], list):
                    return pd.DataFrame(loaded["records"])
                return pd.json_normalize(loaded)
            if fmt == "sas":
                return pd.read_sas(path_or_uri, **kwargs)
            if fmt == "spss":
                return pd.read_spss(path_or_uri, **kwargs)
            if fmt == "stata":
                return pd.read_stata(path_or_uri, **kwargs)

            raise DataIngestionError(
                f"Unsupported file format '{fmt}' for source '{path_or_uri}'.",
                suggestions=[f"Supported formats: {sorted(set(SUPPORTED_FILE_FORMATS.values()))}"],
            )
        except (DataIngestionError, DependencyMissingError):
            raise
        except Exception as exc:
            raise DataIngestionError(
                f"Failed to ingest file '{path_or_uri}' as format '{fmt}'.",
                context={"path": path_or_uri, "format": fmt},
                suggestions=["Verify file integrity, delimiter, encoding, or explicit source_format."],
                cause=exc,
            ) from exc

    def _ingest_database(
        self, uri: str, *, query: str | None = None, **kwargs: Any
    ) -> tuple[pd.DataFrame, str]:
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()
        table_name = kwargs.pop("table", None)

        try:
            if scheme == "sqlite" or uri.lower().endswith((".db", ".sqlite", ".sqlite3")):
                from omniforge.data.connectors import make_sqlite_authorizer, validate_readonly_sql

                db_path = uri.replace("sqlite:///", "") if scheme == "sqlite" else uri
                sqlite_conn = self._open_sqlite_readonly(db_path)
                try:
                    sqlite_conn.set_authorizer(
                        make_sqlite_authorizer(
                            allow_introspection_pragmas=False,
                            allow_read_uncommitted=False,
                        )
                    )
                    if query:
                        sql = validate_readonly_sql(query)
                    elif table_name:
                        canonical = _sqlite_canonical_table_name(sqlite_conn, table_name)
                        sql = f"SELECT * FROM {_quote_sql_identifier(canonical)}"
                    else:
                        sql = self._first_sqlite_table_query(sqlite_conn)
                    return pd.read_sql_query(sql, sqlite_conn, **kwargs), "sqlite"
                finally:
                    sqlite_conn.close()

            if scheme == "duckdb" or uri.lower().endswith(".duckdb"):
                if not is_package_available("duckdb"):
                    raise DependencyMissingError("Install 'duckdb' to ingest from DuckDB.")
                import duckdb

                from omniforge.data.connectors import validate_readonly_sql

                db_path = uri.replace("duckdb:///", "") if scheme == "duckdb" else uri
                with duckdb.connect(
                    db_path, read_only=True, config={"enable_external_access": False}
                ) as conn:
                    if query:
                        return conn.execute(validate_readonly_sql(query)).df(), "duckdb"
                    if table_name:
                        canonical = _duckdb_canonical_table_name(conn, table_name)
                        return conn.execute(f"SELECT * FROM {_quote_sql_identifier(canonical)}").df(), "duckdb"
                    tables = conn.execute("SHOW TABLES").fetchall()
                    if not tables:
                        return pd.DataFrame(), "duckdb"
                    first = str(tables[0][0])
                    return conn.execute(f"SELECT * FROM {_quote_sql_identifier(first)}").df(), "duckdb"

            if scheme.startswith("mongodb"):
                if not is_package_available("pymongo"):
                    raise DependencyMissingError("Install 'pymongo' to ingest from MongoDB.")
                from pymongo import MongoClient

                db_name = kwargs.pop("database", parsed.path.lstrip("/") or "default")
                collection_name = kwargs.pop("collection", table_name or query)
                if not collection_name:
                    raise DataIngestionError("MongoDB ingestion requires 'collection' or 'table'.")
                mongo_filter = kwargs.pop("filter", {})
                client = MongoClient(uri)
                try:
                    docs = list(client[db_name][collection_name].find(mongo_filter))
                finally:
                    client.close()
                for doc in docs:
                    if "_id" in doc:
                        doc["_id"] = str(doc["_id"])
                return pd.DataFrame(docs), "mongodb"

            if not is_package_available("sqlalchemy"):
                raise DependencyMissingError(
                    f"Install 'sqlalchemy' and the database driver for '{scheme}'."
                )
            from sqlalchemy import create_engine, inspect

            engine = create_engine(uri)
            try:
                if query:
                    from omniforge.data.connectors import validate_readonly_sql

                    return pd.read_sql_query(validate_readonly_sql(query), engine, **kwargs), scheme
                if table_name:
                    return pd.read_sql_table(table_name, engine, **kwargs), scheme
                sa_tables = inspect(engine).get_table_names()
                if not sa_tables:
                    return pd.DataFrame(), scheme
                return pd.read_sql_table(sa_tables[0], engine, **kwargs), scheme
            finally:
                engine.dispose()
        except (SecurityError, DataIngestionError, DependencyMissingError):
            raise
        except Exception as exc:
            raise DataIngestionError(
                f"Database ingestion failed for scheme '{scheme or 'sqlite'}'.",
                cause=exc,
                suggestions=["Verify connection URI, credentials, and SQL query syntax."],
            ) from exc

    @classmethod
    def _fetch_http_safe(
        cls,
        client: httpx.Client,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        max_redirects: int = _MAX_HTTP_REDIRECTS,
        max_bytes: int = _MAX_HTTP_RESPONSE_BYTES,
    ) -> tuple[bytes, str]:
        """Fetch an HTTP(S) resource enforcing SSRF validation on every redirect hop and response size bounds."""
        from omniforge.data.connectors import validate_outbound_http_url

        current_url = validate_outbound_http_url(url)
        orig_parsed = urlparse(current_url)
        orig_scheme = orig_parsed.scheme.lower()
        orig_origin = (
            orig_scheme,
            (orig_parsed.hostname or "").lower(),
            orig_parsed.port or (443 if orig_scheme == "https" else 80),
        )
        cur_headers = dict(headers or {})
        cur_params: dict[str, Any] | None = dict(params) if params else None

        for _hop in range(max_redirects + 1):
            with client.stream("GET", current_url, headers=cur_headers, params=cur_params) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location", "").strip()
                    if not location:
                        raise DataIngestionError(
                            f"HTTP redirect ({resp.status_code}) missing Location header for '{cls._sanitize_uri(url)}'."
                        )
                    next_url = validate_outbound_http_url(urljoin(str(resp.request.url), location))
                    next_parsed = urlparse(next_url)
                    next_scheme = next_parsed.scheme.lower()
                    next_origin = (
                        next_scheme,
                        (next_parsed.hostname or "").lower(),
                        next_parsed.port or (443 if next_scheme == "https" else 80),
                    )
                    if next_origin != orig_origin and cur_headers:
                        cur_headers = {
                            k: v
                            for k, v in cur_headers.items()
                            if k.lower() not in _CROSS_ORIGIN_SENSITIVE_HEADERS
                        }
                    current_url = next_url
                    cur_params = None
                    continue

                resp.raise_for_status()
                content_length = resp.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_len = int(content_length)
                    except ValueError:
                        declared_len = -1
                    if declared_len > max_bytes:
                        raise DataIngestionError(
                            f"HTTP response Content-Length ({declared_len} bytes) exceeds safety limit ({max_bytes} bytes)."
                        )

                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise DataIngestionError(
                            f"HTTP response body exceeded safety limit ({max_bytes} bytes)."
                        )
                    chunks.append(chunk)
                content_type = resp.headers.get("content-type", "").lower()
                return b"".join(chunks), content_type

        raise DataIngestionError(
            f"Exceeded maximum HTTP redirects ({max_redirects}) for '{cls._sanitize_uri(url)}'."
        )

    def _ingest_http_file(
        self,
        url: str,
        *,
        fmt: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
        max_redirects: int = _MAX_HTTP_REDIRECTS,
        max_bytes: int = _MAX_HTTP_RESPONSE_BYTES,
        _transport: httpx.BaseTransport | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Download a validated HTTP(S) file into an isolated temp directory and parse via ``_ingest_file``."""
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False, transport=_transport) as client:
                body, _ = self._fetch_http_safe(
                    client,
                    url,
                    headers=headers,
                    params=params,
                    max_redirects=max_redirects,
                    max_bytes=max_bytes,
                )
            raw_suffix = Path(urlparse(url).path).suffix or f".{fmt}"
            clean_chars = "".join(ch for ch in raw_suffix if ch.isalnum() or ch == ".")[:16]
            suffix = clean_chars if clean_chars.startswith(".") and len(clean_chars) > 1 else f".{fmt}"
            with tempfile.TemporaryDirectory(prefix="omniforge_http_") as tmp_dir:
                tmp_file = Path(tmp_dir) / f"payload{suffix}"
                tmp_file.write_bytes(body)
                return self._ingest_file(str(tmp_file), fmt=fmt, **kwargs)
        except (SecurityError, DataIngestionError, DependencyMissingError):
            raise
        except Exception as exc:
            raise DataIngestionError(
                f"HTTP file ingestion failed for '{self._sanitize_uri(url)}'.",
                cause=exc,
                suggestions=["Verify the remote file URL, format, and network reachability."],
            ) from exc

    def _ingest_api(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        source_format: str | None = None,
        timeout: float = 30.0,
        records_key: str | None = None,
        max_pages: int = 1,
        page_param: str = "page",
        max_redirects: int = _MAX_HTTP_REDIRECTS,
        max_bytes: int = _MAX_HTTP_RESPONSE_BYTES,
        _transport: httpx.BaseTransport | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        all_frames: list[pd.DataFrame] = []
        base_params = dict(params or {})
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False, transport=_transport) as client:
                for page in range(1, max_pages + 1):
                    req_params = dict(base_params)
                    if max_pages > 1:
                        req_params[page_param] = page
                    body, content_type = self._fetch_http_safe(
                        client,
                        url,
                        headers=headers,
                        params=req_params,
                        max_redirects=max_redirects,
                        max_bytes=max_bytes,
                    )

                    if source_format == "csv" or "text/csv" in content_type:
                        frame = pd.read_csv(io.StringIO(body.decode(self.default_encoding, errors="replace")))
                    else:
                        payload = json.loads(body.decode(self.default_encoding))
                        if records_key and isinstance(payload, dict):
                            payload = payload.get(records_key, [])
                        elif isinstance(payload, dict):
                            for candidate in ("data", "items", "results", "records"):
                                if candidate in payload and isinstance(payload[candidate], list):
                                    payload = payload[candidate]
                                    break
                        frame = (
                            pd.json_normalize(payload)
                            if isinstance(payload, (dict, list))
                            else pd.DataFrame([{"value": payload}])
                        )
                    if len(frame) == 0:
                        break
                    all_frames.append(frame)
            return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
        except (SecurityError, DataIngestionError, DependencyMissingError):
            raise
        except Exception as exc:
            raise DataIngestionError(
                f"REST API ingestion failed for '{self._sanitize_uri(url)}'.",
                cause=exc,
                suggestions=["Verify endpoint URL, authentication headers, and response schema."],
            ) from exc

    def _ingest_kafka(
        self,
        kafka_uri: str,
        *,
        max_messages: int = 1000,
        timeout_ms: int = 5000,
        group_id: str = "uap-consumer",
        **kwargs: Any,
    ) -> pd.DataFrame:
        parsed = urlparse(kafka_uri)
        bootstrap_servers = parsed.netloc
        topic = parsed.path.lstrip("/")
        if not bootstrap_servers or not topic:
            raise DataIngestionError(
                f"Invalid Kafka URI '{kafka_uri}'. Expected 'kafka://host:port/topic'."
            )
        if not is_package_available("kafka"):
            raise DependencyMissingError(
                "Install 'kafka-python' to consume live Kafka topics.",
                suggestions=["Run: pip install kafka-python"],
            )
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=group_id,
            auto_offset_reset=kwargs.get("auto_offset_reset", "earliest"),
            consumer_timeout_ms=timeout_ms,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        records: list[dict[str, Any]] = []
        try:
            for msg in consumer:
                val = msg.value
                records.append(val if isinstance(val, dict) else {"value": val})
                if len(records) >= max_messages:
                    break
        finally:
            consumer.close()
        return pd.json_normalize(records) if records else pd.DataFrame()

    @staticmethod
    def _detect_format_from_path(path_str: str) -> str:
        suffix = Path(path_str).suffix.lower()
        if suffix in SUPPORTED_FILE_FORMATS:
            return SUPPORTED_FILE_FORMATS[suffix]
        raise DataIngestionError(
            f"Unable to infer file format from extension '{suffix}' on '{path_or_hint(path_str)}'.",
            suggestions=["Pass explicit source_format (e.g. source_format='csv')."],
        )

    @staticmethod
    def _looks_like_api_endpoint(url: str, source_format: str | None) -> bool:
        if source_format and source_format in SUPPORTED_FILE_FORMATS.values():
            return False
        suffix = Path(urlparse(url).path).suffix.lower()
        return suffix not in SUPPORTED_FILE_FORMATS

    @staticmethod
    def _open_sqlite_readonly(db_path: str) -> sqlite3.Connection:
        """Open a SQLite file read-only at the OS/file-URI level (falls back for ':memory:')."""
        if db_path == ":memory:":
            return sqlite3.connect(":memory:")
        from urllib.parse import quote as _urlquote

        try:
            uri = f"file:{_urlquote(str(Path(db_path).expanduser().resolve()), safe='/')}?mode=ro"
            return sqlite3.connect(uri, uri=True)
        except sqlite3.OperationalError:
            # Fallback: open normally — the C-level authorizer installed by the caller still
            # enforces read-only semantics even if the URI form is unavailable on this platform.
            logger.warning("sqlite_readonly_uri_open_failed", path=db_path)
            return sqlite3.connect(db_path)

    @staticmethod
    def _first_sqlite_table_query(conn: sqlite3.Connection) -> str:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            raise DataIngestionError("SQLite database contains no user tables.")
        return f'SELECT * FROM {_quote_sql_identifier(str(row[0]))}'

    @staticmethod
    def _sanitize_uri(uri: str) -> str:
        try:
            parsed = urlparse(uri)
        except ValueError:
            return "<invalid-uri>"
        netloc = parsed.netloc
        if "@" in netloc:
            userinfo, hostinfo = netloc.rsplit("@", 1)
            if ":" in userinfo:
                user, _ = userinfo.split(":", 1)
                netloc = f"{user}:***@{hostinfo}"
        query = parsed.query
        if query:
            pairs = parse_qsl(query, keep_blank_values=True)
            redacted = [
                (k, "***" if k.lower() in _SENSITIVE_QUERY_KEYS else v)
                for k, v in pairs
            ]
            query = urlencode(redacted)
        if netloc != parsed.netloc or query != parsed.query:
            return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, query, parsed.fragment))
        return uri

    @staticmethod
    def _derive_name(source_label: str) -> str:
        if source_label.startswith("in_memory:"):
            return source_label.replace(":", "_")
        stem = Path(urlparse(source_label).path).stem
        return stem or "ingested_dataset"

    @staticmethod
    def _safe_scalar(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        return value if isinstance(value, (str, int, float, bool)) else str(value)

    @staticmethod
    def _is_hashable_series(series: pd.Series) -> bool:
        try:
            series.nunique(dropna=True)
            return True
        except TypeError:
            return False


def path_or_hint(path_str: str) -> str:
    """Return a concise representation of a path for error reporting."""
    return path_str if len(path_str) <= 120 else f"...{path_str[-117:]}"


def fast_read_tabular(path: str | Path, *, backend: str = "auto") -> tuple[pd.DataFrame, str]:
    """Read CSV or Parquet using Polars/DuckDB when available, with transparent pandas fallback."""
    p = Path(path)
    ext = p.suffix.lower()
    use_polars = backend == "polars" or (backend == "auto" and is_package_available("polars"))
    use_duckdb = backend == "duckdb" or (backend == "auto" and not use_polars and is_package_available("duckdb"))

    if use_polars:
        import polars as pl

        if ext == ".parquet":
            return pl.read_parquet(p).to_pandas(), "polars"
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            return pl.read_csv(p, separator=sep).to_pandas(), "polars"

    if use_duckdb:
        import duckdb

        if ext == ".parquet":
            return duckdb.read_parquet(str(p)).df(), "duckdb"
        if ext in (".csv", ".tsv"):
            sep_opt = "\t" if ext == ".tsv" else None
            rel = duckdb.read_csv(str(p), sep=sep_opt) if sep_opt else duckdb.read_csv(str(p))
            return rel.df(), "duckdb"

    if ext == ".parquet":
        return pd.read_parquet(p), "pandas"
    sep = "\t" if ext == ".tsv" else ","
    return pd.read_csv(p, sep=sep), "pandas"

