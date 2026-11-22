"""Platform-wide constants, limits, mappings, and registry defaults for OmniForge."""

from __future__ import annotations

from typing import Final

# Default randomness & reproducibility
DEFAULT_RANDOM_SEED: Final[int] = 42
DEFAULT_TEST_SIZE: Final[float] = 0.2
DEFAULT_CV_FOLDS: Final[int] = 5

# Supported file extensions mapped to canonical format identifiers
SUPPORTED_FILE_FORMATS: Final[dict[str, str]] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".ftr": "feather",
    ".arrow": "arrow",
    ".ipc": "arrow",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".orc": "orc",
    ".avro": "avro",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".hdf": "hdf5",
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sas7bdat": "sas",
    ".xpt": "sas",
    ".sav": "spss",
    ".dta": "stata",
}

SUPPORTED_DB_SCHEMES: Final[tuple[str, ...]] = (
    "postgresql",
    "postgres",
    "mysql",
    "sqlite",
    "mssql",
    "snowflake",
    "bigquery",
    "redshift",
    "clickhouse",
    "duckdb",
    "mongodb",
)

SUPPORTED_CLOUD_SCHEMES: Final[tuple[str, ...]] = (
    "s3",
    "s3a",
    "gs",
    "gcs",
    "az",
    "abfs",
    "abfss",
    "adl",
    "http",
    "https",
)

# Regional grid carbon intensity defaults (gCO2eq per kWh)
CARBON_INTENSITY_BY_COUNTRY: Final[dict[str, float]] = {
    "DEU": 380.0,
    "USA": 379.0,
    "GBR": 233.0,
    "FRA": 56.0,
    "SWE": 41.0,
    "NOR": 28.0,
    "CAN": 120.0,
    "CHE": 35.0,
    "NLD": 325.0,
    "ESP": 175.0,
    "ITA": 315.0,
    "JPN": 462.0,
    "AUS": 549.0,
    "IND": 713.0,
    "CHN": 582.0,
    "BRA": 98.0,
    "SGP": 408.0,
    "GLOBAL": 475.0,
}

# Secret key patterns redacted in structured logs and audit trails
SENSITIVE_KEY_PATTERNS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "authorization",
    "auth_header",
    "credential",
    "ssn",
    "credit_card",
    "bearer",
    "connection_string",
)

# Standard RBAC roles and permissions
RBAC_ROLE_PERMISSIONS: Final[dict[str, frozenset[str]]] = {
    "admin": frozenset(
        {
            "data:read",
            "data:write",
            "data:delete",
            "model:read",
            "model:train",
            "model:deploy",
            "model:promote",
            "model:delete",
            "pipeline:read",
            "pipeline:execute",
            "pipeline:write",
            "mlops:read",
            "mlops:write",
            "copilot:use",
            "report:generate",
            "audit:read",
            "security:manage",
        }
    ),
    # Platform-level operator (P1-12). Same permission set as tenant ``admin`` PLUS the
    # out-of-band privilege to bypass tenant isolation. Tenant ``admin`` is tenant-scoped;
    # cross-tenant access requires this role explicitly. Grant sparingly (IdP group mapping).
    "platform_admin": frozenset(
        {
            "data:read",
            "data:write",
            "data:delete",
            "model:read",
            "model:train",
            "model:deploy",
            "model:promote",
            "model:delete",
            "pipeline:read",
            "pipeline:execute",
            "pipeline:write",
            "mlops:read",
            "mlops:write",
            "copilot:use",
            "report:generate",
            "audit:read",
            "security:manage",
        }
    ),
    "data_scientist": frozenset(
        {
            "data:read",
            "data:write",
            "model:read",
            "model:train",
            "model:deploy",
            "pipeline:read",
            "pipeline:execute",
            "pipeline:write",
            "mlops:read",
            "mlops:write",
            "copilot:use",
            "report:generate",
        }
    ),
    "mlops_engineer": frozenset(
        {
            "data:read",
            "model:read",
            "model:deploy",
            "model:promote",
            "pipeline:read",
            "pipeline:execute",
            "mlops:read",
            "mlops:write",
            "audit:read",
            "report:generate",
        }
    ),
    "analyst": frozenset(
        {
            "data:read",
            "model:read",
            "pipeline:read",
            "pipeline:execute",
            "mlops:read",
            "copilot:use",
            "report:generate",
        }
    ),
    "viewer": frozenset(
        {
            "data:read",
            "model:read",
            "pipeline:read",
            "mlops:read",
        }
    ),
}

# Built-in industry domains
BUILTIN_DOMAINS: Final[tuple[str, ...]] = (
    "finance",
    "energy",
    "healthcare",
    "manufacturing",
    "retail",
    "transportation",
    "agriculture",
    "climate",
    "telecom",
    "hr",
    "education",
    "real_estate",
    "sports",
    "cybersecurity",
    "government",
    "bioinformatics",
    "materials",
    "media",
)
