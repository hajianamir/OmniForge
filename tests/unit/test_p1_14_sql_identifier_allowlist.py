"""P1-14 regression tests: identifier allowlist + readonly SQL in fallback SQL/DuckDB ingestion."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pytest

from omniforge.core.exceptions import DataIngestionError, SecurityError
from omniforge.data.ingestion import DataIngestionEngine, _quote_sql_identifier


@pytest.fixture()
def sqlite_db(tmp_path: Path) -> Path:
    db = tmp_path / "sample.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)")
    conn.execute("INSERT INTO users (email) VALUES ('a@example.com'), ('b@example.com')")
    conn.execute('CREATE TABLE "we""ird" (id INTEGER)')
    conn.commit()
    conn.close()
    return db


@pytest.fixture()
def duckdb_db(tmp_path: Path) -> Path:
    db = tmp_path / "sample.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("CREATE TABLE users (id INTEGER, email VARCHAR)")
    conn.execute("INSERT INTO users VALUES (1, 'a@example.com'), (2, 'b@example.com')")
    conn.close()
    return db


def _rows(engine_result) -> int:
    return len(engine_result.data)


def test_quote_identifier_plain() -> None:
    assert _quote_sql_identifier("users") == '"users"'
    assert _quote_sql_identifier("main.users") == '"main"."users"'
    assert _quote_sql_identifier('we"ird') == '"we""ird"'


@pytest.mark.parametrize("bad", ["", "   ", "a\x00b", "x" * 129, 123])
def test_quote_identifier_rejects_bad(bad) -> None:
    with pytest.raises(SecurityError):
        _quote_sql_identifier(bad)  # type: ignore[arg-type]


def test_sqlite_table_ingest_happy_path(sqlite_db: Path) -> None:
    result = DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
        str(sqlite_db), table="users"
    )
    assert _rows(result) == 2


def test_sqlite_table_ingest_embedded_quote_name(sqlite_db: Path) -> None:
    result = DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
        str(sqlite_db), table='we"ird'
    )
    assert _rows(result) == 0  # table exists but has no rows; ingestion must not crash


def test_sqlite_table_not_found(sqlite_db: Path) -> None:
    with pytest.raises(DataIngestionError, match="not found"):
        DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
            str(sqlite_db), table="users; DROP TABLE users"
        )


def test_sqlite_injection_attempt_rejected_and_table_intact(sqlite_db: Path) -> None:
    with pytest.raises(DataIngestionError):
        DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
            str(sqlite_db), table='users" OR "1"="1'
        )
    conn = sqlite3.connect(sqlite_db)
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 2
    conn.close()


def test_sqlite_readonly_query_selects_only(sqlite_db: Path) -> None:
    with pytest.raises((SecurityError, DataIngestionError)):
        DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
            str(sqlite_db), query="DELETE FROM users"
        )
    with pytest.raises((SecurityError, DataIngestionError)):
        DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
            str(sqlite_db), query="DROP TABLE users"
        )
    with pytest.raises((SecurityError, DataIngestionError)):
        DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
            str(sqlite_db), query="PRAGMA journal_mode=WAL"
        )
    with pytest.raises((SecurityError, DataIngestionError)):
        DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
            str(sqlite_db), query="SELECT * FROM users; DROP TABLE users"
        )
    result = DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
        str(sqlite_db), query="SELECT * FROM users"
    )
    assert _rows(result) == 2
    conn = sqlite3.connect(sqlite_db)
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 2
    conn.close()


def test_duckdb_table_ingest_happy_path(duckdb_db: Path) -> None:
    result = DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
        f"duckdb:///{duckdb_db}", table="users"
    )
    assert _rows(result) == 2


def test_duckdb_table_not_found_and_injection_rejected(duckdb_db: Path) -> None:
    for bad in ("no_such_table", "users; DROP TABLE users", 'users" OR "1"="1'):
        with pytest.raises(DataIngestionError):
            DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
                f"duckdb:///{duckdb_db}", table=bad
            )
    conn = duckdb.connect(str(duckdb_db), read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 2
    conn.close()


def test_duckdb_readonly_query_enforced(duckdb_db: Path) -> None:
    with pytest.raises((SecurityError, DataIngestionError)):
        DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
            f"duckdb:///{duckdb_db}", query="DELETE FROM users"
        )
    with pytest.raises((SecurityError, DataIngestionError)):
        DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
            f"duckdb:///{duckdb_db}", query="ATTACH 'x.db'"
        )
    result = DataIngestionEngine(optimize_dtypes=False, compute_checksum=False).ingest(
        f"duckdb:///{duckdb_db}", query="SELECT * FROM users WHERE id = 1"
    )
    assert _rows(result) == 1
