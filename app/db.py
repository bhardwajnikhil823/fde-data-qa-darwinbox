"""
DuckDB / MotherDuck connection layer.

Design decision: we use DuckDB (locally in-memory, or MotherDuck when a token
is supplied) as the single query engine for every uploaded file. Each uploaded
CSV/Excel file becomes one SQL table. This lets the LLM reason in SQL (which
we can validate and sandbox) instead of generating arbitrary Python/Pandas
code, and lets us answer cross-file questions with ordinary SQL JOINs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import pandas as pd
import streamlit as st


@dataclass
class TableInfo:
    name: str
    columns: list[str]
    dtypes: list[str]
    row_count: int
    sample_rows: pd.DataFrame


def sanitize_table_name(filename: str) -> str:
    """Turn an uploaded filename into a safe, unique SQL identifier."""
    base = filename.rsplit(".", 1)[0]
    base = re.sub(r"[^0-9a-zA-Z_]", "_", base).strip("_").lower()
    if not base:
        base = "table"
    if base[0].isdigit():
        base = f"t_{base}"
    return base


@st.cache_resource(show_spinner=False)
def get_connection(motherduck_token: str | None = None) -> duckdb.DuckDBPyConnection:
    """
    Create (once per session) a DuckDB connection.

    - If a MotherDuck token is provided, connect to a cloud-hosted MotherDuck
      database (persistent, shareable, genuinely "cloud DB").
    - Otherwise fall back to a local in-memory DuckDB, which is still fully
      functional for a demo/single session but not persistent across restarts.
    """
    if motherduck_token:
        con = duckdb.connect(f"md:?motherduck_token={motherduck_token}")
        con.execute("CREATE DATABASE IF NOT EXISTS fde_assignment")
        con.execute("USE fde_assignment")
        return con
    return duckdb.connect(database=":memory:")


def drop_table(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Remove a table (e.g. when its source file is removed from the uploader)."""
    con.execute(f'DROP TABLE IF EXISTS "{table_name}"')


def register_dataframe(
    con: duckdb.DuckDBPyConnection,
    filename: str,
    df: pd.DataFrame,
    preferred_name: str | None = None,
    existing_names: set[str] | None = None,
) -> TableInfo:
    """
    Register a pandas DataFrame as a queryable table and return its schema info.

    `preferred_name`, if given, is used as-is (no uniqueness suffix) — this is
    how the caller tells us "this file was already registered as this table,
    just replace its contents" instead of minting a new `_2`/`_3` table each
    time the same file is re-uploaded.

    `existing_names`, if given, is the set of table names *this session*
    already owns (e.g. from other uploaded files) — used to avoid name clashes
    without being confused by stale/leftover tables from previous sessions
    (which can happen with a persistent cloud DB like MotherDuck).
    """
    table_name = preferred_name or sanitize_table_name(filename)

    if not preferred_name:
        existing = existing_names if existing_names is not None else set()
        original = table_name
        i = 2
        while table_name in existing:
            table_name = f"{original}_{i}"
            i += 1

    con.register("tmp_df_view", df)
    con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM tmp_df_view')
    con.unregister("tmp_df_view")

    info = describe_table(con, table_name)
    save_file_mapping(con, filename, table_name)
    return info


def describe_table(con: duckdb.DuckDBPyConnection, table_name: str) -> TableInfo:
    """Build a TableInfo by introspecting an existing table's schema/sample rows."""
    schema_df = con.execute(f'DESCRIBE "{table_name}"').fetchdf()
    row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    sample = con.execute(f'SELECT * FROM "{table_name}" LIMIT 5').fetchdf()

    return TableInfo(
        name=table_name,
        columns=schema_df["column_name"].tolist(),
        dtypes=schema_df["column_type"].tolist(),
        row_count=row_count,
        sample_rows=sample,
    )


def list_tables_schema(tables: dict[str, TableInfo]) -> str:
    """Render a compact schema description for all registered tables, for use in LLM prompts."""
    chunks = []
    for t in tables.values():
        cols = ", ".join(f"{c} ({d})" for c, d in zip(t.columns, t.dtypes))
        sample = t.sample_rows.to_csv(index=False)
        chunks.append(
            f"Table: {t.name} ({t.row_count} rows)\n"
            f"Columns: {cols}\n"
            f"Sample rows:\n{sample}"
        )
    return "\n\n".join(chunks)


ALLOWED_SQL_PREFIX = re.compile(r"^\s*(WITH|SELECT)\b", re.IGNORECASE)
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|PRAGMA|EXPORT|IMPORT|CALL)\b",
    re.IGNORECASE,
)


def validate_readonly_sql(sql: str) -> tuple[bool, str]:
    """
    Guardrail: only allow read-only SELECT/WITH queries to run.
    Blocks any statement that could mutate data or the filesystem.
    """
    sql_stripped = sql.strip().rstrip(";")
    if ";" in sql_stripped:
        return False, "Multiple statements are not allowed."
    if not ALLOWED_SQL_PREFIX.match(sql_stripped):
        return False, "Only SELECT / WITH (read-only) queries are allowed."
    if FORBIDDEN_KEYWORDS.search(sql_stripped):
        return False, "Query contains a forbidden keyword (only read-only SELECTs are allowed)."
    return True, sql_stripped


def run_sql(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


# ---------------------------------------------------------------------------
# Lightweight session persistence.
#
# The DuckDB/MotherDuck connection is cached at the *server process* level
# (see `get_connection`), so uploaded table data already survives a browser
# page refresh. What does NOT survive is Streamlit's `st.session_state`
# (per-browser-tab bookkeeping). To make a refresh feel like "my workspace is
# still here" instead of an incognito reset, we mirror the minimal state we
# need (filename -> table mapping, and Q&A history) into two small internal
# tables, prefixed `_app_` so they're excluded from the LLM's schema view.
# ---------------------------------------------------------------------------
FILE_REGISTRY_TABLE = "_app_file_registry"
HISTORY_TABLE = "_app_history"
META_PREFIX = "_app_"


def _ensure_meta_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f'CREATE TABLE IF NOT EXISTS "{FILE_REGISTRY_TABLE}" '
        "(filename VARCHAR, table_name VARCHAR)"
    )
    con.execute(
        f'CREATE TABLE IF NOT EXISTS "{HISTORY_TABLE}" '
        "(id INTEGER, ts TIMESTAMP, question VARCHAR, sql VARCHAR)"
    )


def save_file_mapping(con: duckdb.DuckDBPyConnection, filename: str, table_name: str) -> None:
    _ensure_meta_tables(con)
    con.execute(f'DELETE FROM "{FILE_REGISTRY_TABLE}" WHERE filename = ?', [filename])
    con.execute(
        f'INSERT INTO "{FILE_REGISTRY_TABLE}" VALUES (?, ?)', [filename, table_name]
    )


def remove_file_mapping(con: duckdb.DuckDBPyConnection, filename: str) -> None:
    _ensure_meta_tables(con)
    con.execute(f'DELETE FROM "{FILE_REGISTRY_TABLE}" WHERE filename = ?', [filename])


def load_file_mapping(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    _ensure_meta_tables(con)
    rows = con.execute(f'SELECT filename, table_name FROM "{FILE_REGISTRY_TABLE}"').fetchall()
    return {r[0]: r[1] for r in rows}


def rehydrate_tables(con: duckdb.DuckDBPyConnection) -> dict[str, TableInfo]:
    """Rebuild {table_name: TableInfo} for every table we currently own, from the DB itself."""
    mapping = load_file_mapping(con)
    all_tables = {
        r[0]
        for r in con.execute("select table_name from information_schema.tables").fetchall()
    }
    tables: dict[str, TableInfo] = {}
    for table_name in set(mapping.values()):
        if table_name in all_tables:
            tables[table_name] = describe_table(con, table_name)
    return tables


def save_history_entry(con: duckdb.DuckDBPyConnection, question: str, sql: str | None) -> None:
    _ensure_meta_tables(con)
    next_id = con.execute(f'SELECT COALESCE(MAX(id), 0) + 1 FROM "{HISTORY_TABLE}"').fetchone()[0]
    con.execute(
        f'INSERT INTO "{HISTORY_TABLE}" VALUES (?, now(), ?, ?)',
        [next_id, question, sql],
    )


def load_history(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Load past Q&A entries and best-effort re-execute their SQL to regenerate results."""
    _ensure_meta_tables(con)
    rows = con.execute(
        f'SELECT question, sql FROM "{HISTORY_TABLE}" ORDER BY id DESC'
    ).fetchall()
    entries = []
    for question, sql in rows:
        if not sql:
            entries.append({"question": question, "sql": None, "df": None, "attempts": None,
                             "error": "No SQL was recorded for this entry."})
            continue
        try:
            df = run_sql(con, sql)
            entries.append({"question": question, "sql": sql, "df": df, "attempts": 1, "error": None})
        except Exception as exc:  # noqa: BLE001
            entries.append({"question": question, "sql": sql, "df": None, "attempts": None,
                             "error": f"Could not re-run this historical query: {exc}"})
    return entries


def clear_session(con: duckdb.DuckDBPyConnection) -> None:
    """Drop every user-owned table (data + our own metadata tables) — a full workspace reset."""
    tables = con.execute("select table_name from information_schema.tables").fetchall()
    for (table_name,) in tables:
        con.execute(f'DROP TABLE IF EXISTS "{table_name}"')

