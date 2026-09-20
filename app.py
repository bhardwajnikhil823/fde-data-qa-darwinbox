"""
Forward Deployed Engineer take-home: AI-Powered Data Q&A Web App.

Upload multiple CSV/Excel files, ask analytical questions in plain English,
get back correct answers (and charts, when relevant) — powered by an
open-source LLM (Qwen, via Groq) generating validated, read-only SQL
against a DuckDB (optionally MotherDuck cloud) engine.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.charting import build_chart, is_chart_worthy
from app.db import (
    TableInfo,
    clear_session,
    drop_table,
    get_connection,
    list_tables_schema,
    load_file_mapping,
    load_history,
    register_dataframe,
    rehydrate_tables,
    remove_file_mapping,
    run_sql,
    save_history_entry,
    validate_readonly_sql,
)
from app.llm import generate_sql_with_retry

st.set_page_config(page_title="AI Data Q&A", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Secrets / config
# ---------------------------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
MOTHERDUCK_TOKEN = st.secrets.get("MOTHERDUCK_TOKEN", "") or None

con = get_connection(MOTHERDUCK_TOKEN)

# ---------------------------------------------------------------------------
# Session state
#
# Note: the DuckDB/MotherDuck connection itself is cached at the *process*
# level (see get_connection), so table data already survives a page refresh.
# What resets on refresh is st.session_state (per-tab). On the very first run
# of a fresh session we rehydrate our bookkeeping (which files map to which
# tables, and past Q&A) directly from the database, so a refreshed page still
# shows your data and history instead of looking like a wiped/incognito tab.
#
# We deliberately keep this to a single continuous workspace (not multiple
# named/saved chats like a ChatGPT sidebar) — see README "what's next" for
# why that's out of scope for this assignment.
# ---------------------------------------------------------------------------
if "_rehydrated" not in st.session_state:
    st.session_state.tables: dict[str, TableInfo] = rehydrate_tables(con)
    st.session_state.file_to_table = {
        fname: tname
        for fname, tname in load_file_mapping(con).items()
        if tname in st.session_state.tables
    }
    st.session_state.history = load_history(con)
    # Filenames whose contents we've already ingested into the DB *this live
    # session*. Used to avoid re-reading/re-writing every uploaded file on
    # every single rerun (e.g. expanding a preview, typing a question) —
    # only genuinely newly-added files get (re)processed.
    st.session_state._processed_names = set()
    st.session_state._rehydrated = True

# ---------------------------------------------------------------------------
# Sidebar: upload + schema preview
# ---------------------------------------------------------------------------
st.sidebar.title("📁 Data")

if st.sidebar.button("🗑 Clear session (reset everything)"):
    clear_session(con)
    for key in ("tables", "file_to_table", "history", "_processed_names", "_rehydrated"):
        st.session_state.pop(key, None)
    st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV / Excel files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

current_names = {f.name for f in uploaded_files} if uploaded_files else set()

# Only treat a file as "removed" if it was previously seen through the
# uploader widget *in this live session* — rehydrated/persisted files that
# simply aren't re-selected after a refresh should NOT be dropped (the
# browser can't restore a file input's selection after reload, so they'll
# never re-appear in current_names even though the table itself persists).
for removed_name in st.session_state._processed_names - current_names:
    table_name = st.session_state.file_to_table.pop(removed_name, None)
    if table_name:
        drop_table(con, table_name)
        remove_file_mapping(con, removed_name)
        st.session_state.tables.pop(table_name, None)
    st.session_state._processed_names.discard(removed_name)

# Only (re)register files that are newly selected — this is the fix for the
# "expanding a preview takes 10 seconds" issue: previously every rerun
# re-read + re-wrote *every* currently-selected file to MotherDuck, even when
# nothing about the upload actually changed.
newly_added = current_names - st.session_state._processed_names
if newly_added:
    for f in uploaded_files:
        if f.name not in newly_added:
            continue
        try:
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Failed to read {f.name}: {exc}")
            continue

        preferred_name = st.session_state.file_to_table.get(f.name)
        existing_names = set(st.session_state.file_to_table.values())
        info = register_dataframe(
            con, f.name, df, preferred_name=preferred_name, existing_names=existing_names
        )
        st.session_state.file_to_table[f.name] = info.name
        st.session_state.tables[info.name] = info
        st.session_state._processed_names.add(f.name)

if st.session_state.tables:
    st.sidebar.success(f"{len(st.session_state.tables)} table(s) loaded")
    # Reverse lookup so we can offer a per-table remove button, including for
    # tables rehydrated from a previous session (not currently in the
    # uploader widget at all).
    table_to_file = {v: k for k, v in st.session_state.file_to_table.items()}
    for name, info in st.session_state.tables.items():
        col_a, col_b = st.sidebar.columns([5, 1])
        with col_a, st.expander(f"🗂 {name} ({info.row_count} rows) — preview (first 5 rows)"):
            st.dataframe(info.sample_rows, use_container_width=True)
            st.caption("This is just a preview. Questions run against the full table.")
        with col_b:
            if st.button("✖", key=f"remove_{name}", help=f"Remove {name}"):
                drop_table(con, name)
                fname = table_to_file.get(name)
                if fname:
                    remove_file_mapping(con, fname)
                    st.session_state.file_to_table.pop(fname, None)
                    st.session_state._processed_names.discard(fname)
                st.session_state.tables.pop(name, None)
                st.rerun()
else:
    st.sidebar.info("Upload at least one file to get started.")

if not GROQ_API_KEY:
    st.sidebar.warning(
        "No GROQ_API_KEY found in secrets. Add it to `.streamlit/secrets.toml` "
        "(see `.streamlit/secrets.toml.example`) or your Streamlit Cloud app secrets."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📊 AI-Powered Data Q&A")
st.caption(
    "Upload files on the left, then ask questions in plain English — across "
    "one or multiple files. Powered by an open-source LLM generating validated "
    "read-only SQL over DuckDB."
)

question = st.text_input(
    "Ask a question about your data",
    placeholder="e.g. What is the average tenure by department? / Compare headcount across the two files",
)
ask_clicked = st.button("Ask", type="primary", disabled=not (question and st.session_state.tables and GROQ_API_KEY))

if ask_clicked:
    schema_desc = list_tables_schema(st.session_state.tables)

    def executor(sql: str) -> pd.DataFrame:
        ok, cleaned_or_reason = validate_readonly_sql(sql)
        if not ok:
            raise ValueError(f"Blocked by guardrail: {cleaned_or_reason}")
        return run_sql(con, cleaned_or_reason)

    with st.spinner("Thinking..."):
        try:
            sql, result_df, attempts = generate_sql_with_retry(
                GROQ_API_KEY, schema_desc, question, executor, max_retries=2
            )
            entry = {"question": question, "sql": sql, "df": result_df, "attempts": attempts, "error": None}
            save_history_entry(con, question, sql)
        except Exception as exc:  # noqa: BLE001
            entry = {"question": question, "sql": None, "df": None, "attempts": None, "error": str(exc)}
            save_history_entry(con, question, None)

    st.session_state.history.insert(0, entry)

# ---------------------------------------------------------------------------
# History / results (most recent first)
# ---------------------------------------------------------------------------
for entry in st.session_state.history:
    st.markdown(f"### ❓ {entry['question']}")
    if entry["error"]:
        st.error(entry["error"])
        continue

    df = entry["df"]

    # Clarification path: model asked a question back instead of answering.
    if df.shape == (1, 1) and str(df.iloc[0, 0]).startswith("CLARIFY:"):
        st.info(str(df.iloc[0, 0]).replace("CLARIFY:", "").strip())
        with st.expander("Generated SQL"):
            st.code(entry["sql"], language="sql")
        continue

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df, use_container_width=True)
    with col2:
        if is_chart_worthy(df):
            fig = build_chart(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("Generated SQL (transparency)"):
        st.code(entry["sql"], language="sql")
        if entry["attempts"] and entry["attempts"] > 1:
            st.caption(f"Self-repaired after {entry['attempts'] - 1} failed attempt(s).")

    st.divider()
