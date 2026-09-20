# AI-Powered Data Q&A — FDE

**Live app:** https://fde-data-app-darwinbox-mibbpnrv9urnms42ezpmtu.streamlit.app/

Upload multiple CSV/Excel files and ask analytical questions about them in
plain English. The app translates your question into a validated, read-only
SQL query (run over DuckDB), executes it, and shows the answer as a table
and/or chart.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit | Fast to build, easy to host/share |
| Query engine | DuckDB (in-memory, or **MotherDuck** cloud when a token is set) | Each uploaded file becomes a SQL table; SQL is easy to validate/sandbox vs. arbitrary Pandas code; JOINs give cross-file analysis for free |
| NL → Query | **Qwen3.8-27B** (open-weight) via **Groq** hosted inference | Satisfies the "open-source model" requirement while staying deployable on Streamlit Community Cloud (no local model server) |
| Charts | Plotly, chosen by a rule-based heuristic on the result shape (not the LLM) | Cheaper and more deterministic than asking the LLM to also emit chart specs |

## How it works

1. Each uploaded file is registered as its own DuckDB table (`register_dataframe`).
2. On a question, the app sends the LLM the table schemas + a few sample rows
   + the question, and asks it to return **only** a `SELECT`/`WITH` SQL query.
3. Before running, the SQL is checked against a guardrail
   (`validate_readonly_sql`) that blocks any non-read-only statement
   (`INSERT`/`UPDATE`/`DROP`/`ATTACH`/etc.) and rejects multiple statements.
4. If execution fails, the error is fed back to the model and it retries
   (up to 2 times) — a self-repair loop.
5. The generated SQL is always shown to the user for transparency.
6. If the result shape looks chart-worthy (≥2 rows, a numeric column, a
   sensible category/date column), a Plotly chart is rendered automatically.

## Setup & run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and add your GROQ_API_KEY
# (get a free key at https://console.groq.com)

streamlit run app.py
```

Open http://localhost:8501, upload one or more CSV/Excel files, and ask a
question (e.g. *"What is the average tenure by department?"* or, with two
files loaded, *"Compare total headcount between the two files"*).

Sample multi-file HR datasets (designed to exercise cross-file joins) are
included in `sample_data/` — see `sample_data/README` context in the repo,
or just upload all six files and try the example questions in the
Testing section below.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a new app on https://share.streamlit.io pointing at `app.py`.
3. In the app's **Secrets** settings, paste the contents of
   `.streamlit/secrets.toml.example` with your real `GROQ_API_KEY`
   (and `MOTHERDUCK_TOKEN` if you want persistent cloud storage instead of
   per-session in-memory DuckDB).

Note: the deployed app and your local dev environment share the same
MotherDuck database/workspace if you use the same `MOTHERDUCK_TOKEN` in
both — by design, given this project's single-workspace scope (see "What
I'd build next"). Use "🗑 Clear session" to reset between test passes.

## Testing

Try uploading all files from `sample_data/` and asking:
- *"What is the average tenure by department?"*
- *"Show average performance rating by department"* (cross-file join:
  `employees` + `performance_reviews`)
- *"Compare headcount trend over months for Engineering"* (chart)
- *"Delete all employee records"* (should be blocked by the guardrail)

## Guardrails (open-source model execution safety)

- Only `SELECT`/`WITH` statements are allowed to execute — enforced with a
  regex allow/deny list before any query touches DuckDB.
- No arbitrary Python/Pandas code from the model is ever `exec`'d.
- Ambiguous questions cause the model to return a `CLARIFY:` message instead
  of guessing, which the UI surfaces as an info prompt rather than a wrong
  answer.
- Generated SQL is always shown, so answers are auditable — important for an
  HR-data context where trust in numbers matters.

## Delta (beyond a naive AI-generated app)

- **SQL-only, validated guardrail layer** instead of letting the model run
  arbitrary code.
- **Self-repair loop**: failed queries are retried with the error fed back to
  the model, rather than failing the user's question outright.
- **Transparency**: generated SQL is always visible, and multi-attempt
  queries are flagged as "self-repaired."
- **Heuristic (non-LLM) charting**: deterministic, cheaper, and more
  reliable than asking the model for chart specs.

## What I'd build next

- **Multi-session workspaces**: today the app is one continuous workspace
  (single shared table namespace + a single "Clear session" reset). The
  natural next step is named, independently saveable sessions — auto-named
  by default, renameable, deletable, like a ChatGPT-style chat list — each
  with its own isolated tables and Q&A history, scoped per authenticated
  user.
- Enterprise RBAC / row-level security (map Darwinbox org roles to which
  tables/rows a user's questions can touch).
- Result & query caching (hash of file content + question) to cut latency
  and LLM cost on repeat questions.
- Support for larger-than-memory datasets via MotherDuck's cloud storage
  and DuckDB's out-of-core execution, instead of loading everything into
  session memory.
- Multi-turn conversation context (follow-up questions referencing the
  previous result).
- An evaluation harness with a small set of known question→SQL pairs to
  regression-test model/prompt changes.

