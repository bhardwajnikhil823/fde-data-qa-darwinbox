# Write-up — AI-Powered Data Q&A (FDE Take-Home)

## Problem scoping & trade-offs
The core ambiguity was "answer analytical questions in plain English" — this
could mean generating arbitrary Python, or constraining the model to a safer,
narrower surface. I chose to route every question through **SQL over DuckDB**
rather than free-form Pandas/Python execution. Trade-off: SQL can't express
every possible transformation as flexibly as Python, but it's trivially
sandboxable (allow-list `SELECT`/`WITH`, block everything else) and every
uploaded file becomes a table, so cross-file questions are just JOINs — no
custom multi-file merge logic needed. For the database layer, I used
**DuckDB in-memory** as the default (zero-infra, fast for a demo) with an
optional switch to **MotherDuck** (managed cloud DuckDB) via one config token,
which gives a genuinely persistent/shareable cloud database without changing
any query code — useful once this needs to outlive a single session or be
shared across users, which a full production RDBMS would over-engineer for
this scope.

## Handling open-source model execution & guardrails
The app must run on Streamlit Community Cloud, which rules out a locally
hosted model process. I used **Qwen3.8-27B**, an open-weight model, served
through Groq's hosted inference API — this keeps the "open-source model"
requirement honest (it's not a closed-weight model like GPT-4/Claude) while
being deployable without infra. Because LLM-generated SQL can be wrong or
unsafe, I added three guardrails: (1) a regex-based allow/deny list that only
permits read-only `SELECT`/`WITH` statements and rejects DDL/DML and
multi-statement injection; (2) a bounded self-repair loop that feeds execution
errors back to the model (max 2 retries) instead of failing outright; (3) the
model is instructed to return a `CLARIFY:` message rather than hallucinate an
answer when the schema can't support the question. The generated SQL is
always shown to the user, which matters for an HR-data product like
Darwinbox's — analysts need to trust and audit the numbers, not just see them.

## What to build next
Priorities in order: (1) **Multi-session workspaces** — today the app is a
single continuous workspace per deployment (one shared table namespace, one
"Clear session" reset); the natural next step is named, independently
saveable sessions (auto-named, renameable, deletable — like ChatGPT's chat
list), each with its own isolated tables and Q&A history, scoped per
authenticated user; (2) **Enterprise RBAC** — map org/role permissions to
which tables or rows a given user's questions can touch, essential before any
real customer data flows through this; (3) **caching** of identical
question+file-hash pairs to cut latency and inference cost; (4) support for
**larger-than-memory datasets** by leaning further into MotherDuck's cloud
storage and DuckDB's out-of-core execution instead of holding everything in
session memory; (5) multi-turn conversational context for follow-up
questions; (6) a small regression-test harness of known question→SQL pairs to
safely iterate on the prompt/model over time.
