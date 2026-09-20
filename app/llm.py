"""
LLM query pipeline: turns a natural-language question + table schemas into a
read-only SQL query, using an open-source model (Llama 3.x / Qwen) served via
Groq's hosted inference API.

Why an API instead of a local model?
The app targets Streamlit Community Cloud, which cannot run a local Ollama
process. Groq serves genuinely open-weight models (Llama-3.x, etc.) over an
API, so we keep the "open-source model" requirement while being deployable.
"""
from __future__ import annotations

import re

from groq import Groq

MODEL_NAME = "qwen/qwen3.8-27b"  # open-weight Qwen model, hosted on Groq

SYSTEM_PROMPT = """You are a careful data analyst assistant. You are given the \
schema (and sample rows) of one or more DuckDB SQL tables, plus a user's \
natural-language question. Your job is to write a single **read-only** DuckDB \
SQL query (SELECT / WITH only) that answers the question.

Rules:
- Only use tables/columns that actually exist in the provided schema.
- If the question requires combining multiple tables, use JOINs on sensible \
matching columns.
- Never use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/COPY/PRAGMA.
- Return ONLY the SQL query, wrapped in a ```sql code block. No explanation.
- If the question is ambiguous or cannot be answered from the given schema, \
respond with a ```sql block containing: SELECT 'CLARIFY: <your clarifying question>' AS message
- Prefer aggregate queries (GROUP BY) for "average/total/count by X" style questions.
- If the result would sensibly be visualized (a trend over time, a comparison \
across categories, a distribution), make sure the SELECT includes both the \
categorical/time column and the numeric measure so it can be charted directly.
"""


def _extract_sql(text: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: assume the whole response is SQL.
    return text.strip()


def build_user_prompt(schema_description: str, question: str, error_context: str | None = None) -> str:
    prompt = f"Schema:\n{schema_description}\n\nQuestion: {question}\n"
    if error_context:
        prompt += (
            f"\nYour previous SQL attempt failed with this error:\n{error_context}\n"
            "Please fix the query and try again."
        )
    return prompt


def generate_sql(
    api_key: str,
    schema_description: str,
    question: str,
    error_context: str | None = None,
) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(schema_description, question, error_context)},
        ],
        temperature=0,
        max_tokens=800,
    )
    content = response.choices[0].message.content
    return _extract_sql(content)


def generate_sql_with_retry(
    api_key: str,
    schema_description: str,
    question: str,
    executor,
    max_retries: int = 2,
):
    """
    Self-repair loop: generate SQL, try to run it, and if it errors, feed the
    error back to the model and retry (bounded by max_retries).

    `executor` is a callable(sql:str) -> pandas.DataFrame that raises on failure.
    Returns (sql, dataframe, attempts).
    """
    error_context = None
    last_sql = None
    for attempt in range(1, max_retries + 2):
        sql = generate_sql(api_key, schema_description, question, error_context)
        last_sql = sql
        try:
            df = executor(sql)
            return sql, df, attempt
        except Exception as exc:
            if attempt == max_retries + 1:
                raise
            error_context = str(exc)
    raise RuntimeError(f"Failed to produce a working query. Last SQL:\n{last_sql}")
