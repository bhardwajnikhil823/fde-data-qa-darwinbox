"""
Heuristic chart builder.

Rather than asking the LLM to also emit a chart spec (extra cost, extra
failure surface), we use a lightweight rule-based heuristic on the *result
dataframe's shape/dtypes* to decide whether/how to chart it. This is cheaper,
deterministic, and reliable.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px


def is_chart_worthy(df: pd.DataFrame) -> bool:
    if df.shape[0] < 2 or df.shape[1] < 2:
        return False
    numeric_cols = df.select_dtypes(include="number").columns
    return len(numeric_cols) >= 1 and df.shape[1] <= 6


def build_chart(df: pd.DataFrame):
    """Pick a reasonable chart type based on column dtypes and row count."""
    numeric_cols = list(df.select_dtypes(include="number").columns)
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        return None

    y_col = numeric_cols[0]

    # Try to detect a date/time-like column for a trend line.
    date_col = None
    for c in non_numeric_cols:
        if any(k in c.lower() for k in ("date", "month", "year", "week", "time")):
            date_col = c
            break

    if date_col:
        return px.line(df.sort_values(date_col), x=date_col, y=y_col, markers=True,
                        title=f"{y_col} over {date_col}")

    if non_numeric_cols:
        x_col = non_numeric_cols[0]
        if df.shape[0] <= 15:
            return px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        return px.line(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")

    # Only numeric columns: fall back to a histogram of the first one.
    return px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
