"""Load named SQL queries from the shared analysis.sql file."""

from __future__ import annotations

import re
from pathlib import Path

_QUERY_BLOCK = re.compile(
    r"^-- (?P<title>\d+\.\s.+)$\n(?P<body>(?:.*\n)*?)(?=^-- \d+\.|\Z)",
    re.MULTILINE,
)


def load_named_queries(sql_path: Path) -> dict[str, str]:
    """Parse `sql/analysis.sql` into a title -> SQL string map (no trailing semicolon)."""
    text = sql_path.read_text(encoding="utf-8")
    queries: dict[str, str] = {}

    for match in _QUERY_BLOCK.finditer(text):
        title = match.group("title").strip()
        body = match.group("body")
        lines = [line for line in body.splitlines() if not line.strip().startswith("--")]
        sql = "\n".join(lines).strip().rstrip(";")
        if sql.upper().startswith("SELECT"):
            queries[title] = sql

    if not queries:
        raise ValueError(f"No SELECT queries found in {sql_path}")

    return queries
