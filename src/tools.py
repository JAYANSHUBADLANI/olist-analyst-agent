"""The tools the agent is allowed to call.

Three tools, all read-only, all scoped to one SQLite file: list the tables,
describe one table, run a SELECT. The schema is never handed to the model in the
system prompt, so the only way for it to learn the shape of the database is to
call these and read the real answers.

Read-only is enforced in four independent layers rather than trusted to the
prompt: the connection is opened with SQLite's read-only URI flag, the connection
sets query_only, the statement must parse as a single SELECT or WITH, and a
keyword blocklist rejects anything that could attach, mutate or reach outside the
database file.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORBIDDEN = {
    "insert", "update", "delete", "drop", "create", "alter", "replace",
    "truncate", "attach", "detach", "vacuum", "reindex", "pragma",
    "begin", "commit", "rollback", "grant", "revoke",
}

FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}

COMMENT_PATTERN = re.compile(r"(--[^\n]*)|(/\*.*?\*/)", re.DOTALL)
STRING_PATTERN = re.compile(r"'(?:[^']|'')*'")
WORD_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*")


class SqlRejected(Exception):
    """Raised when a statement fails validation before it ever reaches SQLite."""


@dataclass
class ToolResult:
    tool: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_seconds: float = 0.0

    def to_model_text(self, max_cell_chars: int = 200) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        return _render(self.tool, self.payload, max_cell_chars)


def _strip_literals(sql: str) -> str:
    without_comments = COMMENT_PATTERN.sub(" ", sql)
    return STRING_PATTERN.sub("''", without_comments)


def validate_sql(sql: str) -> str:
    if not sql or not sql.strip():
        raise SqlRejected("empty statement")

    stripped = _strip_literals(sql).strip()
    if not stripped:
        raise SqlRejected("statement contains no executable SQL")

    body = stripped.rstrip(";")
    if ";" in body:
        raise SqlRejected("only a single statement is allowed, found a statement separator")

    words = [w.lower() for w in WORD_PATTERN.findall(body)]
    if not words:
        raise SqlRejected("statement contains no SQL keywords")

    if words[0] not in {"select", "with"}:
        raise SqlRejected(f"only SELECT and WITH statements are allowed, statement began with '{words[0]}'")

    hits = sorted(set(words) & FORBIDDEN)
    if hits:
        raise SqlRejected(f"statement contains forbidden keyword(s): {', '.join(hits)}")

    function_hits = sorted(set(words) & FORBIDDEN_FUNCTIONS)
    if function_hits:
        raise SqlRejected(f"statement contains forbidden function(s): {', '.join(function_hits)}")

    return sql.strip().rstrip(";")


class OlistTools:
    def __init__(
        self,
        db_path: Path,
        max_rows: int = 50,
        timeout_seconds: float = 30.0,
        max_cell_chars: int = 200,
    ) -> None:
        if not Path(db_path).exists():
            raise FileNotFoundError(f"database not found at {db_path}, run `python run.py 1` first")
        self.db_path = Path(db_path)
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self.max_cell_chars = max_cell_chars

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.db_path}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
        connection.execute("PRAGMA query_only = ON")
        return connection

    def list_tables(self) -> ToolResult:
        started = time.perf_counter()
        try:
            with self._connect() as connection:
                names = [
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                ]
                tables = []
                for name in names:
                    count = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                    tables.append({"table": name, "rows": count})
            return ToolResult("list_tables", True, {"tables": tables}, elapsed_seconds=time.perf_counter() - started)
        except Exception as exc:
            return ToolResult("list_tables", False, error=str(exc), elapsed_seconds=time.perf_counter() - started)

    def describe_table(self, table: str) -> ToolResult:
        started = time.perf_counter()
        try:
            with self._connect() as connection:
                known = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if table not in known:
                    raise SqlRejected(
                        f"no table named '{table}'. Available tables: {', '.join(sorted(known))}"
                    )

                columns = []
                for cid, name, ctype, notnull, default, pk in connection.execute(
                    f'PRAGMA table_info("{table}")'
                ):
                    sample = connection.execute(
                        f'SELECT "{name}" FROM "{table}" WHERE "{name}" IS NOT NULL LIMIT 3'
                    ).fetchall()
                    columns.append(
                        {
                            "name": name,
                            "type": ctype,
                            "nullable": not notnull,
                            "primary_key": bool(pk),
                            "sample_values": [_truncate(str(s[0]), self.max_cell_chars) for s in sample],
                        }
                    )

                foreign_keys = [
                    {"column": row[3], "references": f"{row[2]}.{row[4]}"}
                    for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
                ]
                rows = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

            payload = {"table": table, "rows": rows, "columns": columns, "foreign_keys": foreign_keys}
            return ToolResult("describe_table", True, payload, elapsed_seconds=time.perf_counter() - started)
        except Exception as exc:
            return ToolResult("describe_table", False, error=str(exc), elapsed_seconds=time.perf_counter() - started)

    def run_sql(self, sql: str) -> ToolResult:
        started = time.perf_counter()
        try:
            statement = validate_sql(sql)
        except SqlRejected as exc:
            return ToolResult("run_sql", False, {"sql": sql}, error=f"rejected before execution: {exc}",
                              elapsed_seconds=time.perf_counter() - started)

        connection = None
        try:
            connection = self._connect()
            deadline = time.perf_counter() + self.timeout_seconds

            def guard() -> int:
                return 1 if time.perf_counter() > deadline else 0

            connection.set_progress_handler(guard, 10_000)
            cursor = connection.execute(statement)
            fetched = cursor.fetchmany(self.max_rows + 1)
            column_names = [d[0] for d in cursor.description] if cursor.description else []
            connection.set_progress_handler(None, 0)

            truncated = len(fetched) > self.max_rows
            rows = fetched[: self.max_rows]
            payload = {
                "sql": statement,
                "columns": column_names,
                "rows": [list(r) for r in rows],
                "row_count": len(rows),
                "truncated": truncated,
            }
            return ToolResult("run_sql", True, payload, elapsed_seconds=time.perf_counter() - started)
        except Exception as exc:
            message = str(exc)
            if "interrupted" in message.lower():
                message = f"query exceeded the {self.timeout_seconds:g}s time limit and was cancelled"
            return ToolResult("run_sql", False, {"sql": sql}, error=message,
                              elapsed_seconds=time.perf_counter() - started)
        finally:
            if connection is not None:
                connection.close()

    def dispatch(self, name: str, arguments: dict) -> ToolResult:
        if name == "list_tables":
            return self.list_tables()
        if name == "describe_table":
            return self.describe_table(str(arguments.get("table", "")))
        if name == "run_sql":
            return self.run_sql(str(arguments.get("sql", "")))
        return ToolResult(name, False, error=f"unknown tool '{name}'")


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _render(tool: str, payload: dict, max_cell_chars: int) -> str:
    if tool == "list_tables":
        lines = ["tables in database:"]
        for entry in payload["tables"]:
            lines.append(f"  {entry['table']} ({entry['rows']:,} rows)")
        return "\n".join(lines)

    if tool == "describe_table":
        lines = [f"table {payload['table']} ({payload['rows']:,} rows)", "columns:"]
        for column in payload["columns"]:
            flags = []
            if column["primary_key"]:
                flags.append("PK")
            if not column["nullable"]:
                flags.append("NOT NULL")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            samples = ", ".join(column["sample_values"])
            lines.append(f"  {column['name']} {column['type']}{suffix} e.g. {samples}")
        if payload["foreign_keys"]:
            lines.append("foreign keys:")
            for fk in payload["foreign_keys"]:
                lines.append(f"  {fk['column']} -> {fk['references']}")
        return "\n".join(lines)

    if tool == "run_sql":
        if payload["row_count"] == 0:
            return "query executed successfully but returned 0 rows"
        header = " | ".join(payload["columns"])
        lines = [header, "-" * len(header)]
        for row in payload["rows"]:
            lines.append(" | ".join(_truncate(_format_cell(c), max_cell_chars) for c in row))
        footer = f"({payload['row_count']} rows"
        if payload["truncated"]:
            footer += f", truncated at {payload['row_count']}, add LIMIT or aggregate to see the rest"
        lines.append(footer + ")")
        return "\n".join(lines)

    return str(payload)


def _format_cell(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List every table in the database with its row count. Call this first to discover what data exists.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Show the columns, types, primary key, foreign keys and a few sample values for one table. Use this to learn real column names before writing SQL.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string", "description": "Exact table name."}},
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute one read-only SQL SELECT against the SQLite database and return the real rows. Only SELECT and WITH are permitted.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "A single SQLite SELECT statement."}},
                "required": ["sql"],
            },
        },
    },
]
