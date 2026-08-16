"""The naive single-shot baseline the agent is measured against.

This is the ordinary text-to-SQL chain: hand the model the schema, ask for one
query, run it once, ask the model to phrase the result. One tool call, no
exploration, no retry when the query fails, no verification of the answer.

The baseline is given the complete schema DDL for free, which the agent never
gets. That is deliberate. A baseline crippled on purpose would make the
comparison meaningless, so this one is the strongest version of the naive
approach and the agent has to earn its margin against it.

The grounding guard is evaluated on baseline answers too, but only as an
observation. Nothing is repaired, so the number reported for the baseline is how
often it would have shipped an ungrounded figure.
"""

from __future__ import annotations

import sqlite3
import time

from src.config import Config
from src.guard import check_answer
from src.llm import LLMClient
from src.tools import OlistTools
from src.trace import RunTrace

SQL_SYSTEM_PROMPT = """You are a text-to-SQL system for a SQLite database. Here is the complete schema:

{schema}

Write exactly one read-only SQLite SELECT statement that answers the user's question.
Reply with the SQL statement and nothing else. No explanation, no markdown fences."""

ANSWER_SYSTEM_PROMPT = """You are a data analyst. You asked a database a question and received a result.
Write a short, direct answer to the user's question in plain business language based on that result."""


def load_schema_ddl(config: Config) -> str:
    connection = sqlite3.connect(f"file:{config.db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        connection.close()
    return "\n\n".join(row[0] for row in rows if row[0])


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("sql\n"):
        cleaned = cleaned[4:].strip()
    return cleaned


def run_baseline(
    question: str,
    config: Config,
    client: LLMClient,
    tools: OlistTools,
    schema_ddl: str,
    question_id: str = "ad-hoc",
    on_step=None,
) -> RunTrace:
    trace = RunTrace(question_id=question_id, question=question, mode="baseline", model=client.model)
    started = time.perf_counter()

    def emit(step) -> None:
        if on_step is not None:
            on_step(step)

    try:
        sql_response = client.complete(
            [
                {"role": "system", "content": SQL_SYSTEM_PROMPT.format(schema=schema_ddl)},
                {"role": "user", "content": question},
            ]
        )
    except Exception as exc:
        step = trace.add(kind="error", content=str(exc), ok=False, error=str(exc))
        emit(step)
        trace.stopped_reason = "llm_error"
        trace.wall_clock_seconds = time.perf_counter() - started
        return trace

    trace.usage.add(sql_response.usage)
    sql = _strip_fences(sql_response.content)
    step = trace.add(kind="plan", content=f"single-shot SQL: {sql}")
    emit(step)

    result = tools.run_sql(sql)
    rendered = result.to_model_text(config.agent.max_cell_chars)
    step = trace.add(
        kind="tool_call",
        tool="run_sql",
        arguments={"sql": sql},
        content=rendered,
        ok=result.ok,
        error=result.error,
        elapsed_seconds=result.elapsed_seconds,
        raw_payload=result.payload,
    )
    emit(step)

    try:
        answer_response = client.complete(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nSQL that was run:\n{sql}\n\nResult:\n{rendered}",
                },
            ]
        )
    except Exception as exc:
        step = trace.add(kind="error", content=str(exc), ok=False, error=str(exc))
        emit(step)
        trace.stopped_reason = "llm_error"
        trace.wall_clock_seconds = time.perf_counter() - started
        return trace

    trace.usage.add(answer_response.usage)
    trace.final_answer = (answer_response.content or "").strip()
    step = trace.add(kind="final", content=trace.final_answer)
    emit(step)
    trace.stopped_reason = "answered"

    guard = check_answer(trace.final_answer, trace, question)
    guard.notes += " (observed only, baseline does not repair)"
    trace.guard = guard
    step = trace.add(
        kind="guard",
        content=guard.notes,
        ok=guard.passed,
        error=None if guard.passed else f"ungrounded: {', '.join(guard.unverified_numbers)}",
    )
    emit(step)

    trace.wall_clock_seconds = time.perf_counter() - started
    return trace
