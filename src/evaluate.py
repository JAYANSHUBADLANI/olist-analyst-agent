"""Run the benchmark and grade it.

Grading is programmatic. A model never judges another model's answer here: the
answer text is scanned for the values that ground truth says must appear, and a
question is correct only if everything the question asked for is present. Numbers
match within 2 percent relative tolerance, so an answer that sensibly rounds
1,258,681.34 to "about 1.26 million" is credited while a wrong figure is not.
"""

from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from src.agent import run_agent
from src.baseline import load_schema_ddl, run_baseline
from src.benchmark import QUESTIONS, Question
from src.config import Config
from src.guard import extract_claims
from src.llm import LLMClient, Usage
from src.tools import OlistTools
from src.trace import RunTrace, append_jsonl

RELATIVE_TOLERANCE = 0.02


@dataclass
class Grade:
    question_id: str
    category: str
    correct: bool
    missing_numbers: list[str] = field(default_factory=list)
    missing_strings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "correct": self.correct,
            "missing_numbers": self.missing_numbers,
            "missing_strings": self.missing_strings,
        }


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _number_found(expected: float, candidates: list[float]) -> bool:
    for candidate in candidates:
        if candidate == expected:
            return True
        scale = max(abs(expected), 1e-9)
        if abs(candidate - expected) / scale <= RELATIVE_TOLERANCE:
            return True
    return False


def _string_found(expected: str, answer: str, aliases: dict[str, list[str]]) -> bool:
    haystack = _normalise(answer)
    options = aliases.get(expected, [expected])
    if expected not in options:
        options = list(options) + [expected]
    return any(_normalise(option) in haystack for option in options)


def grade_answer(question: Question, answer: str) -> Grade:
    truth = question.truth()
    candidates = [c.value for c in extract_claims(answer)]

    missing_numbers: list[str] = []
    missing_strings: list[str] = []

    if question.require in {"numbers", "both"}:
        for expected in truth.get("numbers", []):
            if not _number_found(float(expected), candidates):
                missing_numbers.append(f"{expected}")

    if question.require in {"strings", "both"}:
        for expected in truth.get("strings", []):
            if not _string_found(str(expected), answer, question.aliases):
                missing_strings.append(str(expected))

    return Grade(
        question_id=question.id,
        category=question.category,
        correct=not missing_numbers and not missing_strings,
        missing_numbers=missing_numbers,
        missing_strings=missing_strings,
    )


def run_suite(
    config: Config,
    mode: str,
    questions: list[Question] | None = None,
    model: str | None = None,
    on_progress=None,
) -> dict:
    selected = questions if questions is not None else QUESTIONS
    tools = OlistTools(
        config.db_path,
        max_rows=config.agent.max_rows_returned,
        timeout_seconds=config.agent.sql_timeout_seconds,
        max_cell_chars=config.agent.max_cell_chars,
    )

    trace_path = config.logs_dir / f"{mode}_traces.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    records = []
    totals = Usage()
    started = time.perf_counter()

    with LLMClient(config, model=model) as client:
        schema_ddl = load_schema_ddl(config) if mode == "baseline" else ""

        for question in selected:
            if mode == "agent":
                trace = run_agent(question.text, config, client, tools, question_id=question.id)
            else:
                trace = run_baseline(
                    question.text, config, client, tools, schema_ddl, question_id=question.id
                )

            grade = grade_answer(question, trace.final_answer)
            totals.add(trace.usage)
            append_jsonl(trace_path, trace)
            records.append({"grade": grade, "trace": trace, "question": question})

            if on_progress is not None:
                on_progress(question, trace, grade)

    wall_clock = time.perf_counter() - started
    return {
        "mode": mode,
        "model": model or config.llm.model,
        "records": records,
        "usage": totals,
        "wall_clock_seconds": wall_clock,
        "trace_path": str(trace_path),
    }


def summarise(result: dict) -> dict:
    records = result["records"]
    count = len(records)
    if count == 0:
        return {}

    correct = sum(1 for r in records if r["grade"].correct)
    tool_calls = sum(len(r["trace"].tool_calls) for r in records)
    sql_calls = sum(len(r["trace"].sql_calls) for r in records)
    failed_calls = sum(len(r["trace"].failed_tool_calls) for r in records)
    runs_with_retry = sum(1 for r in records if r["trace"].retried)
    guard_triggered = sum(1 for r in records if r["trace"].guard.checked and not r["trace"].guard.passed)
    guard_checked = sum(1 for r in records if r["trace"].guard.checked)
    guard_repaired = sum(1 for r in records if r["trace"].guard.repaired)
    guard_repaired_clean = sum(
        1 for r in records if r["trace"].guard.repaired and r["trace"].guard.passed
    )

    usage: Usage = result["usage"]
    latencies = sorted(r["trace"].wall_clock_seconds for r in records)

    per_category: dict[str, dict] = {}
    for record in records:
        category = record["grade"].category
        bucket = per_category.setdefault(category, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(record["grade"].correct)
    for bucket in per_category.values():
        bucket["success_rate"] = round(bucket["correct"] / bucket["total"] * 100, 1)

    return {
        "mode": result["mode"],
        "model": result["model"],
        "questions": count,
        "correct": correct,
        "success_rate_pct": round(correct / count * 100, 1),
        "per_category": per_category,
        "avg_tool_calls": round(tool_calls / count, 2),
        "avg_sql_calls": round(sql_calls / count, 2),
        "failed_tool_calls": failed_calls,
        "retry_rate_pct": round(runs_with_retry / count * 100, 1),
        "guard_checked": guard_checked,
        "guard_trigger_rate_pct": round(guard_triggered / count * 100, 1),
        "guard_triggered": guard_triggered,
        "guard_repaired": guard_repaired,
        "guard_repaired_to_clean": guard_repaired_clean,
        "avg_latency_seconds": round(sum(latencies) / count, 2),
        "median_latency_seconds": round(latencies[count // 2], 2),
        "total_wall_clock_seconds": round(result["wall_clock_seconds"], 1),
        "usage": usage.as_dict(),
        "avg_cost_usd_per_question": round(usage.cost_usd / count, 6),
        "avg_tokens_per_question": round((usage.prompt_tokens + usage.completion_tokens) / count, 1),
    }


def save_summary(config: Config, summaries: dict, per_question: list[dict]) -> Path:
    path = config.reports_dir / "benchmark_results.json"
    path.write_text(json.dumps({"summaries": summaries, "per_question": per_question}, indent=2))
    return path


def per_question_rows(results: dict[str, dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for mode, result in results.items():
        for record in result["records"]:
            question = record["question"]
            trace: RunTrace = record["trace"]
            row = rows.setdefault(
                question.id,
                {"question_id": question.id, "category": question.category, "question": question.text},
            )
            row[f"{mode}_correct"] = record["grade"].correct
            row[f"{mode}_answer"] = trace.final_answer
            row[f"{mode}_tool_calls"] = len(trace.tool_calls)
            row[f"{mode}_failed_calls"] = len(trace.failed_tool_calls)
            row[f"{mode}_guard_passed"] = trace.guard.passed
            row[f"{mode}_guard_repaired"] = trace.guard.repaired
            row[f"{mode}_latency_seconds"] = round(trace.wall_clock_seconds, 2)
            row[f"{mode}_cost_usd"] = round(trace.usage.cost_usd, 6)
            row[f"{mode}_missing"] = record["grade"].missing_numbers + record["grade"].missing_strings
    return list(rows.values())
