"""Turn a benchmark run into a readable report.

Everything written here comes from the summaries and traces produced by an actual
run. Nothing is filled in by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import Config
from src.trace import load_jsonl


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def _ratio(agent: float, baseline: float) -> str:
    if baseline <= 0:
        return "n/a"
    return f"{agent / baseline:.1f}x"


def comparison_table(summaries: dict) -> str:
    agent = summaries.get("agent", {})
    baseline = summaries.get("baseline", {})
    if not agent or not baseline:
        return ""

    rows = [
        ("Task success rate", _pct(baseline["success_rate_pct"]), _pct(agent["success_rate_pct"]),
         f"{agent['success_rate_pct'] - baseline['success_rate_pct']:+.1f} pts"),
        ("Questions correct", f"{baseline['correct']}/{baseline['questions']}",
         f"{agent['correct']}/{agent['questions']}",
         f"{agent['correct'] - baseline['correct']:+d}"),
        ("Avg tool calls per question", f"{baseline['avg_tool_calls']}", f"{agent['avg_tool_calls']}",
         _ratio(agent["avg_tool_calls"], baseline["avg_tool_calls"])),
        ("Retry rate", _pct(baseline["retry_rate_pct"]), _pct(agent["retry_rate_pct"]), ""),
        ("Guard trigger rate", _pct(baseline["guard_trigger_rate_pct"]), _pct(agent["guard_trigger_rate_pct"]), ""),
        ("Avg tokens per question", f"{baseline['avg_tokens_per_question']:,.0f}",
         f"{agent['avg_tokens_per_question']:,.0f}",
         _ratio(agent["avg_tokens_per_question"], baseline["avg_tokens_per_question"])),
        ("Avg cost per question", f"${baseline['avg_cost_usd_per_question']:.6f}",
         f"${agent['avg_cost_usd_per_question']:.6f}",
         _ratio(agent["avg_cost_usd_per_question"], baseline["avg_cost_usd_per_question"])),
        ("Avg latency per question", f"{baseline['avg_latency_seconds']:.1f}s",
         f"{agent['avg_latency_seconds']:.1f}s",
         _ratio(agent["avg_latency_seconds"], baseline["avg_latency_seconds"])),
        ("Median latency", f"{baseline['median_latency_seconds']:.1f}s",
         f"{agent['median_latency_seconds']:.1f}s", ""),
    ]

    lines = ["| Metric | Single-shot baseline | Full agent | Difference |", "| --- | --- | --- | --- |"]
    lines += [f"| {name} | {b} | {a} | {d} |" for name, b, a, d in rows]
    return "\n".join(lines)


def category_table(summaries: dict) -> str:
    agent = summaries.get("agent", {})
    baseline = summaries.get("baseline", {})
    if not agent or not baseline:
        return ""

    lines = ["| Question type | Count | Baseline | Agent |", "| --- | ---: | ---: | ---: |"]
    for category in ("single_table", "join", "ambiguous"):
        a = agent["per_category"].get(category)
        b = baseline["per_category"].get(category)
        if not a or not b:
            continue
        lines.append(f"| {category} | {a['total']} | {_pct(b['success_rate'])} | {_pct(a['success_rate'])} |")
    return "\n".join(lines)


def per_question_table(rows: list[dict]) -> str:
    lines = [
        "| id | type | baseline | agent | agent tool calls | agent retries | guard |",
        "| --- | --- | :---: | :---: | ---: | ---: | :---: |",
    ]
    for row in sorted(rows, key=lambda r: r["question_id"]):
        baseline_mark = "pass" if row.get("baseline_correct") else "fail"
        agent_mark = "pass" if row.get("agent_correct") else "fail"
        guard = "clean"
        if row.get("agent_guard_repaired"):
            guard = "repaired"
        elif not row.get("agent_guard_passed", True):
            guard = "flagged"
        lines.append(
            f"| {row['question_id']} | {row['category']} | {baseline_mark} | {agent_mark} "
            f"| {row.get('agent_tool_calls', 0)} | {row.get('agent_failed_calls', 0)} | {guard} |"
        )
    return "\n".join(lines)


def find_examples(config: Config, limit: int = 3) -> dict:
    agent_traces = load_jsonl(config.logs_dir / "agent_traces.jsonl")

    retries = []
    guard_catches = []
    for trace in agent_traces:
        failed = [
            s for s in trace["steps"]
            if s["kind"] == "tool_call" and not s["ok"] and s["tool"] == "run_sql"
        ]
        if failed and len(retries) < limit:
            following = [
                s for s in trace["steps"]
                if s["kind"] == "tool_call" and s["tool"] == "run_sql" and s["index"] > failed[0]["index"] and s["ok"]
            ]
            retries.append(
                {
                    "question_id": trace["question_id"],
                    "question": trace["question"],
                    "failed_sql": failed[0]["arguments"].get("sql", ""),
                    "error": failed[0]["error"],
                    "recovered_sql": following[0]["arguments"].get("sql", "") if following else "",
                }
            )

        guard = trace.get("guard", {})
        if guard.get("repaired") and len(guard_catches) < limit:
            guard_catches.append(
                {
                    "question_id": trace["question_id"],
                    "question": trace["question"],
                    "before": guard.get("answer_before_repair", ""),
                    "flagged": guard.get("unverified_numbers", []),
                    "after": trace["final_answer"],
                }
            )

    return {"retries": retries, "guard_catches": guard_catches}


def build_report(config: Config) -> Path:
    results_path = config.reports_dir / "benchmark_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"no benchmark results at {results_path}. Run `python run.py 3` first, "
            "which needs an API key in .env."
        )
    data = json.loads(results_path.read_text())
    summaries = data["summaries"]
    rows = data["per_question"]
    examples = find_examples(config)

    agent = summaries.get("agent", {})
    baseline = summaries.get("baseline", {})

    lines = ["# Benchmark results", ""]
    lines.append(
        f"Model `{agent.get('model', 'n/a')}`, {agent.get('questions', 0)} questions, "
        f"graded against ground truth computed independently in pandas from the raw CSVs."
    )
    lines.append("")
    lines.append("## Baseline versus agent")
    lines.append("")
    lines.append(comparison_table(summaries))
    lines.append("")
    lines.append("## Success rate by question type")
    lines.append("")
    lines.append(category_table(summaries))
    lines.append("")

    if examples["retries"]:
        lines.append("## Self-correction, taken from real transcripts")
        lines.append("")
        for example in examples["retries"]:
            lines.append(f"**{example['question_id']}**: {example['question']}")
            lines.append("")
            lines.append("Failed attempt:")
            lines.append("")
            lines.append(f"```sql\n{example['failed_sql']}\n```")
            lines.append("")
            lines.append(f"Tool returned: `{example['error']}`")
            lines.append("")
            if example["recovered_sql"]:
                lines.append("Next attempt after reading that error:")
                lines.append("")
                lines.append(f"```sql\n{example['recovered_sql']}\n```")
                lines.append("")

    if examples["guard_catches"]:
        lines.append("## Grounding guard catches, taken from real transcripts")
        lines.append("")
        for example in examples["guard_catches"]:
            lines.append(f"**{example['question_id']}**: {example['question']}")
            lines.append("")
            lines.append(f"First answer: {example['before']}")
            lines.append("")
            lines.append(f"Numbers with no matching tool output: `{', '.join(example['flagged'])}`")
            lines.append("")
            lines.append(f"Answer after the guard forced a rewrite: {example['after']}")
            lines.append("")

    lines.append("## Every question")
    lines.append("")
    lines.append(per_question_table(rows))
    lines.append("")
    lines.append("## Run totals")
    lines.append("")
    for mode, summary in summaries.items():
        usage = summary["usage"]
        lines.append(
            f"- **{mode}**: {usage['total_tokens']:,} tokens across {usage['api_calls']} API calls, "
            f"${usage['cost_usd']:.4f} total, {summary['total_wall_clock_seconds']:.0f}s wall clock."
        )
    lines.append("")
    lines.append(
        "Dollar figures are measured token counts multiplied by the rates in `config/pricing.json`. "
        "Token counts come from the API response; the rates are configuration you should verify against "
        "current provider pricing."
    )
    lines.append("")

    path = config.reports_dir / "benchmark_report.md"
    path.write_text("\n".join(lines))
    return path
