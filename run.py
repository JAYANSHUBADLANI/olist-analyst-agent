"""Single entrypoint for the project.

    python run.py 0                      download the Olist CSVs and verify their checksums
    python run.py 1                      profile the raw CSVs and build the database
    python run.py 2                      smoke test the agent loop on a few questions
    python run.py 3                      run the full benchmark, baseline and agent
    python run.py 4                      rebuild the report from the last saved run
    python run.py ask "your question"    watch the agent answer one question
"""

from __future__ import annotations

import argparse
import json
import sys

from src import build_db, fetch_data, profile_data, report
from src.benchmark import QUESTIONS, categories
from src.config import Config
from src.evaluate import per_question_rows, run_suite, save_summary, summarise

SMOKE_QUESTIONS = ["q02", "q09", "q19"]


def phase0(config: Config, force: bool = False) -> dict:
    print("fetching the Olist CSVs and verifying each against config/data_manifest.json")
    return fetch_data.fetch(config, force=force)


def phase1(config: Config, force: bool) -> dict:
    missing = [name for name, state in fetch_data.verify_all(config).items() if state != "ok"]
    if missing:
        print(f"{len(missing)} raw files missing or failing verification, fetching them first")
        phase0(config)

    profile = profile_data.run(config)
    print(f"profiled {len(profile['tables'])} files, wrote reports/data_profile.md")

    if config.db_path.exists() and not force:
        print(f"database already present at {config.db_path}, pass --force to rebuild")
        return {"skipped": True}

    summary = build_db.build(config, force=True)
    print(f"built {summary['db_path']} ({summary['size_bytes'] / 1e6:.0f} MB)")
    return summary


def phase2(config: Config) -> dict:
    from src.cli import ask

    if not config.db_path.exists():
        print("database not built yet, running phase 1 first")
        phase1(config, force=False)

    print(f"smoke testing the agent loop on {len(SMOKE_QUESTIONS)} questions")
    outcomes = []
    for question_id in SMOKE_QUESTIONS:
        question = next(q for q in QUESTIONS if q.id == question_id)
        trace = ask(question.text, config)
        outcomes.append(
            {
                "question_id": question_id,
                "tool_calls": len(trace.tool_calls),
                "failed_calls": len(trace.failed_tool_calls),
                "guard_passed": trace.guard.passed,
                "cost_usd": round(trace.usage.cost_usd, 6),
                "latency_seconds": round(trace.wall_clock_seconds, 2),
            }
        )
    return {"smoke": outcomes}


def phase3(config: Config, model: str | None = None) -> dict:
    if not config.db_path.exists():
        print("database not built yet, running phase 1 first")
        phase1(config, force=False)

    print(f"benchmark: {len(QUESTIONS)} questions {categories()}")

    results = {}
    summaries = {}
    for mode in ("baseline", "agent"):
        print(f"\nrunning {mode}")

        def progress(question, trace, grade, mode=mode):
            mark = "pass" if grade.correct else "fail"
            guard = "" if trace.guard.passed else "  guard flagged"
            retries = f"  {len(trace.failed_tool_calls)} failed calls" if trace.failed_tool_calls else ""
            print(
                f"  {question.id} {mark}  {len(trace.tool_calls)} calls  "
                f"{trace.wall_clock_seconds:5.1f}s{retries}{guard}"
            )

        result = run_suite(config, mode, model=model, on_progress=progress)
        results[mode] = result
        summaries[mode] = summarise(result)

    rows = per_question_rows(results)
    save_summary(config, summaries, rows)
    path = report.build_report(config)

    print(f"\nbaseline {summaries['baseline']['success_rate_pct']}%  "
          f"agent {summaries['agent']['success_rate_pct']}%")
    print(f"wrote {path}")
    return summaries


def phase4(config: Config) -> dict:
    path = report.build_report(config)
    print(f"rebuilt {path}")
    return {"report": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Grounded data-analyst agent over the Olist database")
    parser.add_argument("phase", help="0, 1, 2, 3, 4, all, or ask")
    parser.add_argument("question", nargs="?", default=None, help="question text when phase is 'ask'")
    parser.add_argument("--force", action="store_true", help="rebuild the database even if it exists")
    parser.add_argument("--baseline", action="store_true", help="with 'ask', use the single-shot baseline")
    parser.add_argument("--model", default=None, help="override the configured model")
    args = parser.parse_args()

    config = Config.load()

    if args.phase in {"2", "3", "ask", "all"} and not config.llm.is_configured:
        print(
            "No API key found. Copy .env.example to .env and set GROQ_API_KEY before running "
            "anything that calls the model. Phases 1 and 4 work without one.",
            file=sys.stderr,
        )
        return 1

    if args.phase == "ask":
        if not args.question:
            parser.error("ask requires a question, for example: python run.py ask \"how many orders shipped late?\"")
        from src.cli import ask

        ask(args.question, config, mode="baseline" if args.baseline else "agent", model=args.model)
        return 0

    phases = ["1", "2", "3", "4"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "0":
            summary = phase0(config, args.force)
        elif phase == "1":
            summary = phase1(config, args.force)
        elif phase == "2":
            summary = phase2(config)
        elif phase == "3":
            summary = phase3(config, model=args.model)
        elif phase == "4":
            summary = phase4(config)
        else:
            parser.error(f"unknown phase '{phase}'")
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
