"""Step by step transcript of a single run.

Every model turn, every tool call, every raw tool result and every guard decision
lands here. The evaluation metrics and the CLI both read this structure, so what
gets reported is exactly what happened rather than a separate summary the model
wrote about itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm import Usage


@dataclass
class Step:
    index: int
    kind: str
    content: str = ""
    tool: str | None = None
    arguments: dict = field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    elapsed_seconds: float = 0.0
    raw_payload: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["elapsed_seconds"] = round(self.elapsed_seconds, 3)
        return data


@dataclass
class GuardResult:
    checked: bool = False
    passed: bool = True
    unverified_numbers: list[str] = field(default_factory=list)
    grounded_numbers: list[str] = field(default_factory=list)
    repaired: bool = False
    answer_before_repair: str | None = None
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunTrace:
    question_id: str
    question: str
    mode: str
    model: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    guard: GuardResult = field(default_factory=GuardResult)
    usage: Usage = field(default_factory=Usage)
    wall_clock_seconds: float = 0.0
    stopped_reason: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(self, **kwargs: Any) -> Step:
        step = Step(index=len(self.steps), **kwargs)
        self.steps.append(step)
        return step

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool_call"]

    @property
    def sql_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool_call" and s.tool == "run_sql"]

    @property
    def failed_tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool_call" and not s.ok]

    @property
    def retried(self) -> bool:
        return len(self.failed_tool_calls) > 0

    @property
    def empty_result_calls(self) -> list[Step]:
        return [
            s
            for s in self.steps
            if s.kind == "tool_call" and s.tool == "run_sql" and s.ok and s.raw_payload.get("row_count") == 0
        ]

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "mode": self.mode,
            "model": self.model,
            "started_at": self.started_at,
            "final_answer": self.final_answer,
            "stopped_reason": self.stopped_reason,
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "usage": self.usage.as_dict(),
            "guard": self.guard.as_dict(),
            "counts": {
                "steps": len(self.steps),
                "tool_calls": len(self.tool_calls),
                "sql_calls": len(self.sql_calls),
                "failed_tool_calls": len(self.failed_tool_calls),
                "empty_result_calls": len(self.empty_result_calls),
            },
            "steps": [s.as_dict() for s in self.steps],
        }


def append_jsonl(path: Path, trace: RunTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace.as_dict(), default=str) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
