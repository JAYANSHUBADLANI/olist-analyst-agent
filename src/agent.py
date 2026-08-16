"""The multi-step agent loop.

Plan, call a tool, read the real result, decide whether another step is needed,
then answer. The loop is explicit: the model gets one turn at a time, tool output
is fed back as an actual observation, and errors come back verbatim so the model
can correct itself rather than being shielded from what went wrong.

The schema is deliberately absent from the system prompt. The model is told the
tools exist and nothing about what is in the database, so any correct answer has
to come from exploration it performed in that run.
"""

from __future__ import annotations

import json
import time

from src.config import Config
from src.guard import check_answer, repair_prompt
from src.llm import LLMClient, LLMResponse
from src.tools import TOOL_SPECS, OlistTools, ToolResult
from src.trace import RunTrace

SYSTEM_PROMPT = """You are a careful data analyst. You answer business questions about a SQLite \
database by querying it. You have never seen this database and you do not know what tables or \
columns it contains.

Work like this:
1. Start by stating a short plan in one or two sentences.
2. Call list_tables to see what exists, then describe_table on the tables that look relevant. \
Never guess a column name. Business language in the question will often not match the real column \
names, so look before you write SQL.
3. Write a single read-only SELECT with run_sql. Only SELECT and WITH are permitted.
4. Read the result you actually got back. If the query errored, read the error and fix the specific \
problem it names. If it returned zero rows or the wrong shape, work out why and try a different \
query rather than repeating the same one.
5. When you have the figures you need, write a short answer in plain business language.

Rules you must follow:
- Every number in your final answer must come from a query result you actually received in this \
conversation. Never state a figure from memory or estimate one.
- If you cannot establish something with a query, say so plainly instead of guessing.
- Answer the question that was asked. Do not pad the answer with extra figures you were not asked for.
- Dates are stored as text in 'YYYY-MM-DD HH:MM:SS' form. SQLite date functions such as julianday \
and strftime work on them.
- Give the final answer as prose, not as a tool call."""


def _assistant_message(response: LLMResponse) -> dict:
    message: dict = {"role": "assistant", "content": response.content or ""}
    if response.tool_calls:
        message["tool_calls"] = response.tool_calls
    return message


def _parse_arguments(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def run_agent(
    question: str,
    config: Config,
    client: LLMClient,
    tools: OlistTools,
    question_id: str = "ad-hoc",
    on_step=None,
) -> RunTrace:
    trace = RunTrace(question_id=question_id, question=question, mode="agent", model=client.model)
    started = time.perf_counter()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    max_steps = config.agent.max_steps
    max_retries = config.agent.max_sql_retries
    consecutive_failures = 0
    guard_repairs_left = 1
    step_budget = max_steps
    plan_only_nudges_left = 1

    def emit(step) -> None:
        if on_step is not None:
            on_step(step)

    turn = 0
    while turn < step_budget:
        turn += 1
        try:
            response = client.complete(messages, tools=TOOL_SPECS)
        except Exception as exc:
            step = trace.add(kind="error", content=str(exc), ok=False, error=str(exc))
            emit(step)
            trace.stopped_reason = "llm_error"
            break

        trace.usage.add(response.usage)

        if response.content and response.tool_calls:
            step = trace.add(kind="plan", content=response.content)
            emit(step)
        elif response.content and not response.tool_calls:
            if not trace.tool_calls and plan_only_nudges_left > 0:
                # The system prompt asks for a short plan before acting, and step 1 of that
                # plan is to call a tool. A model that writes the plan as prose with no tool
                # call in the same turn has not answered anything yet, it has described an
                # intention. Treating that as the final answer ends the run with a "here is
                # what I am about to do" sentence instead of a real answer. Nudged once, not
                # in a loop, so a model that keeps refusing to call a tool still gets to answer.
                plan_only_nudges_left -= 1
                step_budget = min(step_budget + 1, max_steps + 2)
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That describes a plan, not an answer, and no tool has been called yet. "
                            "Call list_tables or describe_table now, or if you already have everything "
                            "you need, give the final answer with no further tool calls."
                        ),
                    }
                )
                continue
            trace.final_answer = response.content.strip()
            step = trace.add(kind="final", content=trace.final_answer)
            emit(step)
            trace.stopped_reason = "answered"
            break

        if not response.tool_calls:
            trace.stopped_reason = "model_returned_nothing"
            break

        messages.append(_assistant_message(response))

        for call in response.tool_calls:
            function = call.get("function", {}) or {}
            name = function.get("name", "")
            arguments = _parse_arguments(function.get("arguments"))

            result: ToolResult = tools.dispatch(name, arguments)
            rendered = result.to_model_text(config.agent.max_cell_chars)

            step = trace.add(
                kind="tool_call",
                tool=name,
                arguments=arguments,
                content=rendered,
                ok=result.ok,
                error=result.error,
                elapsed_seconds=result.elapsed_seconds,
                raw_payload=result.payload,
            )
            emit(step)

            if name == "run_sql":
                consecutive_failures = consecutive_failures + 1 if not result.ok else 0

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": rendered,
                }
            )

        if consecutive_failures >= max_retries:
            trace.stopped_reason = f"gave_up_after_{consecutive_failures}_consecutive_sql_failures"
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"You have had {consecutive_failures} SQL failures in a row. Stop querying and "
                        "answer with what you have established so far, stating plainly what you could not determine."
                    ),
                }
            )
            consecutive_failures = 0
            step_budget = min(step_budget + 1, max_steps + 2)

        if turn >= step_budget and not trace.final_answer:
            messages.append(
                {
                    "role": "user",
                    "content": "You have reached the step limit. Answer now using only figures you have already obtained.",
                }
            )
            try:
                response = client.complete(messages, tools=None)
                trace.usage.add(response.usage)
                trace.final_answer = (response.content or "").strip()
                step = trace.add(kind="final", content=trace.final_answer)
                emit(step)
                trace.stopped_reason = "step_limit_reached"
            except Exception as exc:
                step = trace.add(kind="error", content=str(exc), ok=False, error=str(exc))
                emit(step)
                trace.stopped_reason = "llm_error"
            break

    if trace.final_answer:
        guard = check_answer(trace.final_answer, trace, question)
        step = trace.add(
            kind="guard",
            content=guard.notes,
            ok=guard.passed,
            error=None if guard.passed else f"ungrounded: {', '.join(guard.unverified_numbers)}",
        )
        emit(step)

        if not guard.passed and guard_repairs_left > 0:
            guard_repairs_left -= 1
            guard.answer_before_repair = trace.final_answer
            messages.append({"role": "user", "content": repair_prompt(guard)})

            repaired = _repair(messages, config, client, tools, trace, emit)
            if repaired:
                recheck = check_answer(trace.final_answer, trace, question)
                recheck.repaired = True
                recheck.answer_before_repair = guard.answer_before_repair
                guard = recheck
                step = trace.add(
                    kind="guard",
                    content=f"after repair: {guard.notes}",
                    ok=guard.passed,
                    error=None if guard.passed else f"still ungrounded: {', '.join(guard.unverified_numbers)}",
                )
                emit(step)

        trace.guard = guard

    trace.wall_clock_seconds = time.perf_counter() - started
    return trace


def _repair(messages, config, client, tools, trace, emit) -> bool:
    for _ in range(2):
        try:
            response = client.complete(messages, tools=TOOL_SPECS)
        except Exception as exc:
            step = trace.add(kind="error", content=str(exc), ok=False, error=str(exc))
            emit(step)
            return False

        trace.usage.add(response.usage)

        if not response.tool_calls:
            trace.final_answer = (response.content or "").strip()
            step = trace.add(kind="final", content=trace.final_answer)
            emit(step)
            return True

        if response.content:
            step = trace.add(kind="plan", content=response.content)
            emit(step)

        messages.append(_assistant_message(response))
        for call in response.tool_calls:
            function = call.get("function", {}) or {}
            name = function.get("name", "")
            arguments = _parse_arguments(function.get("arguments"))
            result = tools.dispatch(name, arguments)
            rendered = result.to_model_text(config.agent.max_cell_chars)
            step = trace.add(
                kind="tool_call",
                tool=name,
                arguments=arguments,
                content=rendered,
                ok=result.ok,
                error=result.error,
                elapsed_seconds=result.elapsed_seconds,
                raw_payload=result.payload,
            )
            emit(step)
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": rendered})
    return False
