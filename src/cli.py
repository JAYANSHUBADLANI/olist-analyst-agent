"""Watch the agent work.

Prints the plan, every tool call with the SQL it ran, the real rows that came
back, any error and the retry that followed, then the final answer and the
grounding verdict. Steps appear as they happen rather than being replayed at the
end, so the loop is something you can actually watch rather than take on trust.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src.agent import run_agent
from src.baseline import load_schema_ddl, run_baseline
from src.config import Config
from src.llm import LLMClient
from src.tools import OlistTools
from src.trace import RunTrace, Step

STEP_STYLES = {
    "plan": ("bold cyan", "PLAN"),
    "tool_call": ("bold yellow", "TOOL"),
    "final": ("bold green", "ANSWER"),
    "guard": ("bold magenta", "GUARD"),
    "error": ("bold red", "ERROR"),
}


def _render_step(console: Console, step: Step) -> None:
    style, label = STEP_STYLES.get(step.kind, ("white", step.kind.upper()))

    if step.kind == "tool_call":
        status = "ok" if step.ok else "FAILED"
        status_style = "green" if step.ok else "red"
        header = Text.assemble(
            (f"[{step.index}] {label} ", style),
            (step.tool or "", "bold white"),
            ("  ", ""),
            (status, status_style),
            (f"  {step.elapsed_seconds:.2f}s", "dim"),
        )
        console.print(header)

        if step.tool == "run_sql" and step.arguments.get("sql"):
            console.print(Syntax(step.arguments["sql"].strip(), "sql", theme="ansi_dark", word_wrap=True))
        elif step.arguments:
            console.print(Text(f"    {step.arguments}", style="dim"))

        body = step.error if not step.ok else step.content
        console.print(Panel(body or "", border_style="red" if not step.ok else "grey37", expand=False))
        return

    if step.kind == "guard":
        verdict = "PASSED" if step.ok else "FAILED"
        console.print(
            Text.assemble(
                (f"[{step.index}] {label} ", style),
                (verdict, "green" if step.ok else "red"),
                (f"  {step.content}", "dim"),
            )
        )
        if not step.ok:
            console.print(Text(f"    {step.error}", style="red"))
        return

    console.print(Text.assemble((f"[{step.index}] {label} ", style)))
    console.print(Panel(step.content or "", border_style="grey37", expand=False))


def _render_summary(console: Console, trace: RunTrace) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()

    usage = trace.usage.as_dict()
    table.add_row("mode", trace.mode)
    table.add_row("model", trace.model)
    table.add_row("steps", str(len(trace.steps)))
    table.add_row("tool calls", f"{len(trace.tool_calls)} ({len(trace.failed_tool_calls)} failed)")
    table.add_row("sql queries", str(len(trace.sql_calls)))
    table.add_row("stopped because", trace.stopped_reason or "n/a")
    table.add_row("grounding guard", "passed" if trace.guard.passed else "FAILED")
    if trace.guard.repaired:
        table.add_row("", "answer was repaired after the guard flagged it")
    table.add_row("tokens", f"{usage['total_tokens']:,} in {usage['api_calls']} api calls")
    table.add_row("cost", f"${usage['cost_usd']:.6f}")
    table.add_row("wall clock", f"{trace.wall_clock_seconds:.2f}s")

    console.print(Rule("run summary", style="grey37"))
    console.print(table)


def ask(question: str, config: Config, mode: str = "agent", model: str | None = None) -> RunTrace:
    console = Console()
    console.print(Rule(f"question: {question}", style="bold blue"))

    tools = OlistTools(
        config.db_path,
        max_rows=config.agent.max_rows_returned,
        timeout_seconds=config.agent.sql_timeout_seconds,
        max_cell_chars=config.agent.max_cell_chars,
    )

    def on_step(step: Step) -> None:
        _render_step(console, step)

    with LLMClient(config, model=model) as client:
        if mode == "baseline":
            trace = run_baseline(
                question, config, client, tools, load_schema_ddl(config), on_step=on_step
            )
        else:
            trace = run_agent(question, config, client, tools, on_step=on_step)

    _render_summary(console, trace)
    return trace
