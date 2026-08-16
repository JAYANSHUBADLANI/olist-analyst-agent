"""The agent loop, its retry path and the guard repair path."""

from __future__ import annotations

from src.agent import run_agent
from src.config import Config
from tests.fakes import ScriptedClient, say, tool_call


def test_agent_recovers_from_a_bad_column_name(config: Config, tools):
    client = ScriptedClient(
        [
            say("I will look at the tables first.", [tool_call("c1", "list_tables")]),
            say("", [tool_call("c2", "describe_table", table="orders")]),
            say("", [tool_call("c3", "run_sql", sql="SELECT COUNT(*) AS n FROM orders WHERE status = 'delivered'")]),
            say("", [tool_call("c4", "run_sql", sql="SELECT COUNT(*) AS n FROM orders WHERE order_status = 'delivered'")]),
            say("96,478 orders were delivered."),
        ]
    )

    trace = run_agent("How many orders were delivered?", config, client, tools, question_id="t1")

    assert trace.stopped_reason == "answered"
    assert len(trace.failed_tool_calls) == 1
    assert trace.retried is True
    assert len(trace.sql_calls) == 2
    assert trace.guard.passed
    assert "96,478" in trace.final_answer


def test_failed_call_error_is_visible_to_the_model(config: Config, tools):
    client = ScriptedClient(
        [
            say("", [tool_call("c1", "run_sql", sql="SELECT status FROM orders")]),
            say("I could not determine that."),
        ]
    )
    run_agent("anything", config, client, tools, question_id="t2")

    tool_messages = [
        message
        for turn in client.calls
        for message in turn
        if message.get("role") == "tool"
    ]
    assert any("no such column" in m["content"].lower() for m in tool_messages)


def test_guard_catches_an_invented_figure_and_forces_a_repair(config: Config, tools):
    client = ScriptedClient(
        [
            say("", [tool_call("c1", "run_sql", sql="SELECT COUNT(*) AS n FROM orders")]),
            say("There are 99,441 orders worth R$ 7,400,000 in total."),
            say("There are 99,441 orders. I did not establish a total value."),
        ]
    )

    trace = run_agent("How many orders are there?", config, client, tools, question_id="t3")

    assert trace.guard.repaired is True
    assert trace.guard.passed is True
    assert "7,400,000" in trace.guard.answer_before_repair
    assert "7,400,000" not in trace.final_answer


def test_loop_stops_after_repeated_sql_failures(config: Config, tools):
    bad = "SELECT nope FROM orders"
    client = ScriptedClient(
        [
            say("", [tool_call("c1", "run_sql", sql=bad)]),
            say("", [tool_call("c2", "run_sql", sql=bad)]),
            say("", [tool_call("c3", "run_sql", sql=bad)]),
            say("I could not answer that from the database."),
            say("I could not answer that from the database."),
        ]
    )

    trace = run_agent("something impossible", config, client, tools, question_id="t4")

    assert len(trace.failed_tool_calls) == 3
    assert trace.final_answer


def test_step_budget_is_respected(config: Config, tools):
    script = [say("", [tool_call(f"c{i}", "list_tables")]) for i in range(config.agent.max_steps + 5)]
    script.append(say("Final answer with no figures."))
    client = ScriptedClient(script)

    trace = run_agent("loop forever", config, client, tools, question_id="t5")

    assert len(trace.tool_calls) <= config.agent.max_steps + 2


def test_plan_only_response_is_nudged_not_accepted_as_final(config: Config, tools):
    """Observed against a live model: a plan stated as prose with no tool call in the same
    turn must not end the run. The model gets nudged once to either act or actually answer."""
    client = ScriptedClient(
        [
            say("I will list the tables and then look at the relevant one."),
            say("", [tool_call("c1", "run_sql", sql="SELECT COUNT(*) AS n FROM orders")]),
            say("There are 99,441 orders."),
        ]
    )

    trace = run_agent("How many orders are there?", config, client, tools, question_id="t7")

    assert trace.stopped_reason == "answered"
    assert len(trace.tool_calls) == 1
    assert "99,441" in trace.final_answer


def test_a_second_plan_only_response_is_accepted_so_the_loop_cannot_hang(config: Config, tools):
    client = ScriptedClient(
        [
            say("I will look into this."),
            say("I was unable to determine that from the data available."),
        ]
    )

    trace = run_agent("something unanswerable", config, client, tools, question_id="t8")

    assert trace.stopped_reason == "answered"
    assert trace.final_answer == "I was unable to determine that from the data available."


def test_usage_is_accumulated_across_turns(config: Config, tools):
    client = ScriptedClient(
        [
            say("", [tool_call("c1", "list_tables")]),
            say("Done, no figures stated."),
        ]
    )
    trace = run_agent("hello", config, client, tools, question_id="t6")

    assert trace.usage.api_calls == 2
    assert trace.usage.prompt_tokens == 200
    assert trace.wall_clock_seconds > 0
