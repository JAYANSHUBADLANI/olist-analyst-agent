"""A scripted stand-in for the LLM client.

The agent loop is provider agnostic: it only needs an object with a `model`
attribute and a `complete` method returning an LLMResponse. Scripting that lets
the loop, the retry path and the guard be tested end to end without an API key
and without paying for tokens, which also means these tests are deterministic.
"""

from __future__ import annotations

import json

from src.llm import LLMResponse, Usage


def tool_call(call_id: str, name: str, **arguments) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedClient:
    def __init__(self, script: list[LLMResponse], model: str = "scripted-model") -> None:
        self.script = list(script)
        self.model = model
        self.calls: list[list[dict]] = []

    def complete(self, messages, tools=None, max_attempts: int = 4) -> LLMResponse:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("scripted client ran out of responses")
        return self.script.pop(0)

    def __enter__(self) -> "ScriptedClient":
        return self

    def __exit__(self, *_exc) -> None:
        return None


def say(content: str = "", tool_calls: list[dict] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        message={"role": "assistant", "content": content},
        usage=Usage(prompt_tokens=100, completion_tokens=20, api_calls=1, latency_seconds=0.01, cost_usd=0.0001),
        finish_reason="tool_calls" if tool_calls else "stop",
    )
