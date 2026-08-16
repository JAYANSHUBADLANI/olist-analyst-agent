"""A minimal client for any OpenAI-compatible chat completions endpoint.

This project runs against Groq, but nothing here is Groq specific beyond the
default base URL, so the same code points at any provider that speaks the same
wire format. Token usage comes straight off the API response rather than being
estimated locally; cost is that measured token count multiplied by the rate in
config/pricing.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config import Config, LLMConfig

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

# A provider that is rate limiting by tokens per minute can answer a 429 with a retry-after
# measured in minutes. Sleeping for exactly that long makes a benchmark run look hung: no
# output, no CPU, no way to tell a slow provider from a dead one. The wait is capped so a
# throttled run degrades into visible retries instead.
MAX_RETRY_AFTER_SECONDS = 30.0


class LLMError(Exception):
    """Raised when the provider cannot be reached or returns an unusable response."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.api_calls += other.api_calls
        self.latency_seconds += other.latency_seconds
        self.cost_usd += other.cost_usd

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "api_calls": self.api_calls,
            "latency_seconds": round(self.latency_seconds, 3),
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    message: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""


class LLMClient:
    def __init__(self, config: Config, model: str | None = None) -> None:
        self.settings: LLMConfig = config.llm
        self.model = model or self.settings.model
        self.pricing = config.price_for(self.model)
        if not self.settings.is_configured:
            raise LLMError(
                "no API key found. Put GROQ_API_KEY=... in the project .env file "
                "(copy .env.example) before running anything that calls the model."
            )
        self._client = httpx.Client(timeout=self.settings.request_timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        if not self.pricing:
            return 0.0
        return (
            prompt_tokens * self.pricing.get("input", 0.0)
            + completion_tokens * self.pricing.get("output", 0.0)
        ) / 1_000_000

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_attempts: int = 4,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        last_error = ""
        for attempt in range(max_attempts):
            started = time.perf_counter()
            try:
                response = self._client.post(url, headers=headers, json=body)
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                time.sleep(min(2**attempt, 8))
                continue
            elapsed = time.perf_counter() - started

            if response.status_code in RETRY_STATUS:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                retry_after = response.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    delay = min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
                else:
                    delay = min(2**attempt, 8)
                time.sleep(delay)
                continue

            if response.status_code >= 400:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:500]}")

            payload = response.json()
            choices = payload.get("choices") or []
            if not choices:
                raise LLMError(f"response contained no choices: {json.dumps(payload)[:300]}")

            message = choices[0].get("message", {}) or {}
            raw_usage = payload.get("usage", {}) or {}
            prompt_tokens = int(raw_usage.get("prompt_tokens", 0))
            completion_tokens = int(raw_usage.get("completion_tokens", 0))

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                api_calls=1,
                latency_seconds=elapsed,
                cost_usd=self._cost(prompt_tokens, completion_tokens),
            )
            return LLMResponse(
                content=message.get("content") or "",
                tool_calls=message.get("tool_calls") or [],
                message=message,
                usage=usage,
                finish_reason=choices[0].get("finish_reason", ""),
            )

        raise LLMError(f"provider unavailable after {max_attempts} attempts. last error: {last_error}")
