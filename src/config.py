"""Configuration for the Olist analyst agent."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

RAW_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "product_category_translation": "product_category_name_translation.csv",
}


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    temperature: float
    max_tokens: int
    request_timeout: float = 120.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    max_sql_retries: int = 3
    max_rows_returned: int = 50
    sql_timeout_seconds: float = 30.0
    max_cell_chars: int = 200


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    raw_dir: Path = ROOT / "data" / "raw"
    db_path: Path = ROOT / "data" / "olist.db"
    reports_dir: Path = ROOT / "reports"
    logs_dir: Path = ROOT / "reports" / "runs"
    llm: LLMConfig = field(default_factory=lambda: _load_llm())
    agent: AgentConfig = field(default_factory=AgentConfig)
    pricing: dict = field(default_factory=lambda: _load_pricing())

    @classmethod
    def load(cls, _path: str | None = None) -> "Config":
        load_dotenv(ROOT / ".env")
        cfg = cls(llm=_load_llm(), pricing=_load_pricing())
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    def price_for(self, model: str) -> dict | None:
        return self.pricing.get("models", {}).get(model)


def _vertex_access_token() -> str:
    """Short-lived OAuth token for Vertex, read from an existing gcloud login.

    Vertex has no static API key. The token is fetched once per process and lasts about an
    hour, which is longer than a full benchmark run, so nothing refreshes it mid-run.
    """
    try:
        return subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "GCP_PROJECT_ID is set, so this run expects Vertex AI, but no gcloud access token "
            "could be read. Run `gcloud auth login` first, or unset GCP_PROJECT_ID to fall back "
            "to an OpenAI-compatible provider with a static key."
        ) from exc


def _load_llm() -> LLMConfig:
    load_dotenv(ROOT / ".env")

    # Vertex serves Gemini through an OpenAI-compatible endpoint, so the whole client works
    # unchanged against it. Selecting on GCP_PROJECT_ID keeps the choice in one place rather
    # than asking anyone to hand-assemble the project-scoped URL.
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        region = os.getenv("GCP_REGION", "us-central1")
        return LLMConfig(
            base_url=os.getenv(
                "LLM_BASE_URL",
                f"https://{region}-aiplatform.googleapis.com/v1beta1/"
                f"projects/{project}/locations/{region}/endpoints/openapi",
            ),
            model=os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite"),
            api_key=_vertex_access_token(),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )

    key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or ""
    return LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        api_key=key,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
    )


def _load_pricing() -> dict:
    path = ROOT / "config" / "pricing.json"
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text())
