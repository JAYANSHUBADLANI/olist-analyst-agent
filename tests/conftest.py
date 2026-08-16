"""Shared fixtures."""

from __future__ import annotations

import pytest

from src.config import Config
from src.tools import OlistTools


@pytest.fixture(scope="session")
def config() -> Config:
    return Config.load()


@pytest.fixture(scope="session")
def tools(config: Config) -> OlistTools:
    if not config.db_path.exists():
        pytest.skip("database not built, run `python run.py 1` first")
    return OlistTools(config.db_path, max_rows=10)
