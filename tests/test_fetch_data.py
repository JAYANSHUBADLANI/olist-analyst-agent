"""The data manifest describes the files that are actually on disk."""

from __future__ import annotations

import pytest

from src.config import Config
from src.fetch_data import load_manifest, verify_all


def test_manifest_covers_every_raw_file():
    from src.config import RAW_FILES

    assert set(load_manifest()["files"]) == set(RAW_FILES.values())


def test_manifest_lists_more_than_one_mirror():
    assert len(load_manifest()["mirrors"]) >= 2


def test_local_files_match_the_manifest(config: Config):
    if not config.raw_dir.exists() or not any(config.raw_dir.glob("*.csv")):
        pytest.skip("raw CSVs not downloaded, run `python run.py 0` first")

    failures = {name: state for name, state in verify_all(config).items() if state != "ok"}
    assert not failures, f"files failing verification: {failures}"


def test_manifest_row_counts_match_the_profile(config: Config):
    import json

    profile_path = config.reports_dir / "data_profile.json"
    if not profile_path.exists():
        pytest.skip("profile not generated, run `python run.py 1` first")

    profile = json.loads(profile_path.read_text())
    manifest = load_manifest()["files"]
    for table in profile["tables"].values():
        assert table["rows"] == manifest[table["file"]]["rows"]
        assert table["sha256"] == manifest[table["file"]]["sha256"]
