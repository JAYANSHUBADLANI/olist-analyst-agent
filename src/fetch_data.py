"""Fetch the Olist CSVs and verify them before accepting them.

Kaggle is the canonical source but it needs an authenticated account, which makes
`git clone && run` impossible for anyone who does not have one. This module pulls
the identical files from public mirrors instead and checks every one against the
sha256 recorded in config/data_manifest.json.

Verification is the point, not a nicety. A benchmark built on quietly corrupted or
substituted input would produce numbers that look fine and mean nothing, so a file
whose hash does not match is deleted rather than used, and the next mirror is
tried.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

from src.config import ROOT, Config

CHUNK = 1 << 20


def load_manifest() -> dict:
    return json.loads((ROOT / "config" / "data_manifest.json").read_text())


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: dict) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size != expected["bytes"]:
        return False, f"wrong size, expected {expected['bytes']:,} bytes and found {size:,}"
    actual = sha256_of(path)
    if actual != expected["sha256"]:
        return False, f"sha256 mismatch, expected {expected['sha256'][:16]}... and got {actual[:16]}..."
    return True, "ok"


def _download(url: str, destination: Path, timeout: float) -> None:
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(CHUNK):
                handle.write(chunk)


def fetch(config: Config, force: bool = False, timeout: float = 300.0) -> dict:
    manifest = load_manifest()
    config.raw_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for filename, expected in manifest["files"].items():
        path = config.raw_dir / filename

        if not force:
            ok, reason = verify(path, expected)
            if ok:
                print(f"  {filename:45} already present and verified")
                results[filename] = "cached"
                continue
            if reason != "missing":
                print(f"  {filename:45} {reason}, refetching")

        last_error = "no mirror succeeded"
        for mirror in manifest["mirrors"]:
            try:
                _download(f"{mirror}/{filename}", path, timeout)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            ok, reason = verify(path, expected)
            if ok:
                print(f"  {filename:45} downloaded and verified")
                results[filename] = "downloaded"
                break

            path.unlink(missing_ok=True)
            last_error = reason
        else:
            raise RuntimeError(f"could not obtain a valid {filename}: {last_error}")

    return {"files": results, "raw_dir": str(config.raw_dir)}


def verify_all(config: Config) -> dict:
    manifest = load_manifest()
    report = {}
    for filename, expected in manifest["files"].items():
        ok, reason = verify(config.raw_dir / filename, expected)
        report[filename] = reason if not ok else "ok"
    return report
