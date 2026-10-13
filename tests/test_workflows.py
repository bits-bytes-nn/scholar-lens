"""Contracts between the GitHub workflows and the repository."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _locked_versions() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "poetry.lock").read_text(encoding="utf-8"))
    return {pkg["name"].lower(): pkg["version"] for pkg in lock["package"]}


def test_post_conventions_pins_match_poetry_lock() -> None:
    # The drift check installs two packages with pip; Dependabot only updates
    # poetry.lock, so the pins must follow it.
    workflow = (ROOT / ".github/workflows/post-conventions.yml").read_text(
        encoding="utf-8"
    )
    pins = dict(re.findall(r"\b([A-Za-z][\w.-]*)==([\w.]+)", workflow))
    assert pins, "the workflow no longer pins its pip installs"
    locked = _locked_versions()
    assert {name.lower(): v for name, v in pins.items()} == {
        name.lower(): locked[name.lower()] for name in pins
    }
