"""
``.env.example`` and ``Settings`` must list the same settings (AGENTS.md: add a
setting to both together). Settings rejects unknown keys, so a stale example
line would also break a copied ``.env`` at startup.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"
# Matches set and commented-out assignments, e.g. "API_KEY=x" and "# DATABASE_URL=...".
_KEY = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)=")


def example_keys() -> set[str]:
    keys = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = _KEY.match(line)
        if match:
            keys.add(match.group(1).lower())
    return keys


def test_every_example_key_is_a_setting() -> None:
    assert sorted(example_keys() - set(Settings.model_fields)) == []


def test_every_setting_is_documented_in_the_example() -> None:
    assert sorted(set(Settings.model_fields) - example_keys()) == []
