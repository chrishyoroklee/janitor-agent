"""Locations and user configuration (a small JSON file next to the database)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


def user_home() -> str:
    # JANITOR_USER_HOME exists for tests and sandboxed dry runs.
    return os.path.realpath(os.environ.get("JANITOR_USER_HOME") or os.path.expanduser("~"))


def data_dir() -> Path:
    if env := os.environ.get("JANITOR_DATA_DIR"):
        return Path(env)
    home = Path(user_home())
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "MacJanitor"
    return home / ".local" / "share" / "macjanitor"


@dataclass
class Config:
    # Where to look for node_modules / .venv / build caches (relative to home).
    project_roots: list[str] = field(
        default_factory=lambda: ["Developer", "Projects", "Code", "code", "src", "dev", "repos",
                                 "work", "GitHub", "Documents", "Desktop"]
    )
    project_max_depth: int = 6
    # A project untouched for this long has "stale" dependency directories.
    stale_days: int = 30
    # Candidates smaller than this are noise.
    min_candidate_bytes: int = 10 * 1000 * 1000
    # Unattended runs (launchd) may clean low-risk items when free space drops
    # below this many bytes. 0 disables unattended cleaning entirely.
    auto_clean_below_bytes: int = 0
    # Extra never-touch paths (relative to home or absolute).
    protected: list[str] = field(default_factory=list)
    # Optional LLM planner: which provider to use, and the chosen model per provider
    # (`janitor login`, `janitor use PROVIDER MODEL`).
    llm_provider: str = "anthropic"
    llm_models: dict[str, str] = field(default_factory=dict)
    llm_effort: str = "medium"

    @classmethod
    def load(cls) -> "Config":
        path = data_dir() / "config.json"
        cfg = cls()
        if path.exists():
            raw = json.loads(path.read_text())
            known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
            cfg = cls(**{**asdict(cfg), **known})
        return cfg

    def save(self) -> Path:
        path = data_dir() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path
