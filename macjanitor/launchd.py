"""Periodic maintenance via a per-user launchd agent."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from macjanitor.config import data_dir, user_home

LABEL = "dev.macjanitor.auto"


def plist_path() -> Path:
    return Path(user_home()) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def build_plist(interval_seconds: int) -> dict:
    logs = data_dir()
    return {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "macjanitor", "auto"],
        "StartInterval": interval_seconds,
        "RunAtLoad": False,
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "Nice": 10,
        "StandardOutPath": str(logs / "auto.out.log"),
        "StandardErrorPath": str(logs / "auto.err.log"),
    }


def install(interval_seconds: int) -> Path:
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data_dir().mkdir(parents=True, exist_ok=True)
    if path.exists():
        _launchctl("bootout", f"gui/{os.getuid()}", str(path))
    with open(path, "wb") as f:
        plistlib.dump(build_plist(interval_seconds), f)
    _launchctl("bootstrap", f"gui/{os.getuid()}", str(path))
    return path


def uninstall() -> bool:
    path = plist_path()
    if not path.exists():
        return False
    _launchctl("bootout", f"gui/{os.getuid()}", str(path))
    path.unlink()
    return True


def _launchctl(*args: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["launchctl", *args], capture_output=True)


def notify(title: str, message: str) -> None:
    if sys.platform != "darwin":
        return
    script = f"display notification {_q(message)} with title {_q(title)}"
    subprocess.run(["osascript", "-e", script], capture_output=True)


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
