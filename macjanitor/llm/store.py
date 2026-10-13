"""Credential store: `auth.json` next to the database, readable only by you.

Layout (one entry per provider):

    {"openai":    {"type": "oauth", "access": "...", "refresh": "...", "expires": 1760000000,
                   "client_id": "..."},
     "groq":      {"type": "api_key", "key": "gsk_..."},
     "custom":    {"type": "api_key", "key": "...", "base_url": "https://..."},
     "google-vertex": {"type": "adc", "project": "my-proj", "location": "global"},
     "_device_id": "uuid"}

Writes are atomic (temp file + rename) and the file is created with mode 0600.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from macjanitor.config import data_dir


def auth_path() -> Path:
    return data_dir() / "auth.json"


def load() -> dict:
    p = auth_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def save(data: dict) -> None:
    p = auth_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def get(provider: str) -> dict | None:
    return load().get(provider)


def put(provider: str, cred: dict) -> None:
    data = load()
    data[provider] = cred
    save(data)


def remove(provider: str) -> bool:
    data = load()
    if provider not in data:
        return False
    del data[provider]
    save(data)
    return True


def device_id() -> str:
    """A stable random id for this installation (used by Sign in with ChatGPT)."""
    data = load()
    if not data.get("_device_id"):
        data["_device_id"] = str(uuid.uuid4())
        save(data)
    return data["_device_id"]


def mask(secret: str) -> str:
    return secret[:4] + "…" + secret[-4:] if len(secret) > 12 else "…"
