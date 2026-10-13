"""Tiny stdlib HTTP client for LLM providers: JSON POST/GET, SSE, retries."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

USER_AGENT = "macjanitor"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def _request(method: str, url: str, headers: dict, body: dict | None, timeout: float):
    data = json.dumps(body).encode() if body is not None else None
    h = {"user-agent": USER_AGENT, "accept": "application/json", **headers}
    if data is not None:
        h.setdefault("content-type", "application/json")
    return urllib.request.Request(url, data=data, headers=h, method=method)


def _open(req: urllib.request.Request, timeout: float, retries: int):
    host = urllib.parse.urlsplit(req.full_url).netloc  # req.host becomes the proxy's
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code in RETRY_STATUS and attempt < retries:
                time.sleep(min(float(e.headers.get("retry-after") or 2 ** attempt), 30))
                continue
            raise LLMError(f"HTTP {e.code} from {host}: {_error_message(body)}", e.code, body) from None
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"cannot reach {host}: {e.reason}") from None
    raise AssertionError


def _error_message(body: str) -> str:
    try:
        err = json.loads(body)
        err = err[0] if isinstance(err, list) and err else err
        e = err.get("error", err) if isinstance(err, dict) else err
        if isinstance(e, dict):
            return str(e.get("message") or e)[:400]
        return str(e)[:400]
    except (json.JSONDecodeError, AttributeError):
        return body[:400]


def post_json(url: str, body: dict, headers: dict, *, timeout: float = 180, retries: int = 2) -> dict:
    with _open(_request("POST", url, headers, body, timeout), timeout, retries) as r:
        return json.loads(r.read().decode())


def get_json(url: str, headers: dict, *, timeout: float = 30, retries: int = 1) -> dict:
    with _open(_request("GET", url, headers, None, timeout), timeout, retries) as r:
        return json.loads(r.read().decode())


def post_form(url: str, form: dict, *, timeout: float = 30) -> dict:
    data = urllib.parse.urlencode(form).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "user-agent": USER_AGENT, "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded"})
    with _open(req, timeout, 1) as r:
        return json.loads(r.read().decode())


def post_sse(url: str, body: dict, headers: dict, *, timeout: float = 300) -> Iterator[dict]:
    """POST and yield each `data:` JSON event of a server-sent event stream."""
    req = _request("POST", url, {**headers, "accept": "text/event-stream"}, body, timeout)
    with _open(req, timeout, 2) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue

