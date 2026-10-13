"""Sign in with ChatGPT (OpenAI's public OAuth flow for agents).

Same flow Pi uses: PKCE + a dynamically registered client. OpenAI issues a
client id per login and the access token is sent directly to the OpenAI
Responses API, billed against the user's ChatGPT plan. No client secret.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import secrets
import sys
import threading
import time
import urllib.parse
import webbrowser

from macjanitor.llm import store
from macjanitor.llm.http import LLMError, post_form

DYNAMIC_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME = "MacJanitor"
AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://127.0.0.1:{PORT}{CALLBACK_PATH}"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPE = f"openid profile email offline_access resource.invoke {DIRECT_SCOPE}"
EXPIRY_MARGIN = 180
LOGIN_TIMEOUT = 300

_PAGE = ("<!doctype html><meta charset=utf-8><title>MacJanitor</title>"
         "<body style='font:16px system-ui;margin:4em'><h2>{}</h2><p>{}</p>")


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    return verifier, _b64url(hashlib.sha256(verifier.encode()).digest())


def _parse_callback(url: str, state: str) -> tuple[str, str]:
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "error" in q:
        raise LLMError(f"ChatGPT authorization failed: {q['error'][0]}")
    if q.get("state", [""])[0] != state:
        raise LLMError("OAuth state mismatch")
    code, client_id = q.get("code", [""])[0], q.get("client_id", [""])[0].strip()
    if not code or not client_id:
        raise LLMError("callback is missing the authorization code or issued client id")
    return code, client_id


def _credential(token: dict, client_id: str) -> dict:
    scopes = str(token.get("scope", "")).split()
    if not token.get("access_token") or not token.get("refresh_token"):
        raise LLMError("token response is missing tokens")
    if DIRECT_SCOPE not in scopes:
        raise LLMError(f"OpenAI did not grant {DIRECT_SCOPE}; is your ChatGPT plan eligible?")
    return {"type": "oauth", "access": token["access_token"], "refresh": token["refresh_token"],
            "expires": time.time() + float(token.get("expires_in", 3600)) - EXPIRY_MARGIN,
            "client_id": client_id, "scopes": scopes}


def login() -> dict:
    verifier, challenge = _pkce()
    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    result: dict = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if not self.path.startswith(CALLBACK_PATH):
                self.send_error(404)
                return
            try:
                result["value"] = _parse_callback(self.path, state)
                status, body = 200, _PAGE.format("ChatGPT connected", "You can close this window.")
            except LLMError as e:
                result.setdefault("error", e)
                status, body = 400, _PAGE.format("ChatGPT was not connected", str(e))
            self.send_response(status)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())
            done.set()

        def log_message(self, *args):
            pass

    try:
        server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        raise LLMError(f"port {PORT} is busy (an unfinished login, or the Codex CLI?). "
                       "Close it and try again.") from None
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = AUTHORIZE_URL + "?" + urllib.parse.urlencode({
        "client_id": DYNAMIC_CLIENT_ID, "agent_name_hint": AGENT_NAME,
        "ext_agent_host_id": f"urn:uuid:{store.device_id()}",
        "response_type": "code", "redirect_uri": REDIRECT_URI, "resource": RESOURCE,
        "scope": SCOPE, "state": state, "code_challenge": challenge,
        "code_challenge_method": "S256", "nonce": nonce,
    })
    print("Opening your browser to sign in with ChatGPT. If it does not open, visit:\n\n"
          f"  {url}\n")
    webbrowser.open(url)

    if sys.stdin.isatty():
        # On a remote/SSH machine the redirect can't reach us: accept a pasted URL too.
        def manual():
            try:
                pasted = input("Waiting for the browser… (or paste the final redirect URL here)\n> ")
            except EOFError:
                return
            if pasted.strip() and not done.is_set():
                try:
                    result["value"] = _parse_callback(pasted.strip(), state)
                except LLMError as e:
                    result["error"] = e
                done.set()
        threading.Thread(target=manual, daemon=True).start()

    try:
        if not done.wait(LOGIN_TIMEOUT):
            raise LLMError("timed out waiting for the browser sign-in")
    finally:
        server.shutdown()
        server.server_close()
    if "value" not in result:
        raise result.get("error") or LLMError("sign-in failed")

    code, client_id = result["value"]
    token = post_form(TOKEN_URL, {
        "grant_type": "authorization_code", "client_id": client_id, "code": code,
        "code_verifier": verifier, "redirect_uri": REDIRECT_URI, "resource": RESOURCE,
    })
    if not token.get("id_token"):
        raise LLMError("token response did not include an ID token")
    return _credential(token, client_id)


def refresh(cred: dict) -> dict:
    token = post_form(TOKEN_URL, {"grant_type": "refresh_token", "client_id": cred["client_id"],
                                  "refresh_token": cred["refresh"], "resource": RESOURCE})
    return _credential(token, cred["client_id"])
