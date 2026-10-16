"""Provider registry, credential resolution, and login flows.

Three wire formats cover every provider:

* ``anthropic`` - Claude via the official Anthropic SDK (API key, or an
  ``ant auth login`` OAuth profile, which the SDK reads by itself)
* ``responses`` - OpenAI Responses API (API key, or Sign in with ChatGPT)
* ``chat``      - OpenAI-compatible Chat Completions (Gemini, Vertex AI,
  OpenRouter, Groq, xAI, Mistral, DeepSeek, … and local Ollama / LM Studio)

Credentials resolve in this order: stored login (``janitor login``) → the
provider's environment variables → (Claude only) the SDK's own chain.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from macjanitor.llm import chatgpt_oauth, store
from macjanitor.llm.http import LLMError, get_json

# Login methods
API_KEY, CHATGPT, ANT, ADC, NONE = "api_key", "chatgpt", "ant", "adc", "none"


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    api: str                       # anthropic | responses | chat
    base_url: str = ""
    env: tuple[str, ...] = ()
    logins: tuple[str, ...] = (API_KEY,)
    default_model: str | None = None
    key_url: str = ""


PROVIDERS: dict[str, Provider] = {p.id: p for p in (
    Provider("anthropic", "Anthropic (Claude)", "anthropic", env=("ANTHROPIC_API_KEY",),
             logins=(API_KEY, ANT), default_model="claude-opus-5-5",
             key_url="https://platform.claude.com/settings/keys"),
    Provider("openai", "OpenAI (API key or ChatGPT plan)", "responses", "https://api.openai.com/v1",
             env=("OPENAI_API_KEY",), logins=(CHATGPT, API_KEY),
             key_url="https://platform.openai.com/api-keys"),
    Provider("google", "Google Gemini (AI Studio key)", "chat",
             "https://generativelanguage.googleapis.com/v1beta/openai",
             env=("GEMINI_API_KEY", "GOOGLE_API_KEY"), key_url="https://aistudio.google.com/apikey"),
    Provider("google-vertex", "Google Vertex AI (gcloud login)", "chat", logins=(ADC,)),
    Provider("openrouter", "OpenRouter", "chat", "https://openrouter.ai/api/v1",
             env=("OPENROUTER_API_KEY",), key_url="https://openrouter.ai/keys"),
    Provider("groq", "Groq", "chat", "https://api.groq.com/openai/v1", env=("GROQ_API_KEY",),
             key_url="https://console.groq.com/keys"),
    Provider("xai", "xAI (Grok)", "chat", "https://api.x.ai/v1", env=("XAI_API_KEY",),
             key_url="https://console.x.ai"),
    Provider("mistral", "Mistral", "chat", "https://api.mistral.ai/v1", env=("MISTRAL_API_KEY",),
             key_url="https://console.mistral.ai/api-keys"),
    Provider("deepseek", "DeepSeek", "chat", "https://api.deepseek.com/v1", env=("DEEPSEEK_API_KEY",),
             default_model="deepseek-chat", key_url="https://platform.deepseek.com/api_keys"),
    Provider("cerebras", "Cerebras", "chat", "https://api.cerebras.ai/v1", env=("CEREBRAS_API_KEY",),
             key_url="https://cloud.cerebras.ai"),
    Provider("together", "Together AI", "chat", "https://api.together.ai/v1",
             env=("TOGETHER_API_KEY",), key_url="https://api.together.ai/settings/api-keys"),
    Provider("fireworks", "Fireworks AI", "chat", "https://api.fireworks.ai/inference/v1",
             env=("FIREWORKS_API_KEY",), key_url="https://fireworks.ai/account/api-keys"),
    Provider("huggingface", "Hugging Face Inference", "chat", "https://router.huggingface.co/v1",
             env=("HF_TOKEN",), key_url="https://huggingface.co/settings/tokens"),
    Provider("moonshot", "Moonshot AI (Kimi)", "chat", "https://api.moonshot.ai/v1",
             env=("MOONSHOT_API_KEY",), key_url="https://platform.moonshot.ai/console/api-keys"),
    Provider("ollama", "Ollama (local)", "chat", "http://localhost:11434/v1", logins=(NONE,)),
    Provider("lmstudio", "LM Studio (local)", "chat", "http://localhost:1234/v1", logins=(NONE,)),
    Provider("custom", "Any OpenAI-compatible endpoint", "chat", env=("CUSTOM_LLM_API_KEY",)),
)}

LOGIN_LABELS = {
    API_KEY: "Paste an API key",
    CHATGPT: "Sign in with ChatGPT (uses your Plus/Pro/Business plan)",
    ANT: "Log in with your Claude Console account (`ant auth login`)",
    ADC: "Log in with Google Cloud (`gcloud auth application-default login`)",
    NONE: "No key needed (local server)",
}

# Explained in `janitor login` so nobody wonders why these aren't offered.
NOT_OFFERED = (
    "Claude Pro/Max and Gemini CLI/Antigravity subscription logins are not offered: Anthropic's "
    "and Google's terms only allow those OAuth tokens in their own apps, and accounts using them "
    "elsewhere have been blocked. Use an API key, `ant auth login`, or Vertex AI instead."
)


def get(provider_id: str) -> Provider:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        raise LLMError(f"unknown provider {provider_id!r}; see `janitor login --list`") from None


@dataclass
class Auth:
    kind: str                 # api_key | bearer | sdk | none
    token: str | None
    base_url: str
    source: str
    chatgpt: bool = False


def resolve(p: Provider) -> Auth:
    cred = store.get(p.id) or {}
    base = cred.get("base_url") or p.base_url
    kind = cred.get("type")

    if kind == API_KEY and cred.get("key"):
        return Auth(API_KEY, cred["key"], base, "stored key")
    if kind == "oauth":
        if time.time() >= cred.get("expires", 0):
            cred = {**cred, **chatgpt_oauth.refresh(cred)}
            store.put(p.id, cred)
        return Auth("bearer", cred["access"], base, "Sign in with ChatGPT", chatgpt=True)
    if kind == ADC:
        return Auth("bearer", _gcloud_token(), _vertex_base(cred), f"gcloud ADC ({cred['project']})")
    for var in p.env:
        if os.environ.get(var):
            if p.id == "custom" and not base:
                base = os.environ.get("CUSTOM_LLM_BASE_URL", "")
            return Auth(API_KEY, os.environ[var], base, f"${var}")
    if p.api == "anthropic":
        # Unset key: the SDK falls back to ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
        return Auth("sdk", None, base, "Anthropic SDK credential chain" + (" (ant)" if kind == ANT else ""))
    if NONE in p.logins:
        return Auth("none", None, base, "local")
    raise LLMError(f"{p.name} is not configured. Run `janitor login {p.id}`.")


def status(p: Provider) -> str:
    cred = store.get(p.id) or {}
    if cred.get("type") == API_KEY:
        return f"stored key {store.mask(cred['key'])}"
    if cred.get("type") == "oauth":
        return "Signed in with ChatGPT"
    if cred.get("type") == ADC:
        return f"gcloud ADC · {cred.get('project')} / {cred.get('location')}"
    if cred.get("type") == ANT:
        return "ant auth login profile"
    if cred.get("type") == NONE:
        return f"local server {cred.get('base_url') or p.base_url}"
    for var in p.env:
        if os.environ.get(var):
            return f"${var}"
    return ""


# --- Vertex AI --------------------------------------------------------------
def _gcloud_token() -> str:
    if not shutil.which("gcloud"):
        raise LLMError("gcloud is not installed; see https://cloud.google.com/sdk/docs/install")
    p = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise LLMError("gcloud has no application-default login. Run `janitor login google-vertex`.")
    return p.stdout.strip()


def _vertex_base(cred: dict) -> str:
    loc = cred.get("location") or "global"
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{cred['project']}/locations/{loc}/endpoints/openapi"


# --- headers & models -------------------------------------------------------
def headers(p: Provider, auth: Auth) -> dict:
    h = {}
    if auth.token:
        h["authorization"] = f"Bearer {auth.token}"
    if p.id == "openrouter":
        h |= {"x-title": "MacJanitor", "http-referer": "https://github.com/chrishyoroklee/janitor-agent"}
    return h


def list_models(p: Provider) -> list[str]:
    auth = resolve(p)
    if p.api == "anthropic":
        from macjanitor.llm.chat import anthropic_client

        return [m.id for m in anthropic_client(auth).models.list()]
    if p.id == "google-vertex":
        return []  # no OpenAI-style listing on Vertex; models are named google/<gemini-model-id>
    data = get_json(f"{auth.base_url.rstrip('/')}/models", headers(p, auth))
    rows = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
    ids = [r.get("id") or r.get("name", "") for r in rows if isinstance(r, dict)]
    return sorted({i.removeprefix("models/") for i in ids if i})


# --- login ------------------------------------------------------------------
def _choose(prompt: str, options: list[str]) -> int:
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        raw = input(f"{prompt} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1


def login(p: Provider, *, method: str | None = None, key: str | None = None,
          base_url: str | None = None) -> str:
    """Run the interactive login for `p`. Returns a one-line summary."""
    methods = list(p.logins)
    if method is None and key and API_KEY in methods:
        method = API_KEY
    if method is None:
        method = methods[0] if len(methods) == 1 else methods[_choose(
            f"How do you want to connect {p.name}?", [LOGIN_LABELS[m] for m in methods])]
    if method not in methods:
        raise LLMError(f"{p.name} supports: {', '.join(methods)}")

    if method == NONE:
        cred = {"type": NONE, **({"base_url": base_url} if base_url else {})}
        store.put(p.id, cred)
        return f"{p.name}: using {base_url or p.base_url}"

    if method == API_KEY:
        if p.id == "custom":
            base_url = base_url or input("Base URL (e.g. https://host/v1): ").strip()
            if not base_url.startswith(("http://", "https://")):
                raise LLMError("base URL must start with http:// or https://")
        if key is None:
            if p.key_url:
                print(f"Get a key at {p.key_url}")
            key = getpass.getpass(f"{p.name} API key (input hidden): ") if sys.stdin.isatty() \
                else sys.stdin.readline()
        key = key.strip()
        if not key:
            raise LLMError("no key entered")
        cred = {"type": API_KEY, "key": key, **({"base_url": base_url} if base_url else {})}
        previous = store.get(p.id)
        store.put(p.id, cred)
        try:
            n = len(list_models(p))
            return f"{p.name}: key saved and verified ({n} models available)"
        except Exception as e:  # LLMError, Anthropic SDK errors, missing optional dependency
            status = getattr(e, "status", None) or getattr(e, "status_code", None)
            # Gemini answers a bad key with 400 "API key not valid".
            if status in (401, 403) or (status == 400 and "key" in str(e).lower()):
                if previous:
                    store.put(p.id, previous)
                else:
                    store.remove(p.id)
                raise LLMError(f"{p.name} rejected that key (HTTP {status})") from None
            return f"{p.name}: key saved, but could not verify it now ({e})"

    if method == CHATGPT:
        store.put(p.id, chatgpt_oauth.login())
        return "Signed in with ChatGPT. Usage counts against your ChatGPT plan."

    if method == ANT:
        if not shutil.which("ant"):
            raise LLMError("the Anthropic CLI (`ant`) is not installed. Install it, then run "
                           "`ant auth login`; or choose the API key option.")
        if subprocess.run(["ant", "auth", "login"]).returncode != 0:
            raise LLMError("`ant auth login` did not complete")
        store.put(p.id, {"type": ANT})
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("Note: ANTHROPIC_API_KEY is set and takes precedence over the ant profile.")
        return "Claude Console login saved by `ant`; the Anthropic SDK will use it."

    if method == ADC:
        if not shutil.which("gcloud"):
            raise LLMError("install the Google Cloud CLI first: https://cloud.google.com/sdk/docs/install")
        if subprocess.run(["gcloud", "auth", "application-default", "login"]).returncode != 0:
            raise LLMError("gcloud login did not complete")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or input("Google Cloud project ID: ").strip()
        location = input("Location [global]: ").strip() or "global"
        store.put(p.id, {"type": ADC, "project": project, "location": location})
        return f"Vertex AI ready for project {project} ({location}). Pick a model with " \
               f"`janitor use google-vertex google/<gemini-model-id>`."
    raise AssertionError(method)


def logout(p: Provider) -> bool:
    removed = store.remove(p.id)
    if p.id == "anthropic" and shutil.which("ant"):
        print("To also sign out of the ant profile, run `ant auth logout`.")
    return removed
