"""One tool-calling conversation interface over three wire formats.

Each adapter keeps its provider's *native* message history and only appends
to it, so nothing is lost converting between formats (reasoning items,
thinking blocks and tool-call ids round-trip untouched).

    chat = open_chat(provider, model, system, tools, effort)
    chat.user("question")
    turn = chat.step()                    # -> Turn(text, calls, refused)
    chat.tool_results([(call_id, json_text, is_error), ...])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from macjanitor.llm import providers
from macjanitor.llm.http import LLMError, post_json, post_sse
from macjanitor.llm.providers import Auth, Provider


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    refused: bool = False


def _args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        raise LLMError(f"model sent invalid tool arguments: {e}") from None
    return val if isinstance(val, dict) else {}


def _plain_schema(schema: dict) -> dict:
    """Drop JSON-Schema keywords some OpenAI-compatible backends (Gemini) reject."""
    out = {}
    for k, v in schema.items():
        if k in ("additionalProperties", "maxLength", "minItems"):
            continue
        if isinstance(v, dict):
            v = {kk: _plain_schema(vv) if isinstance(vv, dict) else vv for kk, vv in v.items()} \
                if k == "properties" else _plain_schema(v)
        out[k] = v
    return out


# --- Anthropic (official SDK) ----------------------------------------------
def anthropic_client(auth: Auth):
    try:
        import anthropic
    except ImportError:
        raise LLMError("Claude needs the Anthropic SDK: pip install 'macjanitor[llm]'") from None
    if auth.kind == providers.API_KEY:
        return anthropic.Anthropic(api_key=auth.token)
    return anthropic.Anthropic()  # ANTHROPIC_AUTH_TOKEN or the `ant auth login` profile


# Models that take `output_config.effort` and server-side refusal fallbacks.
_CLAUDE_CURRENT = ("claude-opus-5", "claude-fable-5", "claude-sonnet-5-5")


class AnthropicChat:
    def __init__(self, p: Provider, auth: Auth, model: str, system: str, tools: list[dict], effort: str):
        import anthropic

        self.err = anthropic
        self.client = anthropic_client(auth)
        self.model, self.system, self.effort = model, system, effort
        self.tools = [{"name": t["name"], "description": t["description"],
                       "input_schema": t["parameters"]} for t in tools]
        self.messages: list = []

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self) -> Turn:
        a = self.err
        current = self.model.startswith(_CLAUDE_CURRENT)
        kw = {"output_config": {"effort": self.effort},
              "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if current else {}
        try:
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=self.system, tools=self.tools,
                messages=self.messages, **kw)
        except a.AuthenticationError:
            raise LLMError("Anthropic rejected the credentials; run `janitor login anthropic`") from None
        except a.RateLimitError:
            raise LLMError("rate limited by the Anthropic API; try again shortly") from None
        except a.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}", e.status_code) from None
        except a.APIConnectionError:
            raise LLMError("could not reach the Anthropic API") from None
        if resp.stop_reason == "refusal":
            return Turn(refused=True)
        self.messages.append({"role": "assistant", "content": resp.content})
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return Turn(text, calls if resp.stop_reason == "tool_use" else [])

    def tool_results(self, results) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": cid, "content": out, "is_error": err}
            for cid, out, err in results]})


# --- OpenAI Responses API (API key or Sign in with ChatGPT) ------------------
class ResponsesChat:
    def __init__(self, p: Provider, auth: Auth, model: str, system: str, tools: list[dict], effort: str):
        self.p, self.auth, self.model, self.system, self.effort = p, auth, model, system, effort
        self.url = f"{auth.base_url.rstrip('/')}/responses"
        self.tools = [{"type": "function", "name": t["name"], "description": t["description"],
                       "parameters": t["parameters"], "strict": False} for t in tools]
        self.input: list = []
        self.include_reasoning = True

    def user(self, text: str) -> None:
        self.input.append({"role": "user", "content": text})

    def _body(self) -> dict:
        body = {"model": self.model, "instructions": self.system, "input": self.input,
                "tools": self.tools, "store": False, "stream": True}
        if self.include_reasoning:
            body["include"] = ["reasoning.encrypted_content"]
        return body

    def step(self) -> Turn:
        hdrs = providers.headers(self.p, self.auth)
        try:
            response = self._stream(hdrs)
        except LLMError as e:
            # Non-reasoning models reject `include`; retry once without it.
            if e.status == 400 and self.include_reasoning and "include" in e.body + str(e):
                self.include_reasoning = False
                response = self._stream(hdrs)
            else:
                raise
        turn = Turn()
        for item in response.get("output", []):
            t = item.get("type")
            if t == "reasoning":
                self.input.append(item)            # carries encrypted_content forward
            elif t == "function_call":
                self.input.append({k: item[k] for k in ("type", "call_id", "name", "arguments")})
                turn.calls.append(ToolCall(item["call_id"], item["name"], _args(item.get("arguments"))))
            elif t == "message":
                parts = item.get("content", [])
                turn.text += "".join(c.get("text", "") for c in parts if c.get("type") == "output_text")
                if any(c.get("type") == "refusal" for c in parts):
                    turn.refused = True
                self.input.append({"role": "assistant", "content": [
                    {"type": "output_text", "text": c.get("text", "")}
                    for c in parts if c.get("type") == "output_text"]})
        turn.text = turn.text.strip()
        return turn

    def _stream(self, hdrs: dict) -> dict:
        for event in post_sse(self.url, self._body(), hdrs):
            kind = event.get("type")
            if kind == "response.completed":
                return event["response"]
            if kind in ("response.failed", "response.incomplete", "error"):
                r = event.get("response", event)
                err = r.get("error") or r.get("incomplete_details") or r
                msg = err.get("message", err) if isinstance(err, dict) else err
                raise LLMError(f"OpenAI: {msg}")
        raise LLMError("OpenAI stream ended without a completed response")

    def tool_results(self, results) -> None:
        self.input += [{"type": "function_call_output", "call_id": cid, "output": out}
                       for cid, out, _ in results]


# --- OpenAI-compatible Chat Completions --------------------------------------
class ChatCompletionsChat:
    def __init__(self, p: Provider, auth: Auth, model: str, system: str, tools: list[dict], effort: str):
        if not auth.base_url:
            raise LLMError(f"{p.name} has no base URL; run `janitor login {p.id}`")
        self.p, self.auth, self.model = p, auth, model
        self.url = f"{auth.base_url.rstrip('/')}/chat/completions"
        self.tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": _plain_schema(t["parameters"])}} for t in tools]
        self.messages: list = [{"role": "system", "content": system}]

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self) -> Turn:
        data = post_json(self.url, {"model": self.model, "messages": self.messages,
                                    "tools": self.tools}, providers.headers(self.p, self.auth))
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.p.name} returned no choices: {str(data)[:300]}")
        msg = choices[0].get("message", {})
        calls = msg.get("tool_calls") or []
        kept = {"role": "assistant", "content": msg.get("content") or ""}
        if calls:
            kept["tool_calls"] = calls
        self.messages.append(kept)
        return Turn((msg.get("content") or "").strip(),
                    [ToolCall(c["id"], c["function"]["name"], _args(c["function"].get("arguments")))
                     for c in calls],
                    refused=bool(msg.get("refusal")))

    def tool_results(self, results) -> None:
        self.messages += [{"role": "tool", "tool_call_id": cid, "content": out} for cid, out, _ in results]


ADAPTERS = {"anthropic": AnthropicChat, "responses": ResponsesChat, "chat": ChatCompletionsChat}


def open_chat(provider_id: str, model: str | None, system: str, tools: list[dict], effort: str = "medium"):
    p = providers.get(provider_id)
    model = model or p.default_model
    if not model:
        raise LLMError(f"no model chosen for {p.name}. Run `janitor models {p.id}` and "
                       f"`janitor use {p.id} <model>`.")
    return ADAPTERS[p.api](p, providers.resolve(p), model, system, tools, effort), model
