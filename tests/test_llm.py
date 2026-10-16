"""LLM provider layer tests: credentials, logins, and the three chat adapters (no network)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import threading
import time
import unittest
import urllib.parse
import urllib.request
from unittest import mock

from macjanitor.llm import chatgpt_oauth, providers, store
from macjanitor.llm.chat import ChatCompletionsChat, ResponsesChat, open_chat
from macjanitor.llm.http import LLMError

TOOLS = [{"name": "list_candidates", "description": "d",
          "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}},
                         "additionalProperties": False}}]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        clean = {k: "" for k in os.environ if k.endswith(("_API_KEY", "HF_TOKEN"))}
        self.env = mock.patch.dict(os.environ, {**clean, "JANITOR_DATA_DIR": self.tmp,
                                                "JANITOR_USER_HOME": self.tmp})
        self.env.start()
        for k in clean:
            os.environ.pop(k)

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


class StoreAndResolveTests(Base):
    def test_auth_file_is_private(self):
        store.put("groq", {"type": "api_key", "key": "gsk_1234567890abcdef"})
        mode = stat.S_IMODE(os.stat(store.auth_path()).st_mode)
        self.assertEqual(mode, 0o600)
        self.assertEqual(store.mask("gsk_1234567890abcdef"), "gsk_…cdef")

    def test_stored_key_beats_env(self):
        p = providers.get("groq")
        with self.assertRaises(LLMError):
            providers.resolve(p)
        os.environ["GROQ_API_KEY"] = "from-env"
        self.assertEqual(providers.resolve(p).token, "from-env")
        store.put("groq", {"type": "api_key", "key": "stored"})
        a = providers.resolve(p)
        self.assertEqual((a.token, a.source), ("stored", "stored key"))

    def test_anthropic_falls_back_to_sdk_chain(self):
        a = providers.resolve(providers.get("anthropic"))
        self.assertEqual(a.kind, "sdk")

    def test_custom_endpoint_and_local(self):
        store.put("custom", {"type": "api_key", "key": "k", "base_url": "https://llm.example/v1"})
        self.assertEqual(providers.resolve(providers.get("custom")).base_url, "https://llm.example/v1")
        self.assertEqual(providers.resolve(providers.get("ollama")).kind, "none")

    def test_expired_chatgpt_token_is_refreshed(self):
        store.put("openai", {"type": "oauth", "access": "old", "refresh": "r", "expires": 0,
                             "client_id": "cid"})
        fresh = {"access_token": "new", "refresh_token": "r2", "expires_in": 3600,
                 "scope": f"openid {chatgpt_oauth.DIRECT_SCOPE}"}
        with mock.patch.object(chatgpt_oauth, "post_form", return_value=fresh) as pf:
            a = providers.resolve(providers.get("openai"))
        self.assertEqual(a.token, "new")
        self.assertTrue(a.chatgpt)
        self.assertEqual(pf.call_args[0][1]["grant_type"], "refresh_token")
        self.assertEqual(store.get("openai")["refresh"], "r2")

    def test_rejected_key_is_not_saved(self):
        p = providers.get("mistral")
        err = LLMError("nope", 401)
        with mock.patch.object(providers, "list_models", side_effect=err):
            with self.assertRaises(LLMError):
                providers.login(p, key="bad-key")
        self.assertIsNone(store.get("mistral"))
        with mock.patch.object(providers, "list_models", return_value=["m1", "m2"]):
            self.assertIn("verified", providers.login(p, key="good-key"))
        self.assertEqual(store.get("mistral")["key"], "good-key")

    def test_key_flag_selects_api_key_login(self):
        # OpenAI lists ChatGPT sign-in first; passing --key must not open the browser.
        with mock.patch.object(chatgpt_oauth, "login", side_effect=AssertionError("browser")), \
                mock.patch.object(providers, "list_models", return_value=["m"]):
            providers.login(providers.get("openai"), key="sk-test")
        self.assertEqual(store.get("openai"), {"type": "api_key", "key": "sk-test"})


class ChatGPTOAuthTests(Base):
    def test_callback_validation(self):
        with self.assertRaises(LLMError):
            chatgpt_oauth._parse_callback("/auth/callback?code=c&state=WRONG&client_id=x", "s")
        with self.assertRaises(LLMError):
            chatgpt_oauth._parse_callback("/auth/callback?code=c&state=s", "s")  # no issued client
        self.assertEqual(chatgpt_oauth._parse_callback("/auth/callback?code=c&state=s&client_id=x", "s"),
                         ("c", "x"))

    def test_requires_direct_scope(self):
        with self.assertRaises(LLMError):
            chatgpt_oauth._credential({"access_token": "a", "refresh_token": "r", "scope": "openid"}, "x")

    def test_full_browser_flow(self):
        """Simulate the browser: follow the authorize URL's redirect to the local callback server."""
        def fake_browser(url):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            self.assertEqual(q["client_id"], ["dynamic_agent_client"])
            self.assertEqual(q["code_challenge_method"], ["S256"])
            self.assertTrue(q["ext_agent_host_id"][0].startswith("urn:uuid:"))
            cb = f"{q['redirect_uri'][0]}?code=abc&state={q['state'][0]}&client_id=issued-1"

            def hit():
                time.sleep(0.2)
                urllib.request.urlopen(cb, timeout=5).read()
            threading.Thread(target=hit, daemon=True).start()
            return True

        token = {"access_token": "at", "refresh_token": "rt", "id_token": "it", "expires_in": 3600,
                 "scope": f"openid offline_access {chatgpt_oauth.DIRECT_SCOPE}"}
        with mock.patch("webbrowser.open", fake_browser), \
                mock.patch("sys.stdin.isatty", return_value=False), \
                mock.patch.object(chatgpt_oauth, "post_form", return_value=token) as pf, \
                mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
            cred = chatgpt_oauth.login()
        self.assertEqual((cred["access"], cred["client_id"]), ("at", "issued-1"))
        form = pf.call_args[0][1]
        self.assertEqual((form["code"], form["client_id"]), ("abc", "issued-1"))
        self.assertTrue(form["code_verifier"])


class AdapterTests(Base):
    def test_responses_round_trip(self):
        store.put("openai", {"type": "api_key", "key": "sk-test"})
        chat, model = open_chat("openai", "some-model", "sys", TOOLS)
        self.assertIsInstance(chat, ResponsesChat)
        first = {"type": "response.completed", "response": {"output": [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "xyz", "summary": []},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "list_candidates",
             "arguments": '{"limit": 3}'}]}}
        second = {"type": "response.completed", "response": {"output": [
            {"type": "message", "id": "m1", "content": [{"type": "output_text", "text": "Done."}]}]}}
        with mock.patch("macjanitor.llm.chat.post_sse", side_effect=[iter([first]), iter([second])]) as sse:
            chat.user("hi")
            t = chat.step()
            self.assertEqual((t.calls[0].name, t.calls[0].args), ("list_candidates", {"limit": 3}))
            chat.tool_results([("call_1", "{}", False)])
            self.assertEqual(chat.step().text, "Done.")
        body = sse.call_args_list[1][0][1]
        kinds = [i.get("type", i.get("role")) for i in body["input"]][:4]  # list is shared, grows after
        self.assertEqual(kinds, ["user", "reasoning", "function_call", "function_call_output"])
        self.assertNotIn("id", body["input"][2])  # store:false -> no server item ids
        self.assertEqual(sse.call_args_list[0][0][2]["authorization"], "Bearer sk-test")

    def test_chat_completions_round_trip(self):
        os.environ["GEMINI_API_KEY"] = "g-key"
        chat, _ = open_chat("google", "gemini-x", "sys", TOOLS)
        self.assertIsInstance(chat, ChatCompletionsChat)
        self.assertNotIn("additionalProperties", json.dumps(chat.tools))
        r1 = {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "list_candidates", "arguments": "{}"}}]}}]}
        r2 = {"choices": [{"message": {"content": "All good."}}]}
        with mock.patch("macjanitor.llm.chat.post_json", side_effect=[r1, r2]) as pj:
            chat.user("hi")
            self.assertEqual(chat.step().calls[0].id, "t1")
            chat.tool_results([("t1", '{"ok": true}', False)])
            self.assertEqual(chat.step().text, "All good.")
        url, body, hdrs = pj.call_args[0]
        self.assertTrue(url.startswith("https://generativelanguage.googleapis.com/"))
        self.assertEqual([m["role"] for m in body["messages"]][:4], ["system", "user", "assistant", "tool"])

    def test_model_required_when_no_default(self):
        os.environ["GROQ_API_KEY"] = "k"
        with self.assertRaises(LLMError):
            open_chat("groq", None, "sys", TOOLS)


class AgentWithAnyProviderTests(Base):
    def test_agent_proposes_plan_through_policy(self):
        from macjanitor.agent import Agent
        from macjanitor.analyzer import Analyzer
        from macjanitor.config import Config
        from macjanitor.db import Memory
        from macjanitor.policy import Policy

        cache = os.path.join(self.tmp, "Library/Caches/pip")
        os.makedirs(cache)
        with open(os.path.join(cache, "w"), "wb") as f:
            f.write(os.urandom(2_000_000))
        cfg = Config(min_candidate_bytes=1)
        mem = Memory()
        home = os.path.realpath(self.tmp)
        cands = Analyzer(home, cfg, tools=False).run()
        sid = mem.record_scan(time.time(), 1, 1, cands)
        cid = cands[0].cid
        replies = [
            {"choices": [{"message": {"content": "", "tool_calls": [{"id": "a", "type": "function",
             "function": {"name": "propose_plan", "arguments": json.dumps(
                 {"ids": [cid, "bogus"], "rationale": "pip cache is safe"})}}]}}]},
            {"choices": [{"message": {"content": "Plan saved."}}]},
        ]
        with mock.patch("macjanitor.llm.chat.post_json", side_effect=replies):
            agent = Agent(mem, Policy(home), cfg, home, scan_id=sid, provider="ollama", model="llama")
            res = agent.ask("free space")
        self.assertEqual(res.text, "Plan saved.")
        self.assertEqual(mem.plan_cids(res.plan_id), [cid])
        mem.close()


if __name__ == "__main__":
    unittest.main()
