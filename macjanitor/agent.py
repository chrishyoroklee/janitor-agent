"""Optional LLM planner, over any provider in `macjanitor.llm`.

The model gets five read-mostly tools over the engine's own data. It can
explain, prioritize and *propose* a plan of candidate ids; it cannot name a
path, a command or an action. Proposals go through the same policy engine as
deterministic plans, are saved as a plan, and still need the user's approval
in `janitor clean`.

Connect a provider with `janitor login` (API key, Sign in with ChatGPT,
`ant auth login`, gcloud, or a local Ollama / LM Studio server).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from macjanitor import rules
from macjanitor.analyzer import Candidate
from macjanitor.config import Config
from macjanitor.db import Memory
from macjanitor.fsutil import disk_usage, fmt_bytes
from macjanitor.llm.chat import open_chat
from macjanitor.llm.http import LLMError
from macjanitor.planner import validate_selection
from macjanitor.policy import Policy
from macjanitor.rules import Risk

MAX_TURNS = 12

SYSTEM = """You are MacJanitor, a storage maintenance agent for one person's Mac.

You work only through the tools provided. They expose a deterministic scan of \
cleanup candidates; each candidate has an id, a rule, a risk level and an \
estimated number of reclaimable bytes. You cannot delete anything yourself: \
`propose_plan` stores a plan that the user reviews and approves later, and a \
policy engine may reject items you choose.

How to work:
- Answer from the data. Quote sizes and risks from tool results, don't guess.
- Prefer the least disruptive way to reach the user's goal: low risk before \
medium, medium before high, caches before anything personal.
- Explain *why* space is used in plain language (e.g. "Xcode build products \
for projects you haven't opened since spring").
- Call `propose_plan` only when the user wants space freed or asks for a plan.
- Use `remember` sparingly, for durable preferences the user states (e.g. \
"keeps Docker images for offline work").
Keep the final answer short: a few sentences or a compact list."""

TOOLS = [
    {
        "name": "storage_overview",
        "description": "Disk capacity, free space, free-space trend, and reclaimable bytes per "
                       "category and rule from the latest scan.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_candidates",
        "description": "List cleanup candidates from the latest scan, largest first.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                "rule_id": {"type": "string", "description": "Only candidates of this rule."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_candidate",
        "description": "Full detail for one candidate: rule explanation, size history across "
                       "scans, and whether policy currently allows acting on it.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose_plan",
        "description": "Save a cleanup plan of candidate ids, in execution order, for the user to "
                       "review. Returns which items policy accepted or rejected and the plan id.",
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "rationale": {"type": "string", "description": "One or two sentences for the user."},
            },
            "required": ["ids", "rationale"],
            "additionalProperties": False,
        },
    },
    {
        "name": "remember",
        "description": "Store a short durable note about the user's preferences or machine for "
                       "future sessions.",
        "parameters": {
            "type": "object",
            "properties": {"note": {"type": "string", "maxLength": 300}},
            "required": ["note"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class AgentResult:
    text: str
    plan_id: int | None = None
    model: str = ""


class Agent:
    def __init__(self, mem: Memory, policy: Policy, config: Config, home: str, *,
                 scan_id: int, max_risk: Risk = Risk.MEDIUM, provider: str | None = None,
                 model: str | None = None):
        self.mem, self.policy, self.config, self.home = mem, policy, config, home
        self.scan_id, self.max_risk = scan_id, max_risk
        self.provider = provider or config.llm_provider
        self.model = model or config.llm_models.get(self.provider)
        self.cands = {r["cid"]: Candidate.from_row(r) for r in mem.candidates(scan_id)}
        self.plan_id: int | None = None

    # -- tools ---------------------------------------------------------------
    def _tool(self, name: str, args: dict) -> dict:
        if name == "storage_overview":
            total, _, free = disk_usage(self.home)
            first, last = self.mem.disk_trend(time.time() - 7 * 86400)
            by_rule: dict[str, int] = {}
            for c in self.cands.values():
                by_rule[c.rule_id] = by_rule.get(c.rule_id, 0) + c.reclaimable
            return {
                "disk_total": fmt_bytes(total), "disk_free": fmt_bytes(free),
                "free_change_7d": fmt_bytes(last["free"] - first["free"]) if first and last else None,
                "reclaimable_by_rule": {
                    k: {"bytes": fmt_bytes(v), "title": rules.get(k).title,
                        "category": rules.get(k).category, "risk": str(rules.get(k).risk)}
                    for k, v in sorted(by_rule.items(), key=lambda kv: -kv[1])
                },
            }
        if name == "list_candidates":
            max_risk = Risk.parse(args.get("max_risk", "high"))
            out = [self._brief(c) for c in self.cands.values()
                   if c.risk <= max_risk and args.get("rule_id") in (None, c.rule_id)]
            return {"candidates": out[: args.get("limit", 40)], "total": len(out)}
        if name == "inspect_candidate":
            c = self.cands.get(args["id"])
            if c is None:
                return {"error": "unknown id"}
            h = self.mem.history(c.path)
            v = self.policy.check(c, max_risk=self.max_risk)
            return {**self._brief(c), "explain": c.rule.explain, "extra": c.extra,
                    "history": dict(h) if h else None,
                    "policy": {"allowed": v.allowed, "approval": v.approval, "reasons": v.reasons}}
        if name == "propose_plan":
            plan = validate_selection(args["ids"], self.cands, self.policy, max_risk=self.max_risk)
            if plan.items:
                self.plan_id = self.mem.save_plan(self.scan_id, [c.cid for c in plan.items],
                                                  target=None, max_risk=int(self.max_risk),
                                                  source="agent", rationale=args["rationale"])
            return {"plan_id": self.plan_id, "accepted": [c.cid for c in plan.items],
                    "estimated_freed": fmt_bytes(plan.freed_estimate),
                    "rejected": {c.cid: v.reasons for c, v in plan.rejected},
                    "unknown_ids": [i for i in args["ids"] if i not in self.cands]}
        if name == "remember":
            self.mem.add_note(args["note"], source="agent")
            return {"ok": True}
        return {"error": f"unknown tool {name}"}

    def _brief(self, c: Candidate) -> dict:
        return {"id": c.cid, "rule": c.rule_id, "title": c.rule.title, "path": c.path,
                "risk": str(c.risk), "reclaimable": fmt_bytes(c.reclaimable),
                "reclaimable_bytes": c.reclaimable, "frees_space_now": c.frees_space,
                "reason": c.reason}

    # -- loop ----------------------------------------------------------------
    def _system(self) -> str:
        notes = [f"- {n['text']}" for n in reversed(self.mem.notes())]
        keeps = [f"- keep: {p}" for p, d in self.mem.decisions().items() if d["verdict"] == "keep"]
        context = "\n".join(notes + keeps) or "(none yet)"
        return f"{SYSTEM}\n\nWhat you remember about this user:\n{context}\n\n" \
               f"Highest risk the user allowed this session: {self.max_risk}."

    def ask(self, question: str) -> AgentResult:
        try:
            chat, model = open_chat(self.provider, self.model, self._system(), TOOLS,
                                    self.config.llm_effort)
            chat.user(question)
            for _ in range(MAX_TURNS):
                turn = chat.step()
                if turn.refused:
                    return AgentResult("The model declined this request.", self.plan_id, model)
                if not turn.calls:
                    return AgentResult(turn.text or "(no answer)", self.plan_id, model)
                results = []
                for call in turn.calls:
                    try:
                        out, err = self._tool(call.name, call.args), False
                    except (KeyError, ValueError, TypeError) as e:
                        out, err = {"error": str(e)}, True
                    results.append((call.id, json.dumps(out), err))
                chat.tool_results(results)
            return AgentResult("Stopped after too many steps.", self.plan_id, model)
        except LLMError as e:
            raise RuntimeError(str(e)) from None
