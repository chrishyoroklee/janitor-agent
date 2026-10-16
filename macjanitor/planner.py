"""Deterministic planner: the safest set of actions that reaches a target.

Ordering is "least disruption first": exhaust low-risk candidates before
touching medium, medium before high; within a risk tier prefer the biggest
wins so the plan has as few steps as possible. Trash moves are listed but do
not count toward a target, because they free nothing until the Trash is
emptied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from macjanitor.analyzer import Candidate
from macjanitor.fsutil import is_within
from macjanitor.policy import Policy, Verdict
from macjanitor.rules import Risk


@dataclass
class Plan:
    items: list[Candidate] = field(default_factory=list)
    rejected: list[tuple[Candidate, Verdict]] = field(default_factory=list)
    target: int | None = None

    @property
    def freed_estimate(self) -> int:
        return sum(c.reclaimable for c in self.items if c.frees_space)

    @property
    def reaches_target(self) -> bool:
        return self.target is None or self.freed_estimate >= self.target


def make_plan(cands: list[Candidate], policy: Policy, *, target: int | None = None,
              max_risk: Risk = Risk.MEDIUM, decisions: dict | None = None,
              unattended: bool = False) -> Plan:
    ignored = [p for p, d in (decisions or {}).items() if d["verdict"] == "ignore"]
    plan = Plan(target=target)
    eligible: list[Candidate] = []
    for c in cands:
        if any(is_within(c.path, p) for p in ignored):
            continue
        v = policy.check(c, max_risk=max_risk, unattended=unattended)
        (eligible.append(c) if v.allowed else plan.rejected.append((c, v)))

    eligible.sort(key=lambda c: (c.risk, not c.frees_space, -c.reclaimable))
    for c in eligible:
        if target is not None and plan.freed_estimate >= target:
            break
        if target is not None and not c.frees_space:
            continue
        plan.items.append(c)
    return plan


def validate_selection(cids: list[str], by_cid: dict[str, Candidate], policy: Policy,
                       *, max_risk: Risk) -> Plan:
    """Validate an externally chosen list of candidate ids (e.g. from the LLM)."""
    plan = Plan()
    for cid in dict.fromkeys(cids):  # dedupe, keep order
        c = by_cid.get(cid)
        if c is None:
            continue
        v = policy.check(c, max_risk=max_risk)
        (plan.items.append(c) if v.allowed else plan.rejected.append((c, v)))
    return plan
