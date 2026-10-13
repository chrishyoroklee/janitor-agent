"""Policy engine: the single gate every proposed action must pass.

The policy does not trust the candidate it is handed - candidates may come
from an old scan, from the database, or from an LLM's plan. It re-derives
everything that matters from the rule catalog and the live filesystem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from macjanitor import rules
from macjanitor.fsutil import is_within
from macjanitor.rules import Risk

# Never touched, by any rule, for any reason (home-relative).
PROTECTED = (
    "Library/Mobile Documents",            # iCloud Drive
    "Library/CloudStorage",                # Dropbox / Google Drive / OneDrive
    "Library/Mail", "Library/Messages", "Library/Keychains", "Library/Photos",
    "Library/Application Support/MobileSync",   # iPhone/iPad backups
    "Library/Containers/com.docker.docker",     # Docker VM disk: use the docker CLI
    "Library/Group Containers",
    "Pictures", "Movies", "Music",
    ".ssh", ".gnupg", ".aws", ".kube", ".config",
)
# Personal areas: only regenerable artifacts, or reversible moves to the Trash.
PERSONAL = ("Desktop", "Documents", "Downloads")
APPROVAL = {Risk.LOW: "batch", Risk.MEDIUM: "item", Risk.HIGH: "explicit"}


@dataclass
class Verdict:
    allowed: bool
    approval: str = "batch"   # batch | item | explicit
    reasons: list[str] = field(default_factory=list)

    def deny(self, why: str) -> "Verdict":
        self.allowed = False
        self.reasons.append(why)
        return self


class Policy:
    def __init__(self, home: str, *, decisions: dict | None = None, extra_protected=()):
        self.home = os.path.realpath(home)
        self.decisions = decisions or {}
        self.protected = [os.path.join(self.home, p) for p in PROTECTED]
        self.protected += [p if os.path.isabs(p) else os.path.join(self.home, p) for p in extra_protected]
        self.personal = [os.path.join(self.home, p) for p in PERSONAL]

    def check(self, cand, *, max_risk: Risk = Risk.HIGH, unattended: bool = False,
              live: bool = True) -> Verdict:
        v = Verdict(True)
        rule = rules.BY_ID.get(cand.rule_id)
        if rule is None:
            return v.deny(f"unknown rule {cand.rule_id!r}")
        # The action and risk come from the catalog, never from the candidate.
        if cand.action != rule.action or Risk(cand.risk) != rule.risk:
            return v.deny("candidate action/risk does not match its rule")
        if rule.risk > max_risk:
            return v.deny(f"risk {rule.risk} exceeds allowed {max_risk}")
        if unattended and not (rule.auto_ok and rule.risk == Risk.LOW):
            return v.deny("not eligible for unattended cleaning")
        v.approval = APPROVAL[rule.risk]

        if rule.kind == "tool":
            if cand.path != rules.tool_path(rule.id):
                return v.deny("tool candidate has a filesystem path")
            return v

        path = cand.path
        if not os.path.isabs(path) or os.path.normpath(path) != path:
            return v.deny("path is not absolute and normalized")
        if not is_within(path, self.home) or path == self.home:
            return v.deny("path is outside the home directory")
        if os.path.dirname(path) == self.home:
            return v.deny("refusing to act on a top-level home folder")
        for p in self.protected:
            if is_within(path, p) or is_within(p, path):
                return v.deny(f"protected location: {os.path.relpath(p, self.home)}")
        for p, d in self.decisions.items():
            if d["verdict"] == "keep" and (is_within(path, p) or is_within(p, path)):
                return v.deny("you asked to keep this")
        if any(is_within(path, p) for p in self.personal):
            if not (rule.regenerable or rule.action == rules.TRASH):
                return v.deny("personal folder: only regenerable artifacts or Trash moves")
        if not rule.matches(path, self.home):
            return v.deny(f"path is outside the scope of rule {rule.id}")

        if live:
            try:
                st = os.lstat(path)
            except FileNotFoundError:
                return v.deny("path no longer exists")
            except OSError as e:
                return v.deny(f"cannot stat: {e.strerror}")
            if os.path.islink(path):
                return v.deny("path is a symlink")
            # No symlinked ancestors: the real path must equal the stated path.
            if os.path.realpath(path) != path:
                return v.deny("path traverses a symlink")
            if (cand.dev, cand.ino) != (0, 0) and (st.st_dev, st.st_ino) != (cand.dev, cand.ino):
                return v.deny("file identity changed since the scan")
            if os.path.isdir(os.path.join(path, ".git")):
                return v.deny("contains a git repository")
        return v
