"""Executor: performs typed operations, never free-form shell.

For each item: re-check policy against the live filesystem (identity, symlinks,
scope), act, then measure the real change in free space and write an audit
record. A failure on one item never aborts the others.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass

from macjanitor import rules
from macjanitor.analyzer import Candidate
from macjanitor.db import Memory
from macjanitor.fsutil import disk_usage, is_within
from macjanitor.policy import Policy
from macjanitor.rules import Risk

COMMAND_TIMEOUT = 30 * 60


@dataclass
class Result:
    cand: Candidate
    status: str            # done | failed | skipped | dry_run
    freed: int = 0
    detail: str = ""
    trash_path: str | None = None
    action_id: int | None = None


def _make_writable(func, path, exc):
    """rmtree error hook: read-only dirs (e.g. Go module cache) block deletion."""
    try:
        parent = os.path.dirname(path)
        os.chmod(parent, os.lstat(parent).st_mode | stat.S_IWUSR | stat.S_IXUSR)
        if not os.path.islink(path):
            os.chmod(path, os.lstat(path).st_mode | stat.S_IWUSR)
        func(path)
    except OSError:
        raise exc


def _rmtree(path: str) -> None:
    shutil.rmtree(path, onexc=_make_writable)


def _delete(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        _rmtree(path)
    else:
        os.unlink(path)


def _delete_contents(path: str) -> int:
    failures = 0
    with os.scandir(path) as it:
        entries = list(it)
    for e in entries:
        try:
            if e.is_dir(follow_symlinks=False):
                _rmtree(e.path)
            else:
                os.unlink(e.path)
        except OSError:
            failures += 1
    return failures


def _trash(path: str, home: str) -> str:
    trash = os.path.join(home, ".Trash")
    os.makedirs(trash, exist_ok=True)
    base = os.path.basename(path)
    dest = os.path.join(trash, base)
    n = 1
    while os.path.lexists(dest):
        stem, ext = os.path.splitext(base)
        dest = os.path.join(trash, f"{stem} {time.strftime('%H.%M.%S')}-{n}{ext}")
        n += 1
    os.rename(path, dest)  # same volume: atomic, no copy
    return dest


class Executor:
    def __init__(self, home: str, memory: Memory, policy: Policy):
        self.home = home
        self.mem = memory
        self.policy = policy

    def run(self, cand: Candidate, *, plan_id: int | None, approved_by: str,
            max_risk: Risk, dry_run: bool = False, unattended: bool = False) -> Result:
        rule = rules.get(cand.rule_id)
        started = time.time()
        verdict = self.policy.check(cand, max_risk=max_risk, unattended=unattended)
        if not verdict.allowed:
            return self._log(Result(cand, "skipped", detail="; ".join(verdict.reasons)),
                             plan_id, approved_by, started)
        if dry_run:
            return self._log(Result(cand, "dry_run", detail=self._describe(rule, cand)),
                             plan_id, approved_by, started)

        _, _, free_before = disk_usage(self.home)
        res = Result(cand, "done")
        try:
            argv = self._command(rule, cand)
            if argv is not None:
                p = subprocess.run(argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
                tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
                res.detail = f"$ {' '.join(argv)} -> exit {p.returncode}" + (": " + " | ".join(tail) if tail else "")
                if p.returncode != 0:
                    res.status = "failed"
            elif rule.action == rules.DELETE:
                _delete(cand.path)
                res.detail = "deleted"
            elif rule.action == rules.DELETE_CONTENTS:
                failures = _delete_contents(cand.path)
                res.detail = "emptied" + (f" ({failures} entries could not be removed)" if failures else "")
            elif rule.action == rules.TRASH:
                res.trash_path = _trash(cand.path, self.home)
                res.detail = f"moved to {res.trash_path}"
            else:
                raise AssertionError(rule.action)
        except subprocess.TimeoutExpired:
            res.status, res.detail = "failed", "command timed out"
        except OSError as e:
            res.status, res.detail = "failed", f"{type(e).__name__}: {e}"
        _, _, free_after = disk_usage(self.home)
        res.freed = free_after - free_before
        return self._log(res, plan_id, approved_by, started)

    def _command(self, rule: rules.Rule, cand: Candidate) -> tuple[str, ...] | None:
        """Prefer the owning tool's cleanup command when it is installed *and*
        provably cleans the directory we sized; otherwise fall back to the
        typed filesystem action."""
        if rule.action == rules.COMMAND:
            if not shutil.which(rule.command[0]):
                raise FileNotFoundError(f"{rule.command[0]} is not installed")
            return rule.command
        if not (rule.command and rule.command_dir and shutil.which(rule.command[0])):
            return None
        try:
            p = subprocess.run(rule.command_dir, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        tool_dir = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else ""
        if not os.path.isabs(tool_dir):
            return None
        tool_dir = os.path.realpath(tool_dir)
        if is_within(cand.path, tool_dir) or is_within(tool_dir, cand.path):
            return rule.command
        return None

    def _describe(self, rule: rules.Rule, cand: Candidate) -> str:
        try:
            argv = self._command(rule, cand)
        except FileNotFoundError as e:
            return str(e)
        if argv:
            return "would run: " + " ".join(argv)
        return f"would {rule.action.replace('_', ' ')} {cand.path}"

    def _log(self, res: Result, plan_id, approved_by, started) -> Result:
        c = res.cand
        res.action_id = self.mem.log_action(
            plan_id=plan_id, cid=c.cid, rule_id=c.rule_id, path=c.path, action=c.action,
            risk=int(c.risk), approved_by=approved_by, status=res.status,
            estimated=c.reclaimable, freed=res.freed, trash_path=res.trash_path,
            detail=res.detail, started=started, finished=time.time(),
        )
        return res

    def undo(self, action_id: int) -> str:
        row = self.mem.action(action_id)
        if row is None:
            raise LookupError(f"no action #{action_id}")
        if row["status"] != "done" or not row["trash_path"]:
            raise ValueError("only completed Trash moves can be undone; deletions are permanent")
        if os.path.lexists(row["path"]):
            raise FileExistsError(f"{row['path']} exists again; not overwriting")
        if not os.path.lexists(row["trash_path"]):
            raise FileNotFoundError("the item is no longer in the Trash")
        os.rename(row["trash_path"], row["path"])
        self.mem.mark_undone(action_id)
        return row["path"]
