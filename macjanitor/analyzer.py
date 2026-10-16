"""Scanner + analyzer: turn the rule catalog into sized cleanup candidates.

Scanning is targeted, not exhaustive: we only size the locations rules name
and walk configured project roots for marker directories. Sizing runs on a
thread pool (scandir/stat release the GIL), so a typical developer home is
analyzed in seconds rather than the minutes a full-disk walk takes.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from macjanitor import rules
from macjanitor.config import Config
from macjanitor.fsutil import Usage, canonical, du, is_within
from macjanitor.rules import Risk, Rule

DAY = 86400


@dataclass
class Candidate:
    rule_id: str
    path: str
    risk: Risk
    action: str
    logical: int = 0
    allocated: int = 0
    reclaimable: int = 0
    newest_mtime: float = 0.0
    dev: int = 0
    ino: int = 0
    reason: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def cid(self) -> str:
        return hashlib.sha1(f"{self.rule_id}\0{self.path}".encode()).hexdigest()[:10]

    @property
    def rule(self) -> Rule:
        return rules.get(self.rule_id)

    @property
    def frees_space(self) -> bool:
        return self.action != rules.TRASH

    @classmethod
    def from_row(cls, row) -> "Candidate":
        return cls(rule_id=row["rule_id"], path=row["path"], risk=Risk(row["risk"]),
                   action=row["action"], logical=row["logical"], allocated=row["allocated"],
                   reclaimable=row["reclaimable"], newest_mtime=row["newest_mtime"],
                   dev=row["dev"], ino=row["ino"], reason=row["reason"],
                   extra=json.loads(row["extra"] or "{}"))


def _identity(path: str) -> tuple[int, int] | None:
    try:
        st = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISLNK(st.st_mode):
        return None
    return st.st_dev, st.st_ino


def _age_days(ts: float, now: float) -> float:
    return (now - ts) / DAY if ts else 0.0


class Analyzer:
    def __init__(self, home: str, config: Config, *, workers: int = 8, tools: bool = True):
        self.home = home
        self.config = config
        self.workers = workers
        self.tools = tools
        self.now = time.time()
        self.stats = {"sized": 0, "errors": 0, "seconds": 0.0}

    # -- discovery -----------------------------------------------------------
    def _location_targets(self) -> list[tuple[Rule, str]]:
        out = []
        for rule in rules.RULES:
            if rule.kind != "location" or (rule.darwin_only and sys.platform != "darwin"):
                continue
            for pattern in rule.paths:
                for path in glob.glob(os.path.join(glob.escape(self.home), pattern)):
                    if rule.matches(path, self.home) and _identity(path):
                        out.append((rule, path))
        return out

    def _project_targets(self) -> list[tuple[Rule, str, str]]:
        """Walk project roots for node_modules, virtualenvs and CACHEDIR.TAG dirs."""
        node, venv, cache = rules.get("node-modules"), rules.get("python-venv"), rules.get("build-cache-dir")
        found: list[tuple[Rule, str, str]] = []
        # Canonical spelling so `code` and `Code` (one folder on APFS) count once,
        # and drop roots nested in another root whatever order they are listed in.
        roots = {canonical(os.path.join(self.home, rel)) for rel in self.config.project_roots}
        roots = {r for r in roots if os.path.isdir(r)}
        roots = sorted(r for r in roots if not any(o != r and is_within(r, o) for o in roots))
        for root in roots:
            stack = [(root, 0)]
            while stack:
                path, depth = stack.pop()
                try:
                    entries = list(os.scandir(path))
                except OSError:
                    continue
                for e in entries:
                    if not e.is_dir(follow_symlinks=False):
                        continue
                    if e.name == "node_modules" and rules.is_marker_dir("node_modules", e.path):
                        found.append((node, e.path, path))
                    elif e.name in (".venv", "venv", "env", ".env") and rules.is_marker_dir("venv", e.path):
                        found.append((venv, e.path, path))
                    elif rules.is_marker_dir("cachedir", e.path):
                        found.append((cache, e.path, path))
                    elif depth < self.config.project_max_depth and e.name not in (".git", "Library"):
                        if not e.name.startswith(".") or e.name in (".config",):
                            stack.append((e.path, depth + 1))
        return found

    def _project_last_active(self, project: str, artifact: str) -> float:
        """Newest mtime among the project's own top-level entries (excluding the artifact)."""
        newest = 0.0
        try:
            with os.scandir(project) as it:
                for e in it:
                    if e.path == artifact:
                        continue
                    try:
                        newest = max(newest, e.stat(follow_symlinks=False).st_mtime)
                    except OSError:
                        pass
        except OSError:
            pass
        return newest

    # -- tools ---------------------------------------------------------------
    def _tool_candidates(self) -> list[Candidate]:
        out: list[Candidate] = []
        if not self.tools:
            return out
        if shutil.which("docker"):
            out.extend(self._docker())
        if sys.platform == "darwin" and shutil.which("xcrun"):
            c = self._unavailable_simulators()
            if c:
                out.append(c)
        return out

    def _run_json_lines(self, argv: list[str], timeout: float = 15) -> list[dict]:
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return []
        if p.returncode != 0:
            return []
        rows = []
        for line in p.stdout.splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    def _docker(self) -> list[Candidate]:
        out = []
        for row in self._run_json_lines(["docker", "system", "df", "--format", "{{json .}}"]):
            kind, reclaim = row.get("Type", ""), _parse_docker_size(row.get("Reclaimable", ""))
            rule_id = {"Build Cache": "docker-build-cache", "Images": "docker-unused-images"}.get(kind)
            if rule_id and reclaim:
                r = rules.get(rule_id)
                out.append(Candidate(rule_id, rules.tool_path(rule_id), r.risk, r.action,
                                     logical=reclaim, allocated=reclaim, reclaimable=reclaim,
                                     reason=f"docker reports {row.get('Reclaimable')} reclaimable",
                                     extra={"docker": row}))
        return out

    def _unavailable_simulators(self) -> Candidate | None:
        try:
            p = subprocess.run(["xcrun", "simctl", "list", "devices", "unavailable", "-j"],
                               capture_output=True, text=True, timeout=20)
            data = json.loads(p.stdout or "{}")
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return None
        total, count = Usage(), 0
        for devices in data.get("devices", {}).values():
            for d in devices:
                path = d.get("dataPath") and os.path.dirname(d["dataPath"])
                if path and os.path.isdir(path):
                    total.add(du(path))
                    count += 1
        if not count:
            return None
        r = rules.get("simulators-unavailable")
        return Candidate(r.id, rules.tool_path(r.id), r.risk, r.action, logical=total.logical,
                         allocated=total.allocated, reclaimable=total.reclaimable,
                         reason=f"{count} simulator(s) without an installed runtime",
                         extra={"devices": count})

    # -- main ----------------------------------------------------------------
    def run(self) -> list[Candidate]:
        t0 = time.perf_counter()
        targets: list[tuple[Rule, str, str | None]] = [(r, p, None) for r, p in self._location_targets()]
        targets += self._project_targets()

        with ThreadPoolExecutor(self.workers) as pool:
            usages = list(pool.map(lambda t: du(t[1]), targets))
            tool_future = pool.submit(self._tool_candidates)
            tool_cands = tool_future.result()

        out: list[Candidate] = []
        for (rule, path, project), usage in zip(targets, usages):
            self.stats["sized"] += 1
            self.stats["errors"] += usage.errors
            ident = _identity(path)
            if ident is None or usage.reclaimable < self.config.min_candidate_bytes:
                continue
            age = _age_days(usage.newest_mtime, self.now)
            if rule.min_age_days and age < rule.min_age_days:
                continue
            extra: dict = {"files": usage.files, "errors": usage.errors}
            if project is not None:
                active = self._project_last_active(project, path)
                idle = _age_days(active, self.now)
                if idle < self.config.stale_days:
                    continue
                extra |= {"project": project, "project_idle_days": round(idle)}
                reason = f"project untouched for {idle:.0f} days"
            elif rule.min_age_days:
                reason = f"not modified for {age:.0f} days"
            else:
                reason = rule.explain.split(".")[0]
            if usage.allocated > usage.reclaimable:
                extra["shared_bytes"] = usage.allocated - usage.reclaimable
            out.append(Candidate(rule.id, path, rule.risk, rule.action, logical=usage.logical,
                                 allocated=usage.allocated, reclaimable=usage.reclaimable,
                                 newest_mtime=usage.newest_mtime, dev=ident[0], ino=ident[1],
                                 reason=reason, extra=extra))
        out += [c for c in tool_cands if c.reclaimable >= self.config.min_candidate_bytes]
        out.sort(key=lambda c: (-c.reclaimable, c.path))
        self.stats["seconds"] = time.perf_counter() - t0
        return out


def _parse_docker_size(text: str) -> int:
    """'12.3GB (45%)' -> bytes."""
    from macjanitor.fsutil import parse_size

    token = text.split("(")[0].strip()
    try:
        return parse_size(token) if token else 0
    except ValueError:
        return 0
