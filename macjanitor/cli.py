"""janitor - command line interface.

    janitor status                 disk usage, trend, last scan
    janitor scan                   find cleanup candidates (read-only)
    janitor plan [--target 20GB]   build a dry-run plan from the last scan
    janitor clean [--plan N]       execute a plan with approvals
    janitor free 25GB              scan + plan + approve + clean + verify
    janitor du [PATH]              what is using space under PATH
    janitor ask "question"         ask the optional LLM agent
    janitor login [PROVIDER]       connect an LLM (API key, ChatGPT, Claude, Gemini, local)
    janitor keep|ignore|forget P   remember decisions about a path
    janitor history | undo ID      audit log, restore Trash moves
    janitor schedule install       periodic background checks (launchd)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from macjanitor import __version__, launchd, rules
from macjanitor.analyzer import Analyzer, Candidate
from macjanitor.config import Config, user_home
from macjanitor.db import Memory
from macjanitor.executor import Executor, Result
from macjanitor.fsutil import disk_usage, du, fmt_bytes, parse_size
from macjanitor.planner import Plan, make_plan
from macjanitor.policy import Policy
from macjanitor.rules import Risk

STALE_SCAN_SECONDS = 3600
_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_RISK_COLOR = {Risk.LOW: "32", Risk.MEDIUM: "33", Risk.HIGH: "31"}


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def risk_label(r: Risk) -> str:
    return c(f"{str(r):<6}", _RISK_COLOR[Risk(r)])


def short(path: str, home: str, width: int = 60) -> str:
    p = "~" + path[len(home):] if path.startswith(home + os.sep) else path
    return p if len(p) <= width else "…" + p[-(width - 1):]


class App:
    def __init__(self, args):
        self.args = args
        self.home = user_home()
        self.config = Config.load()
        self.mem = Memory()
        self.policy = Policy(self.home, decisions=self.mem.decisions(),
                             extra_protected=self.config.protected)

    # -- shared --------------------------------------------------------------
    def scan(self, quiet: bool = False) -> tuple[int, list[Candidate]]:
        started = time.time()
        an = Analyzer(self.home, self.config, tools=not getattr(self.args, "no_tools", False))
        cands = an.run()
        total, _, free = disk_usage(self.home)
        scan_id = self.mem.record_scan(started, total, free, cands)
        if not quiet:
            print(c(f"Scanned {an.stats['sized']} locations in {an.stats['seconds']:.1f}s "
                    f"(scan #{scan_id})", "2"))
        return scan_id, cands

    def latest(self, *, rescan_if_stale: bool = True) -> tuple[int, list[Candidate]]:
        row = self.mem.latest_scan()
        if row is None or (rescan_if_stale and time.time() - row["finished"] > STALE_SCAN_SECONDS):
            return self.scan()
        return row["id"], [Candidate.from_row(r) for r in self.mem.candidates(row["id"])]

    def print_candidates(self, cands: list[Candidate], limit: int = 25) -> None:
        if not cands:
            print("No cleanup candidates found.")
            return
        decisions = self.mem.decisions()
        print(f"{'ID':<11}{'SIZE':>10}  {'RISK':<7}{'WHAT':<34}PATH")
        for cand in cands[:limit]:
            mark = c(" (ignored)", "2") if decisions.get(cand.path) else ""
            print(f"{cand.cid:<11}{fmt_bytes(cand.reclaimable):>10}  {risk_label(cand.risk)} "
                  f"{rules.get(cand.rule_id).title[:33]:<34}{short(cand.path, self.home)}{mark}")
        if len(cands) > limit:
            print(c(f"… and {len(cands) - limit} more (use --all)", "2"))

    def print_summary(self, cands: list[Candidate]) -> None:
        by_cat: dict[str, int] = {}
        by_risk: dict[Risk, int] = {}
        for cand in cands:
            cat = rules.get(cand.rule_id).category
            by_cat[cat] = by_cat.get(cat, 0) + cand.reclaimable
            by_risk[cand.risk] = by_risk.get(cand.risk, 0) + cand.reclaimable
        total = sum(by_cat.values())
        print(c(f"\nPotential cleanup: {fmt_bytes(total)}", "1"))
        for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:<12}{fmt_bytes(n):>10}")
        print("  " + " · ".join(f"{risk_label(r).strip()} {fmt_bytes(n)}" for r, n in sorted(by_risk.items())))

    def print_plan(self, plan: Plan, plan_id: int | None) -> None:
        head = f"Plan #{plan_id}" if plan_id else "Plan"
        print(c(f"\n{head}: {len(plan.items)} action(s), ~{fmt_bytes(plan.freed_estimate)} "
                f"estimated", "1"))
        for i, cand in enumerate(plan.items, 1):
            r = rules.get(cand.rule_id)
            how = r.action.replace("_", " ")
            if r.action == rules.COMMAND:
                how = " ".join(r.command)
            elif r.command:
                how += f" (or `{' '.join(r.command)}` if it owns this folder)"
            note = "" if cand.frees_space else c("  (to Trash: frees space once emptied)", "2")
            print(f" {i:>2}. {risk_label(cand.risk)} {fmt_bytes(cand.reclaimable):>9}  {r.title}{note}")
            print(c(f"       {short(cand.path, self.home, 70)}  ·  {how}", "2"))
        if plan.target is not None and not plan.reaches_target:
            print(c(f"\nThis plan falls short of {fmt_bytes(plan.target)}. "
                    "Raise --max-risk or review `janitor scan --all`.", "33"))

    def approve_and_run(self, plan_id: int, items: list[Candidate], max_risk: Risk,
                        assume_yes: bool, dry_run: bool) -> list[Result]:
        ex = Executor(self.home, self.mem, self.policy)
        interactive = sys.stdin.isatty()
        approved: list[tuple[Candidate, str]] = []

        low = [x for x in items if x.risk == Risk.LOW]
        if low:
            size = fmt_bytes(sum(x.reclaimable for x in low))
            if dry_run or assume_yes or (interactive and _confirm(f"Run {len(low)} low-risk action(s), ~{size}?")):
                approved += [(x, "flag" if assume_yes else "user") for x in low]
        for x in items:
            if x.risk == Risk.LOW:
                continue
            label = f"{rules.get(x.rule_id).title}: {short(x.path, self.home)} (~{fmt_bytes(x.reclaimable)})"
            if dry_run:
                approved.append((x, "dry_run"))
            elif x.risk == Risk.MEDIUM and assume_yes and max_risk >= Risk.MEDIUM:
                approved.append((x, "flag"))
            elif x.risk == Risk.MEDIUM and interactive and _confirm(f"[medium] {label}?"):
                approved.append((x, "user"))
            elif x.risk == Risk.HIGH and interactive:
                print(c(f"[high] {label}", "31"))
                print(c(f"       {rules.get(x.rule_id).explain}", "2"))
                if input("       Type 'yes' to approve: ").strip() == "yes":
                    approved.append((x, "user-explicit"))
            elif not interactive:
                print(c(f"skipped (needs interactive approval): {label}", "2"))

        results = []
        for cand, by in approved:
            res = ex.run(cand, plan_id=plan_id, approved_by=by, max_risk=max_risk, dry_run=dry_run)
            results.append(res)
            color = {"done": "32", "failed": "31", "skipped": "33", "dry_run": "36"}[res.status]
            freed = f" freed {fmt_bytes(res.freed)}" if res.status == "done" else ""
            print(f"  {c(res.status, color):<8} #{res.action_id} {rules.get(cand.rule_id).title}{freed}"
                  + c(f"  {res.detail}", "2"))
        if not dry_run and approved:
            self.mem.set_plan_status(plan_id, "executed")
        return results

    # -- commands ------------------------------------------------------------
    def cmd_status(self) -> int:
        total, used, free = disk_usage(self.home)
        self.mem.sample_disk(total, free)
        pct = used / total * 100 if total else 0
        print(f"Disk: {fmt_bytes(free)} free of {fmt_bytes(total)} ({pct:.0f}% used)")
        first, last = self.mem.disk_trend(time.time() - 7 * 86400)
        if first and last and last["ts"] - first["ts"] > 3600:
            days = (last["ts"] - first["ts"]) / 86400
            print(f"Free space change over {days:.1f} days: {fmt_bytes(last['free'] - first['free'])}")
        scan = self.mem.latest_scan()
        if scan:
            age = (time.time() - scan["finished"]) / 3600
            print(f"Last scan #{scan['id']} {age:.1f}h ago: {scan['candidate_count']} candidates, "
                  f"{fmt_bytes(scan['candidate_bytes'])} reclaimable")
        else:
            print("No scans yet. Run `janitor scan`.")
        return 0

    def cmd_scan(self) -> int:
        scan_id, cands = self.scan()
        if self.args.json:
            print(json.dumps([{"id": x.cid, "rule": x.rule_id, "path": x.path, "risk": str(x.risk),
                               "action": x.action, "reclaimable": x.reclaimable,
                               "allocated": x.allocated, "logical": x.logical, "reason": x.reason}
                              for x in cands], indent=2))
            return 0
        self.print_candidates(cands, limit=10**9 if self.args.all else 25)
        self.print_summary(cands)
        print(c("\nNext: `janitor plan --target 20GB` (dry run) or `janitor free 20GB`", "2"))
        return 0

    def cmd_plan(self) -> int:
        scan_id, cands = self.latest()
        target = parse_size(self.args.target) if self.args.target else None
        max_risk = Risk.parse(self.args.max_risk)
        plan = make_plan(cands, self.policy, target=target, max_risk=max_risk,
                         decisions=self.mem.decisions())
        if not plan.items:
            print("Nothing eligible to clean at this risk level.")
            return 0
        pid = self.mem.save_plan(scan_id, [x.cid for x in plan.items], target=target,
                                 max_risk=int(max_risk), source="engine")
        self.print_plan(plan, pid)
        if self.args.verbose and plan.rejected:
            print(c("\nExcluded by policy:", "2"))
            for cand, v in plan.rejected:
                print(c(f"  {short(cand.path, self.home)}: {'; '.join(v.reasons)}", "2"))
        print(c(f"\nDry run only. Execute with `janitor clean --plan {pid}`", "2"))
        return 0

    def cmd_clean(self) -> int:
        row = self.mem.plan(self.args.plan)
        if row is None:
            print("No plan found. Run `janitor plan` first.")
            return 1
        if row["status"] == "executed" and not self.args.dry_run:
            print(f"Plan #{row['id']} was already executed. Run `janitor plan` for a new one.")
            return 1
        cands = {r["cid"]: Candidate.from_row(r) for r in self.mem.candidates(row["scan_id"])}
        items = [cands[cid] for cid in self.mem.plan_cids(row["id"]) if cid in cands]
        max_risk = Risk.parse(self.args.max_risk) if self.args.max_risk else Risk(row["max_risk"])
        plan = Plan(items=items, target=row["target"])
        self.print_plan(plan, row["id"])
        if row["rationale"]:
            print(c(f"Rationale: {row['rationale']}", "2"))
        print()
        results = self.approve_and_run(row["id"], items, max_risk, self.args.yes, self.args.dry_run)
        self.report(results)
        return 0

    def report(self, results: list[Result], target: int | None = None) -> None:
        done = [r for r in results if r.status == "done"]
        if not done:
            return
        est = sum(r.cand.reclaimable for r in done if r.cand.frees_space)
        freed = sum(r.freed for r in done)
        print(c(f"\nReclaimed {fmt_bytes(freed)} (measured free-space change; estimated "
                f"{fmt_bytes(est)})", "1"))
        if freed < est * 0.5:
            print(c("APFS may release space lazily, or blocks were shared (clones/snapshots). "
                    "Local Time Machine snapshots can also hold deleted data: "
                    "`tmutil listlocalsnapshots /`.", "2"))
        if target and freed < target:
            print(c(f"Short of the {fmt_bytes(target)} target by {fmt_bytes(target - freed)}.", "33"))

    def cmd_free(self) -> int:
        target = parse_size(self.args.amount)
        max_risk = Risk.parse(self.args.max_risk)
        scan_id, cands = self.scan()
        if self.args.agent:
            from macjanitor.agent import Agent

            agent = Agent(self.mem, self.policy, self.config, self.home, scan_id=scan_id,
                          max_risk=max_risk, provider=self.args.provider, model=self.args.model)
            res = agent.ask(f"I need to free {fmt_bytes(target)}. Propose the least disruptive plan.")
            print("\n" + res.text)
            if res.plan_id is None:
                return 1
            pid = res.plan_id
            by = {x.cid: x for x in cands}
            plan = Plan(items=[by[i] for i in self.mem.plan_cids(pid)], target=target)
        else:
            plan = make_plan(cands, self.policy, target=target, max_risk=max_risk,
                             decisions=self.mem.decisions())
            if not plan.items:
                print("Nothing eligible to clean at this risk level.")
                return 1
            pid = self.mem.save_plan(scan_id, [x.cid for x in plan.items], target=target,
                                     max_risk=int(max_risk), source="engine")
        self.print_plan(plan, pid)
        print()
        results = self.approve_and_run(pid, plan.items, max_risk, self.args.yes, self.args.dry_run)
        self.report(results, target)
        return 0

    def cmd_du(self) -> int:
        root = os.path.realpath(os.path.expanduser(self.args.path or self.home))
        try:
            entries = [e.path for e in os.scandir(root)]
        except OSError as e:
            print(f"cannot read {root}: {e.strerror}")
            return 1
        t0 = time.perf_counter()
        with ThreadPoolExecutor(16) as pool:
            sizes = list(pool.map(du, entries))
        rows = sorted(zip(entries, sizes), key=lambda t: -t[1].allocated)
        total = sum(s.allocated for s in sizes)
        for path, s in rows[: self.args.top]:
            bar = "█" * int(30 * s.allocated / total) if total else ""
            print(f"{fmt_bytes(s.allocated):>10}  {bar:<30} {short(path, self.home)}")
        errs = sum(s.errors for s in sizes)
        print(c(f"{fmt_bytes(total)} in {sum(s.files for s in sizes):,} files, "
                f"{time.perf_counter() - t0:.1f}s" + (f", {errs} unreadable (grant Full Disk "
                "Access to your terminal to see everything)" if errs else ""), "2"))
        return 0

    def cmd_ask(self) -> int:
        from macjanitor.agent import Agent

        scan_id, _ = self.latest()
        agent = Agent(self.mem, self.policy, self.config, self.home, scan_id=scan_id,
                      max_risk=Risk.parse(self.args.max_risk), provider=self.args.provider,
                      model=self.args.model)
        res = agent.ask(" ".join(self.args.question))
        print(res.text)
        print(c(f"\n— {agent.provider} · {res.model}", "2"))
        if res.plan_id:
            print(c(f"\nSaved plan #{res.plan_id}. Review and run: `janitor clean --plan {res.plan_id}`", "2"))
        return 0

    def _resolve(self, p: str) -> str:
        return os.path.realpath(os.path.expanduser(p))

    def cmd_keep(self) -> int:
        path = self._resolve(self.args.path)
        self.mem.decide(path, "keep", self.args.note or "")
        print(f"Will never touch {short(path, self.home)} or anything inside it.")
        return 0

    def cmd_ignore(self) -> int:
        path = self._resolve(self.args.path)
        self.mem.decide(path, "ignore", self.args.note or "")
        print(f"Plans will skip {short(path, self.home)} (you can still clean it explicitly).")
        return 0

    def cmd_forget(self) -> int:
        ok = self.mem.forget(self._resolve(self.args.path))
        print("Forgotten." if ok else "No decision stored for that path.")
        return 0 if ok else 1

    def cmd_decisions(self) -> int:
        for p, d in sorted(self.mem.decisions().items()):
            print(f"{d['verdict']:<7} {short(p, self.home, 80)}  {d['note'] or ''}")
        for n in self.mem.notes():
            print(c(f"note    {n['text']}", "2"))
        return 0

    def cmd_history(self) -> int:
        rows = self.mem.actions(self.args.limit)
        if not rows:
            print("No actions recorded yet.")
        for r in rows:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["started"]))
            freed = fmt_bytes(r["freed"]) if r["status"] == "done" else ""
            print(f"#{r['id']:<5}{when}  {r['status']:<8}{freed:>10}  {r['rule_id']:<22}"
                  f"{short(r['path'], self.home, 50)}")
        print(c(f"Full audit log: {self.mem.audit_path}", "2"))
        return 0

    def cmd_undo(self) -> int:
        try:
            path = Executor(self.home, self.mem, self.policy).undo(self.args.action_id)
        except (LookupError, ValueError, OSError) as e:
            print(f"Cannot undo: {e}")
            return 1
        print(f"Restored {short(path, self.home)}")
        return 0

    def cmd_schedule(self) -> int:
        if self.args.op == "install":
            every = self.args.every.strip().lower()
            unit = {"h": 3600, "m": 60}.get(every[-1:], 1)
            seconds = max(900, int(float(every.rstrip("hm")) * unit))
            path = launchd.install(seconds)
            print(f"Installed {path} (every {seconds / 3600:g}h).")
            if self.config.auto_clean_below_bytes:
                print(f"Low-risk items will be cleaned automatically when free space drops below "
                      f"{fmt_bytes(self.config.auto_clean_below_bytes)}.")
            else:
                print("Background runs scan and notify only. Set auto_clean_below_bytes in "
                      f"{self.config.save()} to allow unattended low-risk cleaning.")
        else:
            print("Removed." if launchd.uninstall() else "Not installed.")
        return 0

    def cmd_auto(self) -> int:
        """Entry point for launchd: cheap check, scan, optionally clean low-risk items."""
        total, _, free = disk_usage(self.home)
        scan_id, cands = self.scan(quiet=True)
        threshold = self.config.auto_clean_below_bytes
        if threshold and free < threshold:
            plan = make_plan(cands, self.policy, target=threshold - free, max_risk=Risk.LOW,
                             decisions=self.mem.decisions(), unattended=True)
            if plan.items:
                pid = self.mem.save_plan(scan_id, [x.cid for x in plan.items], target=threshold - free,
                                         max_risk=int(Risk.LOW), source="auto")
                ex = Executor(self.home, self.mem, self.policy)
                freed = sum(ex.run(x, plan_id=pid, approved_by="auto", max_risk=Risk.LOW,
                                   unattended=True).freed for x in plan.items)
                self.mem.set_plan_status(pid, "executed")
                launchd.notify("MacJanitor", f"Freed {fmt_bytes(freed)} of low-risk caches.")
                return 0
        low = sum(x.reclaimable for x in cands if x.risk == Risk.LOW)
        if total and free / total < 0.10:
            launchd.notify("MacJanitor", f"Only {fmt_bytes(free)} free. {fmt_bytes(low)} of "
                                         "low-risk cleanup is available: run `janitor free`.")
        return 0

    # -- LLM providers -------------------------------------------------------
    def cmd_login(self) -> int:
        from macjanitor.llm import providers as P

        if self.args.list or not self.args.provider:
            print("Providers:")
            for p in P.PROVIDERS.values():
                st = P.status(p)
                ways = ", ".join(P.LOGIN_LABELS[m].split(" (")[0] for m in p.logins)
                print(f"  {p.id:<14}{p.name:<36}" + (c(f"✓ {st}", "32") if st else c(ways, "2")))
            print(c("\n" + P.NOT_OFFERED, "2"))
            if self.args.list or not sys.stdin.isatty():
                return 0
            ids = list(P.PROVIDERS)
            raw = input("\nProvider to connect: ").strip()
            if raw not in ids:
                print("Cancelled.")
                return 1
            self.args.provider = raw
        p = P.get(self.args.provider)
        print(P.login(p, method=self.args.method, key=self.args.key, base_url=self.args.base_url))
        if self.config.llm_provider != p.id and (self.args.default or not P.status(P.get(self.config.llm_provider))):
            self.config.llm_provider = p.id
        model = self.config.llm_models.get(p.id) or p.default_model
        if not model and sys.stdin.isatty():
            model = self._pick_model(p)
        if model:
            self.config.llm_models[p.id] = model
        self.config.save()
        print(f"Default LLM: {self.config.llm_provider} · "
              f"{self.config.llm_models.get(self.config.llm_provider) or '(no model yet)'}")
        return 0

    def _pick_model(self, p) -> str | None:
        from macjanitor.llm import providers as P
        from macjanitor.llm.http import LLMError

        try:
            models = P.list_models(p)
        except (LLMError, RuntimeError) as e:
            print(c(f"Could not list models: {e}", "33"))
            models = []
        if models:
            shown = models[:40]
            for i, m in enumerate(shown, 1):
                print(f"  {i:>2}. {m}")
            if len(models) > len(shown):
                print(c(f"  … {len(models) - len(shown)} more (`janitor models {p.id}`)", "2"))
        raw = input("Model (number or id, Enter to skip): ").strip()
        if raw.isdigit() and models and 1 <= int(raw) <= min(40, len(models)):
            return models[int(raw) - 1]
        return raw or None

    def cmd_logout(self) -> int:
        from macjanitor.llm import providers as P

        ok = P.logout(P.get(self.args.provider))
        print("Signed out." if ok else "No stored login for that provider.")
        return 0 if ok else 1

    def cmd_models(self) -> int:
        from macjanitor.llm import providers as P

        p = P.get(self.args.provider or self.config.llm_provider)
        current = self.config.llm_models.get(p.id) or p.default_model
        for m in P.list_models(p):
            print(("* " if m == current else "  ") + m)
        return 0

    def cmd_use(self) -> int:
        from macjanitor.llm import providers as P

        p = P.get(self.args.provider)
        self.config.llm_provider = p.id
        if self.args.model:
            self.config.llm_models[p.id] = self.args.model
        self.config.save()
        st = P.status(p)
        print(f"Default LLM: {p.id} · {self.config.llm_models.get(p.id) or p.default_model or '(no model)'}"
              + ("" if st else c(f"  (not connected: run `janitor login {p.id}`)", "33")))
        return 0

    def cmd_config(self) -> int:
        path = self.config.save()
        print(path)
        print(path.read_text())
        return 0


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="janitor", description="Safe, fast storage maintenance for macOS.")
    p.add_argument("--version", action="version", version=f"macjanitor {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="disk usage, trend, last scan")
    s = sub.add_parser("scan", help="find cleanup candidates (read-only)")
    s.add_argument("--all", action="store_true", help="list every candidate")
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-tools", action="store_true", help="skip docker/simctl probes")

    s = sub.add_parser("plan", help="build a dry-run plan")
    s.add_argument("--target", help="amount to free, e.g. 20GB")
    s.add_argument("--max-risk", default="medium", choices=["low", "medium", "high"])
    s.add_argument("-v", "--verbose", action="store_true", help="show items excluded by policy")

    s = sub.add_parser("clean", help="execute a plan with approvals")
    s.add_argument("--plan", type=int, help="plan id (default: latest)")
    s.add_argument("--max-risk", choices=["low", "medium", "high"])
    s.add_argument("-y", "--yes", action="store_true", help="approve low/medium items (never high)")
    s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("free", help="scan, plan, approve, clean and verify")
    s.add_argument("amount", help="e.g. 25GB")
    s.add_argument("--max-risk", default="medium", choices=["low", "medium", "high"])
    s.add_argument("-y", "--yes", action="store_true", help="approve low/medium items (never high)")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--agent", action="store_true", help="let the LLM prioritize the plan")
    s.add_argument("--no-tools", action="store_true")
    s.add_argument("--provider", help="LLM provider for --agent (default from config)")
    s.add_argument("--model")

    s = sub.add_parser("du", help="what is using space under a folder")
    s.add_argument("path", nargs="?")
    s.add_argument("--top", type=int, default=20)

    s = sub.add_parser("ask", help="ask the LLM agent")
    s.add_argument("question", nargs="+")
    s.add_argument("--max-risk", default="medium", choices=["low", "medium", "high"])
    s.add_argument("--provider", help="LLM provider (default from config)")
    s.add_argument("--model")

    s = sub.add_parser("login", help="connect an LLM provider")
    s.add_argument("provider", nargs="?")
    s.add_argument("--method", choices=["api_key", "chatgpt", "ant", "adc", "none"])
    s.add_argument("--key", help="API key (prefer the hidden prompt or stdin)")
    s.add_argument("--base-url", help="for custom / self-hosted endpoints")
    s.add_argument("--default", action="store_true", help="make this the default provider")
    s.add_argument("--list", action="store_true", help="list providers and their status")
    s = sub.add_parser("logout", help="remove a stored LLM login")
    s.add_argument("provider")
    s = sub.add_parser("models", help="list models a provider offers")
    s.add_argument("provider", nargs="?")
    s = sub.add_parser("use", help="set the default LLM provider and model")
    s.add_argument("provider")
    s.add_argument("model", nargs="?")

    for name, hlp in (("keep", "never touch this path"), ("ignore", "leave out of plans")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("path")
        s.add_argument("--note")
    s = sub.add_parser("forget", help="drop a keep/ignore decision")
    s.add_argument("path")
    sub.add_parser("decisions", help="list remembered decisions and agent notes")

    s = sub.add_parser("history", help="audit log of executed actions")
    s.add_argument("--limit", type=int, default=30)
    s = sub.add_parser("undo", help="restore an item moved to the Trash")
    s.add_argument("action_id", type=int)

    s = sub.add_parser("schedule", help="periodic background checks via launchd")
    s.add_argument("op", choices=["install", "uninstall"])
    s.add_argument("--every", default="6h", help="interval, e.g. 6h or seconds")
    sub.add_parser("auto")  # launchd entry point
    sub.add_parser("config", help="write and show the config file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = App(args)
    try:
        return getattr(app, f"cmd_{args.cmd}")()
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        app.mem.close()
