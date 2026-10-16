"""Engine tests against a throwaway home directory. Run: python -m unittest"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from macjanitor import rules
from macjanitor.analyzer import Analyzer, Candidate
from macjanitor.config import Config
from macjanitor.db import Memory
from macjanitor.executor import Executor
from macjanitor.fsutil import du, fmt_bytes, parse_size
from macjanitor.planner import make_plan, validate_selection
from macjanitor.policy import Policy
from macjanitor.rules import Risk

MB = 1000 * 1000
OLD = time.time() - 90 * 86400


def write(path: Path, size: int = 0, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(size))
    if mtime:
        os.utime(path, (mtime, mtime))
    return path


def age(path: Path, mtime: float = OLD) -> None:
    os.utime(path, (mtime, mtime))


def case_insensitive(path: str) -> bool:
    return os.path.exists(path.swapcase())


class Env(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.home = os.path.realpath(os.path.join(self.tmp, "home"))
        os.makedirs(self.home)
        self.data = os.path.join(self.tmp, "data")
        self.env = mock.patch.dict(os.environ, {"JANITOR_USER_HOME": self.home,
                                                "JANITOR_DATA_DIR": self.data})
        self.env.start()
        # Never shell out to real package managers from tests.
        self.which = mock.patch("shutil.which", return_value=None)
        self.which.start()
        self.h = Path(self.home)
        self.config = Config(min_candidate_bytes=1 * MB)
        self.mem = Memory()

    def tearDown(self):
        self.mem.close()
        self.which.stop()
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build_home(self):
        h = self.h
        write(h / "Library/Developer/Xcode/DerivedData/App-x/Build/a.o", 4 * MB)
        write(h / "Library/Caches/pip/wheels/w.whl", 3 * MB)
        write(h / "Library/Caches/com.example.app/blob", 2 * MB)
        # stale project
        write(h / "code/old/node_modules/lodash/index.js", 2 * MB)
        write(h / "code/old/package.json", 10, OLD)
        age(h / "code/old/node_modules")
        # active project: must be skipped
        write(h / "code/new/node_modules/x/index.js", 2 * MB)
        write(h / "code/new/package.json", 10)
        # stale venv inside Documents (regenerable => allowed in personal area)
        write(h / "Documents/proj/.venv/lib/big", 2 * MB)
        write(h / "Documents/proj/.venv/pyvenv.cfg", 10)
        write(h / "Documents/proj/main.py", 10, OLD)
        age(h / "Documents/proj/.venv")
        # old installer
        write(h / "Downloads/tool.dmg", 2 * MB, OLD)
        # personal files and a symlink pointing at them
        write(h / "Pictures/photo.jpg", 5 * MB)
        os.symlink(h / "Pictures", h / "Library/Caches/sneaky")

    def scan(self):
        cands = Analyzer(self.home, self.config, tools=False).run()
        return cands, {c.rule_id + ":" + os.path.relpath(c.path, self.home): c for c in cands}

    def policy(self):
        return Policy(self.home, decisions=self.mem.decisions())


class FsutilTests(unittest.TestCase):
    def test_parse_and_format(self):
        self.assertEqual(parse_size("25GB"), 25 * 10**9)
        self.assertEqual(parse_size("1.5 tb"), 1.5 * 10**12)
        self.assertEqual(parse_size("2GiB"), 2 * 1024**3)
        self.assertEqual(parse_size("500m"), 500 * MB)
        self.assertEqual(fmt_bytes(14_200_000_000), "14.2 GB")
        with self.assertRaises(ValueError):
            parse_size("lots")

    def test_hardlinks_shared_outside_tree_are_not_reclaimable(self):
        with tempfile.TemporaryDirectory() as d:
            inner = Path(d, "tree")
            f = write(inner / "f", 2 * MB)
            os.link(f, Path(d, "outside"))
            write(inner / "g", 1 * MB)
            os.link(inner / "g", inner / "g2")  # both names inside: still reclaimable
            u = du(str(inner))
            self.assertEqual(u.files, 2)
            self.assertGreaterEqual(u.allocated, 3 * MB)
            self.assertLess(u.reclaimable, 2 * MB)
            self.assertGreaterEqual(u.reclaimable, 1 * MB)

    def test_single_hardlinked_file_is_not_reclaimable(self):
        with tempfile.TemporaryDirectory() as d:
            f = write(Path(d, "a.dmg"), 2 * MB)
            os.link(f, Path(d, "elsewhere.dmg"))
            self.assertEqual(du(str(f)).reclaimable, 0)

    def test_does_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            write(Path(d, "big/file"), 3 * MB)
            os.makedirs(Path(d, "tree"))
            os.symlink(Path(d, "big"), Path(d, "tree/link"))
            self.assertLess(du(str(Path(d, "tree"))).allocated, 1 * MB)


class AnalyzerTests(Env):
    def test_finds_expected_candidates(self):
        self.build_home()
        _, by = self.scan()
        self.assertIn("xcode-derived-data:Library/Developer/Xcode/DerivedData", by)
        self.assertIn("pip-cache:Library/Caches/pip", by)
        self.assertIn("app-caches:Library/Caches/com.example.app", by)
        self.assertIn("node-modules:code/old/node_modules", by)
        self.assertIn("python-venv:Documents/proj/.venv", by)
        self.assertIn("downloads-installers:Downloads/tool.dmg", by)
        self.assertNotIn("node-modules:code/new/node_modules", by)       # active project
        self.assertFalse(any("sneaky" in k or "Pictures" in k for k in by))
        self.assertFalse(any(k.startswith("app-caches:Library/Caches/pip") for k in by))

    def test_rescan_updates_memory(self):
        self.build_home()
        cands, _ = self.scan()
        self.mem.record_scan(time.time(), 1, 1, cands)
        write(self.h / "Library/Developer/Xcode/DerivedData/App-x/Build/b.o", 3 * MB)
        cands, _ = self.scan()
        self.mem.record_scan(time.time(), 1, 1, cands)
        h = self.mem.history(str(self.h / "Library/Developer/Xcode/DerivedData"))
        self.assertEqual(h["times_seen"], 2)
        self.assertGreater(h["last_size"], h["prev_size"])

    def test_overlapping_project_roots_scanned_once(self):
        self.build_home()
        roots = ["code/old", "code"]  # child listed before its parent
        if case_insensitive(self.home):
            roots.append("CODE")      # same folder, different spelling
        self.config.project_roots = roots
        cands, _ = self.scan()
        nm = [c for c in cands if c.rule_id == "node-modules"]
        self.assertEqual([os.path.relpath(c.path, self.home) for c in nm], ["code/old/node_modules"])
        self.mem.record_scan(time.time(), 1, 1, cands)  # no duplicate-key crash


class PolicyTests(Env):
    def cand(self, rule_id, rel, **kw):
        r = rules.get(rule_id)
        path = os.path.join(self.home, rel)
        st = os.lstat(path) if os.path.lexists(path) else None
        return Candidate(rule_id, path, r.risk, r.action, reclaimable=MB,
                         dev=st.st_dev if st else 0, ino=st.st_ino if st else 0, **kw)

    def test_rejections(self):
        self.build_home()
        p = self.policy()
        cases = {
            "outside the scope": self.cand("pip-cache", "code/new/package.json"),
            "protected location: Pictures": self.cand("app-caches", "Pictures/photo.jpg"),
            "protected": self.cand("app-caches", "Library/Mobile Documents"),
            "symlink": self.cand("app-caches", "Library/Caches/sneaky"),
            "top-level": self.cand("downloads-installers", "Downloads"),
        }
        for why, cand in cases.items():
            v = p.check(cand)
            self.assertFalse(v.allowed, why)
            self.assertIn(why, " ".join(v.reasons), why)

    def test_tampered_action_or_risk_rejected(self):
        self.build_home()
        c = self.cand("downloads-installers", "Downloads/tool.dmg")
        c.action = rules.DELETE  # e.g. edited database row or LLM output
        self.assertFalse(self.policy().check(c).allowed)
        c = self.cand("downloads-installers", "Downloads/tool.dmg")
        c.risk = Risk.LOW
        self.assertFalse(self.policy().check(c).allowed)

    def test_identity_change_rejected(self):
        self.build_home()
        c = self.cand("pip-cache", "Library/Caches/pip")
        os.rename(c.path, c.path + ".old")
        os.makedirs(c.path)  # same path, different inode
        v = self.policy().check(c)
        self.assertFalse(v.allowed)

    def test_keep_decision_and_risk_ceiling(self):
        self.build_home()
        c = self.cand("pip-cache", "Library/Caches/pip")
        self.assertTrue(self.policy().check(c).allowed)
        self.mem.decide(str(self.h / "Library/Caches"), "keep")
        self.assertFalse(self.policy().check(c).allowed)
        dmg = self.cand("downloads-installers", "Downloads/tool.dmg")
        self.assertFalse(self.policy().check(dmg, max_risk=Risk.MEDIUM).allowed)
        self.assertEqual(self.policy().check(dmg).approval, "explicit")

    def test_unattended_only_low_auto_ok(self):
        self.build_home()
        self.assertTrue(self.policy().check(self.cand("pip-cache", "Library/Caches/pip"),
                                            unattended=True).allowed)
        self.assertFalse(self.policy().check(self.cand("app-caches", "Library/Caches/com.example.app"),
                                             unattended=True).allowed)

    def test_git_repo_never_deleted(self):
        write(self.h / "code/proj/node_modules/.git/HEAD", 10)
        write(self.h / "code/proj/package.json", 10)
        c = self.cand("node-modules", "code/proj/node_modules")
        self.assertIn("git", " ".join(self.policy().check(c).reasons))

    def test_trash_rule_is_usable(self):
        write(self.h / ".Trash/old.zip", 2 * MB)
        _, by = Env.scan(self)
        v = self.policy().check(by["trash:.Trash"])
        self.assertTrue(v.allowed, v.reasons)

    def test_keep_spelled_with_other_case_is_honored(self):
        self.build_home()
        if not case_insensitive(self.home):
            self.skipTest("case-sensitive filesystem")
        from macjanitor.cli import main
        with redirect_stdout(io.StringIO()):
            main(["keep", str(self.h / "CODE/OLD")])
        c = self.cand("node-modules", "code/old/node_modules")
        self.assertIn("keep", " ".join(self.policy().check(c).reasons))


class PlannerTests(Env):
    def test_low_risk_first_and_stops_at_target(self):
        self.build_home()
        cands, _ = self.scan()
        plan = make_plan(cands, self.policy(), target=5 * MB, max_risk=Risk.MEDIUM)
        self.assertTrue(plan.reaches_target)
        self.assertEqual([c.risk for c in plan.items], [Risk.LOW, Risk.LOW])

    def test_trash_moves_do_not_count_toward_target(self):
        self.build_home()
        cands, _ = self.scan()
        plan = make_plan(cands, self.policy(), target=10**12, max_risk=Risk.HIGH)
        self.assertFalse(plan.reaches_target)
        self.assertNotIn("downloads-installers", [c.rule_id for c in plan.items])

    def test_llm_selection_is_validated(self):
        self.build_home()
        cands, by = self.scan()
        dmg = by["downloads-installers:Downloads/tool.dmg"]
        pip = by["pip-cache:Library/Caches/pip"]
        plan = validate_selection([pip.cid, dmg.cid, "nope", pip.cid], {c.cid: c for c in cands},
                                  self.policy(), max_risk=Risk.MEDIUM)
        self.assertEqual([c.cid for c in plan.items], [pip.cid])
        self.assertEqual(len(plan.rejected), 1)

    def test_ignore_covers_everything_inside(self):
        self.build_home()
        cands, _ = self.scan()
        self.mem.decide(str(self.h / "code/old"), "ignore")
        plan = make_plan(cands, self.policy(), max_risk=Risk.MEDIUM, decisions=self.mem.decisions())
        self.assertNotIn("node-modules", [c.rule_id for c in plan.items])


class ExecutorTests(Env):
    def run_one(self, cand, **kw):
        ex = Executor(self.home, self.mem, self.policy())
        return ex.run(cand, plan_id=None, approved_by="test", max_risk=Risk.HIGH, **kw)

    def test_delete_contents_keeps_directory(self):
        self.build_home()
        _, by = self.scan()
        c = by["xcode-derived-data:Library/Developer/Xcode/DerivedData"]
        res = self.run_one(c)
        self.assertEqual(res.status, "done", res.detail)
        self.assertTrue(os.path.isdir(c.path))
        self.assertEqual(os.listdir(c.path), [])
        self.assertEqual(self.mem.actions()[0]["status"], "done")
        self.assertTrue(Path(self.data, "audit.log").read_text().strip())

    def test_dry_run_touches_nothing(self):
        self.build_home()
        _, by = self.scan()
        c = by["node-modules:code/old/node_modules"]
        self.assertEqual(self.run_one(c, dry_run=True).status, "dry_run")
        self.assertTrue(os.path.isdir(c.path))

    def test_delete_read_only_tree(self):
        self.build_home()
        _, by = self.scan()
        c = by["node-modules:code/old/node_modules"]
        os.chmod(os.path.join(c.path, "lodash"), 0o555)
        self.assertEqual(self.run_one(c).status, "done")
        self.assertFalse(os.path.exists(c.path))

    def test_trash_and_undo(self):
        self.build_home()
        _, by = self.scan()
        c = by["downloads-installers:Downloads/tool.dmg"]
        res = self.run_one(c)
        self.assertEqual(res.status, "done")
        self.assertFalse(os.path.exists(c.path))
        self.assertTrue(os.path.exists(res.trash_path))
        Executor(self.home, self.mem, self.policy()).undo(res.action_id)
        self.assertTrue(os.path.exists(c.path))
        self.assertEqual(self.mem.action(res.action_id)["status"], "undone")

    def test_swapped_for_symlink_is_skipped(self):
        self.build_home()
        _, by = self.scan()
        c = by["pip-cache:Library/Caches/pip"]
        shutil.rmtree(c.path)
        os.symlink(self.h / "Pictures", c.path)  # race: path replaced after scan
        res = self.run_one(c)
        self.assertEqual(res.status, "skipped")
        self.assertTrue((self.h / "Pictures/photo.jpg").exists())

    def test_tool_command_only_when_it_owns_the_folder(self):
        self.build_home()
        _, by = self.scan()
        c = by["pip-cache:Library/Caches/pip"]
        ex = Executor(self.home, self.mem, self.policy())
        fake = mock.Mock(returncode=0, stdout="/somewhere/else\n")
        with mock.patch("shutil.which", return_value="/usr/bin/pip3"), \
                mock.patch("subprocess.run", return_value=fake):
            self.assertIsNone(ex._command(c.rule, c))
            fake.stdout = c.path + "\n"
            self.assertEqual(ex._command(c.rule, c), c.rule.command)


class CliTests(Env):
    def run_cli(self, *argv):
        from macjanitor.cli import main

        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch("sys.stdin", io.StringIO("")):
            code = main(list(argv))
        return code, buf.getvalue()

    def test_free_end_to_end(self):
        self.build_home()
        with mock.patch.object(Config, "load", return_value=self.config):
            code, out = self.run_cli("free", "5MB", "--yes", "--no-tools")
        self.assertEqual(code, 0, out)
        self.assertIn("Reclaimed", out)
        self.assertEqual(os.listdir(self.h / "Library/Developer/Xcode/DerivedData"), [])
        self.assertTrue((self.h / "code/old/node_modules").exists())  # not needed for 5 MB
        code, out = self.run_cli("history")
        self.assertIn("xcode-derived-data", out)

    def test_high_risk_never_runs_non_interactively(self):
        self.build_home()
        with mock.patch.object(Config, "load", return_value=self.config):
            self.run_cli("plan", "--max-risk", "high")
            code, out = self.run_cli("clean", "--yes")
        self.assertTrue((self.h / "Downloads/tool.dmg").exists())
        self.assertIn("needs interactive approval", out)

    def test_non_interactive_low_risk_explains_skip(self):
        self.build_home()
        with mock.patch.object(Config, "load", return_value=self.config):
            code, out = self.run_cli("free", "5MB", "--no-tools")
        self.assertIn("needs approval", out)

    def test_bad_size_is_a_usage_error(self):
        from macjanitor.cli import build_parser
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            build_parser().parse_args(["free", "lots"])

    def test_schedule_interval_parsing(self):
        from macjanitor.cli import parse_interval
        self.assertEqual(parse_interval("6h"), 6 * 3600)
        self.assertEqual(parse_interval("1d"), 86400)
        self.assertEqual(parse_interval("45m"), 2700)
        self.assertEqual(parse_interval("3600"), 3600)
        with self.assertRaises(ValueError):
            parse_interval("often")


if __name__ == "__main__":
    unittest.main()
