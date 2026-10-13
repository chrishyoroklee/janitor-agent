"""Local SQLite memory: scans, candidates, path history, user decisions,
plans, the audit trail of executed actions, and agent notes.

Every scan *updates* memory rather than just appending: `path_history` keeps
one row per path with first/last seen and size trend, so the engine (and the
optional LLM) can say "DerivedData grew 6 GB since last week" or "you told me
to keep this". Executed actions are also appended to a JSONL audit log that
lives next to the database.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from macjanitor.config import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY,
    started REAL NOT NULL, finished REAL,
    disk_total INTEGER, disk_free INTEGER,
    candidate_bytes INTEGER DEFAULT 0, candidate_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS candidates (
    scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    cid TEXT NOT NULL,
    rule_id TEXT NOT NULL, path TEXT NOT NULL,
    risk INTEGER NOT NULL, action TEXT NOT NULL,
    logical INTEGER, allocated INTEGER, reclaimable INTEGER,
    newest_mtime REAL, dev INTEGER, ino INTEGER,
    reason TEXT, extra TEXT,
    PRIMARY KEY (scan_id, cid)
);
CREATE TABLE IF NOT EXISTS path_history (
    path TEXT PRIMARY KEY, rule_id TEXT,
    first_seen REAL, last_seen REAL, times_seen INTEGER DEFAULT 0,
    last_size INTEGER, prev_size INTEGER, max_size INTEGER
);
CREATE TABLE IF NOT EXISTS decisions (
    path TEXT PRIMARY KEY,
    verdict TEXT NOT NULL CHECK (verdict IN ('keep', 'ignore')),
    note TEXT, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY,
    created REAL NOT NULL, scan_id INTEGER,
    target INTEGER, max_risk INTEGER,
    source TEXT, rationale TEXT,
    status TEXT NOT NULL DEFAULT 'proposed'
);
CREATE TABLE IF NOT EXISTS plan_items (
    plan_id INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL, cid TEXT NOT NULL,
    PRIMARY KEY (plan_id, seq)
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY,
    plan_id INTEGER, cid TEXT, rule_id TEXT, path TEXT, action TEXT,
    risk INTEGER, approved_by TEXT,
    status TEXT NOT NULL,          -- done | failed | skipped | dry_run | undone
    estimated INTEGER, freed INTEGER,
    trash_path TEXT, detail TEXT,
    started REAL, finished REAL
);
CREATE TABLE IF NOT EXISTS disk_samples (
    ts REAL PRIMARY KEY, total INTEGER, free INTEGER
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, created REAL NOT NULL,
    source TEXT NOT NULL, text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS candidates_by_rule ON candidates(scan_id, rule_id);
"""


class Memory:
    def __init__(self, path: Path | None = None):
        self.dir = data_dir() if path is None else path.parent
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = path or self.dir / "janitor.db"
        self.audit_path = self.dir / "audit.log"
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def q(self, sql: str, *args) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    def one(self, sql: str, *args) -> sqlite3.Row | None:
        return self.conn.execute(sql, args).fetchone()

    # --- scans --------------------------------------------------------------
    def record_scan(self, started: float, total: int, free: int, candidates: list) -> int:
        now = time.time()
        with self.conn:
            self.conn.execute("BEGIN")
            cur = self.conn.execute(
                "INSERT INTO scans (started, finished, disk_total, disk_free, candidate_bytes,"
                " candidate_count) VALUES (?,?,?,?,?,?)",
                (started, now, total, free, sum(c.reclaimable for c in candidates), len(candidates)),
            )
            scan_id = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(scan_id, c.cid, c.rule_id, c.path, int(c.risk), c.action, c.logical,
                  c.allocated, c.reclaimable, c.newest_mtime, c.dev, c.ino, c.reason,
                  json.dumps(c.extra)) for c in candidates],
            )
            for c in candidates:
                self.conn.execute(
                    """INSERT INTO path_history (path, rule_id, first_seen, last_seen, times_seen,
                           last_size, prev_size, max_size)
                       VALUES (?,?,?,?,1,?,NULL,?)
                       ON CONFLICT(path) DO UPDATE SET
                           last_seen=excluded.last_seen, times_seen=times_seen+1,
                           prev_size=last_size, last_size=excluded.last_size,
                           max_size=MAX(max_size, excluded.last_size)""",
                    (c.path, c.rule_id, now, now, c.reclaimable, c.reclaimable),
                )
            self.conn.execute("INSERT OR REPLACE INTO disk_samples VALUES (?,?,?)", (now, total, free))
            # Bounded memory: keep the candidate detail of the last 20 scans only.
            self.conn.execute(
                "DELETE FROM scans WHERE id NOT IN (SELECT id FROM scans ORDER BY id DESC LIMIT 20)"
                " AND id NOT IN (SELECT scan_id FROM plans WHERE scan_id IS NOT NULL)"
            )
        return scan_id

    def sample_disk(self, total: int, free: int) -> None:
        self.conn.execute("INSERT OR REPLACE INTO disk_samples VALUES (?,?,?)", (time.time(), total, free))

    def latest_scan(self) -> sqlite3.Row | None:
        return self.one("SELECT * FROM scans ORDER BY id DESC LIMIT 1")

    def candidates(self, scan_id: int) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM candidates WHERE scan_id=? ORDER BY reclaimable DESC", scan_id)

    def candidate(self, scan_id: int, cid: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM candidates WHERE scan_id=? AND cid=?", scan_id, cid)

    def history(self, path: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM path_history WHERE path=?", path)

    # --- user decisions -----------------------------------------------------
    def decide(self, path: str, verdict: str, note: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?)", (path, verdict, note, time.time())
        )

    def forget(self, path: str) -> bool:
        return self.conn.execute("DELETE FROM decisions WHERE path=?", (path,)).rowcount > 0

    def decisions(self) -> dict[str, sqlite3.Row]:
        return {r["path"]: r for r in self.q("SELECT * FROM decisions")}

    # --- plans --------------------------------------------------------------
    def save_plan(self, scan_id: int, cids: list[str], *, target: int | None, max_risk: int,
                  source: str, rationale: str = "") -> int:
        with self.conn:
            self.conn.execute("BEGIN")
            cur = self.conn.execute(
                "INSERT INTO plans (created, scan_id, target, max_risk, source, rationale)"
                " VALUES (?,?,?,?,?,?)",
                (time.time(), scan_id, target, max_risk, source, rationale),
            )
            pid = cur.lastrowid
            self.conn.executemany("INSERT INTO plan_items VALUES (?,?,?)",
                                  [(pid, i, cid) for i, cid in enumerate(cids)])
        return pid

    def plan(self, plan_id: int | None = None) -> sqlite3.Row | None:
        if plan_id is None:
            return self.one("SELECT * FROM plans ORDER BY id DESC LIMIT 1")
        return self.one("SELECT * FROM plans WHERE id=?", plan_id)

    def plan_cids(self, plan_id: int) -> list[str]:
        return [r["cid"] for r in self.q("SELECT cid FROM plan_items WHERE plan_id=? ORDER BY seq", plan_id)]

    def set_plan_status(self, plan_id: int, status: str) -> None:
        self.conn.execute("UPDATE plans SET status=? WHERE id=?", (status, plan_id))

    # --- audit --------------------------------------------------------------
    def log_action(self, **rec) -> int:
        cols = ", ".join(rec)
        cur = self.conn.execute(
            f"INSERT INTO actions ({cols}) VALUES ({', '.join('?' * len(rec))})", tuple(rec.values())
        )
        rec["id"] = cur.lastrowid
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return cur.lastrowid

    def mark_undone(self, action_id: int) -> None:
        self.conn.execute("UPDATE actions SET status='undone' WHERE id=?", (action_id,))
        with open(self.audit_path, "a") as f:
            f.write(json.dumps({"undo": action_id, "ts": time.time()}) + "\n")

    def actions(self, limit: int = 30) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM actions ORDER BY id DESC LIMIT ?", limit)

    def action(self, action_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM actions WHERE id=?", action_id)

    def disk_trend(self, since: float) -> tuple[sqlite3.Row | None, sqlite3.Row | None]:
        first = self.one("SELECT * FROM disk_samples WHERE ts>=? ORDER BY ts ASC LIMIT 1", since)
        last = self.one("SELECT * FROM disk_samples ORDER BY ts DESC LIMIT 1")
        return first, last

    # --- agent notes --------------------------------------------------------
    def add_note(self, text: str, source: str = "agent") -> None:
        self.conn.execute("INSERT INTO notes (created, source, text) VALUES (?,?,?)",
                          (time.time(), source, text[:500]))
        self.conn.execute("DELETE FROM notes WHERE id NOT IN (SELECT id FROM notes ORDER BY id DESC LIMIT 50)")

    def notes(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM notes ORDER BY id DESC LIMIT ?", limit)
