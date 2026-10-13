"""Filesystem primitives: fast sizing, byte formatting, disk statistics.

Sizing distinguishes three numbers because APFS makes "file size" ambiguous:

* logical     - sum of st_size (what Finder's "size" shows)
* allocated   - sum of st_blocks * 512 (what the files occupy on disk)
* reclaimable - allocated minus blocks we know are shared with something
                outside the tree (hard links whose other names we never saw)

APFS clones and snapshots can still make `reclaimable` optimistic, so the
executor always measures the real free-space delta after acting.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from dataclasses import dataclass

_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]


@dataclass(slots=True)
class Usage:
    logical: int = 0
    allocated: int = 0
    reclaimable: int = 0
    files: int = 0
    dirs: int = 0
    errors: int = 0
    newest_mtime: float = 0.0

    def add(self, other: "Usage") -> None:
        self.logical += other.logical
        self.allocated += other.allocated
        self.reclaimable += other.reclaimable
        self.files += other.files
        self.dirs += other.dirs
        self.errors += other.errors
        self.newest_mtime = max(self.newest_mtime, other.newest_mtime)


def du(root: str, *, one_filesystem: bool = True) -> Usage:
    """Size a tree with os.scandir. Never follows symlinks, never crosses devices."""
    u = Usage()
    try:
        st = os.lstat(root)
    except OSError:
        u.errors += 1
        return u
    if not stat.S_ISDIR(st.st_mode):
        _count_file(u, st, {})
        _settle_links(u, {})
        return u

    dev = st.st_dev
    links: dict[tuple[int, int], list[int]] = {}  # (dev, ino) -> [seen, nlink, blocks]
    stack = [root]
    u.dirs += 1
    u.newest_mtime = st.st_mtime
    while stack:
        path = stack.pop()
        try:
            it = os.scandir(path)
        except OSError:
            u.errors += 1
            continue
        with it:
            for entry in it:
                try:
                    est = entry.stat(follow_symlinks=False)
                except OSError:
                    u.errors += 1
                    continue
                if stat.S_ISDIR(est.st_mode):
                    if one_filesystem and est.st_dev != dev:
                        continue
                    u.dirs += 1
                    if est.st_mtime > u.newest_mtime:
                        u.newest_mtime = est.st_mtime
                    stack.append(entry.path)
                else:
                    _count_file(u, est, links)
    _settle_links(u, links)
    return u


def _count_file(u: Usage, st: os.stat_result, links: dict) -> None:
    blocks = getattr(st, "st_blocks", 0) * 512 or st.st_size
    if st.st_mtime > u.newest_mtime:
        u.newest_mtime = st.st_mtime
    if st.st_nlink > 1 and not stat.S_ISLNK(st.st_mode):
        key = (st.st_dev, st.st_ino)
        rec = links.get(key)
        if rec is not None:  # another name for an inode we already counted
            rec[0] += 1
            return
        links[key] = [1, st.st_nlink, blocks]
    u.files += 1
    u.logical += st.st_size
    u.allocated += blocks


def _settle_links(u: Usage, links: dict) -> None:
    shared = sum(blocks for seen, nlink, blocks in links.values() if seen < nlink)
    u.reclaimable = max(0, u.allocated - shared)


def disk_usage(path: str) -> tuple[int, int, int]:
    """(total, used, free) for the volume holding `path`."""
    d = shutil.disk_usage(path)
    return d.total, d.used, d.free


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit in _UNITS:
        if n < 1000 or unit == _UNITS[-1]:
            return f"{sign}{n:.0f} {unit}" if unit == "B" else f"{sign}{n:.1f} {unit}"
        n /= 1000  # decimal units, like Finder
    raise AssertionError


_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([kmgtp]?i?b?)?\s*$", re.I)
_MULT = {"": 1, "b": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12, "p": 10**15}


def parse_size(text: str) -> int:
    """Parse '25GB', '500m', '1.5 TB', '2GiB' into bytes."""
    m = _SIZE_RE.match(text)
    if not m:
        raise ValueError(f"not a size: {text!r}")
    num, unit = float(m.group(1)), (m.group(2) or "").lower()
    binary = "i" in unit
    unit = unit.replace("i", "").replace("b", "")
    if binary:
        return int(num * 1024 ** "bkmgtp".index(unit or "b"))
    return int(num * _MULT[unit])


def is_within(path: str, root: str) -> bool:
    path, root = os.path.normpath(path), os.path.normpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)
