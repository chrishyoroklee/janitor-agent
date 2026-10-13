"""The cleanup rule catalog.

Every operation the engine can ever perform is declared here. A rule names
*where* it applies (home-relative glob patterns, a marker directory found in
projects, or an external tool), *what* it does (a typed action), and *how
risky* it is. Nothing else - not the database, not the LLM - can introduce a
path pattern or a shell command. Commands are fixed argv tuples, run without a
shell.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from enum import IntEnum


class Risk(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @classmethod
    def parse(cls, s: str) -> "Risk":
        return cls[s.strip().upper()]

    def __str__(self) -> str:
        return self.name.lower()


# Typed actions. The executor implements exactly these.
DELETE = "delete"                    # remove the path permanently
DELETE_CONTENTS = "delete_contents"  # empty the directory, keep it
TRASH = "trash"                      # move to ~/.Trash (reversible, frees nothing yet)
COMMAND = "command"                  # run the rule's fixed argv
ACTIONS = {DELETE, DELETE_CONTENTS, TRASH, COMMAND}


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    category: str
    risk: Risk
    action: str
    explain: str
    paths: tuple[str, ...] = ()          # home-relative globs ("*" never crosses "/")
    marker: str | None = None            # discovered directory name, e.g. "node_modules"
    command: tuple[str, ...] | None = None
    # For location rules: argv printing the directory the tool's command cleans.
    # The command is used only when that directory is the one we scanned.
    command_dir: tuple[str, ...] | None = None
    min_age_days: int = 0                # newest mtime in the tree must be older than this
    regenerable: bool = False            # can be rebuilt by a tool (allowed in personal dirs)
    auto_ok: bool = False                # may run unattended from launchd
    darwin_only: bool = False
    exclude: tuple[str, ...] = field(default=())  # home-relative globs carved out of `paths`

    @property
    def kind(self) -> str:
        if self.marker:
            return "discover"
        if self.paths:
            return "location"
        return "tool"

    def matches(self, path: str, home: str) -> bool:
        """Does `path` fall exactly inside this rule's declared scope?"""
        if self.kind == "tool":
            return path == tool_path(self.id)
        if not os.path.isabs(path):
            return False
        if self.kind == "discover":
            return is_marker_dir(self.marker, path)
        rel = os.path.relpath(path, home)
        if rel.startswith(".."):
            return False
        if any(glob_match(rel, p) for p in self.exclude):
            return False
        return any(glob_match(rel, p) for p in self.paths)


def tool_path(rule_id: str) -> str:
    return f"tool:{rule_id}"


CACHEDIR_SIGNATURE = b"Signature: 8a477f597d28d172789f06886806bc55"


def is_marker_dir(marker: str, path: str) -> bool:
    """Structural check that `path` really is the kind of directory a discover rule names."""
    name = os.path.basename(path)
    if marker == "node_modules":
        return name == "node_modules" and os.path.isfile(os.path.join(os.path.dirname(path), "package.json"))
    if marker == "venv":
        return os.path.isfile(os.path.join(path, "pyvenv.cfg"))
    if marker == "cachedir":
        try:
            with open(os.path.join(path, "CACHEDIR.TAG"), "rb") as f:
                return f.read(len(CACHEDIR_SIGNATURE)) == CACHEDIR_SIGNATURE
        except OSError:
            return False
    return False


def glob_match(rel: str, pattern: str) -> bool:
    """Segment-wise glob: each segment matched with fnmatch, counts must agree."""
    a, b = rel.split(os.sep), pattern.split("/")
    return len(a) == len(b) and all(fnmatch.fnmatchcase(x, y) for x, y in zip(a, b))


DEV, SYSTEM, PERSONAL = "developer", "system", "personal"

RULES: tuple[Rule, ...] = (
    # --- Xcode / Apple developer tooling -------------------------------------
    Rule("xcode-derived-data", "Xcode DerivedData", DEV, Risk.LOW, DELETE_CONTENTS,
         "Build products and indexes. Xcode regenerates them on the next build.",
         paths=("Library/Developer/Xcode/DerivedData",), regenerable=True, auto_ok=True),
    Rule("simulator-caches", "Simulator caches", DEV, Risk.LOW, DELETE_CONTENTS,
         "Dyld and runtime caches for iOS simulators; rebuilt on demand.",
         paths=("Library/Developer/CoreSimulator/Caches",), regenerable=True, auto_ok=True),
    Rule("simulators-unavailable", "Unavailable simulators", DEV, Risk.LOW, COMMAND,
         "Simulator devices whose runtime is no longer installed. Removed with simctl.",
         command=("xcrun", "simctl", "delete", "unavailable"), darwin_only=True, auto_ok=True),
    Rule("xcode-device-support", "Xcode device support", DEV, Risk.MEDIUM, DELETE,
         "Debug symbols for OS versions of devices you once connected. Xcode "
         "re-copies them the next time a device on that version is attached.",
         paths=("Library/Developer/Xcode/iOS DeviceSupport/*",
                "Library/Developer/Xcode/watchOS DeviceSupport/*",
                "Library/Developer/Xcode/tvOS DeviceSupport/*",
                "Library/Developer/Xcode/visionOS DeviceSupport/*"),
         min_age_days=90, regenerable=True),
    Rule("xcode-archives", "Old Xcode archives", DEV, Risk.HIGH, TRASH,
         "Archives hold the dSYMs needed to symbolicate crash reports of shipped builds.",
         paths=("Library/Developer/Xcode/Archives/*",), min_age_days=180),
    # --- package manager caches ---------------------------------------------
    Rule("homebrew-cache", "Homebrew downloads", DEV, Risk.LOW, DELETE_CONTENTS,
         "Downloaded bottles and source tarballs. `brew cleanup` prunes them.",
         paths=("Library/Caches/Homebrew",), command=("brew", "cleanup", "--prune=all", "-s"),
         command_dir=("brew", "--cache"),
         regenerable=True, auto_ok=True),
    Rule("pip-cache", "pip cache", DEV, Risk.LOW, DELETE_CONTENTS,
         "Downloaded wheels; pip re-downloads what it needs.",
         paths=("Library/Caches/pip", ".cache/pip"), command=("pip3", "cache", "purge"),
         command_dir=("pip3", "cache", "dir"),
         regenerable=True, auto_ok=True),
    Rule("uv-cache", "uv cache", DEV, Risk.LOW, DELETE_CONTENTS,
         "uv's package cache; repopulated on the next install.",
         paths=("Library/Caches/uv", ".cache/uv"), command=("uv", "cache", "clean"), command_dir=("uv", "cache", "dir"),
         regenerable=True, auto_ok=True),
    Rule("npm-cache", "npm cache", DEV, Risk.LOW, DELETE_CONTENTS,
         "npm's content-addressed tarball cache.",
         paths=(".npm/_cacache",), command=("npm", "cache", "clean", "--force"),
         command_dir=("npm", "config", "get", "cache"),
         regenerable=True, auto_ok=True),
    Rule("yarn-cache", "Yarn cache", DEV, Risk.LOW, DELETE_CONTENTS,
         "Yarn's global package cache.",
         paths=("Library/Caches/Yarn",), command=("yarn", "cache", "clean"), command_dir=("yarn", "cache", "dir"),
         regenerable=True, auto_ok=True),
    Rule("pnpm-store", "pnpm store", DEV, Risk.MEDIUM, DELETE_CONTENTS,
         "pnpm's content store. Projects hard-link into it, so `pnpm store prune` "
         "only removes packages no project references.",
         paths=("Library/pnpm/store",), command=("pnpm", "store", "prune"),
         command_dir=("pnpm", "store", "path"), regenerable=True),
    Rule("cocoapods-cache", "CocoaPods cache", DEV, Risk.LOW, DELETE_CONTENTS,
         "Downloaded pod sources.",
         paths=("Library/Caches/CocoaPods",),
         regenerable=True, auto_ok=True),
    Rule("gradle-cache", "Gradle caches", DEV, Risk.LOW, DELETE_CONTENTS,
         "Dependency and build caches; Gradle re-resolves them.",
         paths=(".gradle/caches",), regenerable=True),
    Rule("cargo-cache", "Cargo registry cache", DEV, Risk.LOW, DELETE_CONTENTS,
         "Downloaded crate archives.",
         paths=(".cargo/registry/cache",), regenerable=True, auto_ok=True),
    Rule("go-mod-cache", "Go module cache", DEV, Risk.MEDIUM, DELETE_CONTENTS,
         "Downloaded Go modules. Offline builds need them; `go clean -modcache` clears it.",
         paths=("go/pkg/mod",), command=("go", "clean", "-modcache"),
         command_dir=("go", "env", "GOMODCACHE"), regenerable=True),
    # --- Docker (sized and cleaned through the docker CLI only) -------------
    Rule("docker-build-cache", "Docker build cache", DEV, Risk.LOW, COMMAND,
         "BuildKit layer cache; rebuilt by the next `docker build`.",
         command=("docker", "builder", "prune", "-f")),
    Rule("docker-unused-images", "Unused Docker images", DEV, Risk.MEDIUM, COMMAND,
         "Images not used by any container. They are re-pulled when needed.",
         command=("docker", "image", "prune", "-a", "-f")),
    # --- projects (discovered under configured project roots) ----------------
    Rule("node-modules", "Stale node_modules", DEV, Risk.MEDIUM, DELETE,
         "Dependencies of a project untouched for a while; `npm install` restores them.",
         marker="node_modules", regenerable=True),
    Rule("python-venv", "Stale Python virtualenv", DEV, Risk.MEDIUM, DELETE,
         "Virtual environment of a project untouched for a while; recreate it from "
         "requirements/pyproject when you return.",
         marker="venv", regenerable=True),
    Rule("build-cache-dir", "Stale build cache", DEV, Risk.MEDIUM, DELETE,
         "A directory tagged with CACHEDIR.TAG (e.g. Rust `target/`); the build tool "
         "declares it disposable.",
         marker="cachedir", regenerable=True),
    # --- system / user ------------------------------------------------------
    Rule("app-caches", "Application caches", SYSTEM, Risk.MEDIUM, DELETE_CONTENTS,
         "Per-app caches in ~/Library/Caches. Apps rebuild them, but quit the app "
         "first; the next launch may be slower.",
         paths=("Library/Caches/*",),
         exclude=("Library/Caches/Homebrew", "Library/Caches/pip", "Library/Caches/uv",
                  "Library/Caches/Yarn", "Library/Caches/CocoaPods",
                  "Library/Caches/com.apple.*", "Library/Caches/CloudKit",
                  "Library/Caches/FamilyCircle")),
    Rule("user-logs", "Old logs", SYSTEM, Risk.MEDIUM, DELETE,
         "Log folders not written to in 30 days.",
         paths=("Library/Logs/*",), min_age_days=30),
    Rule("trash", "Trash", PERSONAL, Risk.MEDIUM, DELETE_CONTENTS,
         "Items you already moved to the Trash. Emptying it is permanent.",
         paths=(".Trash",)),
    Rule("downloads-installers", "Old installers & archives in Downloads", PERSONAL,
         Risk.HIGH, TRASH,
         "Disk images, installers and archives you downloaded over a month ago. "
         "Moved to the Trash, never deleted outright.",
         paths=("Downloads/*.dmg", "Downloads/*.pkg", "Downloads/*.xip",
                "Downloads/*.iso", "Downloads/*.zip"),
         min_age_days=30),
)

BY_ID = {r.id: r for r in RULES}


def get(rule_id: str) -> Rule:
    return BY_ID[rule_id]
