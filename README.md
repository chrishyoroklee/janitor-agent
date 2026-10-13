# MacJanitor (`janitor`)

A fast, safe, local-first storage maintenance agent for macOS.
**Deterministic engine, optional agentic planner.** Zero dependencies for the engine
(Python 3.12+ standard library only); the LLM is an opt-in extra.

> Tell it how much space you need, and it finds the safest way to reclaim that space
> with the least disruption.

```
$ janitor free 25GB
Scanned 31 locations in 2.4s (scan #12)

Plan #7: 4 action(s), ~26.1 GB estimated
  1. low      14.2 GB  Xcode DerivedData
  2. low       4.3 GB  Docker build cache
  3. low       2.1 GB  Homebrew downloads
  4. medium    5.5 GB  Stale node_modules     ~/code/old-site/node_modules
Run 3 low-risk action(s), ~20.6 GB? [y/N] y
[medium] Stale node_modules: ~/code/old-site/node_modules (~5.5 GB)? [y/N] y
  done  #41 Xcode DerivedData freed 14.1 GB  emptied
  ...
Reclaimed 25.8 GB (measured free-space change; estimated 26.1 GB)
```

## Install

```sh
pipx install .            # or: pip install -e .
pipx install '.[llm]'     # optional: adds the Anthropic SDK for Claude (other providers need nothing)
```

Give your terminal **Full Disk Access** (System Settings → Privacy & Security) to let it
size `~/.Trash` and app containers; without it those locations are simply skipped.

## Commands

| Command | What it does |
|---|---|
| `janitor status` | Free space, 7-day trend, last scan |
| `janitor scan [--all] [--json]` | Read-only: find and size cleanup candidates |
| `janitor plan [--target 20GB] [--max-risk medium]` | Build and save a dry-run plan |
| `janitor clean [--plan N] [--yes] [--dry-run]` | Execute a plan with approvals |
| `janitor free 25GB [--agent]` | Scan → plan → approve → clean → measure |
| `janitor du [PATH]` | Parallel "what is using space here" |
| `janitor ask "why is my disk full?"` | Ask the LLM agent (optional) |
| `janitor login [PROVIDER]` / `logout` / `models` / `use` | Connect and choose an LLM |
| `janitor keep PATH` / `ignore PATH` / `forget PATH` | Remember decisions |
| `janitor history` / `janitor undo ID` | Audit log; restore Trash moves |
| `janitor schedule install --every 6h` | Background checks via launchd |

## Architecture

```
CLI (argparse)            janitor scan · plan · clean · free · ask
   │
Agent harness             planner.py (deterministic)  ·  agent.py (LLM, optional)
   │                      llm/ — providers, logins, Anthropic/Responses/Chat adapters
   │                      policy.py — approves or rejects every action
   │
Engine                    analyzer.py (scan + candidates) · executor.py (typed ops)
   │                      rules.py — the only place paths, actions and commands exist
   │
SQLite memory + audit     db.py — scans, path history, decisions, plans, actions, notes
```

### Safety rules (non-negotiable, enforced in code)

* **Read-only by default.** `scan` and `plan` never modify anything; `clean` only runs a saved plan.
* **No LLM-generated shell.** The model picks *candidate ids*. Paths come from the scan,
  actions and commands come from the rule catalog (`rules.py`), commands run as fixed argv
  without a shell.
* **Policy re-derives everything.** Action and risk must match the rule; the path must sit
  inside the rule's declared scope, inside `$HOME`, outside protected locations (iCloud,
  Mail, Photos, iPhone backups, Docker's VM disk, `~/.ssh`, …), not be a symlink or traverse
  one, and not contain a `.git` repository. Personal folders (Desktop/Documents/Downloads)
  allow only regenerable artifacts (e.g. `node_modules`) or reversible Trash moves.
* **Revalidated at execution time.** Device + inode recorded at scan time must still match;
  a path swapped for a symlink or a new directory is skipped.
* **Tool-native cleanup first.** `brew cleanup`, `pip cache purge`, `npm cache clean`,
  `docker builder prune`, `xcrun simctl delete unavailable` … are used when the tool is
  installed *and* reports that its cache is the folder that was scanned.
* **Approval tiers.** low → one batch confirmation; medium → per item; high → type `yes`
  per item. `--yes` covers low/medium only, never high. Unattended (launchd) runs may only
  clean low-risk rules marked `auto_ok`, and only if you set `auto_clean_below_bytes`.
* **Measured, not assumed.** Every action records the real free-space delta next to the
  estimate, in SQLite and an append-only `audit.log`.

### APFS accounting

The scanner reports three sizes per candidate: *logical* (`st_size`), *allocated*
(`st_blocks × 512`) and *reclaimable* (allocated minus hard-linked blocks also referenced
from outside the tree). Clones and local snapshots can still hold blocks, which is why the
executor measures `statfs` free space before and after each action and reports both.
Trash moves are shown in plans but never counted toward a target, since they free nothing
until the Trash is emptied.

### Memory

`~/Library/Application Support/MacJanitor/janitor.db` (override with `JANITOR_DATA_DIR`):

* `scans`, `candidates` — last 20 scans in detail
* `path_history` — one row per path, updated every scan (first/last seen, size trend)
* `decisions` — your `keep` / `ignore` pins; policy and planner honor them
* `plans`, `plan_items`, `actions` — what was proposed, approved, done, and freed
* `disk_samples` — free-space trend for `status` and the agent
* `notes` — short durable notes the agent saves about your preferences

### Optional LLM agent — bring any provider

`janitor ask` / `janitor free --agent` run a small tool-use loop over five tools:
`storage_overview`, `list_candidates`, `inspect_candidate`, `propose_plan`, `remember`.
`propose_plan` goes through the policy engine and only saves a plan; you still approve it
with `janitor clean`. The model never sees a shell.

```sh
janitor login                 # pick a provider interactively
janitor login openai          # → "Sign in with ChatGPT" or paste an API key
janitor login anthropic       # → paste an API key, or `ant auth login` (Claude Console)
janitor login google          # → Gemini API key from AI Studio
janitor login google-vertex   # → `gcloud auth application-default login`
janitor login custom --base-url https://my-gateway/v1
janitor login --list          # what is connected
janitor models groq           # list a provider's models
janitor use openrouter some/model-id   # set the default
janitor ask --provider ollama --model llama3.2 "what is filling my disk?"
```

| Provider | How you connect |
|---|---|
| OpenAI | **Sign in with ChatGPT** (uses your Plus/Pro/Business plan, OpenAI's official agent OAuth flow) or `OPENAI_API_KEY` |
| Anthropic (Claude) | API key, or **`ant auth login`** (Claude Console OAuth, picked up by the Anthropic SDK) |
| Google Gemini | AI Studio key (`GEMINI_API_KEY`) |
| Google Vertex AI | **gcloud login** (Application Default Credentials) + project/location |
| OpenRouter, Groq, xAI, Mistral, DeepSeek, Cerebras, Together, Fireworks, Hugging Face, Moonshot | API key (or the provider's usual env var) |
| Ollama, LM Studio | Local, no key |
| Custom | Any OpenAI-compatible base URL + key |

Credentials resolve as: stored login → provider env var → (Claude) the SDK's own chain.
Logins live in `auth.json` next to the database, created with mode `0600`; ChatGPT tokens
refresh automatically. `janitor logout PROVIDER` removes one.

**Why no Claude Pro/Max or Gemini CLI subscription login?** Anthropic's and Google's terms
allow those subscription OAuth tokens only in their own apps (Claude Code / claude.ai,
Gemini CLI / Antigravity), and both have blocked accounts that used them in third-party
tools. MacJanitor offers their official developer logins instead (`ant auth login`, gcloud).
OpenAI explicitly supports ChatGPT sign-in for third-party agents, so that one is offered.

Wire formats: Claude uses the official Anthropic SDK (`pip install '.[llm]'`); OpenAI uses
the Responses API; everything else uses OpenAI-compatible Chat Completions over the
standard library — no other dependencies.

## Rules shipped

Xcode DerivedData, device support, archives, simulator caches, unavailable simulators ·
Homebrew, pip, uv, npm, Yarn, pnpm, CocoaPods, Gradle, Cargo, Go module caches ·
Docker build cache and unused images (via the docker CLI only) · stale `node_modules`,
virtualenvs and `CACHEDIR.TAG` build dirs in projects idle for 30+ days · app caches,
old logs, Trash · old installers/archives in Downloads (Trash only).

Add a rule by appending one `Rule(...)` to `rules.py`.

## Config

`janitor config` writes and prints `config.json`: project roots to search, staleness
threshold, minimum candidate size, extra protected paths, unattended-clean threshold,
LLM provider, model per provider, and effort.

## Development

```sh
python -m unittest          # engine tests run against a throwaway $HOME
JANITOR_USER_HOME=/tmp/fakehome JANITOR_DATA_DIR=/tmp/jd python -m macjanitor scan
```
