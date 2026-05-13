# AGENTS.md

Project: `atlas_hetzner_final_project`
Repo: https://github.com/aymank2020/atlas_hetzner_final_project
Server Path: `/srv/atlas/atlas_hetzner_final_project`

## Purpose

Automated Atlas Capture episode labeling pipeline for Hetzner production deployment:

1. Open/reserve an episode on the Atlas UI
2. Extract live segments from the DOM
3. Generate labels (and optional structural repair) via Gemini
4. Validate against labeling policy
5. Apply labels in the UI
6. Verify and submit safely

This is a stateful browser + AI + policy pipeline. The biggest failure mode is
desync between live DOM state, extracted segment snapshots, Gemini plan output,
and final submit state.

## Critical Rule: Source of Truth

When debugging timestamps or duration mismatches:

- **live DOM** is the real current UI state
- **source_segments** is the extracted DOM snapshot used for the current solve pass
- **segment_plan** is AI output and may hallucinate timestamps

For duration enforcement, DOM/source timestamps are ground truth.
Do NOT treat Gemini timestamps as authoritative for policy blocking.

## Project Layout

```
atlas_web_auto_solver.py       # Main entry point
atlas_triplet_compare.py       # Deep Gemini chat-web interaction logic
run_gemini_chat_json.py        # Standalone Gemini chat runner
run_gemini_chat_timed_labels.py
validator.py                   # Offline label validator
submit_gate.py                 # Offline submit gate checker
prompts.py                     # Prompt builder utilities
repair_payload_builder.py      # Structural repair payload builder
_fetch_otp.py                  # Gmail IMAP OTP fetcher
save_gemini_state.py           # Gemini browser state saver
soundcard_patch.py             # Soundcard stub for headless servers

src/
  solver/
    legacy_impl.py             # Main episode runner + integration glue
    orchestrator.py            # Policy gate orchestration, compare/retry flow
    segments.py                # Segment extraction, structural ops, label apply
    chat_only.py               # Chat-web subprocess wrapper, request observability
    gemini.py                  # Gemini API path
    gemini_session.py          # Single-session Gemini manager (v2 path)
    episode_runtime.py         # Episode-scoped runtime container (v2 path)
    browser.py                 # Playwright browser lifecycle
    video.py                   # Video download/processing
    video_core.py              # Dependency-light video helpers
    desync.py                  # Snapshot/checksum helpers for desync detection
    reliability.py             # Retry taxonomy + episode report primitives
    pre_submit_compare.py      # Pre-submit chat-web comparison
    prompting.py               # Prompt assembly
    live_validation.py         # Live DOM validation
    cli.py                     # CLI argument parsing
  rules/
    policy_gate.py             # Final policy validation rules
    consistency.py             # Live DOM vs extracted source consistency
    labels.py                  # Label format/content rules
  infra/
    artifacts.py               # Task-scoped caches, output dumps
    solver_config.py           # Config loading + 300+ keys
    browser_auth.py            # Auth/login helpers
    logging_utils.py           # Logging setup
    runtime.py                 # Runtime environment detection
    session_heartbeat.py       # Heartbeat for long-running sessions
    submit_verify.py           # Submit verification helpers
    utils.py                   # Shared utility functions
    execution_journal.py       # Step-by-step execution journal
    gemini_economics.py        # Cost tracking

configs/
  config_hetzner_production.yaml  # Production config (Hetzner CX43)
  accounts/                        # Per-account overlays

deploy/
  setup_hetzner.sh             # First-time server setup
  install_service.sh           # Install systemd services
  run_solver.sh                # Main solver launcher (Xvfb + Chrome + VNC)
  update_and_restart.sh        # Git pull + restart
  monitor_daemon.py            # Solver watchdog + Telegram alerts
  update_checker.py            # Auto-update checker
  atlas-solver.service         # systemd unit file
  atlas-monitor.service        # systemd unit file
  atlas-updater.service        # systemd unit file

tests/                         # pytest test suite
data/policy/                   # Labeling policy documents
prompts/                       # Domain-specific prompt templates
```

## Quick Start (Hetzner Server)

```bash
# 1. First-time server setup (as root)
bash deploy/setup_hetzner.sh

# 2. Edit credentials
nano .env

# 3. Install systemd services
bash deploy/install_service.sh

# 4. Start solver
systemctl start atlas-solver

# 5. Check status
systemctl status atlas-solver
journalctl -u atlas-solver -f
```

## Manual Run (without systemd)

```bash
sudo -u atlas bash deploy/run_solver.sh
```

## Verification

Quick compile check after edits:

```bash
python -m py_compile src/infra/artifacts.py src/rules/consistency.py src/solver/chat_only.py src/solver/legacy_impl.py src/solver/segments.py
```

Run tests:

```bash
python -m pytest tests/ -x -q
```

## Known Risk Areas

### 1. Desync / false rejection
Typical symptom: UI shows segment duration 5s, policy rejects as 15s.
Causes: stale extracted state, Gemini hallucinated timestamps, DOM changed after extraction.

### 2. Async UI timing
Atlas UI is React-like; prefer stability checks over simple element existence checks.

### 3. Chat session behavior
Two chat-web paths exist:
- Legacy subprocess path (fallback)
- v2 registered-session path (preferred when canary flags enabled)

v2 path is controlled by three flags:
- `run.use_episode_runtime_v2`
- `run.force_episode_browser_isolation`
- `run.strict_single_chat_session`

### 4. Import cycle prevention
The `video.py <-> legacy_impl.py` cycle was broken by extracting pure helpers into `video_core.py`.
- `video_core.py` must stay dependency-light
- `video.py` may lazily import runtime helpers only inside functions
- `legacy_impl.py` may alias from `video.py` / `segments.py`

## Debugging Workflow

To trace a broken segment end-to-end:

1. Find it in `extract_segments(...)` output
2. Find it in `source_segments`
3. Find it in `segment_plan`
4. Inspect `policy_gate` decision
5. Inspect final pre-submit guard output

## What To Preserve

- Fail-closed submit behavior
- DOM-grounded duration checks
- Raw Gemini response persistence before parsing
- Task-scoped cache cleanup between episodes
- Compatibility aliases in `legacy_impl.py`

## What Not To Assume

- Gemini timestamps are NOT reliable
- DOM is NOT stable immediately after UI actions
- One chat request does NOT equal one chat session
- `legacy_impl.py` comments may NOT reflect the real runtime path
