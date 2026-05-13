"""Sequential multi-account runner for Atlas solver deployments."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.infra.solver_config import _deep_merge, load_config


_TASK_ID_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{24})(?![0-9a-f])", re.IGNORECASE)


def _load_yaml_dict(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return raw


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path or "").strip())
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _load_project_env(repo_root: Path, base_env: Dict[str, str] | None = None) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    dotenv_path = repo_root / ".env"
    if not dotenv_path.exists():
        return env
    try:
        from dotenv import dotenv_values  # type: ignore

        values = dotenv_values(dotenv_path)
    except Exception:
        values = {}
        for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    for key, value in dict(values).items():
        if not key or value is None or key in env:
            continue
        env[str(key)] = str(value)
    return env


def load_account_index(index_path: Path) -> Dict[str, Any]:
    raw = _load_yaml_dict(index_path)
    scheduler = raw.get("scheduler", {})
    accounts = raw.get("accounts", [])
    if not isinstance(scheduler, dict):
        raise ValueError("scheduler must be a YAML object")
    if not isinstance(accounts, list) or not accounts:
        raise ValueError("accounts must be a non-empty YAML list")
    return raw


def _enabled_accounts(index_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in index_cfg.get("accounts", []):
        if not isinstance(item, dict):
            continue
        if bool(item.get("enabled", True)):
            out.append(item)
    return out


def build_runner_env(account_cfg: Dict[str, Any], base_env: Dict[str, str] | None = None) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    runner_cfg = account_cfg.get("runner", {})
    if not isinstance(runner_cfg, dict):
        return env
    env_clear = runner_cfg.get("env_clear", [])
    if isinstance(env_clear, list):
        for item in env_clear:
            key = str(item or "").strip()
            if key:
                env.pop(key, None)
    env_map = runner_cfg.get("env_map", {})
    if not isinstance(env_map, dict):
        return env
    for target_env, source_env in env_map.items():
        target = str(target_env or "").strip()
        source = str(source_env or "").strip()
        if not target or not source:
            continue
        # Avoid leaking the previous account's value when the mapped source is absent.
        env.pop(target, None)
        value = env.get(source, "").strip()
        if value:
            env[target] = value
    return env


def build_effective_config(
    base_cfg: Dict[str, Any],
    account_cfg: Dict[str, Any],
    account_name: str,
    *,
    seed_blocked_task_ids: List[str] | None = None,
) -> Dict[str, Any]:
    overlay = {key: value for key, value in account_cfg.items() if key != "runner"}
    merged = _deep_merge(base_cfg, overlay)

    browser_cfg = merged.setdefault("browser", {})
    run_cfg = merged.setdefault("run", {})
    gem_cfg = merged.setdefault("gemini", {})

    browser_cfg["headless"] = bool(browser_cfg.get("headless", True))
    browser_cfg["use_chrome_profile"] = False
    browser_cfg["restore_state_in_profile_mode"] = False
    storage_state_path = str(browser_cfg.get("storage_state_path", "") or "").strip()
    if not storage_state_path or storage_state_path == ".state/atlas_auth.json":
        browser_cfg["storage_state_path"] = f".state/accounts/{account_name}/atlas_auth.json"
    run_cfg.setdefault("output_dir", f"outputs/{account_name}")
    if seed_blocked_task_ids:
        run_cfg["seed_blocked_task_ids"] = [str(item).strip() for item in seed_blocked_task_ids if str(item).strip()]
    gem_cfg.setdefault("api_keys", [])
    gem_cfg.setdefault("rotation_policy", "round_robin")

    return merged


def _latest_task_id_from_output_dir(repo_root: Path, output_dir_raw: str) -> str:
    output_dir = Path(str(output_dir_raw or "").strip())
    if not output_dir:
        return ""
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    if not output_dir.exists() or not output_dir.is_dir():
        return ""
    try:
        files = sorted(
            (p for p in output_dir.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return ""
    for path in files[:80]:
        match = _TASK_ID_RE.search(path.name)
        if match:
            return match.group(1).lower()
    return ""


def _account_unready_reason(repo_root: Path, cfg: Dict[str, Any], env: Dict[str, str]) -> str:
    atlas_email = str(env.get("ATLAS_LOGIN_EMAIL", "") or env.get("ATLAS_EMAIL", "")).strip()
    if not atlas_email:
        return "missing ATLAS_LOGIN_EMAIL/ATLAS_EMAIL"

    browser_cfg = cfg.get("browser", {}) if isinstance(cfg.get("browser"), dict) else {}
    storage_state_raw = str(browser_cfg.get("storage_state_path", "") or "").strip()
    if storage_state_raw:
        storage_state = Path(storage_state_raw)
        if not storage_state.is_absolute():
            storage_state = (repo_root / storage_state).resolve()
        if storage_state.exists():
            return ""

    gmail_password = str(env.get("GMAIL_APP_PASSWORD", "")).strip()
    gmail_email = str(env.get("GMAIL_EMAIL", "") or env.get("GMAIL_USER", "")).strip()
    if gmail_password and gmail_email:
        return ""
    return "missing storage_state and Gmail OTP credentials"


def materialize_account_config(
    repo_root: Path,
    index_path: Path,
    base_cfg: Dict[str, Any],
    account_entry: Dict[str, Any],
    *,
    episodes_per_turn: int,
    seed_blocked_task_ids: List[str] | None = None,
    base_env: Dict[str, str] | None = None,
) -> tuple[str, Path, Dict[str, str]]:
    index_dir = index_path.parent
    account_name = str(account_entry.get("name", "")).strip() or Path(str(account_entry.get("config", ""))).stem
    account_cfg_path = _resolve_path(index_dir, str(account_entry.get("config", "")))
    account_cfg = _load_yaml_dict(account_cfg_path)
    env = build_runner_env(account_cfg, base_env=base_env)
    effective_cfg = build_effective_config(
        base_cfg,
        account_cfg,
        account_name,
        seed_blocked_task_ids=seed_blocked_task_ids,
    )
    if episodes_per_turn > 0:
        run_cfg = effective_cfg.setdefault("run", {})
        run_cfg["max_episodes_per_run"] = int(episodes_per_turn)
        run_cfg["recycle_after_max_episodes"] = False

    generated_dir = repo_root / ".state" / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_path = generated_dir / f"{account_name}.generated.yaml"
    generated_path.write_text(
        yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return account_name, generated_path, env


def _terminate_process_group(
    proc: subprocess.Popen[Any],
    *,
    initial_signal: int = signal.SIGINT,
    grace_sec: float = 15.0,
) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    def _send(sig: int) -> None:
        try:
            if pgid and pgid > 0:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except Exception:
            return

    _send(initial_signal)
    try:
        proc.wait(timeout=max(1.0, grace_sec))
        return
    except Exception:
        pass

    _send(signal.SIGTERM)
    try:
        proc.wait(timeout=5.0)
        return
    except Exception:
        pass

    _send(signal.SIGKILL)
    try:
        proc.wait(timeout=5.0)
    except Exception:
        pass


def run_account_process(
    repo_root: Path,
    account_name: str,
    generated_cfg_path: Path,
    env: Dict[str, str],
    *,
    execute: bool,
    timeout_sec: float,
) -> int:
    cmd = [sys.executable, "atlas_web_auto_solver.py", "--config", str(generated_cfg_path)]
    if execute:
        cmd.insert(2, "--execute")
    print(f"[runner] starting account={account_name} execute={int(execute)} config={generated_cfg_path}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        start_new_session=True,
    )
    try:
        return_code = int(proc.wait(timeout=(timeout_sec if timeout_sec > 0 else None)))
        print(f"[runner] finished account={account_name} exit_code={return_code}")
        return return_code
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc, initial_signal=signal.SIGINT, grace_sec=15.0)
        print(
            f"[runner] timed out account={account_name} after "
            f"{timeout_sec:.1f}s (returning exit_code=124)"
        )
        return 124
    except KeyboardInterrupt:
        print(f"[runner] interrupt received for account={account_name}; stopping child process group...")
        _terminate_process_group(proc, initial_signal=signal.SIGINT, grace_sec=20.0)
        raise


def _normalize_retry_exit_codes(raw_value: Any) -> List[int]:
    if isinstance(raw_value, list):
        out: List[int] = []
        for item in raw_value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    try:
        return [int(raw_value)]
    except Exception:
        return []


def run_scheduler(index_path: Path, *, execute: bool, loop_forever: bool) -> int:
    repo_root = _resolve_repo_root()
    base_env = _load_project_env(repo_root)
    index_cfg = load_account_index(index_path)
    scheduler_cfg = index_cfg.get("scheduler", {})
    mode = str(scheduler_cfg.get("mode", "sequential") or "").strip().lower() or "sequential"
    if mode != "sequential":
        raise ValueError("Only scheduler.mode=sequential is supported in this release.")

    accounts = _enabled_accounts(index_cfg)
    if not accounts:
        raise RuntimeError("No enabled accounts found in scheduler index.")

    base_cfg_path = _resolve_path(index_path.parent, str(index_cfg.get("base_config", "configs/production_hetzner.yaml")))
    base_cfg = load_config(base_cfg_path)
    cooldown_between_accounts_sec = max(0.0, float(scheduler_cfg.get("cooldown_between_accounts_sec", 20)))
    rest_between_turns_sec = max(
        0.0,
        float(
            scheduler_cfg.get(
                "rest_between_turns_sec",
                cooldown_between_accounts_sec,
            )
            or cooldown_between_accounts_sec
        ),
    )
    loop_pause_sec = max(0.0, float(scheduler_cfg.get("loop_pause_sec", 0) or 0))
    continue_on_error = bool(scheduler_cfg.get("continue_on_error", True))
    per_account_timeout_sec = max(0.0, float(scheduler_cfg.get("per_account_timeout_sec", 0)))
    episodes_per_turn = max(1, int(scheduler_cfg.get("episodes_per_account_per_turn", 1) or 1))
    skip_unready_accounts = bool(scheduler_cfg.get("skip_unready_accounts", True))
    account_retry_count = max(0, int(scheduler_cfg.get("account_retry_count", 1) or 1))
    account_retry_delay_sec = max(0.0, float(scheduler_cfg.get("account_retry_delay_sec", 15.0) or 15.0))
    account_retry_exit_codes = _normalize_retry_exit_codes(
        scheduler_cfg.get("account_retry_exit_codes", [1, 124])
    ) or [1, 124]
    recent_task_skip_limit = max(
        1,
        int(scheduler_cfg.get("recent_task_skip_limit", max(2, len(accounts) * 2)) or max(2, len(accounts) * 2)),
    )
    recent_task_ids: List[str] = []

    while True:
        ready_accounts_run = 0
        for idx, account_entry in enumerate(accounts, start=1):
            account_name, generated_cfg_path, env = materialize_account_config(
                repo_root,
                index_path,
                base_cfg,
                account_entry,
                episodes_per_turn=episodes_per_turn,
                seed_blocked_task_ids=recent_task_ids,
                base_env=base_env,
            )
            generated_cfg = _load_yaml_dict(generated_cfg_path)
            unready_reason = _account_unready_reason(repo_root, generated_cfg, env)
            if unready_reason:
                message = f"[runner] skipping account={account_name} reason={unready_reason}"
                print(message)
                if not skip_unready_accounts:
                    raise RuntimeError(message)
                if rest_between_turns_sec > 0 and (idx < len(accounts) or loop_forever):
                    print(f"[runner] rest before next account turn: {rest_between_turns_sec:.1f}s")
                    time.sleep(rest_between_turns_sec)
                continue

            attempt_no = 0
            exit_code = 0
            while True:
                attempt_no += 1
                exit_code = run_account_process(
                    repo_root,
                    account_name,
                    generated_cfg_path,
                    env,
                    execute=execute,
                    timeout_sec=per_account_timeout_sec,
                )
                should_retry = (
                    exit_code != 0
                    and attempt_no <= account_retry_count
                    and exit_code in account_retry_exit_codes
                )
                if not should_retry:
                    break
                print(
                    f"[runner] retrying account={account_name} after exit_code={exit_code} "
                    f"(attempt {attempt_no}/{account_retry_count}) in {account_retry_delay_sec:.1f}s"
                )
                time.sleep(account_retry_delay_sec)
            ready_accounts_run += 1
            latest_task_id = _latest_task_id_from_output_dir(
                repo_root,
                str(((generated_cfg.get("run") or {}).get("output_dir", ""))),
            )
            if latest_task_id:
                if latest_task_id in recent_task_ids:
                    recent_task_ids = [tid for tid in recent_task_ids if tid != latest_task_id]
                recent_task_ids.append(latest_task_id)
                recent_task_ids = recent_task_ids[-recent_task_skip_limit:]
                print(
                    f"[runner] recent task skip list updated: {', '.join(recent_task_ids)}"
                )
            if exit_code != 0 and not continue_on_error:
                return exit_code
            if rest_between_turns_sec > 0 and (idx < len(accounts) or loop_forever):
                print(f"[runner] rest before next account turn: {rest_between_turns_sec:.1f}s")
                time.sleep(rest_between_turns_sec)
        if not loop_forever:
            return 0
        if ready_accounts_run <= 0:
            raise RuntimeError("No ready accounts found in scheduler index.")
        if loop_pause_sec > 0:
            print(f"[runner] completed full account cycle; extra loop pause {loop_pause_sec:.1f}s")
            time.sleep(loop_pause_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential multi-account Atlas runner.")
    parser.add_argument(
        "--index",
        default="configs/accounts/index.yaml",
        help="Scheduler index YAML path.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run solver in execute mode for every enabled account.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop forever over enabled accounts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = run_scheduler(Path(args.index), execute=bool(args.execute), loop_forever=bool(args.loop))
    raise SystemExit(exit_code)


__all__ = [
    "build_effective_config",
    "build_runner_env",
    "load_account_index",
    "materialize_account_config",
    "parse_args",
    "run_account_process",
    "run_scheduler",
    "main",
]
