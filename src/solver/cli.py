"""CLI entrypoint extracted from the legacy solver while preserving compatibility."""

from pathlib import Path

from src.infra.solver_config import _SCRIPT_BUILD, load_config
from src.solver import legacy_impl as _legacy
from src.solver.orchestrator import run

parse_args = _legacy.parse_args
_apply_cli_overrides = _legacy._apply_cli_overrides


def main() -> None:
    print(f"[build] atlas_web_auto_solver {_SCRIPT_BUILD}")
    args = parse_args()
    cfg = load_config(Path(args.config))
    _apply_cli_overrides(cfg, args)
    run(cfg, execute=bool(args.execute))


__all__ = ["parse_args", "_apply_cli_overrides", "main"]

