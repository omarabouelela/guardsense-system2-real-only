"""One-command orchestration for the real-only binary Trigger/Verifier pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:  # direct script execution
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.common.runtime_utils import latest_checkpoint

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Stage:
    """Definition for one pipeline stage."""

    key: str
    title: str
    script_path: Path
    config_arg_name: str
    selected: bool


@dataclass(slots=True)
class StageResult:
    """Execution outcome for one stage."""

    key: str
    status: str
    command: str
    message: str = ""


def _load_yaml(path: Path) -> dict:
    """Load YAML payload from path."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_args() -> argparse.Namespace:
    """Build and parse CLI args."""
    parser = argparse.ArgumentParser(description="Run the real-only binary GuardSense pipeline from one command")

    parser.add_argument("--all", action="store_true", help="Run all stages in canonical order")
    parser.add_argument("--extract-pose", action="store_true", help="Run pose extraction")
    parser.add_argument("--prepare-trigger", action="store_true", help="Run Trigger preprocessing")
    parser.add_argument("--train-trigger", action="store_true", help="Run Trigger training")
    parser.add_argument("--eval-trigger", action="store_true", help="Run Trigger evaluation")
    parser.add_argument("--prepare-verifier", action="store_true", help="Run Verifier preprocessing")
    parser.add_argument("--train-verifier", action="store_true", help="Run Verifier training")
    parser.add_argument("--eval-verifier", action="store_true", help="Run Verifier evaluation")

    parser.add_argument("--trigger-only", action="store_true", help="Run Trigger stages (extract -> prepare -> train -> eval)")
    parser.add_argument("--verifier-only", action="store_true", help="Run Verifier stages (prepare -> train -> eval)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--skip-existing", action="store_true", help="Skip stages when conservative output checks pass")

    parser.add_argument("--pose-config", type=Path, default=Path("configs/pose_extraction.yaml"), help="Pose extraction config")
    parser.add_argument(
        "--trigger-data-config",
        type=Path,
        default=Path("configs/trigger_data.yaml"),
        help="Trigger preprocessing config",
    )
    parser.add_argument(
        "--trigger-train-config",
        type=Path,
        default=Path("configs/trigger_train.yaml"),
        help="Trigger training config",
    )
    parser.add_argument(
        "--trigger-eval-config",
        type=Path,
        default=Path("configs/trigger_eval.yaml"),
        help="Trigger evaluation config",
    )
    parser.add_argument(
        "--verifier-data-config",
        type=Path,
        default=Path("configs/verifier_data.yaml"),
        help="Verifier preprocessing config",
    )
    parser.add_argument(
        "--verifier-train-config",
        type=Path,
        default=Path("configs/verifier_train.yaml"),
        help="Verifier training config",
    )
    parser.add_argument(
        "--verifier-eval-config",
        type=Path,
        default=Path("configs/verifier_eval.yaml"),
        help="Verifier evaluation config",
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for subprocess stages")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")

    args = parser.parse_args()

    if args.trigger_only and args.verifier_only:
        parser.error("--trigger-only and --verifier-only cannot be combined")

    return args


def _ensure_config_exists(path: Path) -> None:
    """Fail fast for missing config overrides."""
    if not path.exists():
        raise FileNotFoundError(f"Config path not found: {path}")


def _should_skip_extract_pose(config_path: Path) -> str | None:
    """Skip pose extraction only if a manifest exists or pose txt files already exist."""
    cfg = _load_yaml(config_path)
    manifest_path = Path(str(cfg.get("manifest_path", "artifacts/pose_extraction_manifest.csv")))
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        return f"manifest exists: {manifest_path}"

    for row in cfg.get("datasets", []):
        output_dir = Path(str(row.get("output_dir", "")))
        if output_dir.exists() and any(output_dir.rglob("*.txt")):
            return f"pose txt files found under: {output_dir}"
    return None


def _should_skip_prepare_trigger(config_path: Path) -> str | None:
    """Skip Trigger preprocessing if HDF5 output already exists."""
    cfg = _load_yaml(config_path)
    output_dir = Path(str(cfg.get("output_dir", "artifacts/trigger_data")))
    hdf5_path = output_dir / "trigger_dataset.hdf5"
    if hdf5_path.exists() and hdf5_path.stat().st_size > 0:
        return f"trigger dataset exists: {hdf5_path}"
    return None


def _should_skip_train_trigger(config_path: Path) -> str | None:
    """Skip Trigger training if latest checkpoint already exists."""
    cfg = _load_yaml(config_path)
    output_dir = Path(str(cfg.get("output_dir", "artifacts/runs")))
    checkpoint = latest_checkpoint(base_dir=output_dir, run_prefix="trigger_")
    if checkpoint is not None:
        return f"latest checkpoint exists: {checkpoint}"
    return None


def _should_skip_prepare_verifier(config_path: Path) -> str | None:
    """Skip Verifier preprocessing if processed manifest exists."""
    cfg = _load_yaml(config_path)
    output_dir = Path(str(cfg.get("output_dir", "artifacts/verifier_data")))
    split_manifest = output_dir / "split_manifest.csv"
    if split_manifest.exists() and split_manifest.stat().st_size > 0:
        return f"verifier split manifest exists: {split_manifest}"
    return None


def _should_skip_train_verifier(config_path: Path) -> str | None:
    """Skip Verifier training if latest checkpoint already exists."""
    cfg = _load_yaml(config_path)
    output_dir = Path(str(cfg.get("output_dir", "artifacts/verifier_runs")))
    checkpoint = latest_checkpoint(base_dir=output_dir, run_prefix="verifier_")
    if checkpoint is not None:
        return f"latest checkpoint exists: {checkpoint}"
    return None


def _selection(args: argparse.Namespace) -> dict[str, bool]:
    """Resolve stage selection from explicit flags and convenience modes."""
    selected = {
        "extract_pose": False,
        "prepare_trigger": False,
        "train_trigger": False,
        "eval_trigger": False,
        "prepare_verifier": False,
        "train_verifier": False,
        "eval_verifier": False,
    }

    any_specific = any(
        [
            args.all,
            args.extract_pose,
            args.prepare_trigger,
            args.train_trigger,
            args.eval_trigger,
            args.prepare_verifier,
            args.train_verifier,
            args.eval_verifier,
            args.trigger_only,
            args.verifier_only,
        ]
    )

    if args.all or not any_specific:
        for key in selected:
            selected[key] = True

    if args.trigger_only:
        for key in ["extract_pose", "prepare_trigger", "train_trigger", "eval_trigger"]:
            selected[key] = True

    if args.verifier_only:
        for key in ["prepare_verifier", "train_verifier", "eval_verifier"]:
            selected[key] = True

    for key, flag in [
        ("extract_pose", args.extract_pose),
        ("prepare_trigger", args.prepare_trigger),
        ("train_trigger", args.train_trigger),
        ("eval_trigger", args.eval_trigger),
        ("prepare_verifier", args.prepare_verifier),
        ("train_verifier", args.train_verifier),
        ("eval_verifier", args.eval_verifier),
    ]:
        if flag:
            selected[key] = True

    return selected


def _run_stage(cmd: list[str], dry_run: bool) -> int:
    """Run subprocess stage with inherited stdio."""
    rendered = " ".join(cmd)
    LOGGER.info("Command: %s", rendered)
    if dry_run:
        LOGGER.info("[dry-run] Skipping execution")
        return 0
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _skip_checks(args: argparse.Namespace) -> dict[str, Callable[[], str | None]]:
    """Build per-stage skip checks."""
    return {
        "extract_pose": lambda: _should_skip_extract_pose(args.pose_config),
        "prepare_trigger": lambda: _should_skip_prepare_trigger(args.trigger_data_config),
        "train_trigger": lambda: _should_skip_train_trigger(args.trigger_train_config),
        "prepare_verifier": lambda: _should_skip_prepare_verifier(args.verifier_data_config),
        "train_verifier": lambda: _should_skip_train_verifier(args.verifier_train_config),
    }


def main() -> int:
    """Entrypoint for one-command binary pipeline orchestration."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    for config_path in [
        args.pose_config,
        args.trigger_data_config,
        args.trigger_train_config,
        args.trigger_eval_config,
        args.verifier_data_config,
        args.verifier_train_config,
        args.verifier_eval_config,
    ]:
        _ensure_config_exists(config_path)

    selected = _selection(args)
    stages = [
        Stage("extract_pose", "Extract Trigger Pose", Path("src/scripts/extract_pose_from_videos.py"), "pose_config", selected["extract_pose"]),
        Stage("prepare_trigger", "Prepare Trigger Dataset", Path("src/scripts/prepare_trigger_data.py"), "trigger_data_config", selected["prepare_trigger"]),
        Stage("train_trigger", "Train Trigger", Path("src/scripts/train_trigger.py"), "trigger_train_config", selected["train_trigger"]),
        Stage("eval_trigger", "Evaluate Trigger", Path("src/scripts/eval_trigger.py"), "trigger_eval_config", selected["eval_trigger"]),
        Stage(
            "prepare_verifier",
            "Prepare Verifier Dataset",
            Path("src/scripts/prepare_verifier_data.py"),
            "verifier_data_config",
            selected["prepare_verifier"],
        ),
        Stage("train_verifier", "Train Verifier", Path("src/scripts/train_verifier.py"), "verifier_train_config", selected["train_verifier"]),
        Stage("eval_verifier", "Evaluate Verifier", Path("src/scripts/eval_verifier.py"), "verifier_eval_config", selected["eval_verifier"]),
    ]

    skip_checks = _skip_checks(args)
    results: list[StageResult] = []

    for idx, stage in enumerate(stages, start=1):
        if not stage.selected:
            continue

        LOGGER.info("=" * 72)
        LOGGER.info("[%d] %s (%s)", idx, stage.title, stage.key)
        LOGGER.info("=" * 72)

        if args.skip_existing and stage.key in skip_checks:
            skip_reason = skip_checks[stage.key]()
            if skip_reason:
                LOGGER.info("Skipping stage due to --skip-existing: %s", skip_reason)
                results.append(StageResult(stage.key, "skipped", "", skip_reason))
                continue

        config_path = getattr(args, stage.config_arg_name)
        cmd = [args.python, str(stage.script_path), "--config", str(config_path)]
        return_code = _run_stage(cmd, dry_run=args.dry_run)
        rendered_cmd = " ".join(cmd)

        if return_code != 0:
            message = f"exit_code={return_code}"
            LOGGER.error("Stage failed: %s | %s", stage.key, message)
            results.append(StageResult(stage.key, "failed", rendered_cmd, message))
            _print_summary(results)
            return return_code

        status = "dry-run" if args.dry_run else "ok"
        results.append(StageResult(stage.key, status, rendered_cmd, ""))
        LOGGER.info("Stage complete: %s", stage.key)

    _print_summary(results)
    return 0


def _print_summary(results: list[StageResult]) -> None:
    """Print a simple end-of-run stage summary."""
    LOGGER.info("\nFinal Summary")
    if not results:
        LOGGER.info("No stages were selected.")
        return
    for row in results:
        suffix = f" ({row.message})" if row.message else ""
        LOGGER.info("- %s: %s%s", row.key, row.status, suffix)


if __name__ == "__main__":
    raise SystemExit(main())
