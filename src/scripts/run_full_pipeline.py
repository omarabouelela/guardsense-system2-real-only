"""End-to-end orchestration for binary real-only Trigger+Verifier(+optional fusion)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:  # direct script execution
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StageResult:
    name: str
    status: str
    command: str
    message: str = ""


def _run(cmd: list[str], dry_run: bool) -> tuple[int, str]:
    rendered = " ".join(cmd)
    if dry_run:
        LOGGER.info("[dry-run] %s", rendered)
        return 0, rendered
    proc = subprocess.run(cmd, check=False, text=True)
    return proc.returncode, rendered


def _latest_run_dir(base_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(base_dir.glob(pattern))
    return candidates[-1] if candidates else None


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if report.get("focus_class_recall", 1.0) < 0.5:
        warnings.append("low focus-class recall")
    if report.get("label_0_false_positives", 0) > 0:
        warnings.append("non-zero label 0 false positives")
    if report.get("cross_class_confusions", 0) > 0:
        warnings.append("cross-class confusion present")
    return warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full GuardSense pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline YAML config")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pipeline run output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--resume", action="store_true", help="Skip stages with marker files")
    parser.add_argument("--skip-stages", nargs="*", default=[], help="Stages to skip")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    stage_defs: list[tuple[str, list[str]]] = [
        ("extract_pose", ["python", "-m", "src.scripts.extract_pose_from_videos", "--config", cfg["pose_extraction_config"]]),
        ("prepare_trigger", ["python", "-m", "src.scripts.prepare_trigger_data", "--config", cfg["trigger_data_config"]]),
        ("prepare_verifier", ["python", "-m", "src.scripts.prepare_verifier_data", "--config", cfg["verifier_data_config"]]),
        ("train_trigger", ["python", "-m", "src.scripts.train_trigger", "--config", cfg["trigger_train_config"]]),
        ("eval_trigger", ["python", "-m", "src.scripts.eval_trigger", "--config", cfg["trigger_eval_config"]]),
        ("train_verifier", ["python", "-m", "src.scripts.train_verifier", "--config", cfg["verifier_train_config"]]),
        ("eval_verifier", ["python", "-m", "src.scripts.eval_verifier", "--config", cfg["verifier_eval_config"]]),
    ]

    if not cfg.get("run_pose_extraction", True):
        stage_defs = [item for item in stage_defs if item[0] != "extract_pose"]

    if cfg.get("run_dual_inference", True):
        stage_defs.append(
            (
                "dual_inference",
                [
                    "python",
                    "-m",
                    "src.scripts.run_dual_inference",
                    "--config",
                    cfg["runtime_config"],
                    "--event-json",
                    cfg["runtime_events_json"],
                    "--output-dir",
                    cfg["runtime_output_dir"],
                ],
            )
        )

    results: list[StageResult] = []
    for stage_name, cmd in stage_defs:
        if stage_name in args.skip_stages:
            results.append(StageResult(stage_name, "skipped", " ".join(cmd), "stage skipped by user"))
            continue

        marker = out_dir / f"{stage_name}.done"
        if args.resume and marker.exists():
            results.append(StageResult(stage_name, "skipped", " ".join(cmd), "resume marker present"))
            continue

        code, rendered = _run(cmd, dry_run=args.dry_run)
        if code != 0:
            results.append(StageResult(stage_name, "failed", rendered, f"exit_code={code}"))
            break
        marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        results.append(StageResult(stage_name, "ok", rendered))

    trigger_runs = Path(cfg.get("trigger_run_dir", "artifacts/runs"))
    verifier_runs = Path(cfg.get("verifier_run_dir", "artifacts/verifier_runs"))
    trigger_latest = _latest_run_dir(trigger_runs, "trigger_*")
    verifier_latest = _latest_run_dir(verifier_runs, "verifier_*")

    trigger_metrics = _read_json_if_exists(trigger_latest / "metrics.json") if trigger_latest else {}
    verifier_metrics = _read_json_if_exists(verifier_latest / "metrics.json") if verifier_latest else {}

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stages": [asdict(result) for result in results],
        "trigger_macro_f1": trigger_metrics.get("macro_f1"),
        "trigger_per_class_f1": {k: v.get("f1") for k, v in (trigger_metrics.get("per_class") or {}).items()},
        "verifier_macro_f1": verifier_metrics.get("macro_f1"),
        "verifier_per_class_f1": {k: v.get("f1") for k, v in (verifier_metrics.get("per_class") or {}).items()},
        "trigger_confusion_matrix_path": str((trigger_latest / "confusion_matrix.png")) if trigger_latest else None,
        "verifier_confusion_matrix_path": str((verifier_latest / "confusion_matrix.png")) if verifier_latest else None,
        "warnings": sorted(set(_warnings(trigger_metrics) + _warnings(verifier_metrics))),
    }

    report_json = out_dir / "final_experiment_report.json"
    report_csv = out_dir / "final_experiment_report.csv"
    report_txt = out_dir / "final_experiment_report.txt"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        for key, value in report.items():
            writer.writerow([key, json.dumps(value) if isinstance(value, (dict, list)) else value])

    report_txt.write_text(
        "\n".join(
            [
                f"Run timestamp: {report['timestamp']}",
                f"Trigger macro F1: {report.get('trigger_macro_f1')}",
                f"Verifier macro F1: {report.get('verifier_macro_f1')}",
                f"Warnings: {', '.join(report.get('warnings', [])) or 'none'}",
                f"JSON report: {report_json}",
                f"CSV report: {report_csv}",
            ]
        ),
        encoding="utf-8",
    )

    LOGGER.info("Pipeline finished. Report: %s", report_json)


if __name__ == "__main__":
    main()
