"""CLI for extracting YOLOv8 pose txt sequences from real videos."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import yaml

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:  # direct script execution
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.common.logging_utils import configure_logging
from src.data.pose_extraction import (
    PoseExtractorConfig,
    build_dataset_config,
    load_yolo_model,
    process_dataset,
    write_manifest_csv,
)

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Extract Trigger-compatible pose txt files from real videos")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    return parser


def main() -> None:
    """Run pose extraction for configured datasets."""
    args = build_arg_parser().parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    configure_logging(level=str(payload.get("log_level", "INFO")), log_file=Path(payload["log_file"]) if payload.get("log_file") else None)

    extractor_cfg = PoseExtractorConfig(**payload.get("extractor", {}))
    dataset_cfgs = [build_dataset_config(row) for row in payload.get("datasets", [])]
    if not dataset_cfgs:
        raise ValueError("No datasets configured under 'datasets'")

    model = load_yolo_model(extractor_cfg.model)

    all_records = []
    for dataset_cfg in dataset_cfgs:
        records = process_dataset(model=model, dataset=dataset_cfg, config=extractor_cfg)
        all_records.extend(records)
        success_count = sum(1 for item in records if item.status == "success")
        LOGGER.info("dataset=%s success=%d total=%d", dataset_cfg.name, success_count, len(records))

    manifest_path = Path(payload.get("manifest_path", "artifacts/pose_extraction_manifest.csv"))
    write_manifest_csv(manifest_path, all_records)

    summary = {
        "manifest_path": str(manifest_path),
        "total": len(all_records),
        "success": sum(1 for item in all_records if item.status == "success"),
        "failed": sum(1 for item in all_records if item.status == "failed"),
        "skipped_existing": sum(1 for item in all_records if item.status == "skipped_existing"),
    }
    LOGGER.info("summary=%s", json.dumps(summary, indent=2))

    summary_path = manifest_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    details_path = manifest_path.with_suffix(".json")
    details_path.write_text(json.dumps([asdict(item) for item in all_records], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
