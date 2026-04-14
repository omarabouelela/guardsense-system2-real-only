"""CLI to prepare GuardSense Verifier video dataset."""

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

from src.data.video_preprocess import (
    RejectionRecord,
    VideoIndexRecord,
    VideoProcessConfig,
    VideoValidationConfig,
    index_video_sources,
    preprocess_clip,
    save_index_csv,
    save_json,
    save_rejections,
    save_split_manifest,
    stratified_split,
    summarize_dataset,
    validate_video,
)
from src.common.runtime_utils import resolve_raw_data_path

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Prepare GuardSense Verifier RGB dataset")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = build_parser().parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    out_dir = Path(cfg["output_dir"])
    clips_dir = out_dir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    process_cfg = VideoProcessConfig(**cfg.get("processing", {}))
    validate_cfg = VideoValidationConfig(**cfg.get("validation", {}))
    split_cfg = cfg.get("split", {})

    indexed: list[VideoIndexRecord] = []
    for source in cfg["sources"]:
        if not bool(source.get("enabled", True)):
            LOGGER.info("Skipping disabled source=%s", source.get("name", "unknown"))
            continue
        indexed.extend(
            index_video_sources(
                source_dirs=[resolve_raw_data_path(p) for p in source["paths"]],
                source_dataset=source["name"],
                label=source.get("label"),
                synthetic_or_real=source.get("synthetic_or_real", "unknown"),
                split=source.get("split", "unspecified"),
                label_rules=source.get("label_rules"),
            )
        )

    valid_rows: list[VideoIndexRecord] = []
    rejections: list[RejectionRecord] = []

    for row in indexed:
        ok, reason = validate_video(row, validate_cfg)
        if not ok:
            rejections.append(RejectionRecord(clip_id=row.clip_id, path=row.original_path, reason=reason or "invalid"))
            continue

        target = clips_dir / f"{row.clip_id}.mp4"
        try:
            processed = preprocess_clip(Path(row.original_path), target, process_cfg)
            row.processed_path = str(target)
            row.codec = str(processed.get("codec") or "")
            row.fps = float(processed.get("fps") or 0.0)
            row.duration_seconds = float(processed.get("duration_seconds") or 0.0)
            row.width = int(processed.get("width") or 0)
            row.height = int(processed.get("height") or 0)
            valid_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rejections.append(RejectionRecord(clip_id=row.clip_id, path=row.original_path, reason=f"preprocess_failed:{exc}"))

    if not valid_rows:
        raise RuntimeError("No valid clips after preprocessing")

    labels = [r.label for r in valid_rows]
    groups = [r.source_dataset for r in valid_rows]
    train_idx, val_idx, test_idx = stratified_split(
        labels,
        groups,
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        seed=int(split_cfg.get("seed", 42)),
        source_aware=bool(split_cfg.get("source_aware", False)),
    )
    split_by_index: dict[int, str] = {int(i): "train" for i in train_idx.tolist()}
    split_by_index.update({int(i): "val" for i in val_idx.tolist()})
    split_by_index.update({int(i): "test" for i in test_idx.tolist()})
    for idx, row in enumerate(valid_rows):
        row.split = split_by_index.get(idx, "unused")

    save_index_csv(valid_rows, out_dir / "metadata.csv")
    save_split_manifest(out_dir / "split_manifest.csv", valid_rows)
    save_rejections(rejections, out_dir / "rejections.csv")

    summary = summarize_dataset(valid_rows, rejections)
    summary_payload = asdict(summary)
    summary_payload["processed_clips"] = len(valid_rows)
    summary_payload["rejected_clips"] = len(rejections)
    summary_payload["total_indexed_clips"] = len(indexed)
    save_json(summary_payload, out_dir / "dataset_report.json")
    label_map = {int(k): str(v) for k, v in (cfg.get("label_map") or {"0": "normal", "1": "fight"}).items()}
    save_json({str(k): v for k, v in sorted(label_map.items())}, out_dir / "label_map.json")

    LOGGER.info("Prepared verifier dataset under %s", out_dir)
    LOGGER.info("Summary: %s", json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
