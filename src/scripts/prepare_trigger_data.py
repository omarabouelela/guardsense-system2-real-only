"""CLI to prepare Trigger dataset from pose sources."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:  # direct script execution
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.pose_preprocess import (
    DatasetIndexRecord,
    NormalizationConfig,
    RejectionRecord,
    SequenceAssemblyConfig,
    apply_normalization,
    export_dataset_hdf5,
    generate_dataset_report,
    index_pose_sources,
    iter_pose_sequences,
    missing_keypoint_ratio,
    one_hot_encode,
    plot_pose_sequence,
    save_index_csv,
    save_json,
    save_rejection_log,
    temporal_windows,
)
from src.common.runtime_utils import resolve_raw_data_path
from src.data.trigger_dataset import SplitConfig, save_split_manifest, stratified_split_indices, subset_arrays

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Prepare GuardSense Trigger pose dataset")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = build_arg_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    normalization = NormalizationConfig(**config.get("normalization", {}))
    temporal = SequenceAssemblyConfig(**config.get("temporal", {}))
    split_cfg = SplitConfig(**config.get("split", {}))
    txt_layout = str(config.get("txt_layout", "auto"))

    index_rows: list[DatasetIndexRecord] = []
    for source in config["sources"]:
        index_rows.extend(
            index_pose_sources(
                source_dirs=[resolve_raw_data_path(p) for p in source["paths"]],
                source_dataset=source["name"],
                label=int(source["label"]),
                synthetic_or_real=source.get("synthetic_or_real", "unknown"),
                split=source.get("split", "unspecified"),
            )
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    accepted_rows: list[DatasetIndexRecord] = []
    rejections: list[RejectionRecord] = []
    missing_ratios: list[float] = []
    sequence_lengths: list[int] = []

    for row, seq, bboxes in iter_pose_sequences(index_rows, txt_layout=txt_layout):
        try:
            normalized = apply_normalization(seq, normalization, bboxes=bboxes)
            windows = temporal_windows(normalized, temporal)
            if windows.shape[0] == 0:
                rejections.append(RejectionRecord(sample_id=row.sample_id, path=row.original_path, reason="too_short_dropped"))
                continue
            for w_idx, window in enumerate(windows):
                features.append(window)
                labels.append(row.label)
                sample_ids.append(f"{row.sample_id}_w{w_idx}")
                groups.append(row.source_dataset)
                accepted_rows.append(row)
                missing_ratios.append(missing_keypoint_ratio(window))
                sequence_lengths.append(int(window.shape[0]))
        except Exception as exc:  # noqa: BLE001
            rejections.append(RejectionRecord(sample_id=row.sample_id, path=row.original_path, reason=str(exc)))

    if not features:
        raise RuntimeError("No valid windows generated. Check source paths and parser settings")

    x = np.stack(features, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    train_idx, val_idx, test_idx = stratified_split_indices(y.tolist(), groups, split_cfg)
    x_train, y_train = subset_arrays(x, y, train_idx)
    x_val, y_val = subset_arrays(x, y, val_idx)
    x_test, y_test = subset_arrays(x, y, test_idx)

    label_map = {int(k): str(v) for k, v in (config.get("label_map") or {"0": "normal", "1": "fight"}).items()}
    num_classes = len(label_map)

    np.save(output_dir / "X.npy", x)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "y_onehot.npy", one_hot_encode(y, num_classes=num_classes))

    export_dataset_hdf5(output_dir / "trigger_dataset.hdf5", x_train, y_train, x_val, y_val, x_test, y_test)
    save_index_csv(accepted_rows, output_dir / "metadata.csv")
    save_split_manifest(output_dir / "split_manifest.csv", sample_ids, y, train_idx, val_idx, test_idx)
    save_rejection_log(rejections, output_dir / "rejections.csv")

    report = generate_dataset_report(accepted_rows, sequence_lengths, missing_ratios, rejections)
    save_json(asdict(report), output_dir / "dataset_report.json")
    save_json({str(k): v for k, v in sorted(label_map.items())}, output_dir / "label_map.json")

    debug_count = min(3, len(x))
    for i in range(debug_count):
        plot_pose_sequence(x[i], title=f"sample={sample_ids[i]} label={y[i]}", output_path=output_dir / f"debug_pose_{i}.png")

    LOGGER.info("Prepared dataset in %s with %d windows", output_dir, len(x))
    LOGGER.info("Report: %s", json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
