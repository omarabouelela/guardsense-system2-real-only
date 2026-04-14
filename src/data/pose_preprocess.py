"""Pose preprocessing utilities for GuardSense Trigger model."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, Iterator, Literal, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

LABEL_MAP: dict[int, str] = {0: "normal", 1: "fight"}
TXT_LAYOUT_VALUES = {"auto", "sequence_per_file", "frame_per_file"}


@dataclass(slots=True)
class DatasetIndexRecord:
    """Metadata row for indexed trigger input sample."""

    sample_id: str
    source_dataset: str
    original_path: str
    label: int
    split: str = "unspecified"
    synthetic_or_real: str = "unknown"
    frame_count: int | None = None
    track_id: str | None = None
    camera_name: str | None = None
    frigate_event_id: str | None = None
    source_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RejectionRecord:
    """Why a sample was rejected from preprocessing."""

    sample_id: str
    path: str
    reason: str


@dataclass(slots=True)
class SequenceAssemblyConfig:
    """Temporal assembly configuration."""

    window_size: int = 32
    overlap: float = 0.5
    short_policy: Literal["pad", "drop", "truncate"] = "pad"

    @property
    def stride(self) -> int:
        stride = int(round(self.window_size * (1.0 - self.overlap)))
        return max(stride, 1)


@dataclass(slots=True)
class NormalizationConfig:
    """Coordinate normalization configuration."""

    mode: Literal["none", "frame", "bbox"] = "frame"
    frame_width: int = 1
    frame_height: int = 1
    bbox_eps: float = 1e-6


@dataclass(slots=True)
class DatasetReport:
    """Aggregated reporting stats for preprocessing."""

    samples_per_class: dict[str, int]
    samples_per_source: dict[str, int]
    synthetic_vs_real: dict[str, int]
    frigate_vs_nonfrigate: dict[str, int]
    missing_keypoints_ratio_mean: float
    sequence_length_mean: float
    sequence_length_median: float
    rejected_count: int


def index_pose_sources(
    source_dirs: Sequence[Path],
    source_dataset: str,
    label: int,
    synthetic_or_real: str,
    split: str = "unspecified",
) -> list[DatasetIndexRecord]:
    """Recursively index supported pose sources (.txt, .npy, .hdf5/.h5)."""
    records: list[DatasetIndexRecord] = []
    supported_suffixes = {".txt", ".npy", ".hdf5", ".h5"}
    for source_dir in source_dirs:
        for path in source_dir.rglob("*"):
            if path.suffix.lower() not in supported_suffixes or not path.is_file():
                continue
            record = DatasetIndexRecord(
                sample_id=path.stem,
                source_dataset=source_dataset,
                original_path=str(path),
                label=label,
                split=split,
                synthetic_or_real=synthetic_or_real,
                source_type=path.suffix.lower().lstrip("."),
            )
            frame_count = infer_frame_count(path)
            record.frame_count = frame_count
            records.append(record)
    LOGGER.info("Indexed %d pose files from %d source directories", len(records), len(source_dirs))
    return records


def infer_frame_count(path: Path) -> int | None:
    """Infer frame count quickly from supported file types."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            return int(arr.shape[0]) if arr.ndim >= 1 else None
        if suffix in {".h5", ".hdf5"}:
            with h5py.File(path, "r") as h5f:
                for key in ("X", "poses", "sequences"):
                    if key in h5f:
                        return int(h5f[key].shape[0])
                return None
        if suffix == ".txt":
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for _ in handle)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed inferring frame_count for %s: %s", path, exc)
    return None


def parse_yolov8_pose_row(row: str, num_keypoints: int = 17) -> tuple[np.ndarray, dict[str, float]]:
    """Parse one YOLOv8 pose row and return keypoints(K,3) + bbox metadata."""
    values = row.strip().split()
    min_len = 5 + (num_keypoints * 3)
    if len(values) < min_len:
        raise ValueError(f"Malformed row: expected >= {min_len} values, got {len(values)}")

    try:
        floats = [float(v) for v in values]
    except ValueError as exc:
        raise ValueError("Malformed row: non-numeric token") from exc

    _, x_center, y_center, width, height, *keypoint_values = floats
    keypoint_array = np.array(keypoint_values[: num_keypoints * 3], dtype=np.float32).reshape(num_keypoints, 3)
    return keypoint_array, {
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }


def parse_yolov8_pose_file(path: Path, num_keypoints: int = 17) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Parse .txt pose file with one pose row per frame."""
    poses: list[np.ndarray] = []
    bboxes: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, row in enumerate(handle, start=1):
            if not row.strip():
                continue
            pose, bbox = parse_yolov8_pose_row(row, num_keypoints=num_keypoints)
            poses.append(pose)
            bboxes.append(bbox)
    if not poses:
        raise ValueError(f"No pose rows found in {path}")
    return np.stack(poses, axis=0), bboxes


def _file_stem_prefix(path: Path) -> str:
    """Derive grouping prefix by removing common frame suffixes."""
    stem = path.stem
    match = re.match(r"^(.*?)(?:[_-]?(?:frame|img|image))?[_-]?\d+$", stem, flags=re.IGNORECASE)
    if match and match.group(1):
        return match.group(1)
    return stem


def _frame_index(path: Path) -> int | None:
    """Extract numeric frame index from file stem."""
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(match.group(1)) if match else None


def _frame_sort_key(path: Path) -> tuple[int, str]:
    """Sort by extracted index, then filename for deterministic ordering."""
    frame_idx = _frame_index(path)
    if frame_idx is not None:
        return frame_idx, path.name
    return 10**9, path.name


def _detect_txt_layout(group_paths: Sequence[Path], frame_counts: Sequence[int | None]) -> str:
    """Infer txt layout for a grouped set of files."""
    valid_counts = [value for value in frame_counts if value is not None]
    mostly_single_line = bool(valid_counts) and (sum(1 for v in valid_counts if v <= 1) / len(valid_counts)) >= 0.8
    if len(group_paths) > 1 and mostly_single_line:
        return "frame_per_file"
    return "sequence_per_file"


def _group_txt_records(
    records: Sequence[DatasetIndexRecord],
    txt_layout: str,
) -> list[tuple[DatasetIndexRecord, list[Path]]]:
    """Group txt records as sequence-level units based on configured layout."""
    grouped: dict[tuple[str, str, int, str], list[DatasetIndexRecord]] = {}
    for record in records:
        source_path = Path(record.original_path)
        group_key = (str(source_path.parent), record.source_dataset, record.label, _file_stem_prefix(source_path))
        grouped.setdefault(group_key, []).append(record)

    assembled: list[tuple[DatasetIndexRecord, list[Path]]] = []
    for (parent, source_dataset, label, stem_prefix), rows in sorted(grouped.items()):
        sorted_rows = sorted(rows, key=lambda row: _frame_sort_key(Path(row.original_path)))
        group_paths = [Path(row.original_path) for row in sorted_rows]
        frame_counts = [row.frame_count for row in sorted_rows]
        effective_layout = txt_layout if txt_layout != "auto" else _detect_txt_layout(group_paths, frame_counts)

        if effective_layout == "sequence_per_file":
            for row in sorted_rows:
                assembled.append((row, [Path(row.original_path)]))
            continue

        if effective_layout != "frame_per_file":
            raise ValueError(f"Unsupported txt_layout: {effective_layout}")

        base = sorted_rows[0]
        frame_indices = [_frame_index(path) for path in group_paths]
        known_indices = [index for index in frame_indices if index is not None]
        missing_indices: list[int] = []
        if known_indices:
            expected = set(range(min(known_indices), max(known_indices) + 1))
            missing_indices = sorted(expected - set(known_indices))
            if missing_indices:
                LOGGER.warning(
                    "Detected %d missing frame files for group=%s in %s. Missing indices sample=%s",
                    len(missing_indices),
                    stem_prefix,
                    parent,
                    missing_indices[:10],
                )

        merged_metadata = dict(base.metadata)
        merged_metadata["txt_layout"] = effective_layout
        merged_metadata["sequence_prefix"] = stem_prefix
        merged_metadata["source_parent"] = parent
        merged_metadata["frame_file_count"] = len(group_paths)
        merged_metadata["frame_files"] = [str(path) for path in group_paths]
        if missing_indices:
            merged_metadata["missing_frame_indices"] = missing_indices

        merged_record = DatasetIndexRecord(
            sample_id=f"{source_dataset}:{stem_prefix}",
            source_dataset=base.source_dataset,
            original_path=str(group_paths[0]),
            label=label,
            split=base.split,
            synthetic_or_real=base.synthetic_or_real,
            frame_count=len(group_paths),
            track_id=base.track_id,
            camera_name=base.camera_name,
            frigate_event_id=base.frigate_event_id,
            source_type=base.source_type,
            metadata=merged_metadata,
        )
        assembled.append((merged_record, group_paths))
    return assembled


def _load_txt_sequence(paths: Sequence[Path], num_keypoints: int = 17) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Load one sequence from one-or-many txt files."""
    if len(paths) == 1:
        return parse_yolov8_pose_file(paths[0], num_keypoints=num_keypoints)

    poses: list[np.ndarray] = []
    bboxes: list[dict[str, float]] = []
    for path in paths:
        frame_seq, frame_bboxes = parse_yolov8_pose_file(path, num_keypoints=num_keypoints)
        if frame_seq.shape[0] > 1:
            LOGGER.warning("Frame txt contains %d rows in %s; using first row only", frame_seq.shape[0], path)
        poses.append(frame_seq[0])
        bboxes.append(frame_bboxes[0])
    return np.stack(poses, axis=0), bboxes


def apply_normalization(
    sequence: np.ndarray,
    config: NormalizationConfig,
    bboxes: Sequence[dict[str, float]] | None = None,
) -> np.ndarray:
    """Normalize (T,K,3) coordinates while preserving visibility channel."""
    out = sequence.astype(np.float32, copy=True)
    if config.mode == "none":
        return out

    xy = out[..., :2]
    if config.mode == "frame":
        xy[..., 0] = xy[..., 0] / max(config.frame_width, 1)
        xy[..., 1] = xy[..., 1] / max(config.frame_height, 1)
    elif config.mode == "bbox":
        if bboxes is None:
            raise ValueError("bbox normalization requested but bboxes were not provided")
        for t in range(min(len(bboxes), out.shape[0])):
            bbox = bboxes[t]
            x0 = bbox["x_center"] - (bbox["width"] / 2)
            y0 = bbox["y_center"] - (bbox["height"] / 2)
            w = max(bbox["width"], config.bbox_eps)
            h = max(bbox["height"], config.bbox_eps)
            xy[t, :, 0] = (xy[t, :, 0] - x0) / w
            xy[t, :, 1] = (xy[t, :, 1] - y0) / h
    else:
        raise ValueError(f"Unknown normalization mode: {config.mode}")

    out[..., :2] = xy
    return out


def temporal_windows(sequence: np.ndarray, config: SequenceAssemblyConfig) -> np.ndarray:
    """Convert (T,K,C) into (N,window,K,C) with overlap and short handling."""
    t = sequence.shape[0]
    w = config.window_size

    if t < w:
        if config.short_policy == "drop":
            return np.empty((0, w, sequence.shape[1], sequence.shape[2]), dtype=sequence.dtype)
        if config.short_policy == "truncate":
            return sequence[:w][None, ...] if t > 0 else np.empty((0, w, sequence.shape[1], sequence.shape[2]), dtype=sequence.dtype)
        padded = np.zeros((w, sequence.shape[1], sequence.shape[2]), dtype=sequence.dtype)
        padded[:t] = sequence
        if t > 0:
            padded[t:] = sequence[t - 1]
        return padded[None, ...]

    stride = config.stride
    windows = [sequence[start : start + w] for start in range(0, t - w + 1, stride)]
    return np.stack(windows, axis=0)


def load_pose_sequence(record: DatasetIndexRecord, num_keypoints: int = 17) -> tuple[np.ndarray, list[dict[str, float]] | None]:
    """Load sequence as (T,K,3)."""
    path = Path(record.original_path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        seq, bboxes = parse_yolov8_pose_file(path, num_keypoints=num_keypoints)
        return seq, bboxes

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[1] != num_keypoints or arr.shape[2] < 3:
            raise ValueError(f"Invalid npy shape {arr.shape} in {path}")
        return arr[..., :3].astype(np.float32), None

    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as h5f:
            for key in ("pose", "poses", "sequence", "sequences", "X"):
                if key in h5f:
                    arr = h5f[key][()]
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim != 3:
                        raise ValueError(f"Invalid hdf5 shape {arr.shape} for key {key}")
                    return arr[..., :3].astype(np.float32), None
        raise ValueError(f"No known pose dataset keys in {path}")

    raise ValueError(f"Unsupported suffix: {suffix}")


def missing_keypoint_ratio(sequence: np.ndarray) -> float:
    """Compute fraction of keypoints not visible (visibility <= 0)."""
    visibility = sequence[..., 2]
    return float(np.mean(visibility <= 0))


def generate_dataset_report(
    index_records: Sequence[DatasetIndexRecord],
    sequence_lengths: Sequence[int],
    missing_ratios: Sequence[float],
    rejections: Sequence[RejectionRecord],
) -> DatasetReport:
    """Aggregate preprocessing report."""
    samples_per_class: dict[str, int] = {}
    samples_per_source: dict[str, int] = {}
    synthetic_vs_real: dict[str, int] = {}
    frigate_vs_nonfrigate = {"frigate": 0, "non_frigate": 0}

    for row in index_records:
        class_name = LABEL_MAP.get(row.label, f"label_{row.label}")
        samples_per_class[class_name] = samples_per_class.get(class_name, 0) + 1
        samples_per_source[row.source_dataset] = samples_per_source.get(row.source_dataset, 0) + 1
        synthetic_vs_real[row.synthetic_or_real] = synthetic_vs_real.get(row.synthetic_or_real, 0) + 1
        if row.frigate_event_id:
            frigate_vs_nonfrigate["frigate"] += 1
        else:
            frigate_vs_nonfrigate["non_frigate"] += 1

    return DatasetReport(
        samples_per_class=samples_per_class,
        samples_per_source=samples_per_source,
        synthetic_vs_real=synthetic_vs_real,
        frigate_vs_nonfrigate=frigate_vs_nonfrigate,
        missing_keypoints_ratio_mean=float(mean(missing_ratios)) if missing_ratios else 0.0,
        sequence_length_mean=float(mean(sequence_lengths)) if sequence_lengths else 0.0,
        sequence_length_median=float(median(sequence_lengths)) if sequence_lengths else 0.0,
        rejected_count=len(rejections),
    )


def save_rejection_log(rejections: Sequence[RejectionRecord], output_path: Path) -> None:
    """Write malformed/rejected samples to csv."""
    if not rejections:
        output_path.write_text("sample_id,path,reason\n", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "path", "reason"])
        writer.writeheader()
        for row in rejections:
            writer.writerow(asdict(row))


def save_index_csv(index_records: Sequence[DatasetIndexRecord], output_path: Path) -> None:
    """Save metadata index csv."""
    frame = pd.DataFrame([asdict(item) for item in index_records])
    frame.to_csv(output_path, index=False)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save JSON with pretty formatting."""
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def export_dataset_hdf5(
    output_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Export train/val/test arrays into a master HDF5 file."""
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("X_train", data=x_train)
        h5f.create_dataset("y_train", data=y_train)
        h5f.create_dataset("X_val", data=x_val)
        h5f.create_dataset("y_val", data=y_val)
        h5f.create_dataset("X_test", data=x_test)
        h5f.create_dataset("y_test", data=y_test)


def one_hot_encode(labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """One-hot encode integer labels."""
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return out


def plot_pose_sequence(
    sequence: np.ndarray,
    title: str,
    output_path: Path,
    max_frames: int = 32,
) -> None:
    """Plot static pose traces with visibility markers for debugging."""
    t = min(sequence.shape[0], max_frames)
    fig, ax = plt.subplots(figsize=(8, 5))
    for frame_id in range(t):
        joints = sequence[frame_id]
        visible = joints[:, 2] > 1
        occluded = joints[:, 2] <= 1
        ax.scatter(joints[visible, 0], joints[visible, 1], c="green", s=12, alpha=0.6)
        ax.scatter(joints[occluded, 0], joints[occluded, 1], c="orange", s=12, alpha=0.4)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def iter_pose_sequences(
    records: Sequence[DatasetIndexRecord],
    txt_layout: Literal["auto", "sequence_per_file", "frame_per_file"] = "sequence_per_file",
) -> Iterator[tuple[DatasetIndexRecord, np.ndarray, list[dict[str, float]] | None]]:
    """Load and yield pose sequences, logging but skipping failures."""
    yield from iter_pose_sequences_with_layout(records, txt_layout=txt_layout)


def iter_pose_sequences_with_layout(
    records: Sequence[DatasetIndexRecord],
    txt_layout: Literal["auto", "sequence_per_file", "frame_per_file"] = "auto",
) -> Iterator[tuple[DatasetIndexRecord, np.ndarray, list[dict[str, float]] | None]]:
    """Load and yield pose sequences with configurable txt assembly layout."""
    if txt_layout not in TXT_LAYOUT_VALUES:
        raise ValueError(f"txt_layout must be one of {sorted(TXT_LAYOUT_VALUES)}, got {txt_layout!r}")

    txt_records = [record for record in records if Path(record.original_path).suffix.lower() == ".txt"]
    non_txt_records = [record for record in records if Path(record.original_path).suffix.lower() != ".txt"]
    txt_units = _group_txt_records(txt_records, txt_layout=txt_layout)

    for record, text_paths in txt_units:
        try:
            seq, bboxes = _load_txt_sequence(text_paths)
            yield record, seq, bboxes
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping sample %s due to parsing error: %s", record.sample_id, exc)

    for record in non_txt_records:
        try:
            seq, bboxes = load_pose_sequence(record)
            yield record, seq, bboxes
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping sample %s due to parsing error: %s", record.sample_id, exc)
