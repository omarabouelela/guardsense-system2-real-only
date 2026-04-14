"""Real-video pose extraction utilities for Trigger data preparation."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)

PERSON_CLASS_ID = 0


@dataclass(slots=True)
class DatasetExtractionConfig:
    """Config for a dataset-specific extraction run."""

    name: str
    input_dir: Path
    output_dir: Path
    recursive: bool = True
    fixed_label: int | None = None
    default_label: int = 0
    label_rules: list[dict[str, Any]] | None = None
    allowed_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


@dataclass(slots=True)
class PoseExtractorConfig:
    """Global extraction controls."""

    model: str = "yolov8n-pose.pt"
    device: str = "auto"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    person_selection: Literal["highest_confidence", "largest_bbox"] = "highest_confidence"
    output_format: Literal["sequence_txt"] = "sequence_txt"
    overwrite: bool = False
    min_detected_frame_ratio: float = 0.05
    low_detection_policy: Literal["reject", "keep"] = "keep"


@dataclass(slots=True)
class ExtractionRecord:
    """Manifest row for one processed video."""

    dataset_name: str
    source_video_path: str
    output_pose_path: str
    label: int
    source_class_name: str
    clip_id: str
    split_hint: str
    status: str
    frame_count: int
    detected_frame_count: int
    avg_detected_persons_per_frame: float
    note: str


def build_dataset_config(raw: dict[str, Any]) -> DatasetExtractionConfig:
    """Parse and validate one dataset extraction config."""
    return DatasetExtractionConfig(
        name=str(raw["name"]),
        input_dir=Path(raw["input_dir"]),
        output_dir=Path(raw["output_dir"]),
        recursive=bool(raw.get("recursive", True)),
        fixed_label=int(raw["fixed_label"]) if raw.get("fixed_label") is not None else None,
        default_label=int(raw.get("default_label", 0)),
        label_rules=list(raw.get("label_rules", [])) or None,
        allowed_extensions=tuple(ext.lower() for ext in raw.get("allowed_extensions", [".mp4", ".avi", ".mov", ".mkv"])),
    )


def resolve_label_for_video(video_path: Path, dataset: DatasetExtractionConfig) -> tuple[int, str, str]:
    """Resolve binary label from relative video path using explicit mapping rules."""
    rel_text = video_path.relative_to(dataset.input_dir).as_posix().lower()
    path_parts = [part.lower() for part in video_path.relative_to(dataset.input_dir).parts]
    split_hint = "unspecified"
    for hint in ("train", "val", "test"):
        if hint in path_parts:
            split_hint = hint
            break

    if dataset.fixed_label is not None:
        source_class = path_parts[0] if path_parts else "unknown"
        return dataset.fixed_label, source_class, split_hint

    for rule in dataset.label_rules or []:
        contains = str(rule.get("contains", "")).lower()
        if contains and contains in rel_text:
            return int(rule["label"]), str(rule.get("source_class_name", contains)), split_hint

    source_class = path_parts[0] if path_parts else "unknown"
    return dataset.default_label, source_class, split_hint


def iter_videos(dataset: DatasetExtractionConfig) -> list[Path]:
    """Collect dataset videos using configured scan mode and extensions."""
    if not dataset.input_dir.exists():
        LOGGER.warning("Input directory missing for dataset=%s: %s", dataset.name, dataset.input_dir)
        return []

    pattern = "**/*" if dataset.recursive else "*"
    videos: list[Path] = []
    for path in dataset.input_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() in dataset.allowed_extensions:
            videos.append(path)
    return sorted(videos)


def _safe_clip_id(video_path: Path, dataset: DatasetExtractionConfig) -> str:
    rel = video_path.relative_to(dataset.input_dir)
    return rel.with_suffix("").as_posix().replace("/", "__")


def _output_pose_path(video_path: Path, dataset: DatasetExtractionConfig) -> Path:
    return dataset.output_dir / f"{_safe_clip_id(video_path, dataset)}.txt"


def _load_yolo_model(model_name_or_path: str) -> Any:
    """Lazy-import Ultralytics to keep --help functional without extra deps."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for pose extraction. Install with: pip install ultralytics"
        ) from exc
    return YOLO(model_name_or_path)


def _select_person_index(results: Any, strategy: str) -> int | None:
    boxes = getattr(results, "boxes", None)
    keypoints = getattr(results, "keypoints", None)
    if boxes is None or keypoints is None or boxes.cls is None:
        return None

    cls_list = boxes.cls.detach().cpu().tolist()
    person_indices = [idx for idx, cls_val in enumerate(cls_list) if int(cls_val) == PERSON_CLASS_ID]
    if not person_indices:
        return None

    if strategy == "largest_bbox":
        xyxy = boxes.xyxy.detach().cpu().numpy()
        best = max(person_indices, key=lambda idx: float(max(0.0, (xyxy[idx][2] - xyxy[idx][0])) * max(0.0, (xyxy[idx][3] - xyxy[idx][1]))))
        return int(best)

    confs = boxes.conf.detach().cpu().tolist() if boxes.conf is not None else [0.0] * len(cls_list)
    best = max(person_indices, key=lambda idx: float(confs[idx]))
    return int(best)


def _empty_pose_row() -> str:
    kp_tokens = ["0.0", "0.0", "0.0"] * 17
    return " ".join(["0", "0.0", "0.0", "0.0", "0.0", *kp_tokens])


def _pose_row_from_result(results: Any, person_index: int) -> str:
    boxes = results.boxes
    keypoints = results.keypoints

    xywhn = boxes.xywhn.detach().cpu().numpy()[person_index]
    kps_xy = keypoints.xyn.detach().cpu().numpy()[person_index]

    if keypoints.conf is not None:
        kps_conf = keypoints.conf.detach().cpu().numpy()[person_index]
    else:
        kps_conf = [1.0] * kps_xy.shape[0]

    tokens = [
        "0",
        f"{float(xywhn[0]):.6f}",
        f"{float(xywhn[1]):.6f}",
        f"{float(xywhn[2]):.6f}",
        f"{float(xywhn[3]):.6f}",
    ]
    for idx in range(min(17, kps_xy.shape[0])):
        tokens.append(f"{float(kps_xy[idx][0]):.6f}")
        tokens.append(f"{float(kps_xy[idx][1]):.6f}")
        tokens.append(f"{float(kps_conf[idx]):.6f}")

    while len(tokens) < 5 + (17 * 3):
        tokens.extend(["0.0", "0.0", "0.0"])

    return " ".join(tokens)


def extract_video_pose(
    model: Any,
    video_path: Path,
    output_path: Path,
    config: PoseExtractorConfig,
) -> tuple[bool, int, int, float, str]:
    """Extract per-frame primary-person pose rows for one video."""
    try:
        import cv2
    except ImportError as exc:
        return False, 0, 0, 0.0, f"opencv_missing:{exc}"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False, 0, 0, 0.0, "video_open_failed"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    detected_frames = 0
    detected_people_total = 0
    rows: list[str] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1

            infer_result = model.predict(
                source=frame,
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                device=config.device,
                verbose=False,
            )
            result = infer_result[0] if infer_result else None
            if result is None or result.boxes is None or result.boxes.cls is None:
                rows.append(_empty_pose_row())
                continue

            person_count = int((result.boxes.cls.detach().cpu().numpy() == PERSON_CLASS_ID).sum())
            detected_people_total += person_count

            selected = _select_person_index(result, config.person_selection)
            if selected is None:
                rows.append(_empty_pose_row())
                continue

            rows.append(_pose_row_from_result(result, selected))
            detected_frames += 1
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Pose extraction failed for %s", video_path)
        return False, frame_count, detected_frames, 0.0, f"inference_error:{exc}"
    finally:
        capture.release()

    if frame_count <= 0:
        return False, 0, 0, 0.0, "no_frames"

    detected_ratio = detected_frames / frame_count
    avg_people = float(detected_people_total / frame_count)
    note = "ok"
    success = True

    if detected_ratio < config.min_detected_frame_ratio:
        note = f"low_detection_ratio:{detected_ratio:.4f}"
        if config.low_detection_policy == "reject":
            success = False

    if success:
        output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return success, frame_count, detected_frames, avg_people, note


def process_dataset(
    model: Any,
    dataset: DatasetExtractionConfig,
    config: PoseExtractorConfig,
) -> list[ExtractionRecord]:
    """Process every video from one dataset and return manifest records."""
    videos = iter_videos(dataset)
    LOGGER.info("Dataset=%s videos_found=%d", dataset.name, len(videos))

    records: list[ExtractionRecord] = []
    for video_path in videos:
        label, source_class_name, split_hint = resolve_label_for_video(video_path, dataset)
        output_path = _output_pose_path(video_path, dataset)
        clip_id = _safe_clip_id(video_path, dataset)

        if output_path.exists() and not config.overwrite:
            records.append(
                ExtractionRecord(
                    dataset_name=dataset.name,
                    source_video_path=str(video_path),
                    output_pose_path=str(output_path),
                    label=label,
                    source_class_name=source_class_name,
                    clip_id=clip_id,
                    split_hint=split_hint,
                    status="skipped_existing",
                    frame_count=0,
                    detected_frame_count=0,
                    avg_detected_persons_per_frame=0.0,
                    note="overwrite_disabled",
                )
            )
            continue

        success, frame_count, detected_count, avg_people, note = extract_video_pose(
            model=model,
            video_path=video_path,
            output_path=output_path,
            config=config,
        )

        records.append(
            ExtractionRecord(
                dataset_name=dataset.name,
                source_video_path=str(video_path),
                output_pose_path=str(output_path),
                label=label,
                source_class_name=source_class_name,
                clip_id=clip_id,
                split_hint=split_hint,
                status="success" if success else "failed",
                frame_count=frame_count,
                detected_frame_count=detected_count,
                avg_detected_persons_per_frame=avg_people,
                note=note,
            )
        )

    return records


def write_manifest_csv(output_path: Path, records: list[ExtractionRecord]) -> None:
    """Write extraction manifest to csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(asdict(records[0]).keys()) if records else list(asdict(ExtractionRecord(
            dataset_name="",
            source_video_path="",
            output_pose_path="",
            label=0,
            source_class_name="",
            clip_id="",
            split_hint="",
            status="",
            frame_count=0,
            detected_frame_count=0,
            avg_detected_persons_per_frame=0.0,
            note="",
        )).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def load_yolo_model(model_name_or_path: str) -> Any:
    """Public wrapper for loading ultralytics pose model."""
    return _load_yolo_model(model_name_or_path)
