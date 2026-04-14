"""Video preprocessing utilities for GuardSense Verifier model."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

LABEL_MAP: dict[int, str] = {0: "normal", 1: "fight"}
SUPPORTED_SUFFIXES = {".mp4", ".avi"}


@dataclass(slots=True)
class FrigateEvent:
    """Internal Frigate event schema for downstream video handling."""

    event_id: str
    camera_name: str
    timestamp_start: str | float
    timestamp_end: str | float | None = None
    tracked_label: str | None = None
    track_id: str | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None
    recording_path: str | None = None
    source: str = "frigate"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoIndexRecord:
    """Metadata for one raw or processed verifier clip."""

    clip_id: str
    label: int
    source_dataset: str
    original_path: str
    split: str = "unspecified"
    synthetic_or_real: str = "unknown"
    camera_name: str | None = None
    frigate_event_id: str | None = None
    track_id: str | None = None
    codec: str | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    processed_path: str | None = None
    source_type: str = "file"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoValidationConfig:
    """Clip validation controls."""

    min_duration_seconds: float = 1.5
    flag_dark: bool = True
    dark_luma_threshold: float = 35.0
    flag_blurry: bool = True
    blurry_laplacian_var_threshold: float = 30.0


@dataclass(slots=True)
class VideoProcessConfig:
    """Video standardization controls."""

    target_fps: int = 30
    target_duration_seconds: float = 3.0
    min_duration_seconds: float = 2.0
    max_duration_seconds: float = 4.0
    target_width: int = 640
    target_height: int = 360
    target_codec: str = "libx264"
    crf: int = 23
    preset: str = "fast"


@dataclass(slots=True)
class RejectionRecord:
    """Rejected clip with explicit reason."""

    clip_id: str
    path: str
    reason: str


@dataclass(slots=True)
class DatasetSummary:
    """Aggregated dataset summary after preprocessing."""

    clips_per_class: dict[str, int]
    clips_per_source: dict[str, int]
    synthetic_vs_real: dict[str, int]
    frigate_vs_nonfrigate: dict[str, int]
    rejected_reasons: dict[str, int]
    average_duration_seconds: float
    average_fps: float


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _run_cmd_bytes(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, check=False, capture_output=True)


def ffprobe_video(path: Path) -> dict[str, Any]:
    """Read media metadata via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_name,avg_frame_rate,width,height",
        "-select_streams",
        "v:0",
        "-of",
        "json",
        str(path),
    ]
    proc = _run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format", {})
    fps_raw = str(stream.get("avg_frame_rate", "0/1"))
    if "/" in fps_raw:
        n, d = fps_raw.split("/", 1)
        fps = (float(n) / float(d)) if float(d) != 0 else 0.0
    else:
        fps = float(fps_raw or 0.0)
    return {
        "codec": stream.get("codec_name"),
        "fps": fps,
        "duration_seconds": float(fmt.get("duration") or 0.0),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def _opencv_video_meta(path: Path) -> dict[str, Any]:
    """Fallback metadata probing using OpenCV when ffprobe is unavailable/incomplete."""
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("OpenCV fallback unavailable for metadata probing") from exc

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return {
            "codec": "opencv",
            "fps": fps,
            "duration_seconds": duration,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def probe_video_metadata(path: Path) -> dict[str, Any]:
    """Probe video metadata with ffprobe first, then OpenCV fallback if needed."""
    try:
        meta = ffprobe_video(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("ffprobe failed for %s; trying OpenCV fallback (%s)", path, exc)
        return _opencv_video_meta(path)

    if meta["duration_seconds"] <= 0 or meta["fps"] <= 0 or meta["width"] <= 0 or meta["height"] <= 0:
        LOGGER.warning("ffprobe returned incomplete metadata for %s; trying OpenCV fallback.", path)
        try:
            fallback = _opencv_video_meta(path)
            for key in ("fps", "duration_seconds", "width", "height"):
                if float(meta.get(key, 0.0) or 0.0) <= 0:
                    meta[key] = fallback[key]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("OpenCV fallback failed for %s (%s); using ffprobe metadata as-is.", path, exc)

    return meta


def _normalize_label_token(value: str) -> str:
    """Normalize a label token by lowercasing and dropping non-alphanumeric chars."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _extract_path_label_tokens(path: Path) -> set[str]:
    """Extract canonical label tokens from path segments without substring matching."""
    tokens: set[str] = set()
    for part in path.parts:
        normalized_part = part.lower()
        compact = _normalize_label_token(normalized_part)
        if compact:
            tokens.add(compact)
        for token in re.split(r"[^a-z0-9]+", normalized_part):
            compact_token = _normalize_label_token(token)
            if compact_token:
                tokens.add(compact_token)
    return tokens


def resolve_label_from_rules(path: Path, label_rules: dict[str, int]) -> int | None:
    """Resolve label from config rules using normalized token equality and specificity.

    Rules are matched against normalized path tokens only (no substring includes), then
    evaluated from most-specific pattern to least-specific pattern.
    """
    if not label_rules:
        return None

    tokens = _extract_path_label_tokens(path)
    sorted_rules = sorted(
        ((str(pattern), int(class_id), _normalize_label_token(str(pattern))) for pattern, class_id in label_rules.items()),
        key=lambda item: (len(item[2]), item[0]),
        reverse=True,
    )
    for pattern, class_id, normalized_pattern in sorted_rules:
        if normalized_pattern and normalized_pattern in tokens:
            LOGGER.debug("Resolved label via rule '%s' for %s -> %d", pattern, path, class_id)
            return class_id
    return None


def parse_label_from_path(path: Path) -> int | None:
    """Parse binary labels from path tokens and common dataset naming conventions."""
    tokens = _extract_path_label_tokens(path)
    if "0" in tokens or "0normal" in tokens:
        return 0
    if "1" in tokens or "1fight" in tokens:
        return 1

    ordered_aliases: list[tuple[str, int]] = [
        ("nonfight", 0),
        ("nonviolent", 0),
        ("nonviolence", 0),
        ("normal", 0),
        ("safe", 0),
        ("negative", 0),
        ("fight", 1),
        ("violence", 1),
        ("violent", 1),
        ("positive", 1),
    ]
    for alias, class_id in ordered_aliases:
        if alias in tokens:
            return class_id
    return None


def index_video_sources(
    source_dirs: Sequence[Path],
    source_dataset: str,
    label: int | None,
    synthetic_or_real: str,
    split: str = "unspecified",
    label_rules: dict[str, int] | None = None,
) -> list[VideoIndexRecord]:
    """Recursively index verifier clips and enrich metadata with ffprobe."""
    rows: list[VideoIndexRecord] = []
    for source_dir in source_dirs:
        for path in source_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            resolved_label: int | None = None
            if label is not None:
                resolved_label = label
            elif label_rules:
                resolved_label = resolve_label_from_rules(path, label_rules)
            if resolved_label is None:
                resolved_label = parse_label_from_path(path)
            if resolved_label is None:
                LOGGER.warning("Skipping unlabeled path %s", path)
                continue
            row = VideoIndexRecord(
                clip_id=path.stem,
                label=resolved_label,
                source_dataset=source_dataset,
                original_path=str(path),
                split=split,
                synthetic_or_real=synthetic_or_real,
                source_type="file",
            )
            try:
                meta = probe_video_metadata(path)
                row.codec = str(meta.get("codec") or "")
                row.fps = float(meta.get("fps") or 0.0)
                row.duration_seconds = float(meta.get("duration_seconds") or 0.0)
                row.width = int(meta.get("width") or 0)
                row.height = int(meta.get("height") or 0)
            except Exception as exc:  # noqa: BLE001
                row.metadata["probe_error"] = str(exc)
            rows.append(row)
    LOGGER.info("Indexed %d video files from %d sources", len(rows), len(source_dirs))
    return rows


def parse_frigate_payload(payload: dict[str, Any]) -> FrigateEvent:
    """Map MQTT-like/API payload into FrigateEvent."""
    data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
    event_id = str(payload.get("id") or payload.get("event_id") or "unknown_event")
    return FrigateEvent(
        event_id=event_id,
        camera_name=str(payload.get("camera") or payload.get("camera_name") or "unknown_camera"),
        timestamp_start=payload.get("start_time") or payload.get("timestamp_start") or 0.0,
        timestamp_end=payload.get("end_time") or payload.get("timestamp_end"),
        tracked_label=payload.get("label") or payload.get("tracked_label"),
        track_id=str(data.get("id") or payload.get("track_id") or "") or None,
        snapshot_path=payload.get("snapshot_path"),
        clip_path=payload.get("clip_path"),
        recording_path=payload.get("recording_path"),
        source="frigate_payload",
        metadata=payload,
    )


def resolve_frigate_paths(event: FrigateEvent, clips_root: Path | None = None, recordings_root: Path | None = None) -> FrigateEvent:
    """Resolve canonical Frigate clip/recording paths if missing."""
    if event.clip_path:
        return event

    clip_candidate = None
    if clips_root is not None:
        clip_candidate = clips_root / f"{event.camera_name}-{event.event_id}.mp4"
        if clip_candidate.exists():
            event.clip_path = str(clip_candidate)

    if not event.recording_path and recordings_root is not None:
        # heuristic best-effort path only; caller may override with API-derived metadata.
        rec_candidate = recordings_root / event.camera_name
        if rec_candidate.exists():
            event.recording_path = str(rec_candidate)

    return event


def validate_video(record: VideoIndexRecord, config: VideoValidationConfig) -> tuple[bool, str | None]:
    """Validate readability, minimum duration, and optional quality flags."""
    path = Path(record.original_path)
    try:
        meta = probe_video_metadata(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable:{exc}"

    duration = float(meta["duration_seconds"])
    if duration < config.min_duration_seconds:
        return False, f"too_short:{duration:.2f}s"

    if config.flag_dark or config.flag_blurry:
        try:
            sampled = sample_frame_stats(path)
        except Exception as exc:  # noqa: BLE001
            return False, f"quality_check_failed:{exc}"
        if config.flag_dark and sampled["avg_luma"] < config.dark_luma_threshold:
            return False, f"too_dark:{sampled['avg_luma']:.2f}"
        if config.flag_blurry and sampled["laplacian_var"] < config.blurry_laplacian_var_threshold:
            return False, f"too_blurry:{sampled['laplacian_var']:.2f}"

    return True, None


def sample_frame_stats(path: Path) -> dict[str, float]:
    """Estimate darkness/blurriness from a center frame via ffmpeg + numpy."""
    probe = probe_video_metadata(path)
    t = max(float(probe["duration_seconds"]) / 2.0, 0.0)
    width = int(probe["width"])
    height = int(probe["height"])
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid frame dimensions")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    proc = _run_cmd_bytes(cmd)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(proc.stderr.strip() or "Could not sample frame")
    frame = np.frombuffer(proc.stdout, dtype=np.uint8)
    expected = width * height
    if frame.size < expected:
        raise RuntimeError("Failed raw frame extraction")

    frame = frame[:expected].reshape(height, width).astype(np.float32)
    avg_luma = float(frame.mean())

    # simple Laplacian variance with numpy finite differences (OpenCV-free).
    lap = (
        -4 * frame
        + np.roll(frame, 1, axis=0)
        + np.roll(frame, -1, axis=0)
        + np.roll(frame, 1, axis=1)
        + np.roll(frame, -1, axis=1)
    )
    lap_var = float(np.var(lap))
    return {"avg_luma": avg_luma, "laplacian_var": lap_var}


def centered_trim_start(duration: float, clip_len: float) -> float:
    """Center trim start timestamp."""
    if duration <= clip_len:
        return 0.0
    return max((duration - clip_len) / 2.0, 0.0)


def preprocess_clip(
    input_path: Path,
    output_path: Path,
    config: VideoProcessConfig,
    roi: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Standardize one clip to target fps/resolution/duration using ffmpeg."""
    meta = probe_video_metadata(input_path)
    duration = float(meta["duration_seconds"])
    clip_len = min(max(config.target_duration_seconds, config.min_duration_seconds), config.max_duration_seconds)
    start = centered_trim_start(duration, clip_len)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={config.target_width}:{config.target_height}:force_original_aspect_ratio=decrease,pad={config.target_width}:{config.target_height}:(ow-iw)/2:(oh-ih)/2"
    if roi:
        x = max(roi.get("x", 0), 0)
        y = max(roi.get("y", 0), 0)
        w = max(roi.get("w", config.target_width), 1)
        h = max(roi.get("h", config.target_height), 1)
        vf = f"crop={w}:{h}:{x}:{y},{vf}"

    codec_candidates = [config.target_codec]
    if config.target_codec != "mpeg4":
        codec_candidates.append("mpeg4")
    last_error = ""
    for codec in codec_candidates:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{clip_len:.3f}",
            "-r",
            str(config.target_fps),
            "-vf",
            vf,
            "-c:v",
            codec,
            "-preset",
            config.preset,
            "-crf",
            str(config.crf),
            "-an",
            "-y",
            str(output_path),
        ]
        proc = _run_cmd(cmd)
        if proc.returncode == 0:
            if codec != config.target_codec:
                LOGGER.warning("Fell back to codec '%s' for %s", codec, input_path)
            break
        last_error = proc.stderr.strip()
    else:
        raise RuntimeError(f"ffmpeg preprocess failed: {last_error}")

    processed_meta = probe_video_metadata(output_path)
    if processed_meta["duration_seconds"] <= 0 or processed_meta["fps"] <= 0:
        raise RuntimeError("processed clip metadata invalid after preprocessing")
    return processed_meta


def extract_event_centered_clip(
    recording_path: Path,
    output_path: Path,
    event_start: float,
    event_end: float | None,
    config: VideoProcessConfig,
) -> dict[str, Any]:
    """Extract a short event-centered clip from a longer recording."""
    if event_end is not None and event_end >= event_start:
        midpoint = (event_start + event_end) / 2.0
    else:
        midpoint = event_start

    clip_len = min(max(config.target_duration_seconds, config.min_duration_seconds), config.max_duration_seconds)
    start = max(midpoint - (clip_len / 2.0), 0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={config.target_width}:{config.target_height}:force_original_aspect_ratio=decrease,pad={config.target_width}:{config.target_height}:(ow-iw)/2:(oh-ih)/2"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(recording_path),
        "-t",
        f"{clip_len:.3f}",
        "-r",
        str(config.target_fps),
        "-vf",
        vf,
        "-c:v",
        config.target_codec,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-an",
        "-y",
        str(output_path),
    ]
    proc = _run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extract failed: {proc.stderr.strip()}")
    return probe_video_metadata(output_path)


def frigate_event_to_clip(
    event: FrigateEvent,
    output_dir: Path,
    process_cfg: VideoProcessConfig,
) -> tuple[Path | None, str]:
    """Resolve or build clip for Frigate event.

    Returns:
        (clip_path, source_kind)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if event.clip_path and Path(event.clip_path).exists():
        return Path(event.clip_path), "frigate_event_clip"

    if event.recording_path and Path(event.recording_path).exists():
        out = output_dir / f"{event.event_id}_from_recording.mp4"
        start = float(event.timestamp_start) if isinstance(event.timestamp_start, (float, int)) else 0.0
        end = float(event.timestamp_end) if isinstance(event.timestamp_end, (float, int)) else None
        extract_event_centered_clip(Path(event.recording_path), out, start, end, process_cfg)
        return out, "frigate_recording_extract"

    return None, "fallback_missing_clip"


def save_index_csv(rows: Sequence[VideoIndexRecord], output_path: Path) -> None:
    """Write index/metadata CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(VideoIndexRecord.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["metadata"] = json.dumps(payload.get("metadata", {}), ensure_ascii=False)
            writer.writerow(payload)


def save_rejections(rows: Sequence[RejectionRecord], output_path: Path) -> None:
    """Write rejection log CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "path", "reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write JSON with indentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_dataset(rows: Sequence[VideoIndexRecord], rejections: Sequence[RejectionRecord]) -> DatasetSummary:
    """Aggregate summary statistics required by Verifier pipeline."""
    clips_per_class: dict[str, int] = {}
    clips_per_source: dict[str, int] = {}
    synthetic_vs_real: dict[str, int] = {}
    frigate_vs_nonfrigate = {"frigate": 0, "non_frigate": 0}
    durations: list[float] = []
    fps_values: list[float] = []

    for row in rows:
        clips_per_class[str(row.label)] = clips_per_class.get(str(row.label), 0) + 1
        clips_per_source[row.source_dataset] = clips_per_source.get(row.source_dataset, 0) + 1
        synthetic_vs_real[row.synthetic_or_real] = synthetic_vs_real.get(row.synthetic_or_real, 0) + 1
        if row.frigate_event_id or row.source_type.startswith("frigate"):
            frigate_vs_nonfrigate["frigate"] += 1
        else:
            frigate_vs_nonfrigate["non_frigate"] += 1
        if row.duration_seconds is not None:
            durations.append(float(row.duration_seconds))
        if row.fps is not None:
            fps_values.append(float(row.fps))

    rejected_reasons: dict[str, int] = {}
    for rej in rejections:
        rejected_reasons[rej.reason] = rejected_reasons.get(rej.reason, 0) + 1

    return DatasetSummary(
        clips_per_class=clips_per_class,
        clips_per_source=clips_per_source,
        synthetic_vs_real=synthetic_vs_real,
        frigate_vs_nonfrigate=frigate_vs_nonfrigate,
        rejected_reasons=rejected_reasons,
        average_duration_seconds=float(mean(durations)) if durations else 0.0,
        average_fps=float(mean(fps_values)) if fps_values else 0.0,
    )


def save_split_manifest(output_path: Path, rows: Sequence[VideoIndexRecord]) -> None:
    """Persist split assignments for processed clips."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "label",
                "split",
                "source_dataset",
                "original_path",
                "processed_path",
                "fps",
                "duration_seconds",
                "resolution",
                "frigate_event_id",
                "camera_name",
                "track_id",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip_id": row.clip_id,
                    "label": row.label,
                    "split": row.split,
                    "source_dataset": row.source_dataset,
                    "original_path": row.original_path,
                    "processed_path": row.processed_path,
                    "fps": row.fps,
                    "duration_seconds": row.duration_seconds,
                    "resolution": f"{row.width}x{row.height}" if row.width and row.height else "",
                    "frigate_event_id": row.frigate_event_id,
                    "camera_name": row.camera_name,
                    "track_id": row.track_id,
                }
            )


def stratified_split(
    labels: Sequence[int],
    groups: Sequence[str] | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    source_aware: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stratified split with optional source grouping."""
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    y = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for class_id in np.unique(y):
        indices = np.where(y == class_id)[0]
        if source_aware and groups is not None:
            class_groups = np.asarray(groups)[indices]
            uniq = np.unique(class_groups)
            rng.shuffle(uniq)
            ordered: list[int] = []
            for g in uniq:
                ordered.extend(indices[class_groups == g].tolist())
            indices = np.asarray(ordered, dtype=np.int64)
        else:
            rng.shuffle(indices)

        n = len(indices)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        n_test = max(0, n - n_train - n_val)

        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train : n_train + n_val].tolist())
        if n_test > 0:
            test_idx.extend(indices[-n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)
