"""Frigate-friendly inference wrappers for Trigger model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from src.data.pose_preprocess import (
    NormalizationConfig,
    SequenceAssemblyConfig,
    apply_normalization,
    load_pose_sequence,
    temporal_windows,
)
from src.data.pose_preprocess import DatasetIndexRecord
from src.common.runtime_utils import resolve_checkpoint_path, resolve_device
from src.trigger.model import TriggerModelConfig, build_trigger_model


@dataclass(slots=True)
class FrigateEvent:
    """Internal normalized Frigate event schema."""

    event_id: str
    camera_name: str
    timestamp_start: str | float
    timestamp_end: str | float | None = None
    tracked_label: str | None = None
    track_id: str | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None
    recording_path: str | None = None
    pose_input_path: str | None = None
    source: str = "frigate"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InferenceConfig:
    """Inference settings."""

    model_path: str
    device: str = "auto"
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    temporal: SequenceAssemblyConfig = field(default_factory=SequenceAssemblyConfig)


def parse_frigate_payload(payload: dict[str, Any]) -> FrigateEvent:
    """Convert MQTT-style/API payload into internal FrigateEvent dataclass."""
    return FrigateEvent(
        event_id=str(payload.get("id") or payload.get("event_id") or "unknown_event"),
        camera_name=str(payload.get("camera") or payload.get("camera_name") or "unknown_camera"),
        timestamp_start=payload.get("start_time") or payload.get("timestamp_start") or 0.0,
        timestamp_end=payload.get("end_time") or payload.get("timestamp_end"),
        tracked_label=payload.get("label") or payload.get("tracked_label"),
        track_id=(payload.get("data", {}) or {}).get("id") if isinstance(payload.get("data"), dict) else payload.get("track_id"),
        snapshot_path=payload.get("snapshot_path"),
        clip_path=payload.get("clip_path"),
        recording_path=payload.get("recording_path"),
        pose_input_path=payload.get("pose_input_path"),
        source="frigate_payload",
        metadata=payload,
    )


class TriggerInferencer:
    """Unified inference API for direct and Frigate-derived sources."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        model_path = resolve_checkpoint_path(config.model_path, base_dir="artifacts/runs", run_prefix="trigger_")
        checkpoint = torch.load(model_path, map_location=self.device)
        model_cfg = TriggerModelConfig(**checkpoint.get("config", {}))
        self.model = build_trigger_model(model_cfg).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.num_classes = int(model_cfg.num_classes)

    def infer_tensor(
        self,
        pose_tensor: np.ndarray,
        event: FrigateEvent | None = None,
        source_type: Literal["tensor", "pose_file", "frigate_event", "frigate_payload"] = "tensor",
    ) -> dict[str, Any]:
        """Run inference on one sequence (T,K,C) or batched window (N,T,K,C)."""
        tensor = pose_tensor
        if tensor.ndim == 3:
            tensor = temporal_windows(tensor, self.config.temporal)
        if tensor.ndim != 4:
            raise ValueError(f"Expected tensor ndim=4 after windowing, got {tensor.ndim}")

        with torch.no_grad():
            x = torch.from_numpy(tensor).float().to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_mean = probs.mean(axis=0)
            pred = int(np.argmax(probs_mean))
            conf = float(np.max(probs_mean))

        return {
            "event_id": event.event_id if event else "direct_input",
            "camera_name": event.camera_name if event else "unknown_camera",
            "track_id": event.track_id if event else None,
            "timestamp_start": event.timestamp_start if event else None,
            "timestamp_end": event.timestamp_end if event else None,
            "class_probabilities": {str(class_id): float(probs_mean[class_id]) for class_id in range(self.num_classes)},
            "predicted_label": pred,
            "confidence": conf,
            "notes": "Real-only binary-first inference output; class ids are configuration-driven.",
            "source_type": source_type,
        }

    def infer_pose_file(self, pose_file: Path, event: FrigateEvent | None = None) -> dict[str, Any]:
        """Run inference for a supported pose file path."""
        record = DatasetIndexRecord(
            sample_id=pose_file.stem,
            source_dataset="inference",
            original_path=str(pose_file),
            label=0,
        )
        sequence, bboxes = load_pose_sequence(record)
        normalized = apply_normalization(sequence, self.config.normalization, bboxes=bboxes)
        return self.infer_tensor(normalized, event=event, source_type="pose_file")

    def infer_frigate_event(self, event: FrigateEvent) -> dict[str, Any]:
        """Run inference from FrigateEvent by resolving pose_input_path first."""
        if not event.pose_input_path:
            raise ValueError("FrigateEvent.pose_input_path is required for trigger inference")
        return self.infer_pose_file(Path(event.pose_input_path), event=event)

    def infer_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Convert MQTT-style payload and infer."""
        event = parse_frigate_payload(payload)
        return self.infer_frigate_event(event)


def infer_events_to_json(events: list[FrigateEvent], inferencer: TriggerInferencer) -> dict[str, Any]:
    """Return JSON record keyed by event_id."""
    out: dict[str, Any] = {}
    for event in events:
        pred = inferencer.infer_frigate_event(event)
        out[event.event_id] = pred
    return out


def event_from_manifest_row(row: dict[str, Any]) -> FrigateEvent:
    """Build FrigateEvent from manifest CSV/JSON row."""
    return FrigateEvent(
        event_id=str(row.get("event_id", "unknown_event")),
        camera_name=str(row.get("camera_name", "unknown_camera")),
        timestamp_start=row.get("timestamp_start", 0.0),
        timestamp_end=row.get("timestamp_end"),
        tracked_label=row.get("tracked_label"),
        track_id=row.get("track_id"),
        snapshot_path=row.get("snapshot_path"),
        clip_path=row.get("clip_path"),
        recording_path=row.get("recording_path"),
        pose_input_path=row.get("pose_input_path"),
        source=str(row.get("source", "manifest")),
        metadata=row,
    )


def event_to_dict(event: FrigateEvent) -> dict[str, Any]:
    """Dict serializer helper."""
    return asdict(event)
