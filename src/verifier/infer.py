"""Frigate-friendly inference wrappers for Verifier model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.common.runtime_utils import resolve_checkpoint_path, resolve_device
from src.data.verifier_dataset import load_video_tensor, sample_clip_frames
from src.data.video_preprocess import FrigateEvent, VideoProcessConfig, frigate_event_to_clip, parse_frigate_payload
from src.verifier.model import VerifierModelConfig, build_verifier_model


@dataclass(slots=True)
class VerifierInferenceConfig:
    """Runtime inference config."""

    model_path: str
    num_frames: int = 16
    temporal_stride: int = 2
    device: str = "auto"


class VerifierInferencer:
    """Inference API for direct clips and Frigate events."""

    def __init__(self, config: VerifierInferenceConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        model_path = resolve_checkpoint_path(config.model_path, base_dir="artifacts/verifier_runs", run_prefix="verifier_")
        checkpoint = torch.load(model_path, map_location=self.device)
        model_cfg = VerifierModelConfig(**checkpoint.get("config", {}))
        self.model = build_verifier_model(model_cfg).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.num_classes = int(model_cfg.num_classes)

    def infer_clip(self, clip_path: Path, event: FrigateEvent | None = None, source_type: str = "direct_clip") -> dict[str, Any]:
        """Inference on one clip file."""
        video = load_video_tensor(clip_path)
        sampled = sample_clip_frames(
            video,
            num_frames=self.config.num_frames,
            temporal_stride=self.config.temporal_stride,
            random_sample=False,
        ).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(self.model(sampled.to(self.device)), dim=-1)[0].cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs.max())

        return {
            "event_id": event.event_id if event else "direct_input",
            "camera_name": event.camera_name if event else "unknown_camera",
            "clip_path_used": str(clip_path),
            "predicted_label": pred,
            "class_probabilities": {str(class_id): float(probs[class_id]) for class_id in range(self.num_classes)},
            "confidence": conf,
            "notes": "Real-only binary-first inference output; class ids are configuration-driven.",
            "source_type": source_type,
            "timestamp_start": event.timestamp_start if event else None,
            "timestamp_end": event.timestamp_end if event else None,
        }

    def infer_frigate_event(
        self,
        event: FrigateEvent,
        extraction_output_dir: Path,
        process_cfg: VideoProcessConfig | None = None,
    ) -> dict[str, Any]:
        """Inference from FrigateEvent with clip resolution/extraction fallback."""
        cfg = process_cfg or VideoProcessConfig()
        clip_path, source_type = frigate_event_to_clip(event, extraction_output_dir, cfg)
        if clip_path is None:
            return {
                "event_id": event.event_id,
                "camera_name": event.camera_name,
                "clip_path_used": None,
                "predicted_label": None,
                "class_probabilities": {str(class_id): 0.0 for class_id in range(self.num_classes)},
                "confidence": 0.0,
                "notes": "Video clip unavailable (snapshot-only or missing media).",
                "source_type": source_type,
                "timestamp_start": event.timestamp_start,
                "timestamp_end": event.timestamp_end,
            }
        return self.infer_clip(clip_path, event=event, source_type=source_type)

    def infer_event_id(
        self,
        event_id: str,
        resolver: Any,
        extraction_output_dir: Path,
        process_cfg: VideoProcessConfig | None = None,
    ) -> dict[str, Any]:
        """Resolve Frigate event_id with external resolver and run inference."""
        event = resolver.resolve_event(event_id)
        return self.infer_frigate_event(event, extraction_output_dir=extraction_output_dir, process_cfg=process_cfg)

    def infer_payload(self, payload: dict[str, Any], extraction_output_dir: Path, process_cfg: VideoProcessConfig | None = None) -> dict[str, Any]:
        """Convert MQTT-like payload to FrigateEvent and run inference."""
        event = parse_frigate_payload(payload)
        return self.infer_frigate_event(event, extraction_output_dir=extraction_output_dir, process_cfg=process_cfg)


def infer_events_to_json(events: list[FrigateEvent], inferencer: VerifierInferencer, extraction_output_dir: Path) -> dict[str, Any]:
    """JSON output keyed by event_id for downstream Frigate integration."""
    out: dict[str, Any] = {}
    for event in events:
        out[event.event_id] = inferencer.infer_frigate_event(event, extraction_output_dir=extraction_output_dir)
    return out


def event_to_dict(event: FrigateEvent) -> dict[str, Any]:
    """Serializer helper."""
    return asdict(event)
