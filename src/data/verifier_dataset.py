"""Dataset utilities for GuardSense Verifier RGB model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VerifierDatasetConfig:
    """Sampling settings for clip loading."""

    num_frames: int = 16
    temporal_stride: int = 2
    random_sample: bool = True


class VerifierVideoDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Torch dataset reading RGB clips from metadata manifest."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        config: VerifierDatasetConfig,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples found for split={split} in {manifest_path}")
        self.config = config
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        clip_path = Path(row["processed_path"])
        label = int(row["label"])
        video = load_video_tensor(clip_path)
        sampled = sample_clip_frames(
            video,
            num_frames=self.config.num_frames,
            temporal_stride=self.config.temporal_stride,
            random_sample=self.config.random_sample,
        )
        if self.transform is not None:
            sampled = self.transform(sampled)
        return sampled, torch.tensor(label, dtype=torch.long), str(row.get("clip_id", clip_path.stem))


def load_video_tensor(path: Path) -> torch.Tensor:
    """Load video and return tensor [C,T,H,W] float32 in [0,1]."""
    try:
        from torchvision.io import read_video
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "torchvision.io.read_video is required for verifier dataset loading. "
            "Install a torch/torchvision pair with video I/O support for your Python runtime."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(path)

    frames, _, _ = read_video(str(path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"Decoded empty video: {path}")
    # [T,H,W,C] -> [C,T,H,W]
    video = frames.permute(3, 0, 1, 2).float() / 255.0
    return video


def sample_clip_frames(
    video: torch.Tensor,
    num_frames: int,
    temporal_stride: int,
    random_sample: bool,
) -> torch.Tensor:
    """Sample fixed-length temporal clip from [C,T,H,W]."""
    if video.ndim != 4:
        raise ValueError(f"Expected [C,T,H,W], got {video.shape}")
    _, t, _, _ = video.shape
    effective = num_frames * temporal_stride

    if t >= effective:
        max_start = t - effective
        if random_sample:
            start = int(np.random.randint(0, max_start + 1))
        else:
            start = max_start // 2
        idx = torch.arange(start, start + effective, temporal_stride)
        return video[:, idx, :, :]

    # pad by repeating last frame.
    idx = torch.arange(0, t, temporal_stride)
    sampled = video[:, idx, :, :] if idx.numel() > 0 else video[:, :1, :, :]
    while sampled.shape[1] < num_frames:
        sampled = torch.cat([sampled, sampled[:, -1:, :, :]], dim=1)
    return sampled[:, :num_frames, :, :]


def build_train_transform() -> Callable[[torch.Tensor], torch.Tensor]:
    """Simple surveillance-appropriate augmentations for train split."""

    def _transform(x: torch.Tensor) -> torch.Tensor:
        # mild brightness jitter + horizontal flip for robustness.
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])
        if torch.rand(1).item() < 0.5:
            alpha = float(torch.empty(1).uniform_(0.85, 1.15).item())
            x = torch.clamp(x * alpha, 0.0, 1.0)
        return x

    return _transform


def build_eval_transform() -> Callable[[torch.Tensor], torch.Tensor]:
    """Identity transform for validation/test."""

    def _transform(x: torch.Tensor) -> torch.Tensor:
        return x

    return _transform


def class_weights_from_manifest(manifest_path: Path, split: str, num_classes: int | None = None) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    df = pd.read_csv(manifest_path)
    counts = df[df["split"] == split]["label"].value_counts().sort_index()
    class_count = int(num_classes if num_classes is not None else (int(df["label"].max()) + 1))
    weights = []
    total = float(counts.sum())
    for class_id in range(class_count):
        c = float(counts.get(class_id, 1.0))
        weights.append(total / (float(class_count) * c))
    return torch.tensor(weights, dtype=torch.float32)
