"""Trigger model definition for pose sequence classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class TriggerModelConfig:
    """Config for temporal CNN classifier."""

    architecture: str = "temporal_cnn"
    num_keypoints: int = 17
    num_channels: int = 3
    num_classes: int = 2
    hidden_size: int = 128
    dropout: float = 0.2

    def __post_init__(self) -> None:
        """Validate architecture compatibility."""
        supported = {"temporal_cnn", "trigger_temporal_cnn"}
        if self.architecture not in supported:
            raise ValueError(
                f"Unsupported Trigger architecture '{self.architecture}'. "
                f"Supported values: {sorted(supported)}"
            )


class TriggerTemporalCNN(nn.Module):
    """Simple maintainable temporal CNN over flattened keypoints."""

    def __init__(self, config: TriggerModelConfig) -> None:
        super().__init__()
        self.config = config
        in_features = config.num_keypoints * config.num_channels

        self.encoder = nn.Sequential(
            nn.Conv1d(in_features, config.hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass with input shape (B,T,K,C)."""
        if x.dim() != 4:
            raise ValueError(f"Expected input shape (B,T,K,C), got {tuple(x.shape)}")
        b, t, k, c = x.shape
        if k != self.config.num_keypoints or c != self.config.num_channels:
            raise ValueError(
                f"Expected K={self.config.num_keypoints}, C={self.config.num_channels}, got K={k}, C={c}"
            )
        flattened = x.reshape(b, t, k * c).transpose(1, 2)  # (B, KC, T)
        encoded = self.encoder(flattened)
        return self.head(encoded)


def build_trigger_model(config: TriggerModelConfig) -> TriggerTemporalCNN:
    """Factory helper for model instantiation."""
    if config.architecture not in {"temporal_cnn", "trigger_temporal_cnn"}:
        raise ValueError(f"Unsupported Trigger architecture: {config.architecture}")
    return TriggerTemporalCNN(config)
