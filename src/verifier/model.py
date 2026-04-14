"""Verifier RGB model definitions."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(slots=True)
class VerifierModelConfig:
    """Config for torchvision video backbones."""

    backbone: str = "r3d_18"
    num_classes: int = 2
    pretrained: bool = False
    dropout: float = 0.2


def build_verifier_model(config: VerifierModelConfig) -> nn.Module:
    """Build configurable torchvision video classifier."""
    try:
        from torchvision.models.video import MC3_18_Weights, R3D_18_Weights, mc3_18, r3d_18
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Verifier model requires torchvision video models. Install a PyTorch/torchvision "
            "compatible pair for your Python version (for Python 3.12, use official matching wheels)."
        ) from exc

    if config.backbone == "r3d_18":
        weights = R3D_18_Weights.DEFAULT if config.pretrained else None
        model = r3d_18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=config.dropout), nn.Linear(in_features, config.num_classes))
        return model

    if config.backbone == "mc3_18":
        weights = MC3_18_Weights.DEFAULT if config.pretrained else None
        model = mc3_18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=config.dropout), nn.Linear(in_features, config.num_classes))
        return model

    raise ValueError(f"Unsupported backbone: {config.backbone}")
