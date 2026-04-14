"""Evaluation entrypoint for trained Verifier checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.verifier_dataset import VerifierDatasetConfig, VerifierVideoDataset, build_eval_transform
from src.common.runtime_utils import resolve_checkpoint_path, resolve_device
from src.verifier.model import VerifierModelConfig, build_verifier_model
from src.verifier.train import macro_metrics


@dataclass(slots=True)
class VerifierEvalConfig:
    """Evaluation config."""

    model_path: str
    manifest_path: str
    split: str = "test"
    batch_size: int = 8
    device: str = "auto"
    num_frames: int = 16
    temporal_stride: int = 2
    output_path: str | None = None


def evaluate_verifier(config: VerifierEvalConfig) -> dict[str, Any]:
    """Evaluate checkpoint on a manifest split."""
    device = resolve_device(config.device)
    model_path = resolve_checkpoint_path(config.model_path, base_dir="artifacts/verifier_runs", run_prefix="verifier_")

    data_cfg = VerifierDatasetConfig(num_frames=config.num_frames, temporal_stride=config.temporal_stride, random_sample=False)
    ds = VerifierVideoDataset(Path(config.manifest_path), config.split, data_cfg, transform=build_eval_transform())
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    checkpoint = torch.load(model_path, map_location=device)
    model_cfg = VerifierModelConfig(**checkpoint.get("config", {}))
    model = build_verifier_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    model.eval()

    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            y_true.append(y.cpu().numpy())
            y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())

    true_np = np.concatenate(y_true)
    pred_np = np.concatenate(y_pred)
    metrics = macro_metrics(true_np, pred_np, num_classes=model_cfg.num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0

    if config.output_path:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
