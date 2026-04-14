"""Training loop for GuardSense Verifier RGB model."""

from __future__ import annotations

import json
import logging
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.verifier_dataset import (
    VerifierDatasetConfig,
    VerifierVideoDataset,
    build_eval_transform,
    build_train_transform,
    class_weights_from_manifest,
)
from src.common.runtime_utils import refresh_latest_alias, resolve_device
from src.verifier.model import VerifierModelConfig, build_verifier_model

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VerifierTrainConfig:
    """Training config values."""

    manifest_path: str
    output_dir: str
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    mixed_precision: bool = True
    model: VerifierModelConfig = field(default_factory=VerifierModelConfig)
    data: VerifierDatasetConfig = field(default_factory=VerifierDatasetConfig)


def set_seed(seed: int) -> None:
    """Deterministic setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> dict[str, Any]:
    """Compute metrics without extra third-party libraries."""
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred, strict=False):
        conf[int(truth), int(pred)] += 1

    precisions, recalls, f1s = [], [], []
    per_class: dict[str, dict[str, float]] = {}
    for class_id in range(num_classes):
        tp = conf[class_id, class_id]
        fp = conf[:, class_id].sum() - tp
        fn = conf[class_id, :].sum() - tp
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        per_class[str(class_id)] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))

    label0_fp = int(conf[:, 0].sum() - conf[0, 0]) if num_classes > 0 else 0
    focus_class = min(1, num_classes - 1) if num_classes > 0 else 0
    cross_class_confusions = int(conf.sum() - np.trace(conf) - label0_fp)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "focus_class_precision": per_class.get(str(focus_class), {}).get("precision", 0.0),
        "focus_class_recall": per_class.get(str(focus_class), {}).get("recall", 0.0),
        "label_0_false_positives": label0_fp,
        "cross_class_confusions": cross_class_confusions,
        "confusion_matrix": conf.tolist(),
    }


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluation helper."""
    model.eval()
    losses: list[float] = []
    yt: list[np.ndarray] = []
    yp: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            yt.append(y.cpu().numpy())
            yp.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return float(np.mean(losses)) if losses else 0.0, np.concatenate(yt), np.concatenate(yp)


def train_verifier(config: VerifierTrainConfig) -> dict[str, Any]:
    """Train verifier model and save full run artifacts."""
    set_seed(config.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    device = resolve_device(config.device)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config.output_dir) / f"verifier_{config.model.backbone}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = VerifierVideoDataset(Path(config.manifest_path), "train", config.data, transform=build_train_transform())
    val_cfg = VerifierDatasetConfig(num_frames=config.data.num_frames, temporal_stride=config.data.temporal_stride, random_sample=False)
    val_ds = VerifierVideoDataset(Path(config.manifest_path), "val", val_cfg, transform=build_eval_transform())
    test_ds = VerifierVideoDataset(Path(config.manifest_path), "test", val_cfg, transform=build_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_verifier_model(config.model).to(device)
    weights = class_weights_from_manifest(
        Path(config.manifest_path),
        split="train",
        num_classes=config.model.num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    scaler = torch.amp.GradScaler("cuda", enabled=(config.mixed_precision and device.type == "cuda"))
    best_f1 = -1.0
    stale = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_model_path = run_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(config.mixed_precision and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss, y_true_val, y_pred_val = _evaluate(model, val_loader, criterion, device)
        val_metrics = macro_metrics(y_true_val, y_pred_val, num_classes=config.model.num_classes)
        val_macro_f1 = float(val_metrics["macro_f1"])
        scheduler.step(val_macro_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)

        LOGGER.info(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            val_macro_f1,
        )

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            stale = 0
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(config.model)}, best_model_path)
        else:
            stale += 1
            if stale >= config.patience:
                LOGGER.info("Early stopping at epoch=%d", epoch)
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, y_true_test, y_pred_test = _evaluate(model, test_loader, criterion, device)
    metrics = macro_metrics(y_true_test, y_pred_test, num_classes=config.model.num_classes)
    metrics["test_loss"] = test_loss

    pred_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for x, y, clip_ids in test_loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_np = y.numpy()
            for i, clip_id in enumerate(clip_ids):
                pred_rows.append(
                    {
                        "clip_id": clip_id,
                        "y_true": int(y_np[i]),
                        "y_pred": int(preds[i]),
                        **{f"prob_{class_id}": float(probs[i, class_id]) for class_id in range(probs.shape[1])},
                    }
                )

    _save_artifacts(config, run_dir, history, metrics, pred_rows)
    refresh_latest_alias(Path(config.output_dir), run_dir)
    return {"run_dir": str(run_dir), "metrics": metrics}


def _save_artifacts(
    config: VerifierTrainConfig,
    run_dir: Path,
    history: dict[str, list[float]],
    metrics: dict[str, Any],
    pred_rows: list[dict[str, Any]],
) -> None:
    """Write required experiment tracking artifacts."""
    config_snapshot = asdict(config)
    config_snapshot["git_hash"] = _git_hash()
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    label_map_path = Path(config.manifest_path).parent / "label_map.json"
    label_map: dict[str, str] = {}
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

    pred_df = pd.DataFrame(pred_rows)
    if not pred_df.empty:
        pred_df["y_true_name"] = pred_df["y_true"].map(lambda value: label_map.get(str(int(value)), str(int(value))))
        pred_df["y_pred_name"] = pred_df["y_pred"].map(lambda value: label_map.get(str(int(value)), str(int(value))))
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    class_dist = pred_df["y_true"].value_counts().sort_index().to_dict() if not pred_df.empty else {}
    (run_dir / "class_distribution.json").write_text(json.dumps(class_dist, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(history["val_macro_f1"], label="val_macro_f1")
    axes[1].legend()
    axes[1].set_title("Val Macro F1")
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png")
    plt.close(fig)

    num_classes = int(config.model.num_classes)
    default_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    conf = np.asarray(metrics.get("confusion_matrix", default_conf.tolist()))
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(conf, cmap="Blues")
    plt.colorbar(im, ax=ax2)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Pred")
    ax2.set_ylabel("True")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax2.text(j, i, str(conf[i, j]), ha="center", va="center")
    fig2.tight_layout()
    fig2.savefig(run_dir / "confusion_matrix.png")
    plt.close(fig2)

    run_log = "\n".join(
        [
            f"run_dir={run_dir}",
            f"macro_f1={metrics.get('macro_f1')}",
            f"focus_class_precision={metrics.get('focus_class_precision')}",
            f"focus_class_recall={metrics.get('focus_class_recall')}",
        ]
    )
    (run_dir / "run.log").write_text(run_log, encoding="utf-8")

    manifest_df = pd.read_csv(config.manifest_path)
    dataset_summary = {
        "manifest_path": config.manifest_path,
        "samples_total": int(len(manifest_df)),
        "labels": label_map,
        "synthetic_vs_real": manifest_df["synthetic_or_real"].value_counts().to_dict() if "synthetic_or_real" in manifest_df.columns else {},
        "source_distribution": manifest_df["source_dataset"].value_counts().to_dict() if "source_dataset" in manifest_df.columns else {},
    }
    (run_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")


def _git_hash() -> str:
    """Best-effort git hash for tracking."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"
