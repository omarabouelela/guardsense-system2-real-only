"""Training loop for Trigger pose model."""

from __future__ import annotations

import json
import logging
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.trigger_dataset import TriggerPoseDataset
from src.common.runtime_utils import refresh_latest_alias, resolve_device
from src.trigger.model import TriggerModelConfig, build_trigger_model

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainConfig:
    """Train configuration values."""

    dataset_path: str
    output_dir: str
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    class_weights: list[float] | None = None
    model: TriggerModelConfig = field(default_factory=TriggerModelConfig)


def set_seed(seed: int) -> None:
    """Enable deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> dict[str, Any]:
    """Compute classification metrics without third-party metric libs."""
    if y_true.size == 0 or y_pred.size == 0:
        empty_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {str(class_id): {"precision": 0.0, "recall": 0.0, "f1": 0.0} for class_id in range(num_classes)},
            "focus_class_precision": 0.0,
            "focus_class_recall": 0.0,
            "label_0_false_positives": 0,
            "cross_class_confusions": 0,
            "confusion_matrix": empty_conf.tolist(),
            "warnings": ["empty_ground_truth_or_predictions"],
        }

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
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
        per_class[str(class_id)] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

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


def load_hdf5_splits(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load expected split arrays from preprocessed HDF5."""
    import h5py

    with h5py.File(path, "r") as h5f:
        return (
            h5f["X_train"][()],
            h5f["y_train"][()],
            h5f["X_val"][()],
            h5f["y_val"][()],
            h5f["X_test"][()],
            h5f["y_test"][()],
        )


def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    """Run model on loader and return loss + predictions."""
    model.eval()
    losses: list[float] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            preds = torch.argmax(logits, dim=-1)
            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    if not all_true:
        return float(np.mean(losses)) if losses else 0.0, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return float(np.mean(losses)) if losses else 0.0, np.concatenate(all_true), np.concatenate(all_pred)


def train_trigger(config: TrainConfig) -> dict[str, Any]:
    """Main training entrypoint with run artifact generation."""
    set_seed(config.seed)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"trigger_binary_real_only_{config.model.architecture}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    device = resolve_device(config.device)
    LOGGER.info("Using device=%s", device)

    x_train, y_train, x_val, y_val, x_test, y_test = load_hdf5_splits(Path(config.dataset_path))
    train_ds = TriggerPoseDataset(x_train, y_train)
    val_ds = TriggerPoseDataset(x_val, y_val)
    test_ds = TriggerPoseDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_trigger_model(config.model).to(device)
    class_weights = torch.tensor(config.class_weights, dtype=torch.float32, device=device) if config.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_train_loss = float("inf")
    stale_epochs = 0
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss, y_true_val, y_pred_val = evaluate_epoch(model, val_loader, criterion, device)
        val_metrics = macro_metrics(y_true_val, y_pred_val, num_classes=config.model.num_classes)
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_has_samples = y_true_val.size > 0
        if val_has_samples:
            scheduler.step(val_macro_f1)
        else:
            LOGGER.warning(
                "Validation loader produced zero samples at epoch %d; using train_loss fallback for checkpoint selection.",
                epoch,
            )

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

        is_better = (val_macro_f1 > best_macro_f1) if val_has_samples else (train_loss < best_train_loss)
        if is_better:
            best_macro_f1 = val_macro_f1 if val_has_samples else best_macro_f1
            best_train_loss = train_loss if not val_has_samples else best_train_loss
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(config.model)}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                LOGGER.info("Early stopping triggered at epoch %d", epoch)
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, y_true_test, y_pred_test = evaluate_epoch(model, test_loader, criterion, device)
    test_metrics = macro_metrics(y_true_test, y_pred_test, num_classes=config.model.num_classes)
    test_metrics["test_loss"] = test_loss
    if y_true_test.size == 0:
        LOGGER.warning("Test loader produced zero samples; returning zeroed test metrics.")

    probs_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        start_idx = 0
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_np = y.numpy()
            for i in range(len(preds)):
                probs_rows.append(
                    {
                        "index": start_idx + i,
                        "y_true": int(y_np[i]),
                        "y_pred": int(preds[i]),
                        **{f"prob_{class_id}": float(probs[i, class_id]) for class_id in range(probs.shape[1])},
                    }
                )
            start_idx += len(preds)

    _save_run_artifacts(config, run_dir, history, test_metrics, probs_rows)
    refresh_latest_alias(output_root, run_dir)
    return {"run_dir": str(run_dir), "best_macro_f1": best_macro_f1, "test_metrics": test_metrics}


def _save_run_artifacts(
    config: TrainConfig,
    run_dir: Path,
    history: dict[str, list[float]],
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    """Save required experiment tracking artifacts."""
    config_path = run_dir / "config_snapshot.yaml"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    class_distribution_path = run_dir / "class_distribution.json"
    curves_path = run_dir / "training_curves.png"
    confusion_path = run_dir / "confusion_matrix.png"
    run_log_path = run_dir / "run.log"
    dataset_summary_path = run_dir / "dataset_summary.json"

    git_hash = _git_hash()
    config_dict = asdict(config)
    config_dict["git_hash"] = git_hash
    config_path.write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    import pandas as pd

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(predictions_path, index=False)

    class_dist = pred_df["y_true"].value_counts().sort_index().to_dict() if not pred_df.empty else {}
    class_distribution_path.write_text(json.dumps(class_dist, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(history["val_macro_f1"], label="val_macro_f1")
    axes[1].legend()
    axes[1].set_title("Val Macro F1")
    fig.tight_layout()
    fig.savefig(curves_path)
    plt.close(fig)

    num_classes = int(config.model.num_classes)
    default_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    conf = np.array(metrics.get("confusion_matrix", default_conf.tolist()))
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(conf, cmap="Blues")
    plt.colorbar(im, ax=ax2)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Pred")
    ax2.set_ylabel("True")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax2.text(j, i, str(conf[i, j]), ha="center", va="center", color="black")
    fig2.tight_layout()
    fig2.savefig(confusion_path)
    plt.close(fig2)

    run_log_path.write_text(
        "\n".join(
            [
                f"run_dir={run_dir}",
                f"macro_f1={metrics.get('macro_f1')}",
                f"focus_class_precision={metrics.get('focus_class_precision')}",
                f"focus_class_recall={metrics.get('focus_class_recall')}",
            ]
        ),
        encoding="utf-8",
    )

    dataset_summary = {
        "dataset_path": config.dataset_path,
        "class_distribution": class_dist,
        "synthetic_vs_real": "available in preprocessing metadata",
    }
    dataset_summary_path.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")


def _git_hash() -> str:
    """Try to capture git hash placeholder for experiment tracking."""
    try:
        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return output
    except Exception:  # noqa: BLE001
        return "unknown"
