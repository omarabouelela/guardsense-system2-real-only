"""Runtime path and device resolution helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    """Resolve a user/device config string to a concrete torch device.

    - ``auto`` selects CUDA when available, else CPU.
    - explicit values (``cpu``, ``cuda``, ``cuda:0``...) are respected.
    """
    normalized = device.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _run_dirs_sorted(base_dir: Path, run_prefix: str) -> list[Path]:
    """Return run directories for a prefix sorted by modification time descending."""
    if not base_dir.exists():
        return []
    candidates = [path for path in base_dir.glob(f"{run_prefix}*") if path.is_dir()]
    if not candidates:
        return []
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _latest_run_dir(base_dir: Path, run_prefix: str) -> Path | None:
    """Return latest run directory by modification time for a run prefix."""
    candidates = _run_dirs_sorted(base_dir, run_prefix)
    return candidates[0] if candidates else None


def latest_checkpoint(base_dir: str | Path, run_prefix: str) -> Path | None:
    """Return best_model.pt from newest valid run directory if available."""
    for run_dir in _run_dirs_sorted(Path(base_dir), run_prefix=run_prefix):
        checkpoint = run_dir / "best_model.pt"
        if checkpoint.exists():
            return checkpoint
    return None


def resolve_checkpoint_path(model_path: str | Path, base_dir: str | Path, run_prefix: str) -> Path:
    """Resolve configured checkpoint path with backwards-compatible fallbacks.

    If ``model_path`` exists, it is returned directly.
    If it points to `.../latest/...`, resolve to latest successful run checkpoint.
    If missing, attempt latest checkpoint under ``base_dir``.
    """
    configured = Path(model_path)
    if configured.exists():
        if configured.is_dir():
            candidate = configured / "best_model.pt"
            if candidate.exists():
                return candidate
        return configured

    configured_posix = configured.as_posix()
    if "/latest/" in configured_posix or configured_posix.endswith("/latest"):
        latest_marker = Path(base_dir) / "latest" / "LATEST_RUN_PATH.txt"
        if latest_marker.exists():
            marker_target = Path(latest_marker.read_text(encoding="utf-8").strip())
            marker_ckpt = marker_target / "best_model.pt"
            if marker_ckpt.exists():
                return marker_ckpt
        latest = latest_checkpoint(base_dir=base_dir, run_prefix=run_prefix)
        if latest is not None:
            return latest

    latest = latest_checkpoint(base_dir=base_dir, run_prefix=run_prefix)
    if latest is not None:
        return latest

    raise FileNotFoundError(
        f"Could not resolve checkpoint path. configured={configured} base_dir={Path(base_dir)} run_prefix={run_prefix}"
    )


def refresh_latest_alias(output_root: str | Path, run_dir: str | Path) -> Path:
    """Update output_root/latest as a symlink (or copied marker path fallback).

    Keeps timestamped directories intact while providing a stable pointer.
    """
    output_root_path = Path(output_root)
    run_dir_path = Path(run_dir)
    latest_path = output_root_path / "latest"
    latest_path.unlink(missing_ok=True)
    try:
        relative_target = os.path.relpath(run_dir_path, output_root_path)
        latest_path.symlink_to(relative_target, target_is_directory=True)
    except OSError:
        # Symlink may fail on some Windows setups; write marker file fallback.
        latest_path.mkdir(parents=True, exist_ok=True)
        (latest_path / "LATEST_RUN_PATH.txt").write_text(str(run_dir_path), encoding="utf-8")
    return latest_path


def resolve_raw_data_path(path_like: str | Path) -> Path:
    """Resolve dataset/raw and data/raw conventions in a compatible way."""
    raw_path = Path(path_like)
    if raw_path.exists():
        return raw_path

    text = raw_path.as_posix()
    candidates: list[Path] = []
    if text.startswith("data/raw/"):
        candidates.append(Path(text.replace("data/raw/", "dataset/raw/", 1)))
    if text.startswith("dataset/raw/"):
        candidates.append(Path(text.replace("dataset/raw/", "data/raw/", 1)))

    for candidate in candidates:
        if candidate.exists():
            LOGGER.info("Resolved raw path %s -> %s", raw_path, candidate)
            return candidate
    return raw_path
