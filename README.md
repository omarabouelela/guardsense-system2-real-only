# GuardSense Fit System 2 (Real-Only Binary V1)

This repository is the **separate real-data-only** GuardSense Fit System 2 codebase.

- **Mode**: `real_only_binary`
- **Primary labels**:
  - `0 = normal / non-fight`
  - `1 = fight`
- **Goal**: train/evaluate Trigger + Verifier on currently collected **real** datasets only.
- **Simuletic/synthetic data**: intentionally excluded from active configs and training flow in this repo.

## Pipelines

### Trigger (pose)
- Input: `.npy`, `.hdf5/.h5`, or YOLO pose `.txt`
- Tensor format: `(B, T, K, C)`
- Defaults: `T=32`, `K=17`, `C=3 (x, y, visibility)`
- Windowing: 32 frames, 50% overlap

### Verifier (RGB clips)
- Input: `.mp4` / `.avi`
- Standardization target: 30 FPS, 640x360 (or compatible), 2–4 seconds

## Real dataset layout target

```text
dataset/
  raw/
    rwf2000/
      videos/
      pose/
    airtlab/
      videos/
      pose/
    xd_violence/
      videos/
      selected_clips/
      pose/
    bullying10k/
      npy/
    classroom_hard_negative/
      videos/
      pose/
```

## Binary-first and future expansion

- Trigger and Verifier `num_classes` are config-driven.
- Label maps are config-driven and exported during preprocessing.
- Metrics/confusion matrices are dynamic by configured class count.
- This binary V1 is designed to remain compatible with later 3-class expansion.

## Runtime/Fusion status

Runtime/fusion code is preserved for downstream use, but **training/evaluation do not depend on it**.
In `configs/runtime.yaml`, `run_dual_inference` is disabled by default for real-only binary training workflows.

## Real video → pose extraction (Trigger)

Use `configs/pose_extraction.yaml` to extract YOLOv8-pose keypoints from real videos before running Trigger preprocessing.

- **Canonical output format**: one `.txt` per video sequence with one YOLO pose row per frame (class, bbox, 17 keypoints as `x y visibility/confidence`).
- **Supported extraction inputs**:
  - `dataset/raw/rwf2000/videos` → `dataset/raw/rwf2000/pose`
  - `dataset/raw/airtlab/videos` → `dataset/raw/airtlab/pose`
  - `dataset/raw/classroom_hard_negative/videos` → `dataset/raw/classroom_hard_negative/pose`
  - `dataset/raw/xd_violence/selected_clips` → `dataset/raw/xd_violence/pose` (optional)
- **Binary labels in extraction manifest**:
  - `0 = normal/non-fight`
  - `1 = fight`
  - Mapping is explicit in per-dataset `label_rules` / `fixed_label` in `configs/pose_extraction.yaml`.

### Extraction command

```bash
python -m src.scripts.extract_pose_from_videos --config configs/pose_extraction.yaml
```

This writes:
- pose `.txt` files under each dataset `pose/` folder
- extraction manifest CSV at `artifacts/pose_extraction/manifest.csv`
- JSON detail and summary files next to the manifest

Run Trigger preprocessing after extraction:

```bash
python -m src.scripts.prepare_trigger_data --config configs/trigger_data.yaml
```

## Main configs

- Pose extraction: `configs/pose_extraction.yaml`
- Trigger data: `configs/trigger_data.yaml`
- Trigger train/eval: `configs/trigger_train.yaml`, `configs/trigger_eval.yaml`
- Verifier data: `configs/verifier_data.yaml`
- Verifier train/eval: `configs/verifier_train.yaml`, `configs/verifier_eval.yaml`
- Full pipeline: `configs/runtime.yaml`

## Example commands

```bash
python -m src.scripts.prepare_trigger_data --config configs/trigger_data.yaml
python -m src.scripts.train_trigger --config configs/trigger_train.yaml
python -m src.scripts.eval_trigger --config configs/trigger_eval.yaml

python -m src.scripts.prepare_verifier_data --config configs/verifier_data.yaml
python -m src.scripts.train_verifier --config configs/verifier_train.yaml
python -m src.scripts.eval_verifier --config configs/verifier_eval.yaml

python -m src.scripts.run_full_pipeline --config configs/runtime.yaml --output-dir artifacts/full_pipeline --dry-run
```
