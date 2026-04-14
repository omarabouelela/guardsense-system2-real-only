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

## Main configs

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
