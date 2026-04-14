# AGENTS.md

## Project overview
This repository contains the real-data-only binary version of GuardSense Fit System 2.

Its purpose is to train and evaluate the Trigger and Verifier models using only the currently collected real datasets, while keeping the original synthetic-ready 3-class repository untouched.

## Core rules
- This is the real-only binary repo.
- Do not assume Simuletic or synthetic data is available in this repository.
- Trigger model is pose-based and trains on .npy or .hdf5.
- Verifier model is video-based and trains on .mp4 or .avi.
- Labels in this repository are:
  - 0 = normal / non-fight
  - 1 = fight
- Keep the code future-compatible with later expansion back to 3 classes.
- Keep the repository modular and production-oriented.
- Do not rewrite the whole project unless absolutely necessary.
- Prefer surgical edits over large rewrites.

## Engineering rules
- Python 3.12.3
- PyTorch
- pathlib
- type hints
- docstrings
- robust logging
- no pseudocode
- prefer maintainable code over fancy code

## Data rules
- Trigger shape: (B, T, K, C)
- T = 32, K = 17 COCO keypoints, C = (x, y, visibility)
- Trigger uses 32-frame windows with 50% overlap
- Trigger should support real pose extracted from real videos
- Verifier clips should be standardized to 30 FPS, 640x360 or 480p, 2 to 4 seconds
- Preserve visibility flags
- Normalize keypoints

## Dataset assumptions
This repository should support these real datasets:

- RWF-2000
  - videos/
  - pose/   (generated from videos)

- AIRTLab
  - videos/
  - pose/   (generated from videos)

- XD-Violence
  - videos/
  - selected_clips/
  - pose/   (optional later)

- Bullying10K
  - npy/

- classroom_hard_negative
  - videos/
  - pose/   (optional later)

## Model rules
- Trigger model num_classes must be configurable
- Verifier model num_classes must be configurable
- Label maps must be dynamic
- Metrics and confusion matrices must be dynamic
- The repository should work correctly in binary mode first

## Runtime rules
- Frigate/runtime code may remain in the repository, but it is not the priority for this binary real-only version.
- Training and evaluation must not depend on runtime/fusion being active.
- If runtime code is preserved, keep it clearly separated from the training flow.
