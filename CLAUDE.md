# CLAUDE.md

## Project contract

SideNet is a post-detection, per-cone boundary-side classifier for FSD. The recommended baseline is DGCNN.

- Target: `track_global` directed boundary side.
- Input frame: `ego` / LiDAR, `+x forward`, `+y left`.
- Default input: `xyz`; generator scores are constants and must not masquerade as detector confidence.
- Output: class probability aligned one-to-one with input detections (`0=Left`, `1=Right`).
- Reverse driving: track-global labels stay fixed; ego-relative labels swap. Do not conflate the two.
- Reflection augmentation: mirror ego Y and swap Left/Right.

The full rationale and unresolved system-level risks are in `docs/ENGINEERING_REVIEW.md`.

## Commands

```bash
# Generate manifest-backed ego-frame data first
cd ../bitfsd-generator
.venv/bin/python main.py collect --config config/perceive.yaml

# Train/evaluate from SideNet
cd ../SideNet
python train_sidenet.py configs/dgcnn.yaml
python infer.py --ckpt runs/dgcnn/dgcnn.pth \
  --data-dir ../bitfsd-generator/output/sidenet_data_ego

# Tests (global ROS plugins may require autoload to be disabled)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Data and split invariants

- `dataset_manifest.yaml` controls which track directories belong to the corpus.
- Every track directory must contain `metadata.yaml` with `coordinate_frame` and `side_semantics`.
- Missing metadata fails closed unless a legacy config explicitly opts in.
- Default split is deterministic `track_holdout`; derived track variants stay in the same family. `training.seed` and `split.seed` are separate so model repeats share one fold.
- `split_manifest.json` records exact frame ids and a dataset SHA-256.
- `configs/dgcnn_legacy_random.yaml` is reproduction-only and must not be used for generalization claims.

## Training invariants

- Do not reintroduce cross-frame zero padding into DGCNN/PointNet/PointNet++.
- `input_mode` must be applied through `select_input_features` before model forward.
- Save both best-selection-metric and last-epoch states.
- Report micro, macro-frame, macro-track, and worst-track metrics.
- The train-reference k-NN baseline must never read validation-frame neighbor GT labels.

## Detector integration

OpenPCDet `pred_boxes[:, :3]` are LiDAR-frame coordinates and can be passed to `infer.SideNetPredictor.predict_detected_cones` with `pred_scores`. A checkpoint coordinate mismatch is an error; never bypass it by relabeling the frame in configuration.

False positives have no binary side ground truth. Use detector filtering, confidence-based abstention, boundary-association rejection, or explicitly upgrade the whole contract to `Cone_Unknown`/three classes.

## Architecture notes

`sidenet.py` contains Transformer, DGCNN, PointNet++, and PointNet plus a shared `build_model` factory. DGCNN is geometric deep learning but not strictly SE(2)-equivariant. Do not add a steerable/equivariant stack until ego-frame track-holdout stress tests isolate a transform-specific failure.
