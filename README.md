# Task-Agnostic Biological Joint Moment Estimator Reproduction

This project reproduces the TCN-style biological joint moment estimator from
Molinaro et al., Nature 2024, using the fully public Camargo 2021 lower-limb
biomechanics dataset instead of the restricted Georgia Tech dataset and
restricted Code Ocean code.

Scope: this repository only builds the wearable-sensor-to-hip/knee-moment
estimator. It does not include reinforcement learning or MuJoCo/Warp
integration.

## Data Location

Do not put the 22 GB dataset inside a cloud-sync folder. The default config
reads the data root from an environment variable:

```text
CAMARGO_DATA_ROOT
```

Recommended local layout:

```text
$CAMARGO_DATA_ROOT/
  raw/
  processed/
```

Set the variable before running the scripts. On PowerShell:

```powershell
$env:CAMARGO_DATA_ROOT = "$env:USERPROFILE\Documents\taskagnostic_repro_data"
```

To persist it for new terminals on Windows:

```powershell
[Environment]::SetEnvironmentVariable("CAMARGO_DATA_ROOT", "$env:USERPROFILE\Documents\taskagnostic_repro_data", "User")
```

Download the public Dropbox release manually:

https://www.dropbox.com/sh/lhurwcy0znonh56/AAAPmVdrxh7M6FW-UYHyPHyza?dl=0

Extract so the folder resembles:

```text
$CAMARGO_DATA_ROOT/raw/
  AB01/
    10_09_18/
      levelground/
        ik/
        id/
        imu/
        fp/
  AB02/
    ...
```

Processed `.npz` caches are written to:

```text
$CAMARGO_DATA_ROOT/processed
```

## Install

Use Python 3.10+ and PyTorch 2.x.

```bash
cd taskagnostic_repro
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

On Linux/macOS, use `source .venv/bin/activate`.

## Inspect One Trial First

The Camargo dataset has appeared in several MATLAB table/object layouts. The
loader therefore supports broad regex-based channel discovery, but you should
inspect one local trial and refine `configs/default.yaml` if needed.

```bash
python data/camargo_loader.py --inspect "$CAMARGO_DATA_ROOT/raw/AB01/path/to/trial.mat"
```

Look for fields corresponding to:

- hip/knee/ankle sagittal joint angles
- hip/knee/ankle angular velocities
- pelvis, thigh, shank, and foot IMU accelerometer/gyroscope channels
- hip and knee inverse-dynamics joint moments
- optional vGRF/COP channels if present
- subject mass or body weight for Nm/kg scaling

Then update these config sections:

```yaml
data:
  feature_patterns:
  target_patterns:
```

The first working Camargo version uses 36 input channels:

- 12 IK channels: left/right hip, knee, ankle angles and finite-difference velocities
- 24 IMU channels: foot, shank, thigh, trunk accelerometer and gyro triples

The output is 2 channels:

- right hip flexion moment
- right knee flexion moment

Targets are scaled to Nm/kg from the subject mass parsed out of `ABxx.osim`.

See `docs/data_pipeline.md` for a diagram of the data flow and tensor shapes.

## Run A Dataset Smoke Check

```bash
python src/dataset.py --config configs/default.yaml --force-cache
```

Expected output includes train/val window counts, input shape `(36, 250)`, and
target shape `(2,)`.

## Train

Quick random split:

```bash
python src/train.py --config configs/default.yaml
```

The default hyperparameters follow the requested estimator:

- causal dilated TCN
- 5 residual blocks
- 64 filters
- kernel size 5
- dropout 0.2
- Adam, learning rate `1e-4`
- 250-sample windows at 200 Hz
- MSE loss

Outputs:

```text
runs/default/                  TensorBoard logs
checkpoints/default/best.pt    best validation checkpoint
checkpoints/default/history.csv
checkpoints/default/run_metadata.json
```

View curves:

```bash
tensorboard --logdir runs
```

## Evaluate

```bash
python src/evaluate.py --config configs/default.yaml
```

This prints and saves:

```text
reports/eval_metrics.csv
```

Metrics are reported for:

- all validation windows
- each validation participant
- each task label inferred from the file path: `cyclic`, `stair`, `ramp`, or `unknown`

Each row includes hip/knee R^2 and hip/knee RMSE in Nm/kg.

## LOPO Evaluation

For leave-one-participant-out, edit:

```yaml
data:
  validation:
    mode: lopo
    lopo_subject: AB01
```

Run training/evaluation once per participant. Keep separate checkpoint/log
directories per fold, for example:

```yaml
train:
  log_dir: runs/lopo_AB01
  checkpoint_dir: checkpoints/lopo_AB01
evaluate:
  checkpoint: checkpoints/lopo_AB01/best.pt
  output_csv: reports/lopo_AB01.csv
```

## Tests

The tests use synthetic data and verify the code path without requiring the
22 GB dataset:

```bash
pytest
```

## Current Loader Notes

`data/camargo_loader.py` can read:

- traditional MATLAB `.mat` via SciPy
- MATLAB v7.3/HDF5 via `mat73` or `h5py`
- MATLAB table objects saved as MCOS opaque records via `mat-io`

The public Camargo release stores modality files separately. The current loader
joins matching `ik`, `id`, and `imu` files by basename under each trial
folder, skips `static` files, and uses the `ABxx.osim` model to compute subject
mass for Nm/kg scaling.

Left-leg mirroring is currently a light-weight augmentation hook. The current
pipeline keeps the bilateral IK channels as features and uses the right-leg
hip/knee moment pair as targets. Once you confirm exact sign conventions, add
regexes to:

```yaml
data:
  mirror_feature_sign_patterns:
  mirror_target_sign_patterns:
```

## Acceptance Path

1. Download and extract at least the first five subjects into
   `$CAMARGO_DATA_ROOT/raw`.
2. Inspect one representative trial and refine feature/target patterns.
3. Run `python src/dataset.py --config configs/default.yaml --force-cache`.
4. Run `python src/train.py --config configs/default.yaml`.
5. Run `python src/evaluate.py --config configs/default.yaml`.
6. Expand to all 22 subjects and run LOPO folds.

The target final performance is average hip R^2 > 0.7 and knee R^2 > 0.7
under LOPO. Some degradation relative to the Nature 2024 paper is expected
because Camargo lacks the original task-agnostic dataset's unstructured task
diversity.
