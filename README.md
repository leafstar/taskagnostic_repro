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

## Pretrained Checkpoint And Prediction

This repository includes one small pretrained checkpoint:

```text
pretrained/camargo_ab06_ab10_lopo_AB10_best.pt
```

This is the best checkpoint from the first five-subject Camargo subset currently
available locally. It was trained on `AB06`, `AB07`, `AB08`, and `AB09`, then
validated on held-out `AB10`. It is useful as a working deployment example, but
it should not be described as the final 22-subject LOPO model.

The checkpoint contains:

- TCN model weights
- model/config metadata
- input feature names
- target names
- training-set feature mean/std for normalization
- validation metrics from the saved epoch

The model expects a numeric feature array shaped:

```text
T x 36
```

where columns must follow this exact order:

```text
0  ik.hip_flexion_r
1  ik.knee_angle_r
2  ik.ankle_angle_r
3  ik.hip_flexion_l
4  ik.knee_angle_l
5  ik.ankle_angle_l
6  ik.hip_flexion_r_vel
7  ik.knee_angle_r_vel
8  ik.ankle_angle_r_vel
9  ik.hip_flexion_l_vel
10 ik.knee_angle_l_vel
11 ik.ankle_angle_l_vel
12 imu.foot_Accel_X
13 imu.foot_Accel_Y
14 imu.foot_Accel_Z
15 imu.foot_Gyro_X
16 imu.foot_Gyro_Y
17 imu.foot_Gyro_Z
18 imu.shank_Accel_X
19 imu.shank_Accel_Y
20 imu.shank_Accel_Z
21 imu.shank_Gyro_X
22 imu.shank_Gyro_Y
23 imu.shank_Gyro_Z
24 imu.thigh_Accel_X
25 imu.thigh_Accel_Y
26 imu.thigh_Accel_Z
27 imu.thigh_Gyro_X
28 imu.thigh_Gyro_Y
29 imu.thigh_Gyro_Z
30 imu.trunk_Accel_X
31 imu.trunk_Accel_Y
32 imu.trunk_Accel_Z
33 imu.trunk_Gyro_X
34 imu.trunk_Gyro_Y
35 imu.trunk_Gyro_Z
```

The prediction output is shaped:

```text
(T - 250 + 1) x 2
```

because the TCN uses a 250-sample causal window and predicts the label at the
last frame of each window. The two output columns are:

```text
id.hip_flexion_r_moment
id.knee_angle_r_moment
```

Both outputs are biological joint moments in `Nm/kg`.

Run prediction on a `.npy`, `.npz`, or numeric `.csv` feature array:

```bash
python src/predict.py \
  --checkpoint pretrained/camargo_ab06_ab10_lopo_AB10_best.pt \
  --input features.npy \
  --output predictions.csv \
  --print-metadata
```

For `.npz` input, the array must be stored under the key `features`. The script
loads the checkpoint, applies the saved training-set normalization, builds
sliding windows, and writes hip/knee moment predictions.

Validation metrics for this checkpoint on held-out `AB10`:

```text
hip R^2      0.773
knee R^2     0.736
hip RMSE     0.184 Nm/kg
knee RMSE    0.203 Nm/kg
```

Benchmark inference latency:

```bash
python tools/benchmark_inference.py \
  --checkpoint pretrained/camargo_ab06_ab10_lopo_AB10_best.pt \
  --device cuda
```

On an RTX 4080 SUPER, the pretrained model takes about `1.9 ms` for a single
`1 x 36 x 250` window and over `100k windows/s` when batched. CPU single-window
latency is about `2.5 ms` on the local test machine.

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
