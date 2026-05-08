#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
Camargo 2021 lower-limb biomechanics dataset
============================================

The data are hosted publicly on Dropbox:
https://www.dropbox.com/sh/lhurwcy0znonh56/AAAPmVdrxh7M6FW-UYHyPHyza?dl=0

This repository does not redistribute the dataset. Download and extract it
manually into a local folder outside cloud-sync directories. Recommended:

  $CAMARGO_DATA_ROOT/raw/

Expected shape:

  $CAMARGO_DATA_ROOT/raw/
    AB01/
      ... trial .mat files ...
    AB02/
      ... trial .mat files ...
    ...

The loader recursively scans each subject folder for .mat files. If your local
folder names differ, update configs/default.yaml:

  data.root
  data.participant_glob
  data.trial_glob

After download, inspect one trial before training:

  python data/camargo_loader.py --inspect data/raw/AB01/path/to/trial.mat

Then run:

  python src/train.py --config configs/default.yaml
  python src/evaluate.py --config configs/default.yaml
MSG
