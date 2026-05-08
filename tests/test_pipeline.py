from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.camargo_loader import TrialData
from src.dataset import CamargoWindowDataset
from src.model import TCNMomentEstimator


def _config() -> dict:
    return {
        "data": {
            "include_left_mirror": True,
            "mirror_feature_sign_patterns": [],
            "mirror_target_sign_patterns": [],
        },
        "window": {"length": 50, "stride": 10, "label_at": "last"},
    }


def test_window_dataset_shapes() -> None:
    rng = np.random.default_rng(0)
    trials = [
        TrialData(
            x=rng.normal(size=(200, 12)).astype(np.float32),
            y=rng.normal(size=(200, 2)).astype(np.float32),
            feature_names=[f"f{i}" for i in range(12)],
            target_names=["hip", "knee"],
            sample_rate_hz=200,
            participant="AB01",
            trial_id="trial_01_left",
            task="cyclic",
            side="left",
        )
    ]
    ds = CamargoWindowDataset(trials, [0], _config(), fit_normalizer=True)
    sample = ds[0]
    assert sample["x"].shape == (12, 50)
    assert sample["y"].shape == (2,)
    assert len(ds) == 32


def test_tcn_forward_shape() -> None:
    model = TCNMomentEstimator(input_dim=12, output_dim=2, filters=16, kernel_size=3, num_blocks=3)
    x = torch.randn(4, 12, 50)
    y = model(x)
    assert y.shape == (4, 2)
