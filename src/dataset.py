from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.camargo_loader import TrialData, load_trials_from_config
from src.utils import compute_stats, load_config, resolve_path, sign_vector, standardize


@dataclass(frozen=True)
class WindowIndex:
    trial_idx: int
    start: int
    target_idx: int
    mirrored: bool


def _config_hash(config: dict[str, Any]) -> str:
    text = repr(
        {
            "data": config.get("data", {}),
            "window": config.get("window", {}),
        }
    ).encode("utf-8")
    return hashlib.sha1(text).hexdigest()[:10]


def cache_trials(config: dict[str, Any], force: bool = False) -> Path:
    cache_dir = resolve_path(config["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"trials_{_config_hash(config)}.npz"
    if cache_path.exists() and not force:
        return cache_path
    trials = load_trials_from_config(config)
    if not trials:
        raise RuntimeError("No trials found. Check data.root and trial_glob in the config.")
    payload: dict[str, Any] = {"num_trials": np.array(len(trials), dtype=np.int64)}
    for i, trial in enumerate(trials):
        payload[f"x_{i}"] = trial.x.astype(np.float32)
        payload[f"y_{i}"] = trial.y.astype(np.float32)
        payload[f"feature_names_{i}"] = np.array(trial.feature_names, dtype=object)
        payload[f"target_names_{i}"] = np.array(trial.target_names, dtype=object)
        payload[f"participant_{i}"] = np.array(trial.participant, dtype=object)
        payload[f"trial_id_{i}"] = np.array(trial.trial_id, dtype=object)
        payload[f"task_{i}"] = np.array(trial.task, dtype=object)
        payload[f"side_{i}"] = np.array(trial.side or "", dtype=object)
        payload[f"sample_rate_hz_{i}"] = np.array(trial.sample_rate_hz, dtype=np.float32)
        payload[f"mass_kg_{i}"] = np.array(np.nan if trial.mass_kg is None else trial.mass_kg, dtype=np.float32)
    np.savez_compressed(cache_path, **payload)
    return cache_path


def load_cached_trials(cache_path: str | Path) -> list[TrialData]:
    data = np.load(cache_path, allow_pickle=True)
    trials: list[TrialData] = []
    for i in range(int(data["num_trials"])):
        mass = float(data[f"mass_kg_{i}"])
        trials.append(
            TrialData(
                x=data[f"x_{i}"].astype(np.float32),
                y=data[f"y_{i}"].astype(np.float32),
                feature_names=[str(v) for v in data[f"feature_names_{i}"].tolist()],
                target_names=[str(v) for v in data[f"target_names_{i}"].tolist()],
                sample_rate_hz=float(data[f"sample_rate_hz_{i}"]),
                participant=str(data[f"participant_{i}"].item()),
                trial_id=str(data[f"trial_id_{i}"].item()),
                task=str(data[f"task_{i}"].item()),
                side=str(data[f"side_{i}"].item()) or None,
                mass_kg=None if np.isnan(mass) else mass,
            )
        )
    return trials


def split_trials(config: dict[str, Any], trials: list[TrialData]) -> tuple[list[int], list[int]]:
    val_cfg = config["data"].get("validation", {})
    mode = val_cfg.get("mode", "random")
    participants = np.array([t.participant for t in trials])
    unique = sorted(set(participants.tolist()))
    if mode == "lopo":
        holdout = val_cfg.get("lopo_subject") or unique[-1]
        val_idx = [i for i, p in enumerate(participants) if p == holdout]
        train_idx = [i for i, p in enumerate(participants) if p != holdout]
        return train_idx, val_idx
    rng = np.random.default_rng(int(config.get("seed", 42)))
    shuffled = unique[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * float(val_cfg.get("val_fraction", 0.2)))))
    val_subjects = set(shuffled[:n_val])
    val_idx = [i for i, p in enumerate(participants) if p in val_subjects]
    train_idx = [i for i, p in enumerate(participants) if p not in val_subjects]
    if not train_idx or not val_idx:
        order = np.arange(len(trials))
        rng.shuffle(order)
        n_val_trials = max(1, int(round(len(order) * float(val_cfg.get("val_fraction", 0.2)))))
        val_idx = order[:n_val_trials].tolist()
        train_idx = order[n_val_trials:].tolist()
    return train_idx, val_idx


class CamargoWindowDataset(Dataset):
    def __init__(
        self,
        trials: list[TrialData],
        trial_indices: list[int],
        config: dict[str, Any],
        *,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        fit_normalizer: bool = False,
        limit_windows: int | None = None,
    ) -> None:
        self.trials = trials
        self.trial_indices = list(trial_indices)
        self.config = config
        self.window_len = int(config["window"]["length"])
        self.stride = int(config["window"].get("stride", 1))
        self.label_at = config["window"].get("label_at", "last")
        self.include_left_mirror = bool(config["data"].get("include_left_mirror", True))

        if fit_normalizer:
            mean, std = compute_stats(trials[i].x for i in self.trial_indices)
        if mean is None or std is None:
            raise ValueError("mean/std must be provided unless fit_normalizer=True")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        if not trials:
            raise ValueError("No trials supplied.")
        self.feature_names = trials[0].feature_names
        self.target_names = trials[0].target_names
        self.feature_signs = sign_vector(self.feature_names, config["data"].get("mirror_feature_sign_patterns", []))
        self.target_signs = sign_vector(self.target_names, config["data"].get("mirror_target_sign_patterns", []))
        self.index = self._build_index()
        if limit_windows is not None:
            self.index = self.index[: int(limit_windows)]

    def _build_index(self) -> list[WindowIndex]:
        out: list[WindowIndex] = []
        for trial_idx in self.trial_indices:
            trial = self.trials[trial_idx]
            n = min(trial.x.shape[0], trial.y.shape[0])
            if n < self.window_len:
                continue
            target_offset = self.window_len - 1 if self.label_at == "last" else self.window_len // 2
            starts = range(0, n - self.window_len + 1, self.stride)
            is_left = (trial.side or "").lower() == "left"
            for start in starts:
                target_idx = start + target_offset
                if not np.isfinite(trial.y[target_idx]).all():
                    continue
                out.append(WindowIndex(trial_idx, start, target_idx, False))
                if self.include_left_mirror and is_left:
                    out.append(WindowIndex(trial_idx, start, target_idx, True))
        if not out:
            raise ValueError("No windows were created. Check window.length versus trial lengths.")
        return out

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        wi = self.index[idx]
        trial = self.trials[wi.trial_idx]
        x = trial.x[wi.start : wi.start + self.window_len].astype(np.float32)
        y = trial.y[wi.target_idx].astype(np.float32)
        if wi.mirrored:
            x = x * self.feature_signs[None, :]
            y = y * self.target_signs
        x = standardize(x, self.mean, self.std)
        return {
            "x": torch.from_numpy(x.T.copy()),
            "y": torch.from_numpy(y.copy()),
            "participant": trial.participant,
            "trial_id": trial.trial_id,
            "task": trial.task,
        }


def build_datasets(config: dict[str, Any], force_cache: bool = False) -> tuple[CamargoWindowDataset, CamargoWindowDataset]:
    cache_path = cache_trials(config, force=force_cache)
    trials = load_cached_trials(cache_path)
    train_idx, val_idx = split_trials(config, trials)
    train_ds = CamargoWindowDataset(
        trials,
        train_idx,
        config,
        fit_normalizer=True,
        limit_windows=config["train"].get("limit_train_windows"),
    )
    val_ds = CamargoWindowDataset(
        trials,
        val_idx,
        config,
        mean=train_ds.mean,
        std=train_ds.std,
        limit_windows=config["train"].get("limit_val_windows"),
    )
    return train_ds, val_ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_ds, val_ds = build_datasets(cfg, force_cache=args.force_cache)
    sample = train_ds[0]
    print(f"train windows: {len(train_ds)}")
    print(f"val windows:   {len(val_ds)}")
    print(f"x shape:       {tuple(sample['x'].shape)}")
    print(f"y shape:       {tuple(sample['y'].shape)}")
    print(f"features:      {len(train_ds.feature_names)}")
    print(f"targets:       {train_ds.target_names}")


if __name__ == "__main__":
    main()
