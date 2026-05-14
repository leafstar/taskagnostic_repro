from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    p = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if p.is_absolute():
        return p
    return project_root() / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def finite_or_zero(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-8)


def compute_stats(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    mean = None
    m2 = None
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.size == 0:
            continue
        batch_count = arr.shape[0]
        batch_mean = arr.mean(axis=0)
        batch_m2 = ((arr - batch_mean) ** 2).sum(axis=0)
        if mean is None:
            count = batch_count
            mean = batch_mean
            m2 = batch_m2
            continue
        delta = batch_mean - mean
        total = count + batch_count
        mean = mean + delta * batch_count / total
        m2 = m2 + batch_m2 + delta**2 * count * batch_count / total
        count = total
    if mean is None or m2 is None or count < 2:
        raise ValueError("Cannot compute normalization statistics from empty data.")
    std = np.sqrt(m2 / max(count - 1, 1))
    return mean.astype(np.float32), np.maximum(std.astype(np.float32), 1e-8)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean(axis=0))


def short_target_label(name: str) -> str:
    base = str(name).split(".")[-1]
    base = re.sub(r"_moment$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"^hip_flexion_", "hip_", base, flags=re.IGNORECASE)
    base = re.sub(r"^knee_angle_", "knee_", base, flags=re.IGNORECASE)
    base = re.sub(r"^ankle_angle_", "ankle_", base, flags=re.IGNORECASE)
    base = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_").lower()
    return base or "target"


def target_metric_labels(target_names: list[str]) -> list[str]:
    labels: list[str] = []
    seen: dict[str, int] = {}
    for name in target_names:
        label = short_target_label(name)
        count = seen.get(label, 0)
        seen[label] = count + 1
        labels.append(label if count == 0 else f"{label}_{count + 1}")
    return labels


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns if p]


def sign_vector(names: list[str], patterns: Iterable[str]) -> np.ndarray:
    compiled = compile_patterns(patterns)
    signs = np.ones(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        if any(p.search(name) for p in compiled):
            signs[i] = -1.0
    return signs


def infer_side(path_or_name: str) -> str | None:
    text = str(path_or_name).lower()
    left_tokens = ["left", "_l_", "-l-", " l ", "lhs", "lt", "_l."]
    right_tokens = ["right", "_r_", "-r-", " r ", "rhs", "rt", "_r."]
    if any(t in text for t in left_tokens):
        return "left"
    if any(t in text for t in right_tokens):
        return "right"
    return None


def normalize_moment_by_mass(moment_nm: np.ndarray, mass_kg: float | None) -> np.ndarray:
    if mass_kg is None or not math.isfinite(float(mass_kg)) or float(mass_kg) <= 0:
        return moment_nm
    return moment_nm / float(mass_kg)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)
