from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CamargoWindowDataset, cache_trials, load_cached_trials, split_trials
from src.model import build_model
from src.train import _collate
from src.utils import choose_device, load_config, r2_score, resolve_path, rmse


def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, list[Any]]:
    model.eval()
    out: dict[str, list[Any]] = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            pred = model(x).cpu().numpy()
            out["y_pred"].append(pred)
            out["y_true"].append(batch["y"].numpy())
            out["participant"].extend(batch["participant"])
            out["trial_id"].extend(batch["trial_id"])
            out["task"].extend(batch["task"])
    out["y_pred"] = [np.concatenate(out["y_pred"], axis=0)]
    out["y_true"] = [np.concatenate(out["y_true"], axis=0)]
    return out


def _metric_rows(preds: dict[str, list[Any]]) -> list[dict[str, Any]]:
    y_true = preds["y_true"][0]
    y_pred = preds["y_pred"][0]
    participants = np.array(preds["participant"])
    tasks = np.array(preds["task"])
    rows: list[dict[str, Any]] = []

    def add_row(group: str, key: str, mask: np.ndarray) -> None:
        if mask.sum() < 2:
            return
        r2 = r2_score(y_true[mask], y_pred[mask])
        e = rmse(y_true[mask], y_pred[mask])
        rows.append(
            {
                "group": group,
                "key": key,
                "n_windows": int(mask.sum()),
                "r2_hip": float(r2[0]),
                "r2_knee": float(r2[1]),
                "rmse_hip": float(e[0]),
                "rmse_knee": float(e[1]),
            }
        )

    add_row("all", "all", np.ones(len(y_true), dtype=bool))
    for participant in sorted(set(participants.tolist())):
        add_row("participant", participant, participants == participant)
    for task in sorted(set(tasks.tolist())):
        add_row("task", task, tasks == task)
    return rows


def evaluate(config: dict[str, Any], checkpoint_path: str | Path | None = None) -> list[dict[str, Any]]:
    ckpt_path = resolve_path(checkpoint_path or config["evaluate"]["checkpoint"])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_config = checkpoint.get("config", config)
    cache_path = cache_trials(ckpt_config, force=False)
    trials = load_cached_trials(cache_path)
    train_idx, val_idx = split_trials(ckpt_config, trials)
    split = config.get("evaluate", {}).get("split", "val")
    indices = train_idx if split == "train" else val_idx
    dataset = CamargoWindowDataset(
        trials,
        indices,
        ckpt_config,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        limit_windows=None,
    )
    device = choose_device(config["train"].get("device", "auto"))
    model = build_model(ckpt_config, input_dim=int(checkpoint["input_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluate"].get("batch_size", 256)),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )
    preds = _predict(model, loader, device)
    rows = _metric_rows(preds)
    out_csv = resolve_path(config["evaluate"]["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = evaluate(cfg, checkpoint_path=args.checkpoint)
    for row in rows:
        print(
            f"{row['group']:>11s} {row['key']:<16s} n={row['n_windows']:<8d} "
            f"R2 hip/knee={row['r2_hip']:.3f}/{row['r2_knee']:.3f} "
            f"RMSE hip/knee={row['rmse_hip']:.3f}/{row['rmse_knee']:.3f}"
        )


if __name__ == "__main__":
    main()
