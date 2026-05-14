from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import build_datasets
from src.model import build_model
from src.utils import AverageMeter, choose_device, load_config, r2_score, resolve_path, rmse, save_json, set_seed, target_metric_labels


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "participant": [b["participant"] for b in batch],
        "trial_id": [b["trial_id"] for b in batch],
        "task": [b["task"] for b in batch],
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    meter = AverageMeter()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if is_train else "val"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            meter.update(float(loss.detach().cpu()), x.shape[0])
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
    return meter.avg, np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    input_dim: int,
    feature_names: list[str],
    target_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "input_dim": input_dim,
            "feature_names": feature_names,
            "target_names": target_names,
            "mean": mean,
            "std": std,
            "metrics": metrics,
        },
        path,
    )


def train(config: dict[str, Any], force_cache: bool = False) -> Path:
    set_seed(int(config.get("seed", 42)))
    device = choose_device(config["train"].get("device", "auto"))
    train_ds, val_ds = build_datasets(config, force_cache=force_cache)
    input_dim = len(train_ds.feature_names)
    model = build_model(config, input_dim=input_dim).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    ckpt_dir = resolve_path(config["train"]["checkpoint_dir"])
    log_dir = resolve_path(config["train"]["log_dir"])
    writer = SummaryWriter(log_dir=str(log_dir))
    save_json(
        ckpt_dir / "run_metadata.json",
        {
            "input_dim": input_dim,
            "feature_names": train_ds.feature_names,
            "target_names": train_ds.target_names,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "device": str(device),
        },
    )

    best_val = float("inf")
    best_path = ckpt_dir / "best.pt"
    history_path = ckpt_dir / "history.csv"
    metric_labels = target_metric_labels(train_ds.target_names)
    fieldnames = ["epoch", "train_loss", "val_loss"]
    for label in metric_labels:
        fieldnames.extend([f"val_r2_{label}", f"val_rmse_{label}"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for epoch in range(1, int(config["train"]["epochs"]) + 1):
            train_loss, _, _ = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                grad_clip_norm=config["train"].get("grad_clip_norm"),
            )
            val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion, device)
            val_r2 = r2_score(y_true, y_pred)
            val_rmse = rmse(y_true, y_pred)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            metric_text: list[str] = []
            for i, label in enumerate(metric_labels):
                row[f"val_r2_{label}"] = float(val_r2[i])
                row[f"val_rmse_{label}"] = float(val_rmse[i])
                writer.add_scalar(f"r2/{label}", float(val_r2[i]), epoch)
                writer.add_scalar(f"rmse/{label}", float(val_rmse[i]), epoch)
                metric_text.append(f"{label}={val_r2[i]:.3f}/{val_rmse[i]:.3f}")
            writer_csv.writerow(row)
            f.flush()
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            print(
                f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
                f"R2/RMSE {' '.join(metric_text)}"
            )

            metrics = {k: float(v) for k, v in row.items() if k != "epoch"}
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    input_dim=input_dim,
                    feature_names=train_ds.feature_names,
                    target_names=train_ds.target_names,
                    mean=train_ds.mean,
                    std=train_ds.std,
                    metrics=metrics,
                )
            save_every = int(config["train"].get("save_every", 0) or 0)
            if save_every and epoch % save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    input_dim=input_dim,
                    feature_names=train_ds.feature_names,
                    target_names=train_ds.target_names,
                    mean=train_ds.mean,
                    std=train_ds.std,
                    metrics=metrics,
                )
    writer.close()
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    best = train(cfg, force_cache=args.force_cache)
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
