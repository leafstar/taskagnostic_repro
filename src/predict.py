from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model
from src.utils import choose_device, standardize


def load_estimator(checkpoint_path: str | Path, device: str = "auto") -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    resolved_device = choose_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model = build_model(checkpoint["config"], input_dim=int(checkpoint["input_dim"])).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, resolved_device


def _window_view(features: np.ndarray, window_length: int, stride: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D array shaped T x C, got {features.shape}.")
    if len(features) < window_length:
        raise ValueError(f"Need at least {window_length} frames, got {len(features)}.")
    starts = np.arange(0, len(features) - window_length + 1, stride)
    return np.stack([features[start : start + window_length] for start in starts], axis=0)


def predict_windows(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    features: np.ndarray,
    *,
    device: torch.device,
    stride: int = 1,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    input_dim = int(checkpoint["input_dim"])
    if features.ndim != 2 or features.shape[1] != input_dim:
        raise ValueError(f"Expected features shaped T x {input_dim}, got {features.shape}.")

    window_length = int(checkpoint["config"]["window"]["length"])
    normalized = standardize(features, checkpoint["mean"], checkpoint["std"]).astype(np.float32)
    windows = _window_view(normalized, window_length, stride)
    label_indices = np.arange(window_length - 1, len(features), stride)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).permute(0, 2, 1).to(device)
            preds.append(model(batch).cpu().numpy())
    return label_indices, np.concatenate(preds, axis=0)


def _load_feature_array(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(p)
    if p.suffix.lower() == ".npz":
        data = np.load(p)
        if "features" not in data:
            raise KeyError("NPZ input must contain an array named 'features'.")
        return data["features"]
    if p.suffix.lower() == ".csv":
        return np.loadtxt(p, delimiter=",", dtype=np.float32)
    raise ValueError("Input must be .npy, .npz with key 'features', or numeric .csv.")


def _write_predictions(path: str | Path, frame_indices: np.ndarray, predictions: np.ndarray, target_names: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".npy":
        np.save(p, predictions)
        return
    if p.suffix.lower() == ".npz":
        np.savez_compressed(p, frame_index=frame_indices, prediction=predictions, target_names=np.array(target_names))
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", *target_names])
        for frame, pred in zip(frame_indices.tolist(), predictions.tolist()):
            writer.writerow([frame, *pred])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained TCN moment estimator on a T x C feature array.")
    parser.add_argument("--checkpoint", default="pretrained/camargo_ab06_ab10_lopo_AB10_best.pt")
    parser.add_argument("--input", required=True, help=".npy, .npz with key 'features', or numeric .csv shaped T x C.")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--print-metadata", action="store_true")
    args = parser.parse_args()

    model, checkpoint, device = load_estimator(args.checkpoint, device=args.device)
    if args.print_metadata:
        print(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "epoch": checkpoint.get("epoch"),
                    "input_dim": checkpoint["input_dim"],
                    "window_length": checkpoint["config"]["window"]["length"],
                    "feature_names": checkpoint["feature_names"],
                    "target_names": checkpoint["target_names"],
                    "metrics": checkpoint.get("metrics", {}),
                },
                indent=2,
            )
        )
    features = _load_feature_array(args.input)
    frame_indices, predictions = predict_windows(
        model,
        checkpoint,
        features,
        device=device,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    _write_predictions(args.output, frame_indices, predictions, checkpoint["target_names"])
    print(f"wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
