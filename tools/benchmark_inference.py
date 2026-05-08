from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import load_estimator


def benchmark(
    checkpoint: Path,
    device_name: str,
    batch_sizes: list[int],
    warmup: int,
    iters: int,
) -> None:
    model, checkpoint_data, device = load_estimator(checkpoint, device=device_name)
    input_dim = int(checkpoint_data["input_dim"])
    window = int(checkpoint_data["config"]["window"]["length"])
    print(f"device={device}")
    print(f"input=batch x {input_dim} x {window}")
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, input_dim, window, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        forward_ms = elapsed / iters * 1000.0
        window_ms = forward_ms / batch_size
        windows_per_second = batch_size * iters / elapsed
        print(
            f"batch={batch_size:4d} "
            f"forward={forward_ms:9.4f} ms "
            f"per_window={window_ms:9.5f} ms "
            f"throughput={windows_per_second:11.1f} windows/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pretrained/camargo_ab06_ab10_lopo_AB10_best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-sizes", default="1,8,32,128,256")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    benchmark(Path(args.checkpoint), args.device, batch_sizes, args.warmup, args.iters)


if __name__ == "__main__":
    main()
