from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        if use_weight_norm:
            conv1 = nn.utils.parametrizations.weight_norm(conv1)
            conv2 = nn.utils.parametrizations.weight_norm(conv2)
        self.net = nn.Sequential(
            conv1,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            conv2,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        modules = list(self.net.modules())
        if self.downsample is not None:
            modules.append(self.downsample)
        for module in modules:
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNMomentEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        filters: int = 64,
        kernel_size: int = 5,
        num_blocks: int = 5,
        dropout: float = 0.2,
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        blocks = []
        channels = [input_dim] + [filters] * num_blocks
        for i in range(num_blocks):
            blocks.append(
                TemporalBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(filters, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        last = features[:, :, -1]
        return self.head(last)


def build_model(config: dict, input_dim: int) -> TCNMomentEstimator:
    model_cfg = config["model"]
    return TCNMomentEstimator(
        input_dim=input_dim,
        output_dim=int(model_cfg.get("output_dim", 2)),
        filters=int(model_cfg.get("filters", 64)),
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        num_blocks=int(model_cfg.get("num_blocks", 5)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        use_weight_norm=bool(model_cfg.get("use_weight_norm", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dim", type=int, default=30)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = build_model(cfg, args.input_dim)
    x = torch.randn(4, args.input_dim, cfg["window"]["length"])
    y = model(x)
    print(model)
    print(tuple(y.shape))


if __name__ == "__main__":
    main()
