"""Shared training helpers: MNIST loading, batching, seeding, stop handling, CLI."""
from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Callable

import numpy as np
import torch

from utils.paths import DATA_DIR, ensure_dirs


class TrainingStopped(Exception):
    """Raised inside a training loop when a stop was requested."""


ProgressFn = Callable[[int, int, dict], None]  # (epoch_done, total_epochs, latest_metrics)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mnist(train: bool) -> torch.Tensor:
    from torchvision import datasets

    ensure_dirs()
    try:
        ds = datasets.MNIST(root=str(DATA_DIR), train=train, download=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not load or download MNIST into {DATA_DIR}. Check your internet connection for the "
            f"first download (the standard torchvision MNIST files only). Original error: {e}"
        ) from e
    return ds.data.clone()  # uint8 [N, 28, 28]


def load_train_data(train_size: int, seed: int) -> torch.Tensor:
    """Random (seeded) subset of the 60k MNIST training images, uint8 [N, 28, 28]."""
    data = _mnist(True)
    if train_size < len(data):
        idx = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))[:train_size]
        data = data[idx]
    return data


def load_test_data() -> torch.Tensor:
    """The official 10k MNIST test set (held out from every model), uint8 [N, 28, 28]."""
    return _mnist(False)


def to_images(u8: torch.Tensor, device) -> torch.Tensor:
    """uint8 [B, 28, 28] -> float [B, 1, 28, 28] in [0, 1] on device."""
    return u8.to(device).float().div_(255.0).unsqueeze(1)


def iterate_batches(data: torch.Tensor, batch_size: int, device, generator: torch.Generator):
    """Shuffled mini-batches; the data stays on CPU (uint8) and only each batch moves to the device."""
    perm = torch.randperm(len(data), generator=generator)
    for i in range(0, len(data) - batch_size + 1, batch_size):  # drop last partial batch
        yield to_images(data[perm[i:i + batch_size]], device)


def check_stop(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise TrainingStopped()


def add_config_args(parser: argparse.ArgumentParser, cfg_cls) -> None:
    for f in fields(cfg_cls):
        default = f.default
        parser.add_argument(f"--{f.name}", type=type(default), default=default)


def cli_progress(epoch: int, total: int, metrics: dict) -> None:
    parts = " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
    print(f"[{epoch}/{total}] {parts}", flush=True)
