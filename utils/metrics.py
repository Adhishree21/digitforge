"""Image similarity metrics between a reference and a reconstruction.

All inputs are float arrays in [0, 1] with identical shape [28, 28] (single image) or
[N, 28, 28] / [N, 1, 28, 28] (batch). data_range is therefore always 1.0.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

# metric -> (label, higher_is_better)
METRICS = {
    "MSE": ("Mean squared error", False),
    "MAE": ("Mean absolute error", False),
    "PSNR": ("Peak signal-to-noise ratio (dB)", True),
    "SSIM": ("Structural similarity", True),
}
PSNR_CAP = 100.0  # reported when the images are identical (MSE = 0)


def _as_batch(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 4:
        a = a[:, 0]
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"Expected [28,28], [N,28,28] or [N,1,28,28]; got shape {a.shape}")
    return a


def _pair(a, b):
    a, b = _as_batch(a), _as_batch(b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def mse(a, b) -> np.ndarray:
    a, b = _pair(a, b)
    return ((a - b) ** 2).mean(axis=(1, 2))


def mae(a, b) -> np.ndarray:
    a, b = _pair(a, b)
    return np.abs(a - b).mean(axis=(1, 2))


def psnr(a, b) -> np.ndarray:
    m = mse(a, b)
    with np.errstate(divide="ignore"):
        out = 10.0 * np.log10(1.0 / m)
    return np.where(m <= 0, PSNR_CAP, np.minimum(out, PSNR_CAP))


def ssim(a, b) -> np.ndarray:
    a, b = _pair(a, b)
    return np.array([structural_similarity(x, y, data_range=1.0) for x, y in zip(a, b)])


def compute_all(a, b) -> dict[str, np.ndarray]:
    """Per-image arrays for every metric."""
    return {"MSE": mse(a, b), "MAE": mae(a, b), "PSNR": psnr(a, b), "SSIM": ssim(a, b)}


def compute_single(a, b) -> dict[str, float]:
    return {k: float(v[0]) for k, v in compute_all(a, b).items()}


def rank_models(values: dict[str, float], higher_is_better: bool) -> list[str]:
    """Model names ordered best -> worst by actual value."""
    return sorted(values, key=lambda k: values[k], reverse=higher_is_better)
