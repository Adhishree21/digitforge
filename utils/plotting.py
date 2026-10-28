"""Matplotlib figures. Every function plots values it is given - nothing is hardcoded or invented."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.metrics import METRICS
from utils.paths import MODEL_LABELS

# Fixed model -> colour mapping (Okabe-Ito, colour-blind safe). Never re-assigned by rank.
COLORS = {"rbm": "#0072B2", "vae": "#E69F00"}
QUALITY_METRICS = ("MSE", "MAE")  # metrics shown in Graph 1
_THEMES = {
    "light": {"fg": "#222222", "muted": "#6b6b6b", "grid": "#d9d9d9"},
    "dark": {"fg": "#eaeaea", "muted": "#a8a8a8", "grid": "#444444"},
}


def use_theme(theme: str = "light") -> dict:
    t = _THEMES.get(theme, _THEMES["light"])
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "axes.edgecolor": t["muted"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.titlecolor": t["fg"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    })
    return t


def _models(data: dict) -> list[str]:
    return [m for m in ("rbm", "vae") if m in data]


# ----------------------------------------------------------------------------- Graph 1
def fig_quality_bars(summary: dict, theme="light"):
    use_theme(theme)
    models = _models(summary)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, metric in zip(axes, QUALITY_METRICS):
        desc, higher = METRICS[metric]
        vals = [summary[m][metric]["mean"] for m in models]
        stds = [summary[m][metric]["std"] for m in models]
        bars = ax.bar([MODEL_LABELS[m] for m in models], vals, color=[COLORS[m] for m in models], width=0.55,
                      yerr=stds, error_kw={"elinewidth": 1, "capsize": 3, "alpha": 0.5})
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}" if metric != "PSNR" else f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
        ax.set_title(f"{metric}  ({'higher' if higher else 'lower'} is better)")
        ax.set_ylabel(desc if metric == "PSNR" else metric)
        ax.set_xlabel("Model")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- Graph 2
def fig_convergence(histories: dict, theme="light"):
    use_theme(theme)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    h = histories.get("rbm")
    if h:
        ep = np.arange(1, len(h["recon_mse"]) + 1)
        ax.plot(ep, h["recon_mse"], color=COLORS["rbm"], lw=2, label="Reconstruction MSE (train)")
        ax.set_ylabel("Reconstruction MSE")
        ax.set_xlabel("Epoch")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("RBM (CD-k)")
    ax = axes[1]
    h = histories.get("vae")
    if h:
        ep = np.arange(1, len(h["total"]) + 1)
        ax.plot(ep, h["total"], lw=2, color=COLORS["vae"], label="Total (-ELBO)")
        ax.plot(ep, h["recon"], lw=2, color=COLORS["vae"], ls="--", label="Reconstruction (BCE)")
        ax.plot(ep, h["kl"], lw=2, color=COLORS["vae"], ls=":", label="KL divergence")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss per image (nats, log scale)")
        ax.legend(fontsize=8)
    ax.set_title("VAE")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- Graph 3
def fig_latency(latency: dict, theme="light"):
    use_theme(theme)
    models = _models(latency)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    vals = [latency[m]["mean_ms"] for m in models]
    errs = [latency[m]["std_ms"] for m in models]
    bars = ax.bar([MODEL_LABELS[m] for m in models], vals, yerr=errs, color=[COLORS[m] for m in models], width=0.5,
                  error_kw={"elinewidth": 1, "capsize": 3, "alpha": 0.6})
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f} ms", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("Latency per image (ms, mean ± sd)")
    ax.set_xlabel("Model")
    ax.set_title("Steady-state inference time (lower is better)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- Graph 4
def fig_error_distribution(arrays: dict, models: list[str], theme="light"):
    use_theme(theme)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    allv = np.concatenate([arrays[f"{m}__MSE"] for m in models])
    bins = np.linspace(0, float(np.percentile(allv, 99.5)), 60)
    for m in models:
        v = arrays[f"{m}__MSE"]
        ax.hist(v, bins=bins, histtype="step", lw=2, color=COLORS[m], label=f"{MODEL_LABELS[m]} (median {np.median(v):.4f})")
        ax.axvline(np.median(v), color=COLORS[m], lw=1, ls=":")
    ax.set_xlabel("Per-image reconstruction MSE (held-out MNIST test set)")
    ax.set_ylabel("Number of test images")
    ax.set_title("Error distribution (dotted = median; lower is better)")
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- Graph 6
def fig_complexity(complexity: dict, latency: dict, theme="light"):
    use_theme(theme)
    models = _models(complexity)
    labels = [MODEL_LABELS[m] for m in models]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    series = [
        ([complexity[m]["params_inference"] / 1e6 for m in models], "Parameters used at inference (millions)", "{:.2f} M"),
        ([complexity[m]["checkpoint_mb"] for m in models], "Checkpoint size (MB)", "{:.1f} MB"),
        ([latency[m]["mean_ms"] for m in models], "Measured latency (ms / image)", "{:.2f} ms"),
    ]
    for ax, (vals, ylabel, fmt) in zip(axes, series):
        bars = ax.bar(labels, vals, color=[COLORS[m] for m in models], width=0.55)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8,
                        xytext=(0, 2), textcoords="offset points")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_to_png(fig, dpi=150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


# ----------------------------------------------------------------------------- digit images
def to_pil(arr: np.ndarray, size: int = 140) -> Image.Image:
    """28x28 float [0,1] -> upscaled grayscale PIL image (no axes, no margins)."""
    a = (np.clip(np.asarray(arr, np.float32), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a, mode="L").resize((size, size), Image.NEAREST)


def png_bytes(arr: np.ndarray, size: int = 280) -> bytes:
    buf = io.BytesIO()
    to_pil(arr, size).save(buf, format="PNG")
    return buf.getvalue()
