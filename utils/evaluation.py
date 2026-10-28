"""Held-out evaluation, latency benchmark and model-complexity measurement.

Everything here is measured from the loaded checkpoints; results are cached to outputs/ and
tagged with the checkpoint signature so stale numbers are never shown for retrained models.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from training.common import load_test_data, to_images
from utils.checkpoints import ckpt_path, file_digest
from utils.device import sync
from utils.metrics import METRICS, compute_all
from utils.paths import MODEL_NAMES, OUTPUT_DIR, ensure_dirs
from utils.reconstruction import ReconstructionEngine

EVAL_JSON = OUTPUT_DIR / "evaluation.json"
EVAL_NPZ = OUTPUT_DIR / "evaluation_arrays.npz"
GALLERY_N = 12

LATENCY_METHOD = (
    "Batch size 1, the same input digit for every model, on the device shown. {warm} warm-up runs are "
    "discarded, then {runs} timed runs, each bracketed by a device synchronisation. One-time checkpoint "
    "loading is excluded. RBM uses {steps} Gibbs step(s); VAE is deterministic (latent mean)."
)


def _signature(engine: ReconstructionEngine) -> list:
    # Content hash (not mtime) so the cache survives git clones / redeploys.
    return [[n, file_digest(ckpt_path(n, engine.directory))] for n in engine.available]


def count_params(module) -> int:
    """Trainable-or-stored parameters (the RBM's parameters are updated by CD, so all are counted)."""
    return int(sum(p.numel() for p in module.parameters()))


def model_complexity(engine: ReconstructionEngine) -> dict:
    out = {}
    for n in engine.available:
        parts = engine.models[n]
        total = sum(count_params(m) for m in parts.values())
        out[n] = {"params_inference": total, "params_total": total,
                  "checkpoint_mb": ckpt_path(n, engine.directory).stat().st_size / 1024**2}
    return out


def benchmark_latency(engine: ReconstructionEngine, x: np.ndarray, runs: int = 100, warmup: int = 10,
                      rbm_steps: int = 1) -> dict:
    res = {}
    for n in engine.available:
        for _ in range(warmup):
            engine.reconstruct(n, x, rbm_steps=rbm_steps)
        times = []
        for _ in range(runs):
            sync(engine.device)
            t0 = time.perf_counter()
            engine.reconstruct(n, x, rbm_steps=rbm_steps)
            sync(engine.device)
            times.append((time.perf_counter() - t0) * 1000.0)
        t = np.array(times)
        res[n] = {"mean_ms": float(t.mean()), "std_ms": float(t.std()), "median_ms": float(np.median(t)),
                  "p95_ms": float(np.percentile(t, 95)), "runs": runs}
    return res


def evaluate(engine: ReconstructionEngine, n_samples: int = 10000, rbm_steps: int = 1, seed: int = 0,
             batch_size: int = 500, progress=None) -> dict:
    """Reconstruct the first n_samples held-out MNIST test images with every available model."""
    if not engine.available:
        raise RuntimeError("No trained models are available to evaluate.")
    ensure_dirs()
    test = load_test_data()[:n_samples]
    per_image = {n: {m: [] for m in METRICS} for n in engine.available}
    gallery = {"original": [], **{n: [] for n in engine.available}}
    total_steps = len(test) // batch_size + (len(test) % batch_size > 0)
    for bi, i in enumerate(range(0, len(test), batch_size)):
        x = to_images(test[i:i + batch_size], engine.device)
        x_np = x.squeeze(1).cpu().numpy()
        if i == 0:
            gallery["original"] = x_np[:GALLERY_N]
        for n in engine.available:
            r = engine.reconstruct_numpy(n, x, rbm_steps=rbm_steps, seed=seed + bi)
            if i == 0:
                gallery[n] = r[:GALLERY_N]
            for m, v in compute_all(x_np, r).items():
                per_image[n][m].append(v)
        if progress:
            progress((bi + 1) / total_steps)

    arrays = {f"{n}__{m}": np.concatenate(v) for n in per_image for m, v in per_image[n].items()}
    arrays["gallery__original"] = np.asarray(gallery["original"])
    for n in engine.available:
        arrays[f"gallery__{n}"] = np.asarray(gallery[n])
    np.savez_compressed(EVAL_NPZ, **arrays)

    summary = {n: {m: {"mean": float(arrays[f"{n}__{m}"].mean()), "median": float(np.median(arrays[f"{n}__{m}"])),
                       "std": float(arrays[f"{n}__{m}"].std())} for m in METRICS} for n in engine.available}
    runs, warm = 100, 10
    result = {
        "signature": _signature(engine), "n_samples": int(len(test)), "rbm_steps": rbm_steps,
        "device": str(engine.device), "created": time.time(), "summary": summary,
        "latency": benchmark_latency(engine, gallery["original"][0], runs, warm, rbm_steps),
        "latency_method": LATENCY_METHOD.format(warm=warm, runs=runs, steps=rbm_steps),
        "complexity": model_complexity(engine),
    }
    EVAL_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_evaluation(engine: ReconstructionEngine | None = None) -> tuple[dict | None, dict | None, str | None]:
    """Returns (result, arrays, problem). `problem` explains why nothing usable was returned."""
    if not EVAL_JSON.exists() or not EVAL_NPZ.exists():
        return None, None, "No evaluation has been run yet."
    try:
        result = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
        with np.load(EVAL_NPZ) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as e:
        return None, None, f"Saved evaluation could not be read ({e}). Re-run it."
    if engine is not None and result.get("signature") != _signature(engine):
        return None, None, "Checkpoints changed since the last evaluation. Re-run it."
    return result, arrays, None
