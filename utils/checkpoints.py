"""Checkpoint saving/loading with a format version and the model config stored alongside the weights.

A checkpoint is a plain dict (loadable with torch.load(weights_only=True)):
    {"format_version", "model", "config", "state": {part_name: state_dict}, "history", "meta"}
"""
from __future__ import annotations

import hashlib
import os
import time
from functools import lru_cache
from pathlib import Path

import torch

from utils.paths import CHECKPOINT_DIR, MODEL_NAMES, ensure_dirs

FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Missing, corrupt or incompatible checkpoint."""


def ckpt_path(name: str, directory: Path | None = None) -> Path:
    return (directory or CHECKPOINT_DIR) / f"{name}.pt"


def save_checkpoint(name: str, config: dict, state: dict, history: dict, meta: dict | None = None,
                    directory: Path | None = None) -> Path:
    ensure_dirs()
    path = ckpt_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model": name,
        "config": config,
        "state": {k: {kk: vv.detach().cpu() for kk, vv in v.items()} for k, v in state.items()},
        "history": history,
        "meta": {"saved_at": time.time(), **(meta or {})},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint
    return path


def load_checkpoint(name: str, directory: Path | None = None, map_location="cpu") -> dict:
    path = ckpt_path(name, directory)
    if not path.exists():
        raise CheckpointError(f"No checkpoint for '{name}' at {path}. Reload the app to train it automatically.")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:  # corrupt file, wrong pickle, etc. - report, never hide
        raise CheckpointError(f"Could not read checkpoint {path.name}: {e}") from e
    if payload.get("format_version") != FORMAT_VERSION or payload.get("model") != name:
        raise CheckpointError(
            f"Checkpoint {path.name} is incompatible (format {payload.get('format_version')}, "
            f"model {payload.get('model')!r}; expected format {FORMAT_VERSION}, model {name!r}). Retrain it."
        )
    return payload


def checkpoint_status(directory: Path | None = None) -> dict[str, dict]:
    """Per-model status for the UI: exists / compatible / size / time / config / error."""
    out = {}
    for name in MODEL_NAMES:
        path = ckpt_path(name, directory)
        st = {"exists": path.exists(), "compatible": False, "size_mb": None, "saved_at": None,
              "config": None, "error": None}
        if st["exists"]:
            st["size_mb"] = path.stat().st_size / 1024**2
            try:
                p = load_checkpoint(name, directory)
                st.update(compatible=True, saved_at=p["meta"].get("saved_at"), config=p["config"])
            except CheckpointError as e:
                st["error"] = str(e)
        out[name] = st
    return out


def file_digest(path: Path) -> str:
    """SHA-1 of a file's content, memoised per (path, size, mtime) within the process."""
    st = path.stat()
    return _digest(str(path), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=16)
def _digest(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_signature(directory: Path | None = None) -> tuple:
    """Changes whenever any checkpoint file changes; used as a cache key by the UI."""
    sig = []
    for name in MODEL_NAMES:
        p = ckpt_path(name, directory)
        sig.append((name, file_digest(p) if p.exists() else None))
    return tuple(sig)
