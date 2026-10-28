"""Streamlit helpers shared by all pages (cached resources, theme, sidebar, status)."""
from __future__ import annotations

import streamlit as st

from utils.checkpoints import checkpoint_signature, checkpoint_status
from utils.device import device_info, get_device
from utils.paths import DATA_DIR, MODEL_LABELS, MODEL_NAMES
from utils.reconstruction import ReconstructionEngine


@st.cache_resource(show_spinner="Loading trained models...")
def _load_engine(signature: tuple) -> ReconstructionEngine:  # signature = cache key only
    return ReconstructionEngine(get_device())


def get_engine() -> ReconstructionEngine:
    """Engine is rebuilt automatically when any checkpoint file changes (e.g. after retraining)."""
    return _load_engine(checkpoint_signature())


def theme_name() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit without st.context.theme
        return "light"


def mnist_downloaded() -> bool:
    return (DATA_DIR / "MNIST" / "raw" / "train-images-idx3-ubyte").exists()


def require_models(engine: ReconstructionEngine, needed=MODEL_NAMES) -> bool:
    """Show a clear error for each missing model; return True if all `needed` are loaded."""
    ok = True
    for n in needed:
        if n not in engine.models:
            ok = False
            st.error(f"**{MODEL_LABELS[n]}** is not available: {engine.errors.get(n, 'not loaded')}")
    if not ok:
        st.info("Reload the page to retry the automatic training.")
    return ok


def sidebar_status() -> None:
    info = device_info()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Device: **{info['device']}** - {info['name']}")
    status = checkpoint_status()
    st.sidebar.caption("Checkpoints: " + "  ".join(
        f"{MODEL_LABELS[n]} {'✅' if status[n]['compatible'] else '❌'}" for n in MODEL_NAMES))


def ensure_trained() -> None:
    """First-run setup: train any missing/incompatible model with the fixed default settings.

    Existing compatible checkpoints are reused, so this never retrains on later reruns.
    """
    from training.configs import RBMConfig, VAEConfig
    from training.train_rbm import train_rbm
    from training.train_vae import train_vae

    status = checkpoint_status()
    todo = [n for n in MODEL_NAMES if not status[n]["compatible"]]
    if not todo:
        return
    trainers = {"rbm": (train_rbm, RBMConfig()), "vae": (train_vae, VAEConfig())}
    st.warning("First-time setup: training " + " and ".join(MODEL_LABELS[n] for n in todo) +
               " on MNIST (downloads the dataset once). This takes a few minutes and is only done once.")
    for n in todo:
        fn, cfg = trainers[n]
        bar = st.progress(0.0, f"Training {MODEL_LABELS[n]}...")

        def progress(epoch, total, metrics, _bar=bar, _n=n):
            _bar.progress(epoch / total, f"Training {MODEL_LABELS[_n]}: epoch {epoch}/{total}")

        try:
            fn(cfg, device=get_device(), progress=progress)
        except Exception as e:
            st.error(f"Training {MODEL_LABELS[n]} failed: {e}")
            st.stop()
    st.rerun()
