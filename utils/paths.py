"""Filesystem locations used across DigitForge (all local, relative to the project root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

MODEL_NAMES = ("rbm", "vae")
MODEL_LABELS = {"rbm": "RBM", "vae": "VAE"}


def ensure_dirs() -> None:
    for d in (CHECKPOINT_DIR, OUTPUT_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
