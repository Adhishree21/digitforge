"""Loads trained checkpoints and reconstructs digits with RBM / VAE."""
from __future__ import annotations

import numpy as np
import torch

from models.rbm import RBM
from models.vae import VAE
from training.configs import config_from_dict
from utils.checkpoints import CheckpointError, load_checkpoint
from utils.device import get_device
from utils.metrics import compute_single
from utils.paths import MODEL_LABELS, MODEL_NAMES


def to_tensor(x, device=None) -> torch.Tensor:
    """numpy [28,28] / [N,28,28] / [N,1,28,28] (or tensor) -> float32 tensor [N, 1, 28, 28]."""
    t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=torch.float32)
    if t.ndim == 2:
        t = t[None, None]
    elif t.ndim == 3:
        t = t[:, None]
    if t.ndim != 4 or t.shape[1:] != (1, 28, 28):
        raise ValueError(f"Expected 28x28 images, got tensor of shape {tuple(t.shape)}")
    return t.to(device) if device is not None else t


class ReconstructionEngine:
    """Holds the trained models on one device. Missing/incompatible checkpoints are recorded in `errors`."""

    def __init__(self, device=None, directory=None):
        self.device = device or get_device()
        self.directory = directory
        self.models: dict[str, dict] = {}
        self.configs: dict[str, object] = {}
        self.histories: dict[str, dict] = {}
        self.errors: dict[str, str] = {}
        for name in MODEL_NAMES:
            try:
                self._load(name)
            except CheckpointError as e:
                self.errors[name] = str(e)

    def _load(self, name: str) -> None:
        ck = load_checkpoint(name, self.directory)
        cfg = config_from_dict(name, ck["config"])
        st = ck["state"]
        try:
            if name == "rbm":
                parts = {"rbm": RBM(784, cfg.n_hidden)}
            else:
                parts = {"vae": VAE(cfg.latent_dim)}
            for k, m in parts.items():
                m.load_state_dict(st[k])
                m.to(self.device).eval()
        except (RuntimeError, KeyError) as e:
            raise CheckpointError(f"Checkpoint '{name}' does not match its saved config: {e}") from e
        self.models[name], self.configs[name], self.histories[name] = parts, cfg, ck["history"]

    # -------------------------------------------------------------------------------
    @property
    def available(self) -> list[str]:
        return [n for n in MODEL_NAMES if n in self.models]

    @torch.no_grad()
    def reconstruct(self, name: str, x, rbm_steps: int | None = None, vae_stochastic: bool = False,
                    seed: int = 0) -> torch.Tensor:
        """x: images in [0,1] (see to_tensor). Returns [N, 1, 28, 28] on the engine's device."""
        if name not in self.models:
            raise CheckpointError(self.errors.get(name, f"Model '{name}' is not loaded."))
        x = to_tensor(x, self.device)
        m = self.models[name]
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        if name == "rbm":
            steps = rbm_steps or self.configs["rbm"].infer_steps
            return m["rbm"].reconstruct(x, steps=steps, generator=gen)
        if vae_stochastic:
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed(int(seed))
        return m["vae"].reconstruct(x, stochastic=vae_stochastic)

    def reconstruct_numpy(self, name: str, x, **kw) -> np.ndarray:
        return self.reconstruct(name, x, **kw).squeeze(1).cpu().numpy()

    def interpolate_vae(self, a, b, steps: int = 8) -> np.ndarray:
        if "vae" not in self.models:
            raise CheckpointError(self.errors.get("vae", "VAE is not loaded."))
        out = self.models["vae"]["vae"].interpolate(to_tensor(a, self.device), to_tensor(b, self.device), steps)
        return out.squeeze(1).cpu().numpy()


def reconstruct_digits(engine: ReconstructionEngine, digits: list[np.ndarray], models: list[str],
                       rbm_steps: int = 1, vae_stochastic: bool = False, seed: int = 0) -> list[dict]:
    """Reconstruct each digit independently with each requested model and score it against the input.

    Returns one dict per digit: {"original", "recon": {model: [28,28]}, "metrics": {model: {...}}}.
    """
    results = []
    for i, d in enumerate(digits):
        rec, met = {}, {}
        for name in models:
            r = engine.reconstruct_numpy(name, d, rbm_steps=rbm_steps, vae_stochastic=vae_stochastic,
                                         seed=seed + i)[0]
            rec[name] = r
            met[name] = compute_single(d, r)
        results.append({"original": np.asarray(d, np.float32), "recon": rec, "metrics": met})
    return results


def label(name: str) -> str:
    return MODEL_LABELS[name]
