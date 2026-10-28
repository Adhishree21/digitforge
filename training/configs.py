"""Training configuration dataclasses. Defaults target a 16 GB RAM laptop with a 6 GB RTX 4050."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class RBMConfig:
    train_size: int = 60000
    epochs: int = 25
    batch_size: int = 128
    lr: float = 0.05
    seed: int = 42
    n_hidden: int = 512
    cd_k: int = 1               # Gibbs steps inside Contrastive Divergence (training)
    weight_decay: float = 1e-4
    infer_steps: int = 1        # default Gibbs steps used at reconstruction time


@dataclass
class VAEConfig:
    train_size: int = 60000
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42
    latent_dim: int = 20


CONFIG_CLASSES = {"rbm": RBMConfig, "vae": VAEConfig}


def config_from_dict(name: str, d: dict):
    """Build a config, ignoring unknown keys (keeps old checkpoints loadable)."""
    cls = CONFIG_CLASSES[name]
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def config_to_dict(cfg) -> dict:
    return asdict(cfg)
