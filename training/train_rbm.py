"""Train the Bernoulli-Bernoulli RBM with CD-k.   CLI: python -m training.train_rbm --epochs 25"""
from __future__ import annotations

import argparse

import torch

from models.rbm import RBM
from training.common import (TrainingStopped, add_config_args, check_stop, cli_progress, iterate_batches,
                             load_test_data, load_train_data, seed_everything, to_images)
from training.configs import RBMConfig, config_to_dict
from utils.checkpoints import save_checkpoint
from utils.device import get_device


def train_rbm(cfg: RBMConfig, device=None, progress=None, should_stop=None, train_data=None, val_data=None,
              checkpoint_dir=None) -> dict:
    device = device or get_device()
    seed_everything(cfg.seed)
    train_data = load_train_data(cfg.train_size, cfg.seed) if train_data is None else train_data
    val_data = (load_test_data()[:5000] if val_data is None else val_data)
    g_cpu = torch.Generator().manual_seed(cfg.seed)
    g_dev = torch.Generator(device=device).manual_seed(cfg.seed)

    rbm = RBM(784, cfg.n_hidden).to(device)
    # initialise visible biases from data statistics (standard RBM practice)
    mean = (train_data[:10000].float() / 255).reshape(-1, 784).mean(0).clamp(1e-3, 1 - 1e-3)
    rbm.v_bias.data = torch.log(mean / (1 - mean)).to(device)

    history = {"recon_mse": [], "free_energy_train": [], "free_energy_val": []}
    fe_train = to_images(train_data[:5000], device).reshape(-1, 784)
    fe_val = to_images(val_data[:5000], device).reshape(-1, 784)

    for epoch in range(1, cfg.epochs + 1):
        momentum = 0.5 if epoch <= 2 else 0.9
        errs = []
        for x in iterate_batches(train_data, cfg.batch_size, device, g_cpu):
            check_stop(should_stop)
            v0 = torch.bernoulli(x.reshape(x.shape[0], -1), generator=g_dev)  # stochastic binarisation
            errs.append(rbm.cd_update(v0, cfg.cd_k, cfg.lr, momentum, cfg.weight_decay, g_dev))
        with torch.no_grad():
            m = {
                "recon_mse": float(sum(errs) / max(1, len(errs))),
                "free_energy_train": rbm.free_energy((fe_train > 0.5).float()).mean().item(),
                "free_energy_val": rbm.free_energy((fe_val > 0.5).float()).mean().item(),
            }
        for k, v in m.items():
            history[k].append(v)
        if progress:
            progress(epoch, cfg.epochs, m)

    path = save_checkpoint("rbm", config_to_dict(cfg), {"rbm": rbm.state_dict()}, history,
                           {"device": str(device), "n_train": len(train_data)}, checkpoint_dir)
    return {"history": history, "path": str(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, RBMConfig)
    cfg = RBMConfig(**vars(p.parse_args()))
    try:
        train_rbm(cfg, progress=cli_progress)
    except TrainingStopped:
        print("stopped")


if __name__ == "__main__":
    main()
