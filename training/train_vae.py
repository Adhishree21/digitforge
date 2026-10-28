"""Train the VAE by maximising the ELBO.   CLI: python -m training.train_vae --epochs 20"""
from __future__ import annotations

import argparse

import torch

from models.vae import VAE, vae_loss
from training.common import (TrainingStopped, add_config_args, check_stop, cli_progress, iterate_batches,
                             load_test_data, load_train_data, seed_everything, to_images)
from training.configs import VAEConfig, config_to_dict
from utils.checkpoints import save_checkpoint
from utils.device import get_device

KEYS = ("total", "recon", "kl", "val_total", "val_recon", "val_kl")


def train_vae(cfg: VAEConfig, device=None, progress=None, should_stop=None, train_data=None, val_data=None,
              checkpoint_dir=None) -> dict:
    device = device or get_device()
    seed_everything(cfg.seed)
    train_data = load_train_data(cfg.train_size, cfg.seed) if train_data is None else train_data
    val_data = (load_test_data()[:5000] if val_data is None else val_data)
    g_cpu = torch.Generator().manual_seed(cfg.seed)

    model = VAE(cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = {k: [] for k in KEYS}
    xval = to_images(val_data[:5000], device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums, n = torch.zeros(3), 0
        for x in iterate_batches(train_data, cfg.batch_size, device, g_cpu):
            check_stop(should_stop)
            mu, logvar = model.encode(x)
            logits = model.decode_logits(model.reparameterize(mu, logvar))
            total, recon, kl = vae_loss(x, logits, mu, logvar)
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            sums += torch.tensor([total.item(), recon.item(), kl.item()])
            n += 1
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(xval)
            vt, vr, vk = vae_loss(xval, model.decode_logits(model.reparameterize(mu, logvar)), mu, logvar)
        vals = (sums / max(1, n)).tolist() + [vt.item(), vr.item(), vk.item()]
        m = dict(zip(KEYS, map(float, vals)))
        for k, v in m.items():
            history[k].append(v)
        if progress:
            progress(epoch, cfg.epochs, m)

    path = save_checkpoint("vae", config_to_dict(cfg), {"vae": model.state_dict()}, history,
                           {"device": str(device), "n_train": len(train_data)}, checkpoint_dir)
    return {"history": history, "path": str(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, VAEConfig)
    cfg = VAEConfig(**vars(p.parse_args()))
    try:
        train_vae(cfg, progress=cli_progress)
    except TrainingStopped:
        print("stopped")


if __name__ == "__main__":
    main()
