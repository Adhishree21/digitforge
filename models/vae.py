"""Convolutional Variational Autoencoder.  Tensor convention: images [B, 1, 28, 28] in [0, 1]."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),   # 14x14
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # 7x7
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 64 * 7 * 7), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1),              # 28x28 (logits)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps, eps ~ N(0, I): keeps sampling differentiable w.r.t. mu and sigma."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_logits(self, z):
        return self.decoder(self.decoder_fc(z).view(-1, 64, 7, 7))

    def decode(self, z):
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x):
        """Returns (reconstruction in [0,1], mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def reconstruct(self, x, stochastic: bool = False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if stochastic else mu
        return self.decode(z)

    @torch.no_grad()
    def interpolate(self, x1, x2, steps: int = 8):
        """Decode points on the line between the latent means of two images -> [steps, 1, 28, 28]."""
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        t = torch.linspace(0, 1, steps, device=mu1.device).view(-1, 1)
        return self.decode((1 - t) * mu1 + t * mu2)


def vae_loss(x, logits, mu, logvar):
    """Negative ELBO and its parts, each averaged per image: (total, recon, kl).

    recon = BCE summed over pixels;  kl = KL(q(z|x) || N(0, I)) in closed form.
    """
    B = x.shape[0]
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / B
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return recon + kl, recon, kl
