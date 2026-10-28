"""Bernoulli-Bernoulli Restricted Boltzmann Machine trained with Contrastive Divergence (CD-k).

Energy:      E(v, h) = -b.v - c.h - v^T W h
Free energy: F(v)    = -b.v - sum_j softplus(c_j + (v^T W)_j)
Tensor convention: the RBM works on flat vectors [B, 784]; `reconstruct` accepts/returns
images [B, 1, 28, 28] and converts explicitly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bernoulli(p: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return (torch.rand(p.shape, device=p.device, generator=generator) < p).to(p.dtype)


class RBM(nn.Module):
    def __init__(self, n_visible: int = 784, n_hidden: int = 512):
        super().__init__()
        self.n_visible, self.n_hidden = n_visible, n_hidden
        # Parameters are updated by the CD rule, not by autograd.
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01, requires_grad=False)
        self.v_bias = nn.Parameter(torch.zeros(n_visible), requires_grad=False)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden), requires_grad=False)
        self.register_buffer("vel_W", torch.zeros(n_visible, n_hidden))
        self.register_buffer("vel_v", torch.zeros(n_visible))
        self.register_buffer("vel_h", torch.zeros(n_hidden))

    # -- conditionals and sampling ------------------------------------------------------
    def p_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(v @ self.W + self.h_bias)

    def p_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(h @ self.W.t() + self.v_bias)

    def sample_h(self, v, generator=None):
        p = self.p_h_given_v(v)
        return p, bernoulli(p, generator)

    def sample_v(self, h, generator=None):
        p = self.p_v_given_h(h)
        return p, bernoulli(p, generator)

    def gibbs_step(self, v, generator=None):
        """One v -> h -> v' step. Returns (p(h|v), p(v'|h), sampled v')."""
        ph, h = self.sample_h(v, generator)
        pv, v_new = self.sample_v(h, generator)
        return ph, pv, v_new

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        wx_b = v @ self.W + self.h_bias
        return -(v @ self.v_bias) - F.softplus(wx_b).sum(dim=1)

    # -- training -----------------------------------------------------------------------
    @torch.no_grad()
    def cd_update(self, v0, k=1, lr=0.05, momentum=0.5, weight_decay=1e-4, generator=None) -> float:
        """One CD-k parameter update on a binary batch v0 [B, 784]. Returns reconstruction MSE."""
        B = v0.shape[0]
        ph0 = self.p_h_given_v(v0)
        h = bernoulli(ph0, generator)
        for _ in range(k):  # explicit Gibbs chain h -> v -> h
            pv = self.p_v_given_h(h)
            v = bernoulli(pv, generator)
            h = bernoulli(self.p_h_given_v(v), generator)
        v_neg = pv  # negative phase uses probabilities (lower variance)
        ph_neg = self.p_h_given_v(v_neg)
        dW = (v0.t() @ ph0 - v_neg.t() @ ph_neg) / B - weight_decay * self.W
        dv = (v0 - v_neg).mean(0)
        dh = (ph0 - ph_neg).mean(0)
        self.vel_W.mul_(momentum).add_(lr * dW)
        self.vel_v.mul_(momentum).add_(lr * dv)
        self.vel_h.mul_(momentum).add_(lr * dh)
        self.W.add_(self.vel_W)
        self.v_bias.add_(self.vel_v)
        self.h_bias.add_(self.vel_h)
        return F.mse_loss(v_neg, v0).item()

    # -- inference ----------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, steps: int = 1, generator=None, threshold: float = 0.5) -> torch.Tensor:
        """Clamp visible units to the (binarised) input, infer hidden, run `steps` Gibbs steps.

        x: [B, 1, 28, 28] in [0, 1]. Returns visible probabilities after the last step, same shape.
        """
        if steps < 1:
            raise ValueError("Gibbs steps must be >= 1")
        B = x.shape[0]
        v = (x.reshape(B, -1) > threshold).to(self.W.dtype)  # clamped observation
        ph = self.p_h_given_v(v)
        for s in range(steps):
            h = bernoulli(ph, generator)
            pv = self.p_v_given_h(h)
            if s < steps - 1:
                v = bernoulli(pv, generator)
                ph = self.p_h_given_v(v)
        return pv.reshape(B, 1, 28, 28)
