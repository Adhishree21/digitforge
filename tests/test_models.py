import numpy as np
import pytest
import torch

from models.rbm import RBM
from models.vae import VAE, vae_loss
from utils.metrics import compute_all, compute_single, mae, mse, psnr, rank_models, ssim

X = torch.rand(5, 1, 28, 28)


def test_rbm_shapes_and_range():
    rbm = RBM(784, 32)
    out = rbm.reconstruct(X, steps=3)
    assert out.shape == X.shape and out.min() >= 0 and out.max() <= 1
    assert rbm.free_energy(torch.rand(5, 784).round()).shape == (5,)
    with pytest.raises(ValueError):
        rbm.reconstruct(X, steps=0)


def test_rbm_reconstruction_is_seeded_and_input_dependent():
    rbm = RBM(784, 32)
    rbm.W.data = torch.randn(784, 32) * 0.1
    a = rbm.reconstruct(X, 2, torch.Generator().manual_seed(1))
    b = rbm.reconstruct(X, 2, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)
    assert not torch.allclose(rbm.reconstruct(X[:1], 1, torch.Generator().manual_seed(1)),
                              rbm.reconstruct(1 - X[:1], 1, torch.Generator().manual_seed(1)))


def test_rbm_cd_learns_a_pattern():
    torch.manual_seed(0)
    rbm = RBM(784, 64)
    pattern = torch.zeros(1, 784)
    pattern[:, 100:300] = 1
    data = pattern.repeat(64, 1)
    first = rbm.cd_update(data, lr=0.1)
    for _ in range(100):
        last = rbm.cd_update(data, lr=0.1)
    assert last < first
    assert rbm.free_energy(pattern) < rbm.free_energy(1 - pattern)


def test_vae_forward_and_loss_parts():
    vae = VAE(8)
    rec, mu, logvar = vae(X)
    assert rec.shape == X.shape and mu.shape == (5, 8) and logvar.shape == (5, 8)
    total, recon, kl = vae_loss(X, vae.decode_logits(vae.reparameterize(mu, logvar)), mu, logvar)
    assert total.item() == pytest.approx((recon + kl).item()) and kl.item() >= 0
    total.backward()


def test_vae_reconstruct_modes_and_interpolation():
    vae = VAE(8).eval()
    assert torch.equal(vae.reconstruct(X), vae.reconstruct(X))             # deterministic
    assert not torch.equal(vae.reconstruct(X, stochastic=True), vae.reconstruct(X, stochastic=True))
    assert vae.reconstruct(X).shape == X.shape
    assert vae.interpolate(X[:1], X[1:2], 6).shape == (6, 1, 28, 28)


def test_metrics_values():
    a = np.zeros((28, 28))
    b = np.full((28, 28), 0.5)
    assert mse(a, b)[0] == pytest.approx(0.25)
    assert mae(a, b)[0] == pytest.approx(0.5)
    assert psnr(a, b)[0] == pytest.approx(10 * np.log10(1 / 0.25))
    img = np.random.rand(28, 28)
    m = compute_single(img, img)
    assert m["MSE"] == 0 and m["MAE"] == 0 and m["PSNR"] == 100.0 and m["SSIM"] == pytest.approx(1.0)
    assert ssim(img, 1 - img)[0] < 0.5
    batch = compute_all(np.random.rand(4, 1, 28, 28), np.random.rand(4, 1, 28, 28))
    assert all(v.shape == (4,) for v in batch.values())
    with pytest.raises(ValueError):
        mse(np.zeros((28, 28)), np.zeros((27, 27)))


def test_rank_models_respects_direction():
    vals = {"rbm": 3.0, "vae": 1.0, "mid": 2.0}
    assert rank_models(vals, higher_is_better=False) == ["vae", "mid", "rbm"]
    assert rank_models(vals, higher_is_better=True) == ["rbm", "mid", "vae"]
