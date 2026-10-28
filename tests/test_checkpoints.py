import numpy as np
import pytest
import torch

from training.configs import RBMConfig, VAEConfig, config_to_dict
from training.train_rbm import train_rbm
from training.train_vae import train_vae
from utils.checkpoints import CheckpointError, checkpoint_status, load_checkpoint, save_checkpoint
from utils.reconstruction import ReconstructionEngine, reconstruct_digits

CPU = torch.device("cpu")
DATA = (torch.rand(256, 28, 28) * 255).to(torch.uint8)


def test_save_load_roundtrip(tmp_path):
    from models.vae import VAE
    m = VAE(4)
    save_checkpoint("vae", {"latent_dim": 4}, {"vae": m.state_dict()}, {"total": [1.0]}, directory=tmp_path)
    ck = load_checkpoint("vae", tmp_path)
    m2 = VAE(4)
    m2.load_state_dict(ck["state"]["vae"])
    x = torch.rand(2, 1, 28, 28)
    assert torch.equal(m.eval()(x)[1], m2.eval()(x)[1])
    assert ck["config"] == {"latent_dim": 4} and ck["history"]["total"] == [1.0]
    assert checkpoint_status(tmp_path)["vae"]["compatible"]


def test_missing_and_incompatible_checkpoint(tmp_path):
    with pytest.raises(CheckpointError):
        load_checkpoint("rbm", tmp_path)
    torch.save({"format_version": 999, "model": "rbm"}, tmp_path / "rbm.pt")
    with pytest.raises(CheckpointError, match="incompatible"):
        load_checkpoint("rbm", tmp_path)
    (tmp_path / "vae.pt").write_bytes(b"garbage")
    with pytest.raises(CheckpointError, match="Could not read"):
        load_checkpoint("vae", tmp_path)


def test_end_to_end_tiny_training_and_engine(tmp_path):
    kw = dict(device=CPU, train_data=DATA, val_data=DATA, checkpoint_dir=tmp_path)
    seen = []
    train_rbm(RBMConfig(epochs=2, batch_size=64, n_hidden=16), progress=lambda e, t, m: seen.append(e), **kw)
    train_vae(VAEConfig(epochs=1, batch_size=64, latent_dim=4), **kw)
    assert seen == [1, 2]
    engine = ReconstructionEngine(CPU, tmp_path)
    assert engine.available == ["rbm", "vae"] and not engine.errors
    digit = np.random.rand(28, 28).astype(np.float32)
    res = reconstruct_digits(engine, [digit, digit], ["rbm", "vae"], rbm_steps=2)
    assert len(res) == 2
    for r in res:
        for n in ("rbm", "vae"):
            assert r["recon"][n].shape == (28, 28) and set(r["metrics"][n]) == {"MSE", "MAE", "PSNR", "SSIM"}
    assert "total" in engine.histories["vae"]


def test_training_can_be_stopped(tmp_path):
    from training.common import TrainingStopped
    with pytest.raises(TrainingStopped):
        train_vae(VAEConfig(epochs=3, batch_size=32), device=CPU, train_data=DATA, val_data=DATA,
                  checkpoint_dir=tmp_path, should_stop=lambda: True)
    assert not (tmp_path / "vae.pt").exists()
