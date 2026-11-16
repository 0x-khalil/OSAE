"""Fast sanity checks that need no dataset:  python tests/test_core.py   (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from osae.data import INPUT_DIM, corrupt, make_splits, random_crop_flip
from osae.metrics import SparsityAccumulator, classification_report, linear_cka
from osae.models import Classifier, ConvAutoencoder, ResNet18Trunk

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_shapes_and_budgets():
    x = torch.randn(2, 3, 96, 96, device=DEV)
    assert ResNet18Trunk().to(DEV)(x).shape == (2, 512, 6, 6)
    assert Classifier().to(DEV)(x).shape == (2, 10)
    ae, oae = ConvAutoencoder(128).to(DEV), ConvAutoencoder(2048).to(DEV)
    for m, c in [(ae, 128), (oae, 2048)]:
        x_hat, z = m(x)
        assert x_hat.shape == x.shape and z.shape == (2, c, 6, 6) and (z >= 0).all()
    assert ae.latent_numel() < INPUT_DIM < oae.latent_numel()  # under- vs over-complete
    pa, po = ae.param_breakdown(), oae.param_breakdown()
    assert pa["encoder_trunk"] == po["encoder_trunk"]  # identical transferred trunk
    print("AE ", pa, f"{ae.overcompleteness():.3f}x")
    print("OAE", po, f"{oae.overcompleteness():.3f}x")
    print(f"total param ratio OAE/AE = {po['total'] / pa['total']:.3f}")


def test_trunk_transfer_drops_decoder():
    ae, clf = ConvAutoencoder(2048), Classifier()
    clf.encoder.load_state_dict(ae.encoder.state_dict(), strict=True)
    names = [n for n, _ in clf.named_parameters()]
    assert not any(k in n for n in names for k in ("decoder", "from_latent", "to_latent"))


def test_unit_norm_decoder():
    m = ConvAutoencoder(64)
    with torch.no_grad():
        m.from_latent.weight.mul_(7.0)
    m.renorm_decoder()
    n = torch.linalg.vector_norm(m.from_latent.weight, dim=(0, 2, 3))
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)


def test_augment_and_corrupt():
    g = torch.Generator(device=DEV); g.manual_seed(0)
    x = torch.arange(2 * 3 * 96 * 96, dtype=torch.float32, device=DEV).view(2, 3, 96, 96)
    y = random_crop_flip(x, 12, g)
    assert y.shape == x.shape
    m = corrupt(torch.ones(4, 3, 96, 96, device=DEV), "mask", g, mask_ratio=0.5)
    frac = (m == 0).float().mean().item()
    assert 0.3 < frac < 0.7, frac
    n = corrupt(torch.zeros(4, 3, 96, 96, device=DEV), "gaussian", g, noise_std=0.5)
    assert abs(n.std().item() - 0.5) < 0.02
    # crop without padding = identity up to flip
    z = random_crop_flip(x, 0, g)
    assert all(torch.equal(z[i], x[i]) or torch.equal(z[i], x[i].flip(2)) for i in range(2))


def test_splits():
    y = np.repeat(np.arange(10), 500)
    s = make_splits(y, [0.1, 0.25, 0.5, 1.0], 50, 0, 1000, 100)
    val = set(s["val"])
    assert len(val) == 500 and np.all(np.bincount(y[s["val"]]) == 50)
    prev = set()
    for k in ["0.1", "0.25", "0.5", "1"]:
        cur = set(s["labeled_subsets"][k])
        assert not cur & val and prev <= cur  # disjoint from val, nested
        counts = np.bincount(y[list(cur)], minlength=10)
        assert counts.min() == counts.max()  # stratified
        prev = cur
    assert len(prev) == 4500
    assert not set(s["unlabeled_holdout"]) & set(s["unlabeled_train"])


def test_metrics():
    yt = np.array([0, 0, 1, 1, 2, 2]); yp = np.array([0, 1, 1, 1, 2, 0])
    r = classification_report(yt, yp, k=3)
    assert abs(r["top1"] - 4 / 6) < 1e-9 and r["per_class_accuracy"] == [0.5, 1.0, 0.5]
    X = np.random.randn(200, 16)
    assert abs(linear_cka(X, X) - 1) < 1e-6 and abs(linear_cka(X, 3 * X @ np.linalg.qr(np.random.randn(16, 16))[0]) - 1) < 1e-6
    acc = SparsityAccumulator(4)
    z = torch.zeros(2, 4, 2, 2); z[0, 0, 0, 0] = 1.0
    acc.update(z)
    s = acc.summary()
    assert s["fraction_active"] == 1 / 32 and s["dead_channel_fraction"] == 0.75


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
