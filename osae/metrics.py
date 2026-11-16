"""Evaluation metrics: classification, sparsity, representation geometry, compute cost."""
from __future__ import annotations

import time

import numpy as np
import torch

from .data import NUM_CLASSES


# ----------------------------------------------------------------------------- classification

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> dict:
    cm = confusion_matrix(y_true, y_pred, k)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    predicted = cm.sum(0).astype(float)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)  # = per-class accuracy
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    return {
        "top1": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class_accuracy": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "n": int(cm.sum()),
    }


# ----------------------------------------------------------------------------- sparsity

class SparsityAccumulator:
    """Streaming statistics of a non-negative latent code z of shape (B, C, H, W)."""

    def __init__(self, channels: int, hist_bins: np.ndarray | None = None):
        self.C = channels
        self.n = 0
        self.coeffs = 0
        self.active = 0
        self.active_per_sample = []
        self.hoyer = []
        self.l1_mean = []
        self.channel_freq = torch.zeros(channels, dtype=torch.float64)
        self.channel_mean = torch.zeros(channels, dtype=torch.float64)
        self.bins = hist_bins if hist_bins is not None else np.concatenate([[0.0], np.logspace(-4, 2, 61)])
        self.hist = np.zeros(len(self.bins) - 1, dtype=np.int64)
        self.zeros = 0

    @torch.no_grad()
    def update(self, z: torch.Tensor):
        z = z.float()
        B = z.shape[0]
        flat = z.flatten(1)
        act = flat > 0
        self.n += B
        self.coeffs += flat.numel()
        self.active += int(act.sum())
        self.active_per_sample.append(act.sum(1).cpu().numpy())
        d = flat.shape[1]
        l1 = flat.abs().sum(1)
        l2 = flat.norm(dim=1).clamp_min(1e-12)
        self.hoyer.append(((d ** 0.5 - l1 / l2) / (d ** 0.5 - 1)).cpu().numpy())
        self.l1_mean.append(flat.abs().mean(1).cpu().numpy())
        self.channel_freq += (z > 0).double().mean(dim=(2, 3)).sum(0).cpu()
        self.channel_mean += z.double().mean(dim=(2, 3)).sum(0).cpu()
        v = flat[act]
        self.zeros += int((~act).sum())
        edges = torch.as_tensor(self.bins, device=v.device, dtype=v.dtype)
        idx = torch.bucketize(v, edges, right=True) - 1  # bins[i] <= v < bins[i+1]
        idx = idx[(idx >= 0) & (idx < len(self.bins) - 1)]
        self.hist += torch.bincount(idx, minlength=len(self.bins) - 1).cpu().numpy()

    def summary(self) -> dict:
        aps = np.concatenate(self.active_per_sample)
        freq = (self.channel_freq / max(self.n, 1)).numpy()
        d = self.coeffs // max(self.n, 1)
        return {
            "latent_dim": int(d),
            "fraction_active": self.active / max(self.coeffs, 1),
            "active_units_per_sample_mean": float(aps.mean()),
            "active_units_per_sample_std": float(aps.std()),
            "active_channels_per_location_mean": float(aps.mean() / max(d // self.C, 1)),
            "hoyer_sparsity_mean": float(np.concatenate(self.hoyer).mean()),
            "mean_abs_activation": float(np.concatenate(self.l1_mean).mean()),
            "dead_channel_fraction": float((freq == 0).mean()),
            "rarely_active_channel_fraction(<1%)": float((freq < 0.01).mean()),
            "channel_activation_frequency": freq.tolist(),
            "hist_bins": self.bins.tolist(),
            "hist_counts_nonzero": self.hist.tolist(),
            "count_exact_zero": int(self.zeros),
        }


# ----------------------------------------------------------------------------- geometry

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019) between two feature matrices with the same rows."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro") + 1e-12))


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """trace(S_between) / trace(S_within) on standardised features (scale-free separability)."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    mu = X.mean(0)
    sb = sw = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        mc = Xc.mean(0)
        sb += len(Xc) * float(((mc - mu) ** 2).sum())
        sw += float(((Xc - mc) ** 2).sum())
    return sb / max(sw, 1e-12)


def separability(Xtr, ytr, Xva, yva, seed: int = 0) -> dict:
    """Label-efficiency-free probes of class structure in a fixed representation.

    Fits on the labelled training pool and scores on the validation split (never the test split).
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xva)
    logreg = LogisticRegression(max_iter=3000, C=1.0)
    logreg.fit(A, ytr)
    knn = KNeighborsClassifier(n_neighbors=20, metric="cosine").fit(A, ytr)
    k = min(50, A.shape[1], A.shape[0] - 1)
    P = PCA(n_components=k, random_state=seed).fit(A)
    Pva = P.transform(B)
    return {
        "linear_probe_val_acc": float(logreg.score(B, yva)),
        "knn20_val_acc": float(knn.score(B, yva)),
        "silhouette_val_pca50_cosine": float(silhouette_score(Pva, yva, metric="cosine")),
        "fisher_ratio_val": fisher_ratio(Xva, yva),
        "pca50_explained_variance": float(P.explained_variance_ratio_.sum()),
    }


# ----------------------------------------------------------------------------- compute cost

@torch.no_grad()
def inference_benchmark(model: torch.nn.Module, device, amp_dtype=None, batch_sizes=(1, 256),
                        iters: int = 100, warmup: int = 20) -> dict:
    from .utils import autocast

    model.eval()
    out = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 96, 96, device=device).contiguous(memory_format=torch.channels_last)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        times = []
        n_it = iters if bs == 1 else max(10, iters // 5)
        for i in range(warmup + n_it):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            with autocast(device, amp_dtype):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - t)
        t = np.asarray(times)
        out[f"bs{bs}"] = {
            "latency_ms_median": float(np.median(t) * 1e3),
            "latency_ms_p90": float(np.percentile(t, 90) * 1e3),
            "throughput_img_s": float(bs / np.median(t)),
            "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
        }
    return out
