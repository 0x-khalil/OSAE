"""STL-10 loading, split construction and a GPU-side batch loader.

Design notes
------------
* Images are stored once as uint8 ``.npy`` arrays of shape (N, 3, 96, 96) and opened with
  ``mmap_mode='r'``; the OS page cache keeps them in RAM after the first epoch.
* All augmentation (reflect-pad random crop + horizontal flip), normalisation and denoising
  corruption run on the GPU. This avoids slow multi-process DataLoaders on Windows and keeps
  the pipeline identical for every experiment.
* The official test split is only ever opened by ``scripts/evaluate.py`` (final evaluation).
  :func:`load_labeled` refuses to return it unless ``allow_test=True`` is passed explicitly.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .utils import fmt_key, load_json

CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
NUM_CLASSES = 10
IMG_SIZE = 96
INPUT_DIM = 3 * IMG_SIZE * IMG_SIZE  # 27,648 input coefficients


# ----------------------------------------------------------------------------- files

def array_path(data_dir: str | Path, name: str) -> Path:
    return Path(data_dir) / f"{name}.npy"


def load_array(data_dir: str | Path, name: str, mmap: bool = True) -> np.ndarray:
    return np.load(array_path(data_dir, name), mmap_mode="r" if mmap else None)


def load_stats(data_dir: str | Path) -> tuple[list[float], list[float]]:
    s = load_json(Path(data_dir) / "stats.json")
    return s["mean"], s["std"]


def load_splits(data_dir: str | Path) -> dict:
    return load_json(Path(data_dir) / "splits.json")


def load_labeled(data_dir: str | Path, split: str, allow_test: bool = False):
    if split == "test" and not allow_test:
        raise RuntimeError("The STL-10 test split is reserved for final evaluation (scripts/evaluate.py).")
    X = load_array(data_dir, f"X_{split}")
    y = np.load(array_path(data_dir, f"y_{split}"))
    return X, y


def labeled_indices(splits: dict, fraction: float) -> np.ndarray:
    return np.asarray(splits["labeled_subsets"][fmt_key(fraction)], dtype=np.int64)


# ----------------------------------------------------------------------------- splits

def make_splits(y_train: np.ndarray, fractions, val_per_class: int, split_seed: int,
                n_unlabeled: int, unlabeled_holdout: int) -> dict:
    """Stratified validation split + nested stratified labelled subsets + unlabeled holdout.

    * ``val``: ``val_per_class`` images per class, never used for gradient updates.
    * ``labeled_subsets[f]``: the first ``round(f * n_c)`` images of a fixed per-class permutation
      of the remaining pool, so subsets are nested (10% ⊂ 25% ⊂ 50% ⊂ 100%) and class-balanced.
    * ``unlabeled_holdout``: images of the unlabeled split used only to monitor reconstruction.
    """
    rng = np.random.default_rng(split_seed)
    val, pool = [], {}
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(y_train == c)
        perm = rng.permutation(idx)
        val.append(perm[:val_per_class])
        pool[c] = perm[val_per_class:]
    subsets = {}
    for f in fractions:
        parts = [pool[c][: int(round(float(f) * len(pool[c])))] for c in range(NUM_CLASSES)]
        subsets[fmt_key(f)] = np.sort(np.concatenate(parts)).tolist()
    urng = np.random.default_rng(split_seed + 1)
    uperm = urng.permutation(n_unlabeled)
    return {
        "split_seed": split_seed,
        "val_per_class": val_per_class,
        "val": np.sort(np.concatenate(val)).tolist(),
        "labeled_subsets": subsets,
        "unlabeled_holdout": np.sort(uperm[:unlabeled_holdout]).tolist(),
        "unlabeled_train": np.sort(uperm[unlabeled_holdout:]).tolist(),
    }


# ----------------------------------------------------------------------------- GPU ops

def random_crop_flip(x: torch.Tensor, pad: int, gen: torch.Generator) -> torch.Tensor:
    """Per-sample reflect-padded random crop (translation of up to ±pad px) + horizontal flip."""
    B, C, H, W = x.shape
    dev = x.device
    if pad > 0:
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        oy = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
        ox = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
        rows = oy[:, None] + torch.arange(H, device=dev)[None]
        cols = ox[:, None] + torch.arange(W, device=dev)[None]
        bidx = torch.arange(B, device=dev)[:, None, None]
        x = xp.permute(0, 2, 3, 1)[bidx, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    return x.contiguous()


def corrupt(x: torch.Tensor, noise: str, gen: torch.Generator, noise_std: float = 0.5,
            mask_ratio: float = 0.5, mask_patch: int = 8) -> torch.Tensor:
    """Denoising corruption applied to the *normalised* input; the target stays clean.

    * ``gaussian``: additive N(0, noise_std^2) noise (normalised units, i.e. ~noise_std channel stds).
    * ``mask``: random square patches (``mask_patch`` px) are zeroed (= dataset mean colour) with
      probability ``mask_ratio``.
    """
    if noise in (None, "none"):
        return x
    if noise == "gaussian":
        return x + noise_std * torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype)
    if noise == "mask":
        B, _, H, W = x.shape
        gh, gw = H // mask_patch, W // mask_patch
        keep = (torch.rand(B, 1, gh, gw, device=x.device, generator=gen) >= mask_ratio).to(x.dtype)
        keep = F.interpolate(keep, size=(H, W), mode="nearest")
        return x * keep
    raise ValueError(f"unknown noise type {noise!r}")


# ----------------------------------------------------------------------------- loader

class GPULoader:
    """Iterates over ``X[indices]`` in batches, returning normalised float tensors on ``device``.

    A background thread gathers uint8 batches from the (memory-mapped) array into pinned memory
    so disk/RAM reads overlap with GPU compute. Shuffling and augmentation are driven by the
    loader's own seeded generators, so two loaders with the same seed see identical streams.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None, indices, batch_size: int, device,
                 mean, std, shuffle: bool = False, drop_last: bool = False, augment: bool = False,
                 pad: int = 12, seed: int = 0, prefetch: int = 6):
        self.X, self.y = X, y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.bs, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.augment, self.pad = augment, pad
        self.device = torch.device(device)
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.prefetch = prefetch
        self.pin = self.device.type == "cuda"

    def __len__(self) -> int:
        n = len(self.indices)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def _batches(self):
        idx = self.rng.permutation(self.indices) if self.shuffle else self.indices
        n = len(idx)
        stop = (n // self.bs) * self.bs if self.drop_last else n
        for i in range(0, stop, self.bs):
            yield np.sort(idx[i:i + self.bs])  # sorted -> sequential memmap reads

    def _producer(self, q: queue.Queue, batches):
        try:
            for b in batches:
                xb = torch.from_numpy(np.ascontiguousarray(self.X[b]))
                yb = torch.from_numpy(np.asarray(self.y[b], dtype=np.int64)) if self.y is not None else None
                if self.pin:
                    xb = xb.pin_memory()
                q.put((xb, yb, b))
        except Exception as e:  # surface errors in the consumer
            q.put(e)
        q.put(None)

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        t = threading.Thread(target=self._producer, args=(q, list(self._batches())), daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            xb, yb, b = item
            x = xb.to(self.device, non_blocking=True).float().div_(255.0)
            if self.augment:
                x = random_crop_flip(x, self.pad, self.gen)
            x = (x - self.mean) / self.std
            x = x.contiguous(memory_format=torch.channels_last)
            if yb is not None:
                yb = yb.to(self.device, non_blocking=True)
            yield x, yb, b
        t.join()

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std + self.mean).clamp(0, 1)


def infinite(loader: GPULoader):
    """Endless stream of batches; reshuffles each pass."""
    while True:
        for batch in loader:
            yield batch
