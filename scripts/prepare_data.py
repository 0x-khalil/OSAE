"""Download STL-10, convert to uint8 .npy arrays, compute normalisation stats and fixed splits.

Outputs in --data_dir:
  X_train.npy (5000,3,96,96) uint8   y_train.npy (5000,) int64 in [0,9]
  X_test.npy  (8000,3,96,96) uint8   y_test.npy  (8000,)
  X_unlabeled.npy (100000,3,96,96) uint8           (no labels exist / none are used)
  stats.json   per-channel mean/std computed on the *unlabeled* split only
  splits.json  validation indices, nested stratified labelled subsets, unlabeled holdout
  splits_summary.md  human-readable report of the exact splits
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np

from osae.data import CLASSES, NUM_CLASSES, array_path, make_splits
from osae.utils import fmt_key, save_json

SHAPE = (3, 96, 96)
MIRRORS = [  # same official archive (MD5 91f7769df0f17e558f3565bffb0c7dfb)
    "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
    "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
]


def convert_images(src: Path, dst: Path, n: int, chunk: int = 5000) -> None:
    # STL-10 binaries store each image channel-major and column-major -> transpose H/W.
    raw = np.memmap(src, dtype=np.uint8, mode="r", shape=(n, *SHAPE))
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.uint8, shape=(n, *SHAPE))
    for i in range(0, n, chunk):
        out[i:i + chunk] = raw[i:i + chunk].transpose(0, 1, 3, 2)
    out.flush()
    del out, raw


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/stl10")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--unlabeled_holdout", type=int, default=2000)
    ap.add_argument("--delete_raw", action="store_true", help="remove the downloaded archive/binaries afterwards")
    args = ap.parse_args()

    d = Path(args.data_dir)
    d.mkdir(parents=True, exist_ok=True)
    names = {"X_train": 5000, "X_test": 8000, "X_unlabeled": 100000}
    if not all(array_path(d, k).exists() for k in names):
        import torchvision

        raw = d / "raw"
        print("Downloading / extracting STL-10 (~2.6 GB) via torchvision (MD5-verified) ...", flush=True)
        for url in MIRRORS:
            torchvision.datasets.STL10.url = url
            try:
                torchvision.datasets.STL10(root=str(raw), split="train", download=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  mirror failed ({url}): {e}", flush=True)
        else:
            raise RuntimeError("could not download STL-10 from any official mirror")
        b = raw / "stl10_binary"
        for split, n in [("train", 5000), ("test", 8000), ("unlabeled", 100000)]:
            print(f"converting {split} ({n} images)", flush=True)
            convert_images(b / f"{split}_X.bin", array_path(d, f"X_{split}"), n)
            if split != "unlabeled":
                y = np.fromfile(b / f"{split}_y.bin", dtype=np.uint8).astype(np.int64) - 1
                assert y.shape == (n,) and y.min() == 0 and y.max() == NUM_CLASSES - 1
                np.save(array_path(d, f"y_{split}"), y)
        if args.delete_raw:
            shutil.rmtree(raw, ignore_errors=True)

    # --- normalisation statistics: unlabeled split only (never the test split)
    if not (d / "stats.json").exists():
        U = np.load(array_path(d, "X_unlabeled"), mmap_mode="r")
        s = np.zeros(3)
        ss = np.zeros(3)
        cnt = 0
        for i in range(0, len(U), 5000):
            x = U[i:i + 5000].astype(np.float64) / 255.0
            s += x.sum(axis=(0, 2, 3))
            ss += (x ** 2).sum(axis=(0, 2, 3))
            cnt += x.shape[0] * x.shape[2] * x.shape[3]
        mean = s / cnt
        std = np.sqrt(ss / cnt - mean ** 2)
        save_json({"mean": mean.tolist(), "std": std.tolist(), "computed_on": "unlabeled (100,000 images)"},
                  d / "stats.json")
        print("stats:", mean.round(4), std.round(4))

    # --- splits
    y_train = np.load(array_path(d, "y_train"))
    n_unl = np.load(array_path(d, "X_unlabeled"), mmap_mode="r").shape[0]
    splits = make_splits(y_train, args.fractions, args.val_per_class, args.split_seed, n_unl, args.unlabeled_holdout)
    splits["checksums_sha256"] = {f"{k}.npy": sha256(array_path(d, k)) for k in ["X_train", "y_train", "y_test"]}
    save_json(splits, d / "splits.json")

    lines = ["# STL-10 splits", "",
             f"split_seed = {args.split_seed}; validation = {args.val_per_class} images/class drawn from the "
             "official 5,000-image training split; the official 8,000-image test split is untouched.", "",
             "| subset | images | " + " | ".join(CLASSES) + " |",
             "|---|---|" + "---|" * NUM_CLASSES]

    def row(name, idx):
        counts = np.bincount(y_train[np.asarray(idx)], minlength=NUM_CLASSES)
        return f"| {name} | {len(idx)} | " + " | ".join(map(str, counts)) + " |"

    lines.append(row("validation", splits["val"]))
    for f in args.fractions:
        lines.append(row(f"train {int(round(f * 100))}%", splits["labeled_subsets"][fmt_key(f)]))
    lines += ["", f"Unlabeled: {len(splits['unlabeled_train'])} images for AE training, "
                  f"{len(splits['unlabeled_holdout'])} held out for reconstruction monitoring.",
              "", "Labelled subsets are nested (10% ⊂ 25% ⊂ 50% ⊂ 100%) and identical for every method and seed.",
              "Exact indices (into the official train / unlabeled arrays) are in `splits.json`."]
    (d / "splits_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
