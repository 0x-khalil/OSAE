"""FINAL evaluation on the official STL-10 test split (8,000 images).

This is the only script that opens the test split. Run it once all training and validation-based
selection is complete. For every finished downstream run (``metrics.json`` + ``best.pt``) it writes
``test_metrics.json`` with top-1 accuracy, macro-F1, per-class accuracy, confusion matrix, and
inference cost (latency, throughput, peak GPU memory, parameter count).
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import torch

from osae.data import GPULoader, load_labeled, load_stats
from osae.metrics import classification_report, inference_benchmark
from osae.models import Classifier, load_trunk_into
from osae.utils import amp_dtype, autocast, count_params, get_device, load_json, save_json


def resolve_encoder(run_dir: Path, stored: str) -> Path:
    """The stored encoder path is absolute; after moving / restoring runs (e.g. a new Kaggle session) fall
    back to the same pre-training folder under this run's runs/ root
    (runs/downstream/<variant>/<protocol>/f<frac>/s<seed> -> runs/pretrain/<name>/encoder.pt)."""
    p = Path(stored)
    if p.exists():
        return p
    alt = Path(run_dir).resolve().parents[4] / "pretrain" / p.parent.name / "encoder.pt"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"pre-trained encoder not found: {stored} (also tried {alt})")


def run_amp(run_dir: Path, adt):
    """The frozen random-trunk control is evaluated in fp32 when fp16 would be used (overflow risk)."""
    cfg = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["config"]
    return None if (cfg.get("init") == "random" and adt == torch.float16) else adt


def build(run_dir: Path, dev) -> Classifier:
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = Classifier()
    if ck["head_only"]:
        load_trunk_into(model, str(resolve_encoder(run_dir, ck["pretrained"])))
        model.head.load_state_dict(ck["state_dict"])
    else:
        model.load_state_dict(ck["state_dict"])
    return model.to(dev).to(memory_format=torch.channels_last).eval()


@torch.no_grad()
def predict(model, loader, dev, adt):
    ys, ps = [], []
    for x, y, _ in loader:
        with autocast(dev, adt):
            ps.append(model(x).argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_dir", default="runs/downstream")
    ap.add_argument("--data_dir", default="data/stl10")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--amp", default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dev = get_device()
    adt = amp_dtype(args.amp)
    mean, std = load_stats(args.data_dir)
    X, y = load_labeled(args.data_dir, "test", allow_test=True)
    loader = GPULoader(X, y, np.arange(len(y)), 500, dev, mean, std)

    runs = sorted(p.parent for p in Path(args.runs_dir).rglob("metrics.json") if (p.parent / "best.pt").exists())
    todo = [r for r in runs if args.force or not (r / "test_metrics.json").exists()]
    print(f"{len(runs)} finished runs, {len(todo)} to evaluate on the test split")

    bench_path = Path(args.results_dir) / "inference_benchmark.json"
    bench = load_json(bench_path) if bench_path.exists() else None
    for i, r in enumerate(todo):
        model = build(r, dev)
        if bench is None:  # identical architecture for all runs -> benchmark once
            bench = {"params_total": count_params(model), "amp": args.amp,
                     "benchmark": inference_benchmark(model, dev, adt)}
            save_json(bench, bench_path)
        yt, yp = predict(model, loader, dev, run_amp(r, adt))
        rep = classification_report(yt, yp)
        rep["inference"] = bench
        save_json(rep, r / "test_metrics.json")
        np.save(r / "test_predictions.npy", yp.astype(np.int8))  # enables paired (McNemar) tests
        print(f"[{i + 1}/{len(todo)}] {r}: top1 {rep['top1']:.4f} macroF1 {rep['macro_f1']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
