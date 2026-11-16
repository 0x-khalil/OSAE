"""Publish the study's results (and optionally checkpoints) to a Hugging Face Hub repository.

The token is read from the HF_TOKEN environment variable (or a prior `huggingface-cli login`).
It is never written to disk or into the repository.

    export HF_TOKEN=...            # in the pod / shell only
    python scripts/publish_hf.py --repo_id 0x-Khalil/OSAE

Uploaded:
  README.md                      model card generated from results/RESULTS.md
  results/**                     tables, statistics, figures, representation analysis, selections
  encoders/<variant>/encoder.pt  pre-trained ResNet-18 trunks (one per AE variant) + metrics/history/recon
  configs/, data_splits/         exact config and split indices for reproducibility
  downstream/**                  per-run metrics (+ best checkpoints with --include_downstream_ckpts, ~10 GB)
"""
import _bootstrap  # noqa: F401

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from osae.utils import ROOT, load_json, load_yaml


def model_card(repo_id: str, cfg: dict, results_md: str, lambda_sel: dict | None, github: str | None) -> str:
    ls = f"λ* = {lambda_sel['lambda_star']:g} (selected on validation)" if lambda_sel else "λ* not available"
    src = f"\nCode: {github}\n" if github else ""
    return f"""---
tags:
- autoencoder
- sparse-coding
- representation-learning
- self-supervised
- stl10
- resnet18
---

# Overcomplete Sparse Autoencoders for Visual Representation Learning (STL-10)

Controlled study of whether an **overcomplete, L1-sparse convolutional autoencoder** learns features that transfer
better to STL-10 classification than a **conventional undercomplete autoencoder**, against a **supervised
ResNet-18 trained from scratch**, at 10 / 25 / 50 / 100 % of the labelled data.
{src}
* Shared feature extractor: ResNet-18 adapted to 96×96 (3×3/2 stem, no max-pool); identical downstream classifier for all models.
* Standard AE latent: 128×6×6 = 4,608 coefficients (0.17× input). Overcomplete AE latent: 2048×6×6 = 73,728 (2.67× input).
* Pre-training on the 100,000 unlabeled images; {ls}.
* Downstream: full fine-tuning and frozen-encoder protocols, {len(cfg['seeds'])} seeds, identical stratified subsets,
  validation-only model selection, test split used once for final evaluation.

## Files

| Path | Content |
|---|---|
| `results/RESULTS.md` | headline table, ablations, paired statistics, costs |
| `results/figures/` | label-efficiency curves, confusion matrices, per-class accuracy |
| `results/representation/` | sparsity, reconstruction, separability, PCA / t-SNE / UMAP, CKA |
| `encoders/<variant>/encoder.pt` | pre-trained ResNet-18 trunk (`torch.load(...)["encoder"]`) |
| `downstream/` | per-run metrics (and checkpoints if uploaded) |
| `configs/`, `data_splits/` | exact configuration and split indices |

Load a pre-trained trunk with the code from the repository:

```python
import torch
from osae.models import Classifier
clf = Classifier()
clf.encoder.load_state_dict(torch.load("encoders/oae_sparse/encoder.pt")["encoder"])
```

---

{results_md}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", required=True, help="e.g. 0x-Khalil/OSAE")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--repo_type", default="model", choices=["model", "dataset"])
    ap.add_argument("--private", action="store_true", help="create the repo as private if it does not exist")
    ap.add_argument("--include_downstream_ckpts", action="store_true", help="also upload every best.pt (~45 MB each)")
    ap.add_argument("--github", default=None, help="code URL to cite in the model card")
    ap.add_argument("--commit_message", default="Upload OSAE study results")
    ap.add_argument("--dry_run", action="store_true", help="stage files locally and list them without uploading")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    cfg = load_yaml(ROOT / args.config)
    runs, results = ROOT / cfg["runs_dir"], ROOT / cfg["results_dir"]
    if not (results / "RESULTS.md").exists():
        raise SystemExit(f"{results / 'RESULTS.md'} not found - run the study (scripts/run_all.py) first")

    token = os.environ.get("HF_TOKEN")  # falls back to cached login if unset
    api = HfApi(token=token)
    if not args.dry_run:
        who = api.whoami()
        print(f"authenticated as {who.get('name')}")

    stage = Path(tempfile.mkdtemp(prefix="osae_hf_"))
    try:
        shutil.copytree(results, stage / "results")
        shutil.copytree(ROOT / "configs", stage / "configs", ignore=shutil.ignore_patterns("_*"))
        dd = ROOT / cfg["data_dir"]
        (stage / "data_splits").mkdir()
        for f in ["splits.json", "splits_summary.md", "stats.json"]:
            if (dd / f).exists():
                shutil.copy2(dd / f, stage / "data_splits" / f)

        variants = load_json(results / "variants.json")["variants"] if (results / "variants.json").exists() else {}
        seen = set()
        # main variant names (e.g. oae_sparse) win over the lambda-sweep alias (oae_l0.01) of the same checkpoint
        for name, info in sorted(variants.items(), key=lambda kv: kv[0].startswith("oae_l")):
            pdir = Path(info.get("pretrain_dir", ""))
            if not info.get("pretrain_dir") or info["canonical"] in seen or not (pdir / "encoder.pt").exists():
                continue
            seen.add(info["canonical"])
            dst = stage / "encoders" / name
            dst.mkdir(parents=True)
            for f in ["encoder.pt", "metrics.json", "history.json", "recon.png"]:
                if (pdir / f).exists():
                    shutil.copy2(pdir / f, dst / f)
            (dst / "canonical.txt").write_text(info["canonical"] + "\n", encoding="utf-8")

        down = runs / "downstream"
        keep = ["metrics.json", "test_metrics.json", "test_predictions.npy"]
        if args.include_downstream_ckpts:
            keep.append("best.pt")
        for m in down.rglob("metrics.json"):
            rel = m.parent.relative_to(down)
            dst = stage / "downstream" / rel
            dst.mkdir(parents=True, exist_ok=True)
            for f in keep:
                if (m.parent / f).exists():
                    shutil.copy2(m.parent / f, dst / f)

        lam = results / "lambda_selection.json"
        card = model_card(args.repo_id, cfg, (results / "RESULTS.md").read_text(encoding="utf-8"),
                          load_json(lam) if lam.exists() else None, args.github)
        (stage / "README.md").write_text(card, encoding="utf-8")

        files = [p for p in stage.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files) / 2**20
        print(f"staged {len(files)} files ({size:.1f} MB) in {stage}")
        if args.dry_run:
            for p in sorted(files)[:40]:
                print("  ", p.relative_to(stage))
            return

        api.create_repo(args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
        if size > 2000:  # resumable, chunked upload for multi-GB folders (e.g. with all checkpoints)
            api.upload_large_folder(repo_id=args.repo_id, repo_type=args.repo_type, folder_path=str(stage))
        else:
            api.upload_folder(repo_id=args.repo_id, repo_type=args.repo_type, folder_path=str(stage),
                              commit_message=args.commit_message)
        print(f"published: https://huggingface.co/{args.repo_id}")
    finally:
        if not args.dry_run:
            shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
