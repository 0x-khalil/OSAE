"""What do the autoencoders learn? Representation analysis on labelled train-pool + validation images.

For each model (``name=path``; path = pre-training dir with model.pt, downstream run dir with best.pt,
or the literal ``random``) this computes, without touching the test split:

* reconstruction quality (MSE in normalised space, PSNR) on the unlabeled holdout and validation images
* sparsity of the latent code z: fraction of active units, active units per image, Hoyer sparsity,
  dead / rarely-used channels, activation-value histogram, per-channel activation frequency
* class structure of two fixed representations - GAP(trunk) (the 512-d feature the classifier uses)
  and GAP(z) - via a logistic-regression probe and 20-NN (fit on the train pool, scored on validation),
  silhouette score and Fisher ratio
* 2-D embeddings (PCA, t-SNE, UMAP if installed) coloured by class
* linear CKA between trunk features of all models, and a complementarity probe on concatenated
  features of model pairs
* correlation between reconstruction quality and separability across autoencoder variants
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from osae.data import CLASSES, GPULoader, labeled_indices, load_array, load_labeled, load_splits, load_stats
from osae.metrics import SparsityAccumulator, linear_cka, separability
from osae.models import Classifier, ConvAutoencoder
from osae.utils import amp_dtype, autocast, get_device, save_json, set_seed


def load_model(path: str, dev):
    if path == "random":
        return "classifier", Classifier().to(dev).eval()
    p = Path(path)
    if (p / "model.pt").exists():
        ck = torch.load(p / "model.pt", map_location="cpu", weights_only=False)
        m = ConvAutoencoder(ck["config"]["latent_channels"])
        m.load_state_dict(ck["model"])
        return "ae", m.to(dev).eval()
    if (p / "best.pt").exists():
        from evaluate import build  # same directory
        return "classifier", build(p, dev)
    raise FileNotFoundError(f"no model.pt / best.pt in {p}")


@torch.no_grad()
def extract(kind, model, loader, dev, adt, sparsity: SparsityAccumulator | None):
    trunk, lat, mse, psnr = [], [], 0.0, []
    n = 0
    for x, _, _ in loader:
        with autocast(dev, adt):
            h = model.encoder(x)
            if kind == "ae":
                z = F.relu(model.to_latent(h))
                x_hat = model.decode(z)
        trunk.append(F.adaptive_avg_pool2d(h.float(), 1).flatten(1).cpu())
        if kind == "ae":
            z = z.float()
            lat.append(z.mean(dim=(2, 3)).cpu())
            if sparsity is not None:
                sparsity.update(z)
            x_hat = x_hat.float()
            mse += F.mse_loss(x_hat, x, reduction="sum").item()
            n += x.numel()
            a, b = loader.denormalize(x), loader.denormalize(x_hat)
            per = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
            psnr.append((10 * torch.log10(1 / per)).cpu().numpy())
    out = {"trunk": torch.cat(trunk).numpy()}
    if kind == "ae":
        out["latent"] = torch.cat(lat).numpy()
        out["recon_mse_normalized"] = mse / n
        out["psnr_db"] = float(np.concatenate(psnr).mean())
    return out


def scatter_grid(embs: dict, y, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(embs)
    cols = min(4, k)
    rows = -(-k // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, (name, E) in zip(axes.flat, embs.items()):
        for c in range(10):
            m = y == c
            ax.scatter(E[m, 0], E[m, 1], s=3, color=cmap(c), label=CLASSES[c], alpha=0.7, linewidths=0)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in list(axes.flat)[k:]:
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(c), label=CLASSES[c]) for c in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=path entries")
    ap.add_argument("--pairs", nargs="*", default=[], help="name1,name2 pairs for the complementarity probe")
    ap.add_argument("--data_dir", default="data/stl10")
    ap.add_argument("--out_dir", default="results/representation")
    ap.add_argument("--embed_samples", type=int, default=2000)
    ap.add_argument("--max_pool", type=int, default=None, help="limit train-pool images (smoke tests)")
    ap.add_argument("--no_umap", action="store_true")
    ap.add_argument("--amp", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = get_device()
    adt = amp_dtype(args.amp)
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    mean, std = load_stats(args.data_dir)
    splits = load_splits(args.data_dir)
    X, y = load_labeled(args.data_dir, "train")
    pool = labeled_indices(splits, 1.0)
    if args.max_pool:
        pool = np.random.default_rng(0).choice(pool, args.max_pool, replace=False)
    val = np.asarray(splits["val"])
    U = load_array(args.data_dir, "X_unlabeled")
    hold = np.asarray(splits["unlabeled_holdout"])

    def L(arr, idx):
        return GPULoader(arr, None, np.sort(idx), 250, dev, mean, std)

    ytr = y[np.sort(pool)]
    yva = y[np.sort(val)]
    feats, summary = {}, {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        kind, model = load_model(path, dev)
        print(f"== {name} ({kind}) from {path}", flush=True)
        sp = SparsityAccumulator(model.latent_channels) if kind == "ae" else None
        madt = None if (path == "random" and adt == torch.float16) else adt  # random trunk: fp32 (overflow)
        ftr = extract(kind, model, L(X, pool), dev, madt, sp)
        fva = extract(kind, model, L(X, val), dev, madt, sp)
        res = {"kind": kind, "path": path}
        if kind == "ae":
            fh = extract(kind, model, L(U, hold), dev, madt, None)
            res.update({
                "latent_channels": model.latent_channels,
                "latent_dim": model.latent_numel(),
                "overcompleteness": model.overcompleteness(),
                "recon_mse_holdout": fh["recon_mse_normalized"], "psnr_holdout_db": fh["psnr_db"],
                "recon_mse_val": fva["recon_mse_normalized"], "psnr_val_db": fva["psnr_db"],
                "sparsity": sp.summary(),
            })
        for rep in (["trunk", "latent"] if kind == "ae" else ["trunk"]):
            s = separability(ftr[rep], ytr, fva[rep], yva, seed=args.seed)
            res[f"sep_{rep}"] = s
            print(f"   {rep:6s} probe {s['linear_probe_val_acc']:.4f} knn {s['knn20_val_acc']:.4f} "
                  f"sil {s['silhouette_val_pca50_cosine']:.3f} fisher {s['fisher_ratio_val']:.3f}", flush=True)
        feats[name] = {"tr": ftr, "va": fva}
        summary[name] = res
        del model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- CKA + complementarity
    names = list(feats)
    allf = {n: np.concatenate([feats[n]["tr"]["trunk"], feats[n]["va"]["trunk"]]) for n in names}
    cka = np.array([[linear_cka(allf[a], allf[b]) for b in names] for a in names])
    comp = {}
    for pair in args.pairs:
        a, b = pair.split(",")
        if a in feats and b in feats:
            cat_tr = np.concatenate([feats[a]["tr"]["trunk"], feats[b]["tr"]["trunk"]], 1)
            cat_va = np.concatenate([feats[a]["va"]["trunk"], feats[b]["va"]["trunk"]], 1)
            s = separability(cat_tr, ytr, cat_va, yva)
            comp[pair] = {"concat_probe": s["linear_probe_val_acc"],
                          a: summary[a]["sep_trunk"]["linear_probe_val_acc"],
                          b: summary[b]["sep_trunk"]["linear_probe_val_acc"], "cka": float(cka[names.index(a), names.index(b)])}

    # ---------------------------------------------------------------- reconstruction vs separability
    ae_names = [n for n in names if summary[n]["kind"] == "ae"]
    recon_vs_sep = None
    if len(ae_names) >= 3:
        from scipy.stats import spearmanr
        r = [summary[n]["psnr_holdout_db"] for n in ae_names]
        for rep in ["trunk", "latent"]:
            s = [summary[n][f"sep_{rep}"]["linear_probe_val_acc"] for n in ae_names]
            rho, p = spearmanr(r, s)
            recon_vs_sep = recon_vs_sep or {}
            recon_vs_sep[rep] = {"spearman_rho_psnr_vs_probe": float(rho), "p_value": float(p), "n_models": len(ae_names)}

    save_json({"models": summary, "cka_trunk": {"names": names, "matrix": cka.tolist()},
               "complementarity": comp, "reconstruction_vs_separability": recon_vs_sep,
               "n_pool": int(len(pool)), "n_val": int(len(val)), "n_holdout": int(len(hold))},
              out / "summary.json")

    # ---------------------------------------------------------------- tables
    import pandas as pd
    rows = []
    for n, r in summary.items():
        row = {"model": n, "kind": r["kind"]}
        if r["kind"] == "ae":
            sp = r["sparsity"]
            row.update({"latent_dim": r["latent_dim"], "overcomplete_x": round(r["overcompleteness"], 3),
                        "psnr_holdout": r["psnr_holdout_db"], "mse_holdout": r["recon_mse_holdout"],
                        "frac_active": sp["fraction_active"], "active_units/img": sp["active_units_per_sample_mean"],
                        "hoyer": sp["hoyer_sparsity_mean"], "dead_ch": sp["dead_channel_fraction"]})
            for k, v in r["sep_latent"].items():
                row[f"latent_{k}"] = v
        for k, v in r["sep_trunk"].items():
            row[f"trunk_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(args.seed)
    yall = np.concatenate([ytr, yva])
    sub = rng.choice(len(yall), min(args.embed_samples, len(yall)), replace=False)
    try:
        import umap  # noqa: F401
        has_umap = not args.no_umap
    except ImportError:
        has_umap = False
    for rep in ["trunk", "latent"]:
        mats = {n: np.concatenate([feats[n]["tr"][rep], feats[n]["va"][rep]])[sub]
                for n in names if rep in feats[n]["tr"]}
        if not mats:
            continue
        std_m = {n: StandardScaler().fit_transform(M) for n, M in mats.items()}
        scatter_grid({n: PCA(2, random_state=args.seed).fit_transform(M) for n, M in std_m.items()},
                     yall[sub], f"PCA of GAP({rep}) features", out / "figures" / f"pca_{rep}.png")
        pcs = {n: PCA(min(50, M.shape[1]), random_state=args.seed).fit_transform(M) for n, M in std_m.items()}
        scatter_grid({n: TSNE(2, init="pca", perplexity=30, random_state=args.seed).fit_transform(P)
                      for n, P in pcs.items()}, yall[sub], f"t-SNE of GAP({rep}) features",
                     out / "figures" / f"tsne_{rep}.png")
        if has_umap:
            import umap
            scatter_grid({n: umap.UMAP(n_components=2, random_state=args.seed).fit_transform(P)
                          for n, P in pcs.items()}, yall[sub], f"UMAP of GAP({rep}) features",
                         out / "figures" / f"umap_{rep}.png")

    if ae_names:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for n in ae_names:
            sp = summary[n]["sparsity"]
            bins = np.asarray(sp["hist_bins"])
            cnt = np.asarray(sp["hist_counts_nonzero"], dtype=float)
            total = cnt.sum() + sp["count_exact_zero"]
            centers = np.sqrt(np.maximum(bins[:-1], 1e-5) * bins[1:])
            axes[0].plot(centers, cnt / max(total, 1), label=f"{n} (zeros {sp['count_exact_zero'] / max(total, 1):.0%})")
            freq = np.sort(np.asarray(sp["channel_activation_frequency"]))[::-1]
            axes[1].plot(np.linspace(0, 1, len(freq)), freq, label=n)
        axes[0].set_xscale("log"); axes[0].set_xlabel("activation value (z > 0)"); axes[0].set_ylabel("fraction of all coefficients")
        axes[0].set_title("Latent activation distribution")
        axes[1].set_xlabel("channel rank (normalised)"); axes[1].set_ylabel("P(channel active)"); axes[1].set_yscale("log")
        axes[1].set_title("Per-channel activation frequency")
        for a in axes:
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / "latent_activations.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, rep in zip(axes, ["trunk", "latent"]):
            for n in ae_names:
                ax.scatter(summary[n]["psnr_holdout_db"], summary[n][f"sep_{rep}"]["linear_probe_val_acc"])
                ax.annotate(n, (summary[n]["psnr_holdout_db"], summary[n][f"sep_{rep}"]["linear_probe_val_acc"]), fontsize=7)
            ax.set_xlabel("holdout PSNR (dB)"); ax.set_ylabel("linear-probe val acc"); ax.set_title(f"Reconstruction vs separability: {rep}")
        fig.tight_layout()
        fig.savefig(out / "figures" / "recon_vs_separability.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(names), 0.8 + 0.6 * len(names)))
    im = ax.imshow(cka, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cka[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if cka[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Linear CKA of GAP(trunk) features")
    fig.tight_layout()
    fig.savefig(out / "figures" / "cka_trunk.png", dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
