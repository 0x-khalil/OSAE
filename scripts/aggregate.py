"""Collect downstream runs into result tables, statistical comparisons and figures.

Selection rule: whenever a table row has several candidate variants (e.g. "Sparse OAE" may be the
clean, Gaussian-denoising or masking-denoising variant), the variant with the highest *mean validation*
top-1 accuracy (over all fractions and seeds of that protocol) is chosen. Test numbers are never used
for any choice.

Statistics: per labelled fraction, methods are compared with a paired t-test over seeds (seeds share
the labelled subset and the data order), Holm-corrected across the four fractions, plus Cohen's d_z and
an exact McNemar test per seed on the test predictions (the largest p over seeds is reported). With 3 seeds the t-test has little
power; p-values should be read together with effect sizes and the per-seed numbers.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from osae.data import CLASSES
from osae.utils import load_json, save_json


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for m in runs_dir.rglob("metrics.json"):
        d = m.parent
        # layout: runs/downstream/<canonical>/<protocol>/f<fraction>/s<seed>
        try:
            canonical, protocol, f, s = d.relative_to(runs_dir).parts[-4:]
        except ValueError:
            continue
        mj = load_json(m)
        row = {"canonical": canonical, "protocol": protocol, "fraction": float(f[1:]), "seed": int(s[1:]),
               "val_top1": mj["best"]["val_top1"], "val_macro_f1": mj["best"].get("val_macro_f1"),
               "best_step": mj["best"]["step"], "train_time_s": mj["train_time_s"],
               "peak_train_mem_mb": mj["peak_gpu_mem_mb"], "params_total": mj["params_total"],
               "params_trainable": mj["params_trainable"], "n_train": mj["n_train"], "dir": str(d)}
        t = d / "test_metrics.json"
        if t.exists():
            tj = load_json(t)
            row.update({"test_top1": tj["top1"], "test_macro_f1": tj["macro_f1"],
                        "per_class_accuracy": tj["per_class_accuracy"], "confusion_matrix": tj["confusion_matrix"]})
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "–"
    return f"{100 * x.mean():.2f} ± {100 * x.std(ddof=1):.2f}" if len(x) > 1 else f"{100 * x.mean():.2f}"


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    from scipy.stats import binomtest
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def compare(df: pd.DataFrame, a: str, b: str, protocol: str, fractions, y_test) -> list[dict]:
    from scipy.stats import ttest_rel
    out = []
    for f in fractions:
        A = df[(df.canonical == a) & (df.protocol == protocol) & (df.fraction == f)].set_index("seed")
        B = df[(df.canonical == b) & (df.protocol == protocol) & (df.fraction == f)].set_index("seed")
        seeds = sorted(set(A.index) & set(B.index))
        if "test_top1" not in A or "test_top1" not in B or len(seeds) < 2:
            continue
        xa, xb = A.loc[seeds, "test_top1"].values, B.loc[seeds, "test_top1"].values
        diff = xb - xa
        t = ttest_rel(xb, xa)
        dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")
        mc = None
        if y_test is not None:
            pa = [Path(A.loc[s, "dir"]) / "test_predictions.npy" for s in seeds]
            pb = [Path(B.loc[s, "dir"]) / "test_predictions.npy" for s in seeds]
            if all(p.exists() for p in pa + pb):
                # one exact McNemar test per seed (predictions of different seeds are not independent);
                # report the largest p, i.e. the difference must hold in every seed
                mc = max(mcnemar_exact(np.load(x) == y_test, np.load(z) == y_test) for x, z in zip(pa, pb))
        out.append({"A": a, "B": b, "protocol": protocol, "fraction": f, "n_seeds": len(seeds),
                    "mean_A": float(xa.mean()), "mean_B": float(xb.mean()), "diff_B_minus_A": float(diff.mean()),
                    "diff_std": float(diff.std(ddof=1)), "t": float(t.statistic), "p_paired_t": float(t.pvalue),
                    "cohens_dz": dz, "p_mcnemar_max_over_seeds": mc})
    if out:
        adj = holm([r["p_paired_t"] if np.isfinite(r["p_paired_t"]) else 1.0 for r in out])
        for r, p in zip(out, adj):
            r["p_paired_t_holm"] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default="runs/downstream")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--variants", default="results/variants.json", help="written by run_all.py")
    ap.add_argument("--data_dir", default="data/stl10")
    args = ap.parse_args()
    res = Path(args.results_dir)
    (res / "figures").mkdir(parents=True, exist_ok=True)

    V = load_json(args.variants)
    variants, main_rows, comparisons = V["variants"], V["main_table"], V.get("comparisons", [])
    df = collect(Path(args.runs_dir))
    if df.empty:
        print("no runs found")
        return
    canon2var = {}
    for v, info in variants.items():
        canon2var.setdefault(info["canonical"], []).append(v)
    df["variant"] = df.canonical.map(lambda c: "/".join(canon2var.get(c, [c])))
    df.drop(columns=[c for c in ["per_class_accuracy", "confusion_matrix"] if c in df]).to_csv(res / "all_runs.csv", index=False)
    fractions = sorted(df.fraction.unique())
    has_test = "test_top1" in df
    y_test = None
    try:
        y_test = np.load(Path(args.data_dir) / "y_test.npy")
    except FileNotFoundError:
        pass

    report = ["# Results", "", "All numbers: mean ± std over seeds, in %. Test = official STL-10 test split (8,000 images); "
              "every choice below was made on the validation split.", ""]
    selections = {}

    for protocol in sorted(df.protocol.unique()):
        dp = df[df.protocol == protocol]
        # --------------------------------------------------- main table
        chosen = {}
        for row_name, cands in main_rows.items():
            cands = [variants[c]["canonical"] for c in cands if c in variants]
            avail = [c for c in cands if c in set(dp.canonical)]
            if not avail:
                continue
            val = {c: dp[dp.canonical == c].val_top1.mean() for c in avail}
            chosen[row_name] = max(val, key=val.get)
            selections[f"{protocol}/{row_name}"] = {"chosen": chosen[row_name], "mean_val_top1": val}
        for metric in (["test_top1", "test_macro_f1"] if has_test else []) + ["val_top1"]:
            tab = []
            for row_name, c in chosen.items():
                r = {"Model": row_name, "variant": c}
                for f in fractions:
                    r[f"{int(round(f * 100))}%"] = mean_std(dp[(dp.canonical == c) & (dp.fraction == f)][metric])
                tab.append(r)
            report += [f"## {protocol} — {metric}", "", pd.DataFrame(tab).to_markdown(index=False), ""]

        # --------------------------------------------------- ablation table (every variant)
        metric = "test_top1" if has_test else "val_top1"
        abl = []
        for c in sorted(dp.canonical.unique()):
            info = next((variants[v] for v in canon2var.get(c, []) if v in variants), {})
            r = {"variant": "/".join(canon2var.get(c, [c])), "latent": info.get("latent_channels", "–"),
                 "lambda": info.get("l1_lambda", "–"), "noise": info.get("noise", "–")}
            for f in fractions:
                r[f"{int(round(f * 100))}%"] = mean_std(dp[(dp.canonical == c) & (dp.fraction == f)][metric])
            r["val (all f)"] = f"{100 * dp[dp.canonical == c].val_top1.mean():.2f}"
            abl.append(r)
        report += [f"## {protocol} — all variants ({metric})", "", pd.DataFrame(abl).to_markdown(index=False), ""]

        # --------------------------------------------------- cost table
        cost = dp.groupby("canonical").agg(train_time_s=("train_time_s", "mean"), peak_train_mem_mb=("peak_train_mem_mb", "mean"),
                                            params_total=("params_total", "first"), params_trainable=("params_trainable", "first"))
        report += [f"## {protocol} — downstream training cost (mean over runs)", "", cost.round(1).to_markdown(), ""]

        # --------------------------------------------------- statistics
        if has_test:
            stats = []
            for a, b in comparisons:
                ca = chosen.get(a, variants.get(a, {}).get("canonical", a))
                cb = chosen.get(b, variants.get(b, {}).get("canonical", b))
                stats += compare(df, ca, cb, protocol, fractions, y_test)
            if stats:
                st = pd.DataFrame(stats)
                st.to_csv(res / f"stats_{protocol}.csv", index=False)
                show = st[["A", "B", "fraction", "n_seeds", "diff_B_minus_A", "diff_std", "p_paired_t", "p_paired_t_holm",
                           "cohens_dz", "p_mcnemar_max_over_seeds"]].copy()
                show["diff_B_minus_A"] *= 100
                show["diff_std"] *= 100
                report += [f"## {protocol} — paired comparisons (test top-1, B − A in points)", "",
                           show.to_markdown(index=False, floatfmt=".4g"), ""]

        # --------------------------------------------------- figures
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        metric = "test_top1" if has_test else "val_top1"
        fig, ax = plt.subplots(figsize=(6, 4))
        for row_name, c in chosen.items():
            g = dp[dp.canonical == c].groupby("fraction")[metric]
            m, s = g.mean(), g.std(ddof=1).fillna(0)
            ax.errorbar(m.index * 100, m.values * 100, yerr=s.values * 100, marker="o", capsize=3, label=f"{row_name} ({c})")
        ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
        ax.set_xlabel("% of labelled training pool"); ax.set_ylabel(f"{metric} (%)"); ax.set_title(f"STL-10 — {protocol}")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(res / "figures" / f"label_efficiency_{protocol}.png", dpi=140); plt.close(fig)

        if has_test:
            for f in [min(fractions), max(fractions)]:
                fig, axes = plt.subplots(1, len(chosen), figsize=(4.2 * len(chosen), 4), squeeze=False)
                for ax, (row_name, c) in zip(axes[0], chosen.items()):
                    cms = dp[(dp.canonical == c) & (dp.fraction == f)].confusion_matrix.dropna()
                    if len(cms) == 0:
                        ax.axis("off"); continue
                    cm = np.mean([np.asarray(x, dtype=float) for x in cms], axis=0)
                    cm = cm / cm.sum(1, keepdims=True)
                    ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
                    ax.set_xticks(range(10), CLASSES, rotation=90, fontsize=6); ax.set_yticks(range(10), CLASSES, fontsize=6)
                    for i in range(10):
                        for j in range(10):
                            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=5,
                                    color="w" if cm[i, j] > 0.5 else "k")
                    ax.set_title(f"{row_name}", fontsize=8)
                fig.suptitle(f"Row-normalised test confusion matrix, {protocol}, {int(f * 100)}% labels (mean over seeds)", fontsize=9)
                fig.tight_layout(); fig.savefig(res / "figures" / f"confusion_{protocol}_f{f:g}.png", dpi=140); plt.close(fig)

                pcs = []
                for row_name, c in chosen.items():
                    v = dp[(dp.canonical == c) & (dp.fraction == f)].per_class_accuracy.dropna()
                    if len(v):
                        pcs.append((row_name, np.mean([np.asarray(x) for x in v], axis=0)))
                if pcs:
                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    w = 0.8 / len(pcs)
                    for k, (row_name, acc) in enumerate(pcs):
                        ax.bar(np.arange(10) + k * w, acc * 100, w, label=row_name)
                    ax.set_xticks(np.arange(10) + 0.4 - w / 2, CLASSES, fontsize=7)
                    ax.set_ylabel("per-class test acc (%)"); ax.legend(fontsize=7)
                    ax.set_title(f"Per-class accuracy, {protocol}, {int(f * 100)}% labels")
                    fig.tight_layout(); fig.savefig(res / "figures" / f"per_class_{protocol}_f{f:g}.png", dpi=140); plt.close(fig)

    # frozen vs fine-tuned per variant
    if set(df.protocol) >= {"frozen", "finetune"}:
        metric = "test_top1" if has_test else "val_top1"
        piv = df.groupby(["canonical", "protocol", "fraction"])[metric].mean().unstack("protocol").dropna()
        piv["finetune_minus_frozen"] = piv["finetune"] - piv["frozen"]
        report += [f"## Frozen vs fine-tuned ({metric}, mean over seeds)", "", (piv * 100).round(2).to_markdown(), ""]

    # pre-training cost
    pre = []
    for v, info in variants.items():
        p = Path(info.get("pretrain_dir", "")) / "metrics.json"
        if info.get("pretrain_dir") and p.exists():
            m = load_json(p)
            pre.append({"variant": v, "latent_dim": m["latent_dim"], "overcomplete_x": round(m["overcompleteness"], 3),
                        "params_encoder": m["params"]["encoder_total"], "params_decoder": m["params"]["decoder_total"],
                        "train_time_h": m["train_time_s"] / 3600, "img_per_s": m["train_throughput_img_s"],
                        "peak_mem_mb": m["peak_gpu_mem_mb"], "holdout_psnr": m["holdout_final"]["psnr_db"],
                        "frac_active": m["holdout_final"]["fraction_active"]})
    if pre:
        report += ["## Autoencoder pre-training", "", pd.DataFrame(pre).to_markdown(index=False, floatfmt=".4g"), ""]
    bench = res / "inference_benchmark.json"
    if bench.exists():
        b = load_json(bench)
        report += ["## Inference cost (identical downstream architecture for all models)", "",
                   f"Parameters: {b['params_total']:,}", "",
                   pd.DataFrame(b["benchmark"]).T.to_markdown(floatfmt=".3g"), ""]
    report += ["## Selections (validation only)", "", "```", str(selections), "```"]
    (res / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    save_json(selections, res / "selections.json")
    print("\n".join(report))


if __name__ == "__main__":
    main()
