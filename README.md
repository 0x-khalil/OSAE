# Overcomplete Sparse Autoencoders for Visual Representation Learning (STL-10)

**Hypothesis under test.** An *overcomplete, sparse* convolutional code preserves more useful local
visual structure than a conventional *compressed* autoencoder code, and therefore transfers better to
image classification, especially when few labels are available.

This is a hypothesis, not a premise. The code is built to give a fair, controlled and reproducible
answer either way: identical downstream architectures, identical labelled subsets, validation-only
model selection, a sealed test set, multiple seeds, paired statistics, and factorial ablations that
separate the effects of overcompleteness, sparsity and denoising.

---

## 1. Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python tests/test_core.py
```

```bash
.venv/Scripts/python scripts/run_all.py --config configs/smoke.yaml
```

```bash
.venv/Scripts/python scripts/run_all.py --config configs/default.yaml
```

The smoke config exercises every stage in about 30 minutes, with meaningless numbers. The default config
is the full study.

**Measured cost on an RTX 3070 Laptop (8 GB), bf16:**

| Job | Throughput | Time per job | Number | Total |
|---|---|---|---|---|
| AE pre-training (peak memory ~4.5 GB) | ~1,060 img/s | ~31 min (20 epochs) | 11 | **~6 h** |
| Fine-tuning (2,000 steps) | — | ~3.5 min | 120 | ~7 h |
| Frozen-encoder runs | — | ~1.5 min | 120 | ~3 h |

The full study takes roughly **16–17 h** of GPU time. To cut it, lower `pretrain.epochs` or
`downstream.common.steps`, or run stages selectively with `--stages` / `--only`. On Linux/macOS, use `.venv/bin/python` instead of `.venv/Scripts/python`.

### One command on a rented pod: full study + publish to Hugging Face

```bash
git clone https://github.com/0x-khalil/OSAE.git && cd OSAE
```

```bash
export HF_TOKEN=hf_your_token_here
```

```bash
tmux new -s osae
```

```bash
bash scripts/run_pod.sh
```

`scripts/run_pod.sh` does the following:
1. Creates the venv, with PyTorch for CUDA 12.8.
2. Checks the GPU and runs the unit tests.
3. Runs every stage of the study with 4 concurrent downstream jobs and 2 concurrent pre-trainings.
4. Uploads the results to <https://huggingface.co/0x-Khalil/OSAE> with `scripts/publish_hf.py`.

The upload contains:
- `RESULTS.md` as the model card
- tables, statistics and figures
- the representation analysis
- one pre-trained encoder per variant
- per-run metrics and predictions
- the exact configs and split indices

Optional settings are environment variables: `HF_REPO`, `JOBS`, `PRETRAIN_JOBS`, `CONFIG`,
`SKIP_PUBLISH=1`, `UPLOAD_CKPTS=1` (adds all ~10 GB of downstream checkpoints). The script is safe to
re-run after an interruption. The full log is written to `runs/pod_<timestamp>.log`.

**The token is only ever read from the `HF_TOKEN` environment variable.** Never put it in a file in
this repository.

### Running on a large rented GPU (e.g. RTX 4090 / 5090)

Small jobs leave a big GPU mostly idle. Each downstream run needs under 1 GB of GPU memory, and each AE
pre-training about 4.5 GB. `run_all.py` can therefore run several jobs on the same GPU at once:

- `--jobs N` sets how many downstream runs execute concurrently.
- `--pretrain_jobs M` sets how many AE pre-trainings execute concurrently.
- Both can also be set under `parallel:` in the config.
- A failed job does not stop the others. All failures are listed at the end, and a rerun redoes only the unfinished jobs.

On a 5090 (Blackwell), PyTorch must be built for CUDA 12.8 or later. The `cu126` wheel above will not run.
On a Linux box:

```bash
python -m venv .venv && .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && .venv/bin/python -m pip install -r requirements.txt
```

Start the run inside `tmux` so an SSH disconnect does not stop it:

```bash
tmux new -s osae
```

```bash
.venv/bin/python scripts/run_all.py --config configs/default.yaml --jobs 4 --pretrain_jobs 2
```

Estimated time on a 5090: about 4–6 h sequentially, or about 2.5–3.5 h with `--jobs 4 --pretrain_jobs 2`.
These figures are extrapolated from the laptop measurements, not measured on a 5090.

Running jobs concurrently does not change the protocol. Every run keeps its own seed, data order and
hyper-parameters. Only wall-clock time and the measured `train_time_s` change: per-run timings measured
under concurrency are inflated by GPU sharing. If you report training time as a cost metric, take it from
a sequential run (`--jobs 1`).

Every stage is **resumable**: finished runs are detected and skipped, and an interrupted AE
pre-training resumes from `last.pt`. Useful flags:

```bash
.venv/Scripts/python scripts/run_all.py --config configs/default.yaml --dry_run
```

```bash
.venv/Scripts/python scripts/run_all.py --config configs/default.yaml --stages downstream --only oae_sparse
```

Outputs:
- `results/RESULTS.md` — headline table, ablations, statistics, costs
- `results/representation/` — representation analysis and figures
- `results/figures/` — label-efficiency curves, confusion matrices, per-class accuracy
- `runs/` — every run with its config, history and checkpoints
- `runs/logs/` — full logs

---

## 2. Data protocol (STL-10)

| Split | Size | Used for |
|---|---|---|
| unlabeled | 100,000 | AE pre-training (98,000) + reconstruction monitoring (2,000 held out). No labels exist or are used. |
| train | 5,000 | 500 stratified **validation** images (50/class) + 4,500-image **labelled pool** |
| test | 8,000 | **final evaluation only** |

* `scripts/prepare_data.py` downloads the official archive (MD5-verified) and converts it to uint8 `.npy`. It also computes the normalisation mean/std **on the unlabeled split only** and writes `data/stl10/splits.json` with the exact indices, plus a readable `splits_summary.md`.
* The labelled subsets of 10/25/50/100 % of the pool (450/1,120/2,250/4,500 images) are **stratified** (equal per-class counts) and **nested**. They come from one fixed `split_seed`, so they are **identical for every method and every seed**.
* **Test-set isolation is enforced in code.** `osae.data.load_labeled("test")` raises unless `allow_test=True` is passed explicitly, and only `scripts/evaluate.py` passes it. Training, early stopping, λ selection, variant selection and representation analysis never see test images.

---

## 3. Models

### Shared feature extractor: ResNet-18 adapted to 96×96

The standard ResNet-18 has 2-2-2-2 BasicBlocks with widths 64/128/256/512. The ImageNet stem
(7×7/2 conv + max-pool) is replaced with a **3×3/2 conv without max-pool**, so the spatial sizes are
96 → 48 → 24 → 12 → 6. The trunk has 11.17M parameters and outputs 512×6×6.

**The downstream classifier is identical for every experiment:** trunk → global average pooling
(512-d) → BatchNorm1d (no affine parameters) → Linear(512, 10). The latent projection and the decoder
exist only during pre-training. They are discarded at transfer: only the trunk weights are copied, with
a strict key match. As a result, downstream parameter count (11.18M), FLOPs and inference latency are the
same for all models by construction.

### Autoencoders (Experiments B and C)

```
x (3×96×96) ──ResNet-18 trunk──► h (512×6×6) ──1×1 conv + ReLU──► z (C_z×6×6) ──decoder──► x̂
```

| | C_z | latent coefficients | vs. input (27,648) | encoder params | total params |
|---|---|---|---|---|---|
| Standard (undercomplete) AE | 128 | 4,608 | 0.17× | 11.23M | 12.87M |
| Overcomplete AE | 2,048 | 73,728 | **2.67×** | 12.22M (+8.8%) | 14.84M (+15%) |

* Both models have **the same encoder depth and the same trunk**. They differ only in the width of the 1×1 latent projection. The extra 1M parameters of the OAE are the unavoidable cost of a wider code, and that part is discarded before classification.
* The latent code is spatial (6×6 grid), so local structure is kept.
* The **decoder** has 4× (nearest upsample + 3×3 conv + ReLU) stages, 6 → 96, then a final 3×3 conv. It uses **no normalisation layers**. Its first 1×1 layer has **unit-L2-norm columns**, re-projected after every step. This matters: otherwise the L1 penalty could be removed trivially by shrinking z and scaling up the decoder, or by a BatchNorm cancelling the scale.

### Objectives

```
L_AE  = MSE(x, x̂)
L_OAE = MSE(x, x̂) + λ · mean(|z|)          (= λ/|z| · ||z||_1)
```

The L1 term is normalised by the number of latent coefficients, so a given λ applies the same
per-coefficient pressure to both code sizes. For **denoising**, the input is corrupted and the target
stays clean. Two corruptions are available:
- Gaussian noise: σ = 0.5 in normalised units
- random masking: 50% of 8×8 patches set to the mean colour

Pre-training settings: AdamW, lr 1e-3, warm-up + cosine, batch 128, 20 epochs, crop + flip augmentation, bf16 autocast.

### Downstream protocols

| init \ protocol | `finetune` (all params) | `frozen` (trunk frozen, BN stats fixed, head trained) |
|---|---|---|
| `scratch` (random) | **Experiment A: supervised ResNet-18 baseline** | — |
| `random` (random) | — | **random-feature control** |
| `pretrained` (AE trunk) | Experiments B/C, full fine-tuning | Experiments B/C, frozen encoder |

Identical for all methods, fractions and seeds:
- SGD + Nesterov, momentum 0.9, batch 128
- **fixed 2,000-step budget** with 100 warm-up steps and cosine decay
- the same reflect-pad-12 random crop + horizontal flip
- the same normalisation

Learning rate and weight decay are fixed per protocol:
- fine-tune: lr 0.05, weight decay 5e-4
- frozen: lr 0.1, weight decay 0

Validation accuracy is computed every 100 steps, and the **best-validation checkpoint** is kept (early
stopping by model selection).

Why a fixed step budget instead of fixed epochs: with 450 images, a fixed epoch count would give the
10% runs about 10× fewer updates than the 100% runs. A fixed number of steps gives every run the same
optimisation budget.

---

## 4. Experimental design

**Stage 1: sparsity sweep.**
- Pre-train the OAE (C_z = 2048, clean inputs) with λ ∈ {0, 0.001, 0.01, 0.1}.
- Pick **λ\*** (λ > 0) by the mean **validation** accuracy of the frozen-encoder probe at 10% and 100% labels. The choice is written to `results/lambda_selection.json`.
- Effects of λ on sparsity, reconstruction and separability for every λ are reported by the representation analysis.

**Stage 2: factorial ablation** (all with λ\* from stage 1):

| variant | latent | sparsity | input | isolates |
|---|---|---|---|---|
| `ae` | under (128) | λ=0 | clean | **Experiment B** |
| `ae_sparse` | under | λ\* | clean | sparsity without overcompleteness |
| `ae_denoise` | under | λ=0 | Gaussian | denoising without overcompleteness |
| `ae_sparse_denoise` | under | λ\* | Gaussian | |
| `oae` | over (2048) | λ=0 | clean | overcompleteness alone |
| `oae_sparse` | over | λ\* | clean | **Experiment C** (sparse OAE) |
| `oae_denoise` | over | λ=0 | Gaussian | |
| `oae_sparse_denoise` | over | λ\* | Gaussian | **Experiment C** (denoising) |
| `oae_sparse_mask` | over | λ\* | masking | **Experiment C** (denoising) |

The 2×2×2 core ({under, over} × {λ=0, λ\*} × {clean, Gaussian}) allows main effects and interactions
to be read directly. Examples:
- `oae` − `ae`: overcompleteness alone
- `oae_sparse` − `oae`: sparsity given overcompleteness
- `ae_sparse` − `ae`: sparsity given undercompleteness

Every variant runs under both protocols × 4 fractions × 3 seeds. With the baseline and the random
control, that makes **9×2×4×3 + 12 + 12 = 240 downstream runs**.

**Headline table.**
- "Sparse OAE" is the best of {`oae_sparse`, `oae_sparse_denoise`, `oae_sparse_mask`}, chosen by **mean validation accuracy**. The choice is recorded in `results/selections.json`.
- Every variant is also reported separately in the ablation tables, so this selection cannot hide anything.

**Seeds.**
- Downstream seeds {0, 1, 2} control head/model initialisation, data order and augmentation.
- By default, each AE variant is pre-trained once (seed 0). Set `pretrain_seeds: [0, 1, 2]` to pre-train one AE per downstream seed, which captures pre-training variance at 3× the pre-training cost.

---

## 5. Metrics and statistics

For each downstream run, `runs/downstream/<variant>/<protocol>/f<fraction>/s<seed>/` contains:
- `metrics.json`: validation curve, best step, **training time**, **peak GPU memory**, **parameter counts**
- `test_metrics.json`, written by `evaluate.py`: **top-1**, **macro-F1**, balanced accuracy, **per-class accuracy**, **confusion matrix**, and **inference latency/throughput/memory** at batch 1 and 256
- `test_predictions.npy`

Pre-training cost (time, throughput, peak memory, encoder/decoder parameters) is in
`runs/pretrain/*/metrics.json`.

`aggregate.py` reports, per protocol:
- mean ± std over seeds for every method × fraction
- **paired t-tests** over seeds, Holm-corrected across fractions
- **Cohen's d_z**
- an **exact McNemar test per seed** on test predictions (largest p over seeds reported)

With 3 seeds the t-test has low power. Read p-values together with effect sizes and the per-seed
numbers in `results/all_runs.csv`.

## 6. Representation analysis (`scripts/analyze_representations.py`, no test data)

The analysis covers every AE variant, a random-init trunk, and the supervised 100%-label model as a
reference. It reports:

- **Reconstruction:** MSE and PSNR on the unlabeled holdout and validation images; reconstruction grids in `runs/pretrain/*/recon.png`.
- **Sparsity of z:**
  - fraction of active units
  - **average number of active latent units per image**
  - active channels per spatial location
  - Hoyer sparsity
  - dead and rarely-active channels
  - **activation-value distributions** and per-channel activation frequency (`figures/latent_activations.png`)
- **Class separability:** computed for both GAP(trunk) (the feature the classifier sees) and GAP(z):
  - logistic-regression probe and 20-NN, fit on the labelled pool and scored on validation
  - silhouette score
  - Fisher ratio
- **Embeddings:** **PCA**, **t-SNE** and **UMAP** (`figures/{pca,tsne,umap}_{trunk,latent}.png`).
- **Complementarity:**
  - linear **CKA** between all models (`figures/cka_trunk.png`)
  - a probe on concatenated features of model pairs, e.g. AE + OAE: if the concatenation beats both, the features are complementary
- **Reconstruction ≠ classification:** Spearman correlation between holdout PSNR and probe accuracy across AE variants (`figures/recon_vs_separability.png`). This tests the claim directly instead of assuming it.

---

## 7. Repository layout

```
configs/default.yaml        full protocol (single source of truth for shared hyper-parameters)
configs/smoke.yaml          minutes-long end-to-end check
osae/data.py                STL-10 arrays, stratified nested splits, GPU loader/augmentation/corruption
osae/models.py              ResNet-18 trunk (96x96), Classifier, ConvAutoencoder (unit-norm decoder)
osae/metrics.py             classification report, sparsity accumulator, CKA, separability, latency
scripts/prepare_data.py     download + convert + normalisation stats + splits
scripts/pretrain.py         AE / OAE pre-training (L1 sparsity, Gaussian/mask denoising)
scripts/finetune.py         downstream training (scratch / random / pretrained × finetune / frozen)
scripts/evaluate.py         FINAL test evaluation (the only code that reads the test split)
scripts/analyze_representations.py
scripts/aggregate.py        tables, paired statistics, figures -> results/RESULTS.md
scripts/run_all.py          resumable orchestration of all stages
tests/test_core.py          dataset-free sanity checks
```

Individual scripts can also be run directly. See `--help` on each.

## 8. Reproducibility notes

* The configuration, environment (Python/PyTorch/CUDA/GPU) and full training history are stored with every run. Data checksums are in `splits.json`.
* `cudnn.benchmark` is on for speed. Pass `--deterministic` to `pretrain.py` / `finetune.py` for bit-reproducible (slower) runs.
* Known limitations:
  - one pre-training seed per variant by default (see §4)
  - fixed downstream learning rates rather than a per-method LR search
  - 20 pre-training epochs, which is short by self-supervised standards. Absolute accuracies are therefore below published STL-10 numbers. The comparison is controlled, but conclusions apply to this budget.
