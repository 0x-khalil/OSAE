"""Pre-train a convolutional (under- or over-complete, optionally sparse / denoising) autoencoder
on the STL-10 unlabeled split.

    L = MSE(x, x_hat) + lambda * mean(|z|)

``mean(|z|)`` is the L1 norm ||z||_1 divided by the number of latent coefficients, so the same lambda
applies the same per-coefficient pressure to the undercomplete and the overcomplete code.
For denoising runs the *input* is corrupted and the reconstruction target is the clean image.

Outputs (in --out_dir): encoder.pt (trunk weights for transfer), model.pt (full AE), metrics.json,
history.json, recon.png. ``last.pt`` allows resuming an interrupted run.
"""
import _bootstrap  # noqa: F401

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from osae.data import GPULoader, corrupt, load_array, load_splits, load_stats
from osae.metrics import SparsityAccumulator
from osae.models import ConvAutoencoder
from osae.utils import (Timer, amp_dtype, amp_name, atomic_torch_save, autocast, cosine_with_warmup, env_info,
                        get_device, grad_scaler, save_json, set_lr, set_seed)

EXIT_DEADLINE = 3  # run_all.py treats this exit code as "paused, resume later", not as a failure


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", default="data/stl10")
    ap.add_argument("--latent_channels", type=int, required=True, help="128 = undercomplete, 2048 = overcomplete")
    ap.add_argument("--l1_lambda", type=float, default=0.0)
    ap.add_argument("--noise", choices=["none", "gaussian", "mask"], default="none")
    ap.add_argument("--noise_std", type=float, default=0.5)
    ap.add_argument("--mask_ratio", type=float, default=0.5)
    ap.add_argument("--mask_patch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=None, help="cap total steps (smoke tests)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"],
                    help="auto = bf16 on Ampere+, fp16 (with loss scaling) on older GPUs such as T4")
    ap.add_argument("--deadline", type=float, default=None,
                    help="unix time; past it, save a resumable checkpoint at the next epoch end and exit 3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--holdout_max", type=int, default=None, help="limit holdout images (smoke tests)")
    return ap.parse_args()


@torch.no_grad()
def evaluate_holdout(model, loader, l1_lambda) -> dict:
    model.eval()
    acc = SparsityAccumulator(model.latent_channels)
    mse_sum, n, psnr = 0.0, 0, []
    for x, _, _ in loader:
        x_hat, z = model(x)
        x_hat = x_hat.float()
        mse_sum += F.mse_loss(x_hat, x, reduction="sum").item()
        n += x.numel()
        a, b = loader.denormalize(x), loader.denormalize(x_hat)
        per = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
        psnr.append((10 * torch.log10(1.0 / per)).cpu().numpy())
        acc.update(z)
    s = acc.summary()
    out = {
        "recon_mse_normalized": mse_sum / n,
        "psnr_db": float(np.concatenate(psnr).mean()),
        "l1_mean": s["mean_abs_activation"],
        "objective": mse_sum / n + l1_lambda * s["mean_abs_activation"],
    }
    out.update({k: v for k, v in s.items() if not isinstance(v, list)})
    model.train()
    return out


def save_recon_grid(model, loader, path, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    x, _, _ = next(iter(loader))
    with torch.no_grad():
        x_hat, _ = model(x[:n])
    a = loader.denormalize(x[:n]).float().permute(0, 2, 3, 1).cpu().numpy()
    b = loader.denormalize(x_hat.float()).permute(0, 2, 3, 1).cpu().numpy()
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    for i in range(n):
        axes[0, i].imshow(a[i]); axes[1, i].imshow(b[i])
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_title("input", fontsize=8, loc="left"); axes[1, 0].set_title("reconstruction", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    model.train()


def main():
    args = parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "encoder.pt").exists():
        print(f"[skip] {out} already finished")
        return
    set_seed(args.seed, args.deterministic)
    dev = get_device()
    adt = amp_dtype(args.amp)
    scaler = grad_scaler(adt)

    mean, std = load_stats(args.data_dir)
    splits = load_splits(args.data_dir)
    U = load_array(args.data_dir, "X_unlabeled")  # no labels are loaded for pre-training
    hold = splits["unlabeled_holdout"][: args.holdout_max] if args.holdout_max else splits["unlabeled_holdout"]
    train_loader = GPULoader(U, None, splits["unlabeled_train"], args.batch_size, dev, mean, std,
                             shuffle=True, drop_last=True, augment=True, seed=args.seed)
    hold_loader = GPULoader(U, None, hold, 256, dev, mean, std, shuffle=False, augment=False)

    model = ConvAutoencoder(args.latent_channels).to(dev).to(memory_format=torch.channels_last)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n_.startswith("from_latent") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.999))
    steps_per_epoch = len(train_loader)
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = int(args.warmup_epochs * steps_per_epoch)
    warmup = min(warmup, total // 10) if args.max_steps else warmup

    step, history, start_epoch = 0, [], 0
    if (out / "last.pt").exists():  # resume
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        step, history, start_epoch = ck["step"], ck["history"], ck["epoch"]
        train_loader.rng = np.random.default_rng(args.seed + 1000 * start_epoch)
        print(f"resumed at epoch {start_epoch} step {step}")

    print(f"AE C_z={args.latent_channels} latent={model.latent_numel()} "
          f"({model.overcompleteness():.2f}x input) lambda={args.l1_lambda} noise={args.noise} "
          f"steps={total} params={model.param_breakdown()}", flush=True)

    timer = Timer()
    if history:  # resumed: keep counting training time from where it stopped
        timer.total = history[-1]["train_time_s"]
    print(f"mixed precision: {amp_name(adt)}", flush=True)
    bad_steps = 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    epoch = start_epoch
    t_log = time.perf_counter()
    while step < total:
        timer.start()
        for x, _, _ in train_loader:
            if step >= total:
                break
            set_lr(opt, cosine_with_warmup(step, total, warmup, args.lr))
            x_in = corrupt(x, args.noise, train_loader.gen, args.noise_std, args.mask_ratio, args.mask_patch)
            with autocast(dev, adt):
                x_hat, z = model(x_in)
            rec = F.mse_loss(x_hat.float(), x)       # target is always the clean image
            l1 = z.float().abs().mean()
            loss = rec + args.l1_lambda * l1
            opt.zero_grad(set_to_none=True)
            if not math.isfinite(loss.item()):
                # fp16 forward overflow: skip this batch; abort only if it keeps happening
                bad_steps += 1
                if bad_steps >= 20:
                    raise FloatingPointError("20 consecutive non-finite losses")
                step += 1
                continue
            bad_steps = 0
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            model.renorm_decoder()
            step += 1
            if step % args.log_every == 0:
                act = (z > 0).float().mean().item()
                dt = time.perf_counter() - t_log
                t_log = time.perf_counter()
                print(f"ep {epoch} step {step}/{total} rec {rec.item():.4f} l1 {l1.item():.4f} "
                      f"active {act:.3f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{args.log_every * args.batch_size / dt:.0f} img/s", flush=True)
        timer.stop()
        epoch += 1
        ev = evaluate_holdout(model, hold_loader, args.l1_lambda)
        ev.update({"epoch": epoch, "step": step, "train_time_s": timer.total})
        history.append(ev)
        print(f"== epoch {epoch}: holdout mse {ev['recon_mse_normalized']:.4f} psnr {ev['psnr_db']:.2f} "
              f"active {ev['fraction_active']:.3f} dead {ev['dead_channel_fraction']:.3f}", flush=True)
        save_json(history, out / "history.json")
        atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                           "step": step, "epoch": epoch, "history": history}, out / "last.pt")
        if args.deadline and time.time() > args.deadline and step < total:
            print(f"deadline reached after epoch {epoch}: checkpoint saved, exiting to resume later", flush=True)
            sys.exit(EXIT_DEADLINE)

    final = evaluate_holdout(model, hold_loader, args.l1_lambda)
    save_recon_grid(model, hold_loader, out / "recon.png")
    cfg = vars(args)
    atomic_torch_save({"model": model.state_dict(), "config": cfg}, out / "model.pt")
    save_json({
        "config": cfg,
        "latent_shape": model.latent_shape(),
        "latent_dim": model.latent_numel(),
        "input_dim": 3 * 96 * 96,
        "overcompleteness": model.overcompleteness(),
        "params": model.param_breakdown(),
        "steps": step,
        "train_time_s": timer.total,
        "amp_resolved": amp_name(adt),
        "train_throughput_img_s": step * args.batch_size / max(timer.total, 1e-9),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
        "holdout_final": final,
        "env": env_info(),
    }, out / "metrics.json")
    # encoder.pt is the "finished" marker, so it is written last
    atomic_torch_save({"encoder": model.encoder.state_dict(), "config": cfg}, out / "encoder.pt")
    (out / "last.pt").unlink(missing_ok=True)
    print("done:", out)


if __name__ == "__main__":
    main()
