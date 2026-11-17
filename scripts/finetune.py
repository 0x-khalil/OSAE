"""Downstream classification on a stratified labelled subset of STL-10 (validation-based model selection).

Initialisations (``--init``)
  scratch     random init, all parameters trained          -> Experiment A (supervised baseline)
  random      random init, trunk frozen (random-feature control for the frozen protocol)
  pretrained  trunk copied from an autoencoder's encoder.pt -> Experiments B / C

Protocols (``--protocol``)
  finetune    all parameters trained
  frozen      trunk frozen in eval mode (BN statistics fixed), only the head is trained

Every run uses the same architecture, optimiser, schedule, step budget, batch size, augmentation and
normalisation (set via the experiment YAML). Model selection uses the validation split only; the test
split is never loaded here. The best-validation checkpoint is evaluated later by scripts/evaluate.py.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from osae.data import GPULoader, infinite, labeled_indices, load_labeled, load_splits, load_stats
from osae.metrics import classification_report
from osae.models import Classifier, load_trunk_into
from osae.utils import (Timer, amp_dtype, amp_name, atomic_torch_save, autocast, cosine_with_warmup, count_params,
                        env_info, get_device, grad_scaler, save_json, set_lr, set_seed)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", default="data/stl10")
    ap.add_argument("--init", choices=["scratch", "random", "pretrained"], required=True)
    ap.add_argument("--pretrained", default=None, help="path to encoder.pt")
    ap.add_argument("--protocol", choices=["finetune", "frozen"], required=True)
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=0, help="early stop after N evals w/o val improvement (0=off)")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"],
                    help="auto = bf16 on Ampere+, fp16 (with loss scaling) on older GPUs such as T4")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    if args.init == "pretrained" and not args.pretrained:
        ap.error("--init pretrained requires --pretrained")
    if args.init == "scratch" and args.protocol == "frozen":
        ap.error("scratch+frozen is the 'random' init")
    if args.init == "random" and args.protocol == "finetune":
        ap.error("random+finetune is identical to 'scratch'")
    return args


@torch.no_grad()
def evaluate(model, loader, dev, adt) -> tuple[dict, float]:
    model.eval()
    ys, ps, loss, n = [], [], 0.0, 0
    for x, y, _ in loader:
        with autocast(dev, adt):
            logits = model(x)
        loss += F.cross_entropy(logits.float(), y, reduction="sum").item()
        n += len(y)
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    return classification_report(np.concatenate(ys), np.concatenate(ps)), loss / n


def main():
    args = parse()
    out = Path(args.out_dir)
    if (out / "metrics.json").exists():
        print(f"[skip] {out} already finished")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.deterministic)
    dev = get_device()
    adt = amp_dtype(args.amp)
    # A frozen *random* trunk has untrained BN statistics, so its activations grow layer by layer and can
    # overflow fp16. That control is forward-only and cheap, so it runs in fp32 (bf16 has fp32's range).
    if args.init == "random" and adt == torch.float16:
        adt = None
    scaler = grad_scaler(adt)

    mean, std = load_stats(args.data_dir)
    splits = load_splits(args.data_dir)
    X, y = load_labeled(args.data_dir, "train")  # test split is not accessible here
    tr_idx = labeled_indices(splits, args.fraction)
    va_idx = np.asarray(splits["val"])
    assert not set(tr_idx.tolist()) & set(va_idx.tolist()), "train/val overlap"
    train_loader = GPULoader(X, y, tr_idx, args.batch_size, dev, mean, std, shuffle=True,
                             drop_last=len(tr_idx) > args.batch_size, augment=True, seed=args.seed)
    val_loader = GPULoader(X, y, va_idx, 500, dev, mean, std)

    model = Classifier()
    if args.init == "pretrained":
        load_trunk_into(model, args.pretrained)
    model = model.to(dev).to(memory_format=torch.channels_last)

    frozen = args.protocol == "frozen"
    if frozen:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.weight_decay},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=args.momentum, nesterov=True)

    def set_train_mode():
        model.train()
        if frozen:
            model.encoder.eval()  # keep pre-trained BN statistics fixed

    timer, history = Timer(), []
    best = {"val_top1": -1.0, "step": 0}
    bad_evals = 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stream = infinite(train_loader)
    set_train_mode()
    timer.start()
    for step in range(1, args.steps + 1):
        set_lr(opt, cosine_with_warmup(step - 1, args.steps, args.warmup_steps, args.lr))
        x, yb, _ = next(stream)
        with autocast(dev, adt):
            if frozen:
                with torch.no_grad():
                    f = model.features(x)
                logits = model.head(f.float())
            else:
                logits = model(x)
        loss = F.cross_entropy(logits.float(), yb, label_smoothing=args.label_smoothing)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)  # skips the update if fp16 gradients overflowed
        scaler.update()
        if step % args.eval_every == 0 or step == args.steps:
            timer.stop()
            rep, vloss = evaluate(model, val_loader, dev, adt)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": vloss, "val_top1": rep["top1"],
                            "val_macro_f1": rep["macro_f1"], "lr": opt.param_groups[0]["lr"],
                            "train_time_s": timer.total})
            if rep["top1"] > best["val_top1"]:  # strict '>' -> earliest best step wins ties
                best = {"val_top1": rep["top1"], "val_macro_f1": rep["macro_f1"], "val_loss": vloss, "step": step}
                state = model.head.state_dict() if (frozen and args.init == "pretrained") else model.state_dict()
                atomic_torch_save({"state_dict": state, "head_only": frozen and args.init == "pretrained",
                                   "pretrained": args.pretrained, "config": vars(args)}, out / "best.pt")
                bad_evals = 0
            else:
                bad_evals += 1
            print(f"step {step}/{args.steps} loss {loss.item():.3f} val_top1 {rep['top1']:.4f} "
                  f"best {best['val_top1']:.4f}@{best['step']}", flush=True)
            set_train_mode()
            if args.patience and bad_evals >= args.patience:
                print("early stopping")
                break
            timer.start()
    if timer._t is not None:
        timer.stop()

    counts = np.bincount(y[tr_idx], minlength=10)
    save_json({
        "config": vars(args),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "train_class_counts": counts.tolist(),
        "best": best,
        "history": history,
        "train_time_s": timer.total,
        "amp_resolved": amp_name(adt),
        "steps_run": history[-1]["step"] if history else 0,
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
        "params_total": count_params(model),
        "params_trainable": count_params(model, trainable_only=True),
        "env": env_info(),
    }, out / "metrics.json")
    print("done:", out, "best val", best)


if __name__ == "__main__":
    main()
