"""Small shared helpers: seeding, JSON/YAML IO, timing, parameter counting, environment info."""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(name: str | None):
    """Mixed-precision dtype. 'auto' = bf16 on Ampere or newer (RTX 30xx/40xx/50xx, A100...), fp16 on
    older GPUs without bf16 support (e.g. Kaggle's Tesla T4 / P100), None on CPU."""
    name = (name or "none").lower()
    if name == "auto":
        if not torch.cuda.is_available():
            return None
        return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None, "fp32": None}[name]


def amp_name(dtype) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", None: "fp32"}[dtype]


def grad_scaler(dtype):
    """Loss scaling is needed for fp16 (narrow exponent range), not for bf16 / fp32."""
    return torch.amp.GradScaler("cuda", enabled=dtype == torch.float16 and torch.cuda.is_available())


def atomic_torch_save(obj, path: str | Path) -> None:
    """Write to a temp file and rename, so a concurrent reader / uploader never sees a partial file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_config(path: str | Path) -> dict:
    """Load a YAML config. A ``base: other.yaml`` key (relative to this file) is loaded first and
    recursively overridden by this file's values, so variant configs only list what they change."""
    path = Path(path)
    cfg = load_yaml(path)
    base = cfg.pop("base", None)
    if base:
        return _deep_merge(load_config(path.parent / base), cfg)
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def autocast(device: torch.device, dtype):
    if dtype is None or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def fmt_key(fraction: float) -> str:
    """Canonical string key for a labelled-data fraction (0.1 -> '0.1')."""
    return f"{float(fraction):g}"


class Timer:
    """Wall-clock timer that synchronises CUDA so GPU work is fully counted."""

    def __init__(self):
        self.total = 0.0
        self._t = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t = time.perf_counter()

    def stop(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t
        self.total += dt
        self._t = None
        return dt


def cosine_with_warmup(step: int, total: int, warmup: int, base_lr: float, min_lr: float = 0.0) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)
