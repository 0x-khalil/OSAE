#!/usr/bin/env bash
# One command for a rented Linux GPU pod: set up, run the FULL study (train + final test evaluation +
# analysis + tables), then publish everything to the Hugging Face Hub.
#
#   git clone https://github.com/0x-khalil/OSAE.git && cd OSAE
#   export HF_TOKEN=hf_...                  # your token: set it in the shell, never commit it
#   tmux new -s osae                        # survives SSH disconnects
#   bash scripts/run_pod.sh
#
# Environment overrides (all optional):
#   HF_REPO=0x-Khalil/OSAE   JOBS=4   PRETRAIN_JOBS=2   CONFIG=configs/default.yaml
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128   (CUDA 12.8+ is required for RTX 50xx / Blackwell)
#   SKIP_PUBLISH=1           run the study only
#   UPLOAD_CKPTS=1           also upload every downstream checkpoint (~10 GB)
#
# Safe to re-run: finished runs are skipped, so after an interruption just run it again.
set -euo pipefail

cd "$(dirname "$0")/.."
HF_REPO="${HF_REPO:-0x-Khalil/OSAE}"
JOBS="${JOBS:-4}"
PRETRAIN_JOBS="${PRETRAIN_JOBS:-2}"
CONFIG="${CONFIG:-configs/default.yaml}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
GITHUB_URL="${GITHUB_URL:-https://github.com/0x-khalil/OSAE}"
PY=.venv/bin/python
mkdir -p runs
LOG="runs/pod_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

if [[ -z "${SKIP_PUBLISH:-}" && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Run: export HF_TOKEN=hf_...   (or SKIP_PUBLISH=1 to only run the study)"
  exit 1
fi

echo "== $(date)  config=$CONFIG jobs=$JOBS pretrain_jobs=$PRETRAIN_JOBS repo=$HF_REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------- environment
if [[ ! -x "$PY" ]]; then
  python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q torch torchvision --index-url "$TORCH_INDEX"
  "$PY" -m pip install -q -r requirements.txt
fi
"$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")  # fails fast on unsupported GPU arch
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "OK")
EOF
"$PY" tests/test_core.py

# ---------------------------------------------------------------- full study
"$PY" scripts/run_all.py --config "$CONFIG" --jobs "$JOBS" --pretrain_jobs "$PRETRAIN_JOBS"

# ---------------------------------------------------------------- publish
if [[ -z "${SKIP_PUBLISH:-}" ]]; then
  EXTRA=()
  [[ -n "${UPLOAD_CKPTS:-}" ]] && EXTRA+=(--include_downstream_ckpts)
  "$PY" scripts/publish_hf.py --repo_id "$HF_REPO" --config "$CONFIG" --github "$GITHUB_URL" "${EXTRA[@]}"
fi
echo "== finished $(date). Log: $LOG"
