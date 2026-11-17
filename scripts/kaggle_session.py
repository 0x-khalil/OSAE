"""Run the full study on Kaggle's free GPUs, across as many 12-hour sessions as it takes.

Each session:
  1. checks the GPU(s) and the Hugging Face token (fails in minute 1, not hour 11)
  2. restores the previous sessions' progress from a PRIVATE HF dataset repo (<repo>-state)
  3. runs scripts/run_all.py over all GPUs until a time budget that leaves room to save
     (pre-training pauses itself at an epoch boundary; finished runs are skipped next time)
  4. pushes the progress back every --sync_every_min minutes and at the end
  5. when everything is done: publishes the final results to the PUBLIC repo (<repo>)

Usage in a Kaggle notebook (Accelerator: "GPU T4 x2", Internet: on, Secret "HF_TOKEN" attached):

    !git clone https://github.com/0x-khalil/OSAE.git
    !cd OSAE && python scripts/kaggle_session.py --hf_repo 0x-Khalil/OSAE

then "Save Version" -> "Save & Run All (Commit)". When it prints SESSION PAUSED, run it again
(new version); when it prints STUDY COMPLETE, the results are on the Hub.
"""
import _bootstrap  # noqa: F401

import argparse
import importlib.util
import os
import subprocess
import sys
import threading
import time

from osae.utils import ROOT

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 3600:5.2f} h] {msg}", flush=True)


def get_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:  # Kaggle: Add-ons -> Secrets -> label "HF_TOKEN", attached to this notebook
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001
        return None


def ensure_packages():
    need = [pkg for mod, pkg in [("yaml", "pyyaml"), ("tabulate", "tabulate"), ("sklearn", "scikit-learn"),
                                 ("umap", "umap-learn"), ("huggingface_hub", "huggingface_hub"),
                                 ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("scipy", "scipy")]
            if importlib.util.find_spec(mod) is None]
    if need:  # torch / torchvision are preinstalled on Kaggle and deliberately left alone
        log(f"installing {need}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])


def check_gpus() -> int:
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        sys.exit("No GPU. In the notebook sidebar set Accelerator = 'GPU T4 x2' (and Internet = on).")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        try:
            (torch.randn(256, 256, device=f"cuda:{i}") @ torch.randn(256, 256, device=f"cuda:{i}")).sum().item()
        except Exception as e:  # e.g. P100 with a PyTorch build that dropped sm_60
            sys.exit(f"GPU {i} ({name}, sm_{cap[0]}{cap[1]}) is not usable with this PyTorch build: {e}\n"
                     "Choose the 'GPU T4 x2' accelerator.")
        log(f"GPU {i}: {name} (sm_{cap[0]}{cap[1]}, {torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GB)")
    log(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    return n


class Syncer(threading.Thread):
    """Pushes progress to the state repo periodically while the study runs."""

    def __init__(self, repo, cfg, every_min):
        super().__init__(daemon=True)
        self.repo, self.cfg, self.every = repo, cfg, every_min * 60
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        while not self.stop_evt.wait(self.every):
            self.push("periodic sync")

    def push(self, why):
        from hf_state import push
        with self.lock:
            try:
                push(self.repo, self.cfg, f"{why} at {(time.time() - T0) / 3600:.2f} h")
                log(f"state saved ({why})")
            except Exception as e:  # noqa: BLE001  - never let a sync hiccup kill training
                log(f"WARNING: state sync failed ({why}): {e}")
                return False
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_repo", default="0x-Khalil/OSAE", help="public repo for the final results")
    ap.add_argument("--state_repo", default=None, help="private dataset repo for progress (default <hf_repo>-state)")
    ap.add_argument("--config", default="configs/kaggle.yaml")
    ap.add_argument("--session_hours", type=float, default=12.0, help="Kaggle's hard limit")
    ap.add_argument("--reserve_hours", type=float, default=1.5,
                    help="stop starting new work this long before the limit (running jobs + final sync)")
    ap.add_argument("--sync_every_min", type=float, default=40)
    ap.add_argument("--jobs_per_gpu", type=int, default=2, help="concurrent downstream runs per GPU")
    ap.add_argument("--github", default="https://github.com/0x-khalil/OSAE")
    ap.add_argument("--no_publish", action="store_true")
    args = ap.parse_args()
    state_repo = args.state_repo or f"{args.hf_repo}-state"

    log(f"OSAE Kaggle session in {ROOT}")
    ensure_packages()
    token = get_token()
    if not token:
        sys.exit("HF_TOKEN not found. Kaggle: Add-ons -> Secrets -> add 'HF_TOKEN' and attach it to this notebook.")
    os.environ["HF_TOKEN"] = token  # for this process and the scripts it starts; never printed or saved

    from hf_state import check, pull
    from osae.utils import load_config
    cfg = load_config(ROOT / args.config)
    n_gpu = check_gpus()
    check(state_repo)
    pull(state_repo, cfg)

    deadline = T0 + 3600 * (args.session_hours - args.reserve_hours)
    gpus = ",".join(str(i) for i in range(n_gpu))
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "run_all.py"), "--config", args.config,
           "--gpus", gpus, "--jobs", str(args.jobs_per_gpu * n_gpu), "--pretrain_jobs", str(n_gpu),
           "--deadline_ts", str(deadline)]
    log(f"running the study until {time.strftime('%H:%M', time.localtime(deadline))} "
        f"({args.session_hours - args.reserve_hours:.1f} h budget) on GPUs {gpus}")
    syncer = Syncer(state_repo, cfg, args.sync_every_min)
    syncer.start()
    rc = subprocess.call(cmd, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    syncer.stop_evt.set()
    saved = syncer.push("end of session")
    if not saved:  # one retry; Kaggle also keeps /kaggle/working as the notebook's output
        time.sleep(30)
        syncer.push("end of session (retry)")

    if rc == 3:
        log("SESSION PAUSED: time budget reached, progress saved. Run the notebook again to continue.")
        return
    if rc != 0:
        log(f"STUDY FAILED (exit {rc}). Progress is saved; see runs/logs/ for the failing job, fix, and rerun.")
        sys.exit(rc)
    log("STUDY COMPLETE")
    if not args.no_publish:
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "publish_hf.py"), "--repo_id", args.hf_repo,
                               "--config", args.config, "--github", args.github], cwd=ROOT)
        log(f"results published: https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
