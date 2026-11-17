"""Carry the study's progress between short-lived sessions (e.g. Kaggle's 12-hour limit) via a
**private** Hugging Face dataset repository.

    python scripts/hf_state.py pull --repo_id 0x-Khalil/OSAE-state --config configs/kaggle.yaml
    python scripts/hf_state.py push --repo_id 0x-Khalil/OSAE-state --config configs/kaggle.yaml

Synced: the config's runs_dir (checkpoints, metrics, logs) and results_dir. Not synced: the dataset
(re-downloaded each session) and temp files. Files already on the Hub are not re-uploaded (the Hub
deduplicates by content hash), so a push only sends what changed since the last one.

The token comes from HF_TOKEN (or a cached login) and is never written anywhere.
"""
import _bootstrap  # noqa: F401

import argparse
import os
import time
from pathlib import Path

from osae.utils import ROOT, load_config

IGNORE = ["**/*.tmp", "**/__pycache__/**", "**/.cache/**"]


def _dirs(cfg) -> list[str]:
    out = []
    for key in ("runs_dir", "results_dir"):
        p = Path(cfg[key])
        if p.is_absolute():
            raise ValueError(f"{key} must be relative to the repository for syncing, got {p}")
        out.append(p.as_posix().strip("/"))
    return out


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def ensure_repo(repo_id: str) -> None:
    _api().create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)


def pull(repo_id: str, cfg: dict) -> int:
    """Download the saved state into the repository. Returns the number of files restored."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    api = _api()
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        print(f"state repo {repo_id} does not exist yet: starting from scratch")
        return 0
    dirs = _dirs(cfg)
    wanted = [f for f in files if any(f.startswith(d + "/") for d in dirs)]
    if not wanted:
        print("state repo is empty: starting from scratch")
        return 0
    t = time.time()
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(ROOT), token=os.environ.get("HF_TOKEN"),
                      allow_patterns=[f"{d}/**" for d in dirs], max_workers=8)
    print(f"restored {len(wanted)} files from {repo_id} in {time.time() - t:.0f}s")
    return len(wanted)


def push(repo_id: str, cfg: dict, message: str = "sync study state") -> None:
    api = _api()
    ensure_repo(repo_id)
    dirs = [d for d in _dirs(cfg) if (ROOT / d).exists()]
    if not dirs:
        print("nothing to push")
        return
    t = time.time()
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(ROOT),
                      allow_patterns=[f"{d}/**" for d in dirs], ignore_patterns=IGNORE,
                      commit_message=message)
    print(f"pushed {dirs} to {repo_id} in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pull", "push", "check"])
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--config", default="configs/kaggle.yaml")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    if args.action == "pull":
        pull(args.repo_id, cfg)
    elif args.action == "push":
        push(args.repo_id, cfg)
    else:
        check(args.repo_id)


def check(repo_id: str) -> None:
    """Fail fast: prove the token can create *and write* the private repo before hours of training."""
    ensure_repo(repo_id)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    _api().upload_file(path_or_fileobj=f"last session start: {stamp}\n".encode(), path_in_repo="SESSION.txt",
                       repo_id=repo_id, repo_type="dataset", commit_message="session start")
    print(f"ok: can write to private dataset repo {repo_id}")


if __name__ == "__main__":
    main()
