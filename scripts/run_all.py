"""Orchestrate the full study from one YAML config. Every stage is resumable: finished runs are skipped.

Stages (default: all, in order)
  prepare     download STL-10, build arrays / normalisation stats / splits
  sweep       pre-train the overcomplete AE for every lambda in the sweep; frozen probes on the selection grid
  select      choose lambda* (> 0) by mean validation accuracy -> results/lambda_selection.json
  pretrain    pre-train every variant of the factorial design (lambda_star resolved)
  downstream  baseline / control / every variant x protocol x fraction x seed
  evaluate    FINAL test-split evaluation of every finished downstream run
  analyze     representation analysis (no test data)
  aggregate   result tables, statistics, figures -> results/RESULTS.md

Examples
  python scripts/run_all.py --config configs/default.yaml
  python scripts/run_all.py --config configs/default.yaml --stages downstream --only oae_sparse
  python scripts/run_all.py --config configs/default.yaml --dry_run
  python scripts/run_all.py --config configs/default.yaml --jobs 4 --pretrain_jobs 2   # big GPU (e.g. 32 GB)
"""
import _bootstrap  # noqa: F401

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from osae.utils import ROOT, fmt_key, load_json, load_yaml, save_json

STAGES = ["prepare", "sweep", "select", "pretrain", "downstream", "evaluate", "analyze", "aggregate"]
PY = sys.executable
SCRIPTS = ROOT / "scripts"


class Runner:
    def __init__(self, cfg, dry_run=False, downstream_jobs=None, pretrain_jobs=None):
        self.cfg, self.dry = cfg, dry_run
        par = cfg.get("parallel", {}) or {}
        self.downstream_jobs = max(1, int(downstream_jobs or par.get("downstream_jobs", 1)))
        self.pretrain_jobs = max(1, int(pretrain_jobs or par.get("pretrain_jobs", 1)))
        self._print_lock = threading.Lock()
        self.runs = ROOT / cfg["runs_dir"]
        self.results = ROOT / cfg["results_dir"]
        self.logs = self.runs / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    def say(self, msg):
        with self._print_lock:
            print(msg, flush=True)

    def run(self, script, args, log_name, n_parallel=1):
        """Run one script as a subprocess, logging to runs/logs/<log_name>.log."""
        cmd = [PY, "-u", str(SCRIPTS / script)] + [str(a) for a in args]
        tag = f"[{log_name}] " if n_parallel > 1 else ""
        self.say(f"{tag}$ " + " ".join(cmd[2:]))
        if self.dry:
            return
        t = time.time()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        if n_parallel > 1:  # share CPU threads between concurrent jobs instead of oversubscribing
            env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 4) // n_parallel))
        with open(self.logs / f"{log_name}.log", "a", encoding="utf-8") as log:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", cwd=ROOT, bufsize=1, env=env)
            for line in p.stdout:
                log.write(line)
                log.flush()
                if any(k in line for k in ("==", "done", "skip", "Error", "error", "Traceback", "best val")):
                    self.say(f"   {tag}" + line.rstrip())
            p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"{script} failed (exit {p.returncode}); see {self.logs / (log_name + '.log')}")
        self.say(f"   {tag}({time.time() - t:.0f}s)")

    def execute(self, jobs, n_parallel=1):
        """Run (script, args, log_name) jobs, up to n_parallel at a time on the same GPU.

        A failed job does not stop the others; all failures are reported together at the end, and a
        rerun picks up only the unfinished jobs.
        """
        jobs = [j for j in jobs if j is not None]
        if not jobs:
            return
        n = min(n_parallel, len(jobs))
        if n <= 1 or self.dry:
            for j in jobs:
                self.run(*j)
            return
        self.say(f"running {len(jobs)} jobs, {n} at a time")
        failures, done = [], 0
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = {pool.submit(self.run, *j, n_parallel=n): j for j in jobs}
            for fut in as_completed(futs):
                done += 1
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    failures.append(f"{futs[fut][2]}: {e}")
                self.say(f"-- progress {done}/{len(jobs)} ({len(failures)} failed)")
        if failures:
            raise RuntimeError(f"{len(failures)} job(s) failed:\n  " + "\n  ".join(failures))

    # ------------------------------------------------------------------ naming
    @staticmethod
    def canonical(spec: dict, pcfg: dict) -> str:
        noise = spec.get("noise", "none")
        tag = {"none": "clean", "gaussian": f"gauss{spec.get('noise_std', pcfg['noise_std']):g}",
               "mask": f"mask{spec.get('mask_ratio', pcfg['mask_ratio']):g}"}[noise]
        return f"C{spec['latent_channels']}_l{float(spec['l1_lambda']):g}_{tag}"

    def pretrain_dir(self, canonical: str, pseed: int) -> Path:
        return self.runs / "pretrain" / f"{canonical}_ps{pseed}"

    def pretrain_seed_for(self, seed: int) -> int:
        ps = self.cfg["pretrain_seeds"]
        return ps[0] if len(ps) == 1 else ps[self.cfg["seeds"].index(seed)]

    def down_dir(self, canonical, protocol, fraction, seed) -> Path:
        return self.runs / "downstream" / canonical / protocol / f"f{fmt_key(fraction)}" / f"s{seed}"

    # ------------------------------------------------------------------ steps
    def pretrain_jobs_for(self, spec: dict) -> list:
        p = self.cfg["pretrain"]
        can = self.canonical(spec, p)
        jobs = []
        for ps in self.cfg["pretrain_seeds"]:
            d = self.pretrain_dir(can, ps)
            if (d / "encoder.pt").exists():
                continue
            args = ["--out_dir", d, "--data_dir", self.cfg["data_dir"], "--latent_channels", spec["latent_channels"],
                    "--l1_lambda", spec["l1_lambda"], "--noise", spec.get("noise", "none"),
                    "--noise_std", spec.get("noise_std", p["noise_std"]), "--mask_ratio", spec.get("mask_ratio", p["mask_ratio"]),
                    "--mask_patch", p["mask_patch"], "--epochs", p["epochs"], "--batch_size", p["batch_size"],
                    "--lr", p["lr"], "--weight_decay", p["weight_decay"], "--warmup_epochs", p["warmup_epochs"],
                    "--amp", self.cfg["amp"], "--seed", ps]
            if p.get("max_steps"):
                args += ["--max_steps", p["max_steps"]]
            if p.get("holdout_max"):
                args += ["--holdout_max", p["holdout_max"]]
            jobs.append(("pretrain.py", args, f"pretrain_{can}_ps{ps}"))
        return jobs

    def downstream_job(self, canonical, init, protocol, fraction, seed):
        d = self.down_dir(canonical, protocol, fraction, seed)
        if (d / "metrics.json").exists():
            return None
        dc = self.cfg["downstream"]
        hp = {**dc["common"], **dc[protocol]}
        args = ["--out_dir", d, "--data_dir", self.cfg["data_dir"], "--init", init, "--protocol", protocol,
                "--fraction", fraction, "--seed", seed, "--amp", self.cfg["amp"]]
        if init == "pretrained":
            enc = self.pretrain_dir(canonical, self.pretrain_seed_for(seed)) / "encoder.pt"
            if not enc.exists() and not self.dry:
                raise FileNotFoundError(f"missing pre-trained encoder {enc}")
            args += ["--pretrained", enc]
        for k, v in hp.items():
            args += [f"--{k}", v]
        return ("finetune.py", args, f"down_{canonical}_{protocol}_f{fmt_key(fraction)}_s{seed}")

    # ------------------------------------------------------------------ variants
    def sweep_specs(self) -> dict:
        s = self.cfg["lambda_sweep"]
        return {f"oae_l{v:g}": {"latent_channels": s["latent_channels"], "l1_lambda": float(v), "noise": "none"}
                for v in s["values"]}

    def lambda_star(self):
        p = self.results / "lambda_selection.json"
        return load_json(p)["lambda_star"] if p.exists() else None

    def resolved_variants(self, require_lambda=True) -> dict:
        ls = self.lambda_star()
        out = {}
        for name, spec in self.cfg["variants"].items():
            spec = dict(spec)
            if spec["l1_lambda"] == "lambda_star":
                if ls is None:
                    if require_lambda:
                        raise RuntimeError("lambda* not selected yet: run the 'sweep' and 'select' stages first")
                    continue
                spec["l1_lambda"] = ls
            spec["l1_lambda"] = float(spec["l1_lambda"])
            out[name] = spec
        return out

    def write_variants(self):
        pcfg = self.cfg["pretrain"]
        V = {}
        allspecs = {**self.sweep_specs(), **self.resolved_variants(require_lambda=False)}
        for name, spec in allspecs.items():
            can = self.canonical(spec, pcfg)
            V[name] = {**spec, "canonical": can, "pretrain_dir": str(self.pretrain_dir(can, self.cfg["pretrain_seeds"][0]))}
        V["scratch"] = {"canonical": "scratch"}
        V["random"] = {"canonical": "random"}
        save_json({"variants": V, "main_table": self.cfg["main_table"], "comparisons": self.cfg.get("comparisons", []),
                   "lambda_star": self.lambda_star()}, self.results / "variants.json")
        return V

    # ------------------------------------------------------------------ stages
    def stage_prepare(self):
        sp = self.cfg["splits"]
        if (ROOT / self.cfg["data_dir"] / "splits.json").exists():
            print("data ready")
            return
        self.run("prepare_data.py", ["--data_dir", self.cfg["data_dir"], "--fractions", *sp["fractions"],
                                     "--val_per_class", sp["val_per_class"], "--split_seed", sp["split_seed"],
                                     "--unlabeled_holdout", sp["unlabeled_holdout"]], "prepare")

    def stage_sweep(self, only=None):
        sel = self.cfg["lambda_sweep"]["selection"]
        specs = {"oae": {"latent_channels": self.cfg["lambda_sweep"]["latent_channels"], "l1_lambda": 0.0, "noise": "none"},
                 **self.sweep_specs()}
        specs = {n: sp for n, sp in specs.items() if not only or only in n}
        self.execute([j for sp in specs.values() for j in self.pretrain_jobs_for(sp)], self.pretrain_jobs)
        self.execute([self.downstream_job(self.canonical(sp, self.cfg["pretrain"]), "pretrained", sel["protocol"], f, s)
                      for sp in specs.values() for f in sel["fractions"] for s in sel["seeds"]], self.downstream_jobs)

    def stage_select(self):
        import numpy as np
        sel = self.cfg["lambda_sweep"]["selection"]
        table = {}
        for name, spec in self.sweep_specs().items():
            can = self.canonical(spec, self.cfg["pretrain"])
            accs = []
            for f in sel["fractions"]:
                for s in sel["seeds"]:
                    m = self.down_dir(can, sel["protocol"], f, s) / "metrics.json"
                    if m.exists():
                        accs.append(load_json(m)["best"]["val_top1"])
            if accs:
                table[spec["l1_lambda"]] = float(np.mean(accs))
        if not table:
            if self.dry:
                print("(dry run) lambda* would be selected here")
                return
            raise RuntimeError("no sweep results found")
        best = max(sorted(table), key=lambda k: table[k])  # ties -> smallest lambda
        save_json({"lambda_star": best, "mean_val_top1_by_lambda": {f"{k:g}": v for k, v in table.items()},
                   "rule": f"argmax mean validation top-1 of {sel['protocol']} probe over fractions {sel['fractions']}, "
                           f"seeds {sel['seeds']}; lambda > 0 only; test split not used"},
                  self.results / "lambda_selection.json")
        print(f"lambda* = {best:g}  ({table})")

    def stage_pretrain(self, only=None):
        specs = [sp for n, sp in self.resolved_variants(require_lambda=not self.dry).items() if not only or only in n]
        self.execute([j for sp in specs for j in self.pretrain_jobs_for(sp)], self.pretrain_jobs)

    def stage_downstream(self, only=None):
        dc = self.cfg["downstream"]
        fr, seeds = self.cfg["splits"]["fractions"], self.cfg["seeds"]
        jobs = []
        if not only or only in "scratch":
            jobs += [("scratch", "scratch", "finetune")]
        if not only or only in "random":
            jobs += [("random", "random", "frozen")]
        specs = dict(self.resolved_variants(require_lambda=not self.dry))
        if self.cfg["lambda_sweep"].get("full_downstream_grid"):
            specs.update(self.sweep_specs())
        for name, spec in specs.items():
            if only and only not in name:
                continue
            can = self.canonical(spec, self.cfg["pretrain"])
            jobs += [(can, "pretrained", p) for p in dc["protocols"]]
        seen, queue = set(), []
        # fraction-major order: low-label results (the focus of the study) arrive first
        for f in fr:
            for s in seeds:
                for can, init, proto in jobs:
                    if (can, proto, f, s) in seen:
                        continue
                    seen.add((can, proto, f, s))
                    queue.append(self.downstream_job(can, init, proto, f, s))
        self.execute(queue, self.downstream_jobs)

    def stage_evaluate(self):
        self.run("evaluate.py", ["--runs_dir", self.runs / "downstream", "--data_dir", self.cfg["data_dir"],
                                 "--results_dir", self.results, "--amp", self.cfg["amp"]], "evaluate")

    def stage_analyze(self):
        V = self.write_variants()
        a = self.cfg.get("analysis", {})
        models, seen = [], set()
        for name, info in V.items():
            pd_ = Path(info.get("pretrain_dir", ""))
            if info.get("pretrain_dir") and (pd_ / "model.pt").exists() and info["canonical"] not in seen:
                seen.add(info["canonical"])
                models.append(f"{name}={pd_}")
        models.append("random_init=random")
        ref = self.down_dir("scratch", "finetune", 1.0, self.cfg["seeds"][0])
        if a.get("include_supervised_reference", True) and (ref / "best.pt").exists():
            models.append(f"scratch_100={ref}")
        args = ["--models", *models, "--data_dir", self.cfg["data_dir"], "--out_dir", self.results / "representation",
                "--embed_samples", a.get("embed_samples", 2000), "--amp", self.cfg["amp"]]
        if a.get("pairs"):
            args += ["--pairs", *a["pairs"]]
        if a.get("max_pool"):
            args += ["--max_pool", a["max_pool"]]
        self.run("analyze_representations.py", args, "analyze")

    def stage_aggregate(self):
        self.write_variants()
        self.run("aggregate.py", ["--runs_dir", self.runs / "downstream", "--results_dir", self.results,
                                  "--variants", self.results / "variants.json", "--data_dir", self.cfg["data_dir"]],
                 "aggregate")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--only", default=None, help="substring filter on variant names (sweep/pretrain/downstream)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--jobs", type=int, default=None,
                    help="downstream runs to execute concurrently on the GPU (overrides parallel.downstream_jobs)")
    ap.add_argument("--pretrain_jobs", type=int, default=None,
                    help="AE pre-trainings to execute concurrently (overrides parallel.pretrain_jobs)")
    args = ap.parse_args()
    cfg = load_yaml(ROOT / args.config)
    r = Runner(cfg, args.dry_run, args.jobs, args.pretrain_jobs)
    t0 = time.time()
    for st in STAGES:
        if st not in args.stages:
            continue
        print(f"\n######## stage: {st}", flush=True)
        fn = getattr(r, f"stage_{st}")
        fn(args.only) if st in ("sweep", "pretrain", "downstream") else fn()
    print(f"\nall requested stages finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
