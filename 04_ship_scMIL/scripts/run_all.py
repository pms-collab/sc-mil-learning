#!/usr/bin/env python3
# scripts/run_all.py
# End-to-end runner for scMIL pipeline:
# download -> preprocess -> build_bags -> train -> eval -> check_leakage

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml


def find_repo_root(start: Path) -> Path:
    """
    Find repo root by walking parents until we find pyproject.toml or configs/ or src/.
    """
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "pyproject.toml").exists():
            return p
        if (p / "configs").is_dir() and (p / "src").is_dir():
            return p
    raise RuntimeError(f"Could not locate repo root from: {start}")


def abspath(p: Path) -> Path:
    return p.expanduser().resolve()


def load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must parse to a YAML mapping (dict).")
    return cfg


def parse_dataset(cfg: dict) -> tuple[str, str]:
    d = (cfg.get("data", {}) or {})
    root = str(d.get("root", "data")).strip() or "data"
    dataset = str(d.get("dataset", "")).strip()
    if not dataset:
        raise ValueError("Config parse produced empty data.dataset")
    return root, dataset


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def tee_subprocess(cmd: List[str], log_path: Path, cwd: Path) -> None:
    ensure_dir(log_path.parent)
    if log_path.exists():
        log_path.unlink()

    # write cmd header
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd] " + " ".join(map(shlex_quote, cmd)) + "\n")

    print(f"==> {log_path.stem}")
    print("    " + " ".join(map(shlex_quote, cmd)))
    print(f"    log: {log_path}")

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None

    with log_path.open("a", encoding="utf-8") as f:
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)

    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Step failed (rc={rc}). See log: {log_path}")


def shlex_quote(s: str) -> str:
    # minimal safe quoting for display
    if not s:
        return "''"
    if any(c in s for c in " \t\n\"'\\$`"):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


def assert_file(p: Path, msg: str) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"{msg}\nMissing: {p}")


def wipe_run_outputs(rundir: Path) -> None:
    # only wipe run artifacts; never touch data/raw here.
    targets = [
        rundir / "preprocess",
        rundir / "bags",
        rundir / "train",
        rundir / "eval",
        rundir / "leakage",
        rundir / "logs",
    ]
    for t in targets:
        if t.exists():
            shutil.rmtree(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    ap.add_argument("--rundir", required=True, help="runs/<dataset>/<run_name> (e.g., runs/gse96583_batch2/e2e)")
    ap.add_argument("--python", default=None, help="Optional python executable. Default: current interpreter.")
    ap.add_argument("--force", action="store_true", help="Wipe rundir outputs and pass --force to steps.")
    ap.add_argument(
        "--exp",
        default="baseline",
        help="Train experiment subdir name under <rundir>/train/<exp>. Default: baseline",
    )
    ap.add_argument(
        "--eval_splits",
        default="test",
        help="Comma-separated splits for eval. Default: test (example: test,val)",
    )
    args = ap.parse_args()

    # repo root normalization (cwd-independent)
    repo_root = find_repo_root(Path(__file__).resolve())
    os.chdir(repo_root)

    py = args.python or sys.executable
    cfg_path = abspath(Path(args.config))
    rundir = abspath(Path(args.rundir))

    ensure_dir(rundir)
    logdir = rundir / "logs"
    ensure_dir(logdir)

    cfg = load_cfg(cfg_path)
    data_root, dataset = parse_dataset(cfg)

    # derived paths (for assertions)
    raw_h5ad = repo_root / data_root / "raw" / dataset / "raw.h5ad"
    raw_ok = raw_h5ad.with_suffix(raw_h5ad.suffix + ".ok")

    preprocess_out = rundir / "preprocess"
    processed_h5ad = preprocess_out / "artifacts" / "processed.h5ad"
    preprocess_ok = preprocess_out / "preprocess.ok"

    bags_out = rundir / "bags"
    bags_npz = bags_out / "bags.npz"
    bags_ok = bags_out / "bags.ok"
    bags_meta = bags_out / "bags_meta.csv"
    split_bags = bags_out / "split_bags.csv"
    bags_table = bags_out / "artifacts" / "bags_table.csv.gz"

    train_out = rundir / "train" / args.exp
    best_ckpt = train_out / "checkpoints" / "best.pt"

    eval_out = rundir / "eval" / "test"  # keep convention; can be parameterized later
    eval_pred = eval_out / "predictions.csv"
    eval_metrics = eval_out / "metrics.json"

    leak_out = rundir / "leakage" / "report.json"

    if args.force:
        print("==> FORCE enabled: wiping run outputs under rundir.")
        wipe_run_outputs(rundir)
        ensure_dir(rundir)
        ensure_dir(logdir)

    # ---- Step 01: download ----
    cmd = [py, "scripts/download.py", "--config", str(cfg_path)]
    if args.force:
        cmd.append("--force")
    tee_subprocess(cmd, logdir / "01_download.log", cwd=repo_root)
    assert_file(raw_h5ad, "download did not produce raw.h5ad as expected.")
    assert_file(raw_ok, "download did not produce raw marker (.ok) as expected.")

    # ---- Step 02: preprocess ----
    cmd = [py, "scripts/preprocess.py", "--config", str(cfg_path), "--out", str(preprocess_out)]
    if args.force:
        cmd.append("--force")
    tee_subprocess(cmd, logdir / "02_preprocess.log", cwd=repo_root)
    assert_file(processed_h5ad, "preprocess did not produce artifacts/processed.h5ad.")
    assert_file(preprocess_ok, "preprocess did not produce preprocess.ok marker.")

    # ---- Step 03: build_bags ----
    cmd = [
        py,
        "scripts/build_bags.py",
        "--config",
        str(cfg_path),
        "--out",
        str(bags_out),
        "--preprocess_out",
        str(preprocess_out),
    ]
    if args.force:
        cmd.append("--force")
    tee_subprocess(cmd, logdir / "03_build_bags.log", cwd=repo_root)
    assert_file(bags_npz, "build_bags did not produce bags.npz.")
    assert_file(bags_meta, "build_bags did not produce bags_meta.csv.")
    assert_file(split_bags, "build_bags did not produce split_bags.csv.")
    assert_file(bags_table, "build_bags did not produce artifacts/bags_table.csv.gz.")
    assert_file(bags_ok, "build_bags did not produce bags.ok marker.")

    # ---- Step 04: train ----
    cmd = [
        py,
        "scripts/train.py",
        "--config",
        str(cfg_path),
        "--rundir",
        str(rundir),
        "--out",
        str(train_out),
    ]
    if args.force:
        cmd.append("--force")
    tee_subprocess(cmd, logdir / "04_train.log", cwd=repo_root)
    assert_file(best_ckpt, "train did not produce checkpoints/best.pt.")

    # ---- Step 05: eval ----
    cmd = [
        py,
        "scripts/eval.py",
        "--bags_dir",
        str(bags_out),
        "--ckpt",
        str(best_ckpt),
        "--out",
        str(eval_out),
        "--splits",
        str(args.eval_splits),
    ]
    tee_subprocess(cmd, logdir / "05_eval.log", cwd=repo_root)
    assert_file(eval_pred, "eval did not produce predictions.csv.")
    assert_file(eval_metrics, "eval did not produce metrics.json.")

    # ---- Step 06: check_leakage ----
    cmd = [
        py,
        "scripts/check_leakage.py",
        "--bags_dir",
        str(bags_out),
        "--out",
        str(leak_out),
    ]
    tee_subprocess(cmd, logdir / "06_check_leakage.log", cwd=repo_root)
    assert_file(leak_out, "check_leakage did not produce report.json.")

    print("")
    print("DONE")
    print(f"RunDir : {rundir}")
    print(f"Logs   : {logdir}")
    print(f"Raw    : {raw_h5ad}")
    print(f"Proc   : {processed_h5ad}")
    print(f"Bags   : {bags_out}")
    print(f"Ckpt   : {best_ckpt}")
    print(f"Eval   : {eval_out}")
    print(f"Leak   : {leak_out}")


if __name__ == "__main__":
    main()

