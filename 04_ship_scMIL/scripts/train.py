# scripts/train.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil

from scmil.config import load_yaml
from scmil.paths import resolve_paths
from scmil.train.train_core import run_train

def setup_logging(out_dir: Path) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None, help="Optional override output dir (default: <rundir>/train/baseline)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    paths = resolve_paths(cfg, repo_root=repo_root, run_dir=Path(args.rundir))
    out_dir = Path(args.out) if args.out else paths.train_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir)
    logging.info("config=%s", str(cfg_path))
    logging.info("rundir=%s", str(paths.run_dir))
    logging.info("out=%s", str(out_dir))

    # snapshot config (CLI responsibility)
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, art / "config_train.yaml")

    run_train(cfg, paths, out_dir=out_dir, force=bool(args.force))

if __name__ == "__main__":
    main()

