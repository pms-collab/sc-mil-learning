
# scripts/eval.py
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scmil.eval.eval_core import run_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags_dir", required=True, help="runs/<dataset>/bags (contains bags.npz, split_bags.csv, bags_meta.csv)")
    ap.add_argument("--ckpt", required=True, help="runs/<dataset>/train/<exp>/checkpoints/best.pt")
    ap.add_argument("--out", required=True, help="runs/<dataset>/eval/<name>")
    ap.add_argument("--splits", default="test", help="Comma-separated splits. e.g. test or test,val")
    ap.add_argument("--allow_train", action="store_true", help="Allow evaluating train split (default: False)")
    ap.add_argument("--config", default=None, help="Optional config snapshot path to copy into out/artifacts/")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # optional reproducibility snapshot
    if args.config:
        art = out_dir / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, art / "config_eval.yaml")

    run_eval(
        bags_dir=Path(args.bags_dir),
        ckpt_path=Path(args.ckpt),
        out_dir=out_dir,
        splits=args.splits,
        allow_train=bool(args.allow_train),
        write_outputs=True,
    )


if __name__ == "__main__":
    main()
