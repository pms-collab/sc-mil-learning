# scripts/build_bags.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scmil.bags.build_bags_core import run_build_bags


def setup_logging(out_dir: Path) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "build_bags.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument(
        "--preprocess_out",
        default=None,
        type=str,
        help="Path to runs/<dataset>/preprocess (dir) or processed.h5ad (file). Optional.",
    )
    ap.add_argument(
        "--processed_h5ad",
        default=None,
        type=str,
        help="Direct path to processed.h5ad. Optional; overrides --preprocess_out.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    logging.info("Starting build_bags")
    logging.info("config=%s", str(config_path))
    logging.info("out=%s", str(out_dir))

    res = run_build_bags(
        config_path=config_path,
        out_dir=out_dir,
        preprocess_out=Path(args.preprocess_out) if args.preprocess_out else None,
        processed_h5ad=Path(args.processed_h5ad) if args.processed_h5ad else None,
        force=bool(args.force),
    )

    logging.info("Outputs:")
    for k, v in res.items():
        logging.info("  %s: %s", k, str(v))


if __name__ == "__main__":
    main()

