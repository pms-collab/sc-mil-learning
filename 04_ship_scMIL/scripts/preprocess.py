# scripts/preprocess.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scmil.preprocess.preprocess_core import run_preprocess


def setup_logging(out_dir: Path) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocess.log"
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
    ap.add_argument("--config", required=True, type=str, help="Path to YAML config")
    ap.add_argument("--out", required=True, type=str, help="Output dir (e.g., runs/.../preprocess)")
    ap.add_argument(
        "--raw_h5ad",
        default=None,
        type=str,
        help="Optional override path to raw.h5ad (file). If omitted, resolved from config.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    logging.info("Starting preprocess")
    logging.info("config=%s", str(config_path))
    logging.info("out=%s", str(out_dir))

    res = run_preprocess(
        config_path=config_path,
        out_dir=out_dir,
        raw_h5ad=Path(args.raw_h5ad) if args.raw_h5ad else None,
        force=bool(args.force),
    )

    logging.info("Outputs:")
    for k, v in res.items():
        logging.info("  %s: %s", k, str(v))


if __name__ == "__main__":
    main()

