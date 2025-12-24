# scripts/download.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from scmil.data.download_core import run_download


def _peek_paths_from_config(config_path: Path) -> Path:
    """
    We set log file under: <data.root>/raw/<dataset>/_download/logs/download.log
    This function reads minimal fields to compute that path before core runs.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict.")
    data = cfg.get("data", {}) or {}
    root = Path(str(data.get("root", "data")))
    dataset = str(data.get("dataset", "")).strip()
    if not dataset:
        raise KeyError("Missing config: data.dataset")
    return root / "raw" / dataset / "_download"


def setup_logging(download_dir: Path) -> None:
    log_dir = download_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "download.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.info("log_file=%s", str(log_file))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to YAML config")
    ap.add_argument("--force", action="store_true", help="Redownload/rebuild even if outputs exist")
    args = ap.parse_args()

    config_path = Path(args.config)

    download_dir = _peek_paths_from_config(config_path)
    download_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(download_dir)

    logging.info("Starting download/build raw.h5ad")
    logging.info("config=%s", str(config_path))
    logging.info("force=%s", str(bool(args.force)))

    res = run_download(config_path=config_path, force=bool(args.force))

    logging.info("Outputs:")
    for k, v in res.items():
        logging.info("  %s: %s", k, str(v))


if __name__ == "__main__":
    main()

