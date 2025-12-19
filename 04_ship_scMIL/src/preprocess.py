# src/preprocess.py
import argparse
from pathlib import Path
import shutil
import yaml

import numpy as np
import scanpy as sc


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return cfg


def resolve_raw_h5ad(cfg: dict) -> Path:
    dataset = cfg["data"]["dataset"]
    root = cfg["data"].get("root", "data")

    # canonical path
    candidates = [
        Path(root) / "raw" / dataset / "raw.h5ad",
        Path(root) / "raw" / str(dataset).lower() / "raw.h5ad",
        Path(root) / "raw" / str(dataset).upper() / "raw.h5ad",
    ]
    for p in candidates:
        if p.exists():
            return p

    # give best error message
    raise FileNotFoundError(
        "raw.h5ad not found. Expected one of:\n"
        + "\n".join([f"  - {p}" for p in candidates])
    )


def infer_mt_mask(var_names) -> np.ndarray:
    # Human: MT-*, Mouse: mt-* (sometimes)
    up = var_names.astype(str).str.upper()
    mask = up.str.startswith("MT-").to_numpy()
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    sc.settings.verbosity = 2

    out = Path(args.out)
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    # Reproducibility: store the exact config used for this run
    shutil.copyfile(args.config, art / "config_resolved.yaml")

    raw_h5ad = resolve_raw_h5ad(cfg)
    adata = sc.read_h5ad(str(raw_h5ad))

    # Ensure a stable cell identifier
    if "cell_id" not in adata.obs.columns:
        adata.obs["cell_id"] = adata.obs_names.astype(str)

    # Required obs columns for later steps (bags/split)
    bags_cfg = cfg.get("bags", {})
    bag_id_col = bags_cfg.get("bag_id_col", "sample_id")
    label_col = bags_cfg.get("label_col", "condition")
    group_id_col = bags_cfg.get("group_id_col", "donor_id")

    missing = [c for c in [bag_id_col, label_col, group_id_col] if c not in adata.obs.columns]
    if missing:
        raise ValueError(
            "raw.h5ad is missing required obs columns for bagging/splitting:\n"
            f"  missing: {missing}\n"
            f"  expected (from config): bag_id_col={bag_id_col}, label_col={label_col}, group_id_col={group_id_col}\n"
            "Fix: ensure raw.h5ad.obs contains these columns (or change configs/base.yaml)."
        )

    # QC params
    qc = cfg.get("qc", {})
    min_genes = int(qc.get("min_genes", 200))
    max_genes = int(qc.get("max_genes", 2500))
    max_mt_pct = float(qc.get("max_mt_pct", 5.0))
    min_cells_per_gene = int(qc.get("min_cells_per_gene", 3))

    # Identify mitochondrial genes and compute QC metrics
    adata.var["mt"] = infer_mt_mask(adata.var_names)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Basic filters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    if "n_genes_by_counts" in adata.obs.columns:
        adata = adata[adata.obs["n_genes_by_counts"] <= max_genes].copy()

    # Only filter on mt% if mt genes exist
    if adata.var["mt"].any() and "pct_counts_mt" in adata.obs.columns:
        adata = adata[adata.obs["pct_counts_mt"] <= max_mt_pct].copy()

    # Preserve raw counts BEFORE normalization/log1p.
    # If download already stored raw counts in layers["counts"], keep it.
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Normalize + log1p (analysis-ready expression in adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Save
    processed_path = art / "processed.h5ad"
    adata.write_h5ad(str(processed_path))

    print(f"[preprocess] raw_h5ad={raw_h5ad}")
    print(f"[preprocess] wrote {processed_path}")
    print(
        f"[preprocess] n_cells={adata.n_obs}, n_genes={adata.n_vars}, "
        f"filters: min_genes={min_genes}, max_genes={max_genes}, "
        f"max_mt_pct={max_mt_pct}, min_cells_per_gene={min_cells_per_gene}"
    )
    print(f"[preprocess] required obs cols OK: {bag_id_col}, {label_col}, {group_id_col}")


if __name__ == "__main__":
    main()
