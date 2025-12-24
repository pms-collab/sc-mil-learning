# src/scmil/preprocess/preprocess_core.py
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scanpy as sc
import yaml


# -------------------------
# Config
# -------------------------
def load_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict.")
    return cfg


def get_seed(cfg: dict) -> int:
    seed = cfg.get("seed", 42)
    try:
        return int(seed)
    except Exception as e:
        raise ValueError(f"seed must be int-like; got {seed!r}") from e


def resolve_raw_h5ad(cfg: dict, *, raw_h5ad: Optional[Path] = None) -> Path:
    """
    Priority:
      1) raw_h5ad override
      2) data.root/raw/<dataset>/raw.h5ad (plus case variants)
    """
    if raw_h5ad is not None:
        p = Path(raw_h5ad)
        if not p.exists():
            raise FileNotFoundError(f"--raw_h5ad not found: {p}")
        if p.suffix != ".h5ad":
            raise ValueError(f"--raw_h5ad must be .h5ad; got: {p}")
        return p

    data = cfg.get("data", {}) or {}
    root = Path(str(data.get("root", "data")))
    dataset = str(data.get("dataset", "")).strip()
    if not dataset:
        raise KeyError("Missing config: data.dataset")

    candidates = [
        root / "raw" / dataset / "raw.h5ad",
        root / "raw" / dataset.lower() / "raw.h5ad",
        root / "raw" / dataset.upper() / "raw.h5ad",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "raw.h5ad not found. Expected one of:\n" + "\n".join([f"  - {p}" for p in candidates])
    )


# -------------------------
# QC helpers
# -------------------------
def infer_mt_mask(var_names) -> np.ndarray:
    """
    Human: MT-*
    - var_names may be pd.Series or Index; treat as string series.
    """
    s = var_names.astype(str)
    up = s.str.upper()
    return up.str.startswith("MT-").to_numpy()


def ensure_required_obs_cols(adata, cfg: dict) -> None:
    bags = cfg.get("bags", {}) or {}
    bag_id_col = str(bags.get("bag_id_col", "sample_id"))
    label_col = str(bags.get("label_col", "condition"))
    group_id_col = str(bags.get("group_id_col", "donor_id"))

    required = [bag_id_col, label_col, group_id_col]
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        raise ValueError(
            "raw.h5ad is missing required obs columns for bagging/splitting:\n"
            f"  missing: {missing}\n"
            f"  expected (from config): bag_id_col={bag_id_col}, label_col={label_col}, group_id_col={group_id_col}\n"
            "Fix: ensure raw.h5ad.obs contains these columns (or change config)."
        )


def _safe_force_cleanup(out_dir: Path) -> None:
    """
    Targeted delete: only preprocess artifacts under out_dir.
    """
    targets = [
        out_dir / "artifacts" / "processed.h5ad",
        out_dir / "artifacts" / "config_preprocess.yaml",
        out_dir / "preprocess.ok",
    ]
    for p in targets:
        if p.exists():
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p, ignore_errors=True)


# -------------------------
# Core entrypoint
# -------------------------
def run_preprocess(
    *,
    config_path: Path,
    out_dir: Path,
    raw_h5ad: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Path]:
    cfg = load_cfg(config_path)
    seed = get_seed(cfg)

    # deterministic-ish
    np.random.seed(seed)
    sc.settings.verbosity = 2

    out_dir.mkdir(parents=True, exist_ok=True)
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(__name__)

    processed_path = art / "processed.h5ad"
    marker = out_dir / "preprocess.ok"

    if force:
        _safe_force_cleanup(out_dir)

    if processed_path.exists() and marker.exists() and not force:
        log.info("Skip preprocess: outputs exist (use --force to rebuild). out=%s", str(out_dir))
        return {
            "processed_h5ad": processed_path,
            "marker": marker,
            "config_snapshot": art / "config_preprocess.yaml",
        }

    # snapshot config used
    shutil.copyfile(config_path, art / "config_preprocess.yaml")

    raw_path = resolve_raw_h5ad(cfg, raw_h5ad=raw_h5ad)
    log.info("raw_h5ad=%s", str(raw_path))

    adata = sc.read_h5ad(raw_path)

    # Stable cell id
    if "cell_id" not in adata.obs.columns:
        adata.obs["cell_id"] = adata.obs_names.astype(str)

    # Hard requirement for downstream (bagging/splitting)
    ensure_required_obs_cols(adata, cfg)

    # QC params
    qc = cfg.get("qc", {}) or {}
    min_genes = int(qc.get("min_genes", 200))
    max_genes = int(qc.get("max_genes", 2500))
    max_mt_pct = float(qc.get("max_mt_pct", 5.0))
    min_cells_per_gene = int(qc.get("min_cells_per_gene", 3))

    # mito mask
    if "symbol" in adata.var.columns:
        mt_source = adata.var["symbol"].astype(str)
    else:
        mt_source = adata.var_names.to_series().astype(str)

    adata.var["mt"] = infer_mt_mask(mt_source)

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Filters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    # max genes
    if "n_genes_by_counts" in adata.obs.columns:
        adata = adata[adata.obs["n_genes_by_counts"] <= max_genes].copy()

    # multiplets (optional)
    if "multiplets" in adata.obs.columns:
        vc = adata.obs["multiplets"].astype(str).str.lower().value_counts().to_dict()
        log.info("multiplets counts: %s", str(vc))
        adata = adata[adata.obs["multiplets"].astype(str).str.lower().eq("singlet")].copy()

    # mt% filter only if mt exists and metric exists
    if (
        bool(adata.var.get("mt", False).any())
        and "pct_counts_mt" in adata.obs.columns
        and float(adata.obs["pct_counts_mt"].max()) > 0.0
    ):
        adata = adata[adata.obs["pct_counts_mt"] <= max_mt_pct].copy()

    # Preserve raw counts before normalization/log1p
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Normalize + log1p into adata.X
    norm = cfg.get("normalize", {}) or {}
    target_sum = float(norm.get("target_sum", 1e4))
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Write
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(processed_path)

    marker.write_text("ok\n", encoding="utf-8")

    log.info("wrote %s", str(processed_path))
    log.info("marker %s", str(marker))
    log.info(
        "n_cells=%d n_genes=%d | filters: min_genes=%d max_genes=%d max_mt_pct=%.3f min_cells_per_gene=%d | target_sum=%.1f",
        int(adata.n_obs),
        int(adata.n_vars),
        min_genes,
        max_genes,
        max_mt_pct,
        min_cells_per_gene,
        target_sum,
    )

    return {
        "processed_h5ad": processed_path,
        "marker": marker,
        "config_snapshot": art / "config_preprocess.yaml",
    }
