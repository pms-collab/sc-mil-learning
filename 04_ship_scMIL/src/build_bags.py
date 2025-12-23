# src/build_bags.py
from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import scanpy as sc
except ImportError as e:
    raise ImportError("scanpy is required to read .h5ad. Install scanpy/anndata.") from e


# ============================================================
# Decorators (cross-cutting concerns)
# ============================================================
def log_step(step_name: str) -> Callable:
    """Log start/end + elapsed time for a function."""
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            logging.info("START %s", step_name)
            out = fn(*args, **kwargs)
            dt = time.time() - t0
            logging.info("END %s (%.2fs)", step_name, dt)
            return out
        return wrapper
    return deco


def requires_file_exists(param_name: str) -> Callable:
    """
    Validate a Path-like argument exists before function body runs.
    Expects the function to be called with keyword arg `param_name=Path(...)`.
    """
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if param_name not in kwargs:
                raise TypeError(
                    f"requires_file_exists expects '{param_name}' to be passed as keyword argument."
                )
            p = Path(kwargs[param_name])
            if not p.exists():
                raise FileNotFoundError(f"Required input not found: {p}")
            return fn(*args, **kwargs)
        return wrapper
    return deco


def requires_obs_cols(cols: List[str], obs_param: str = "obs") -> Callable:
    """Validate required columns exist in a pandas DataFrame before running function."""
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if obs_param not in kwargs:
                raise TypeError(
                    f"requires_obs_cols expects '{obs_param}' to be passed as keyword argument."
                )
            obs = kwargs[obs_param]
            if not isinstance(obs, pd.DataFrame):
                raise TypeError(f"{obs_param} must be a pandas DataFrame; got {type(obs)}")
            missing = [c for c in cols if c not in obs.columns]
            if missing:
                raise ValueError(f"Missing required adata.obs columns: {missing}")
            return fn(*args, **kwargs)
        return wrapper
    return deco


# ============================================================
# Config helpers
# ============================================================
@dataclass(frozen=True)
class SplitCfg:
    method: str
    group_col: str
    test_size: float
    val_size: float


@dataclass(frozen=True)
class BagsCfg:
    bag_id_col: str
    label_col: str
    group_id_col: str


@dataclass(frozen=True)
class FeaturesCfg:
    hvg: int
    pca_dim: int


def load_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict.")
    return cfg


def get_seed(cfg: dict) -> int:
    seed = cfg.get("seed", 42)
    if not isinstance(seed, int):
        raise ValueError(f"seed must be int; got {type(seed)}")
    return seed


def get_bags_cfg(cfg: dict) -> BagsCfg:
    bags = cfg.get("bags", {})
    return BagsCfg(
        bag_id_col=bags["bag_id_col"],
        label_col=bags["label_col"],
        group_id_col=bags["group_id_col"],
    )


def get_split_cfg(cfg: dict) -> SplitCfg:
    split = cfg.get("split", {})
    return SplitCfg(
        method=split["method"],
        group_col=split["group_col"],
        test_size=float(split["test_size"]),
        val_size=float(split["val_size"]),
    )


def get_features_cfg(cfg: dict) -> FeaturesCfg:
    feat = cfg.get("features", {})
    return FeaturesCfg(
        hvg=int(feat.get("hvg", 2000)),
        pca_dim=int(feat.get("pca_dim", 50)),
    )


def resolve_processed_h5ad(cfg: dict, *, preprocess_out: Optional[Path], processed_h5ad: Optional[Path]) -> Path:
    """
    Priority:
      1) --processed_h5ad (file)
      2) --preprocess_out (dir -> expects artifacts/processed.h5ad; or direct file)
      3) conventional: runs/<dataset>/preprocess/artifacts/processed.h5ad
    """
    if processed_h5ad is not None:
        p = processed_h5ad
        if not p.exists():
            raise FileNotFoundError(f"--processed_h5ad not found: {p}")
        if p.suffix != ".h5ad":
            raise ValueError(f"--processed_h5ad must be .h5ad; got: {p}")
        return p

    if preprocess_out is not None:
        p = preprocess_out
        if p.is_dir():
            cand = p / "artifacts" / "processed.h5ad"
            if cand.exists():
                return cand
            raise FileNotFoundError(f"--preprocess_out given but processed not found: {cand}")
        if p.is_file() and p.suffix == ".h5ad":
            return p
        raise FileNotFoundError(f"--preprocess_out not usable: {p}")

    dataset = str(cfg["data"]["dataset"])
    cand = Path("runs") / dataset / "preprocess" / "artifacts" / "processed.h5ad"
    if cand.exists():
        return cand

    raise FileNotFoundError(
        "processed.h5ad not found. Provide one of:\n"
        "  - --processed_h5ad <file.h5ad>\n"
        "  - --preprocess_out <runs/<dataset>/preprocess>\n"
        f"Or ensure conventional path exists: {cand}"
    )


# ============================================================
# Split logic (group_holdout)
# ============================================================
def _validate_split_fracs(test_size: float, val_size: float) -> None:
    for name, x in [("test_size", test_size), ("val_size", val_size)]:
        if not (0.0 <= x < 1.0):
            raise ValueError(f"{name} must be in [0, 1); got {x}")
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1; got {test_size + val_size}")


def group_holdout_split(
    groups: Iterable[str],
    *,
    test_size: float,
    val_size: float,
    seed: int,
) -> Dict[str, str]:
    """
    Deterministic group-wise split (donor-level holdout).
    Fractions apply to number of UNIQUE groups.
    """
    _validate_split_fracs(test_size, val_size)

    uniq = sorted(set(groups))
    n = len(uniq)
    if n == 0:
        raise ValueError("No groups found (empty group list).")

    s = pd.Series(uniq).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled = s.tolist()

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))

    if n_test >= n:
        n_test = n - 1
    if n_val >= n - n_test:
        n_val = max(0, (n - n_test) - 1)

    if n - (n_test + n_val) <= 0:
        if n >= 3:
            n_test, n_val = 1, 1
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test, n_val = 0, 0

    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test: n_test + n_val])
    train_groups = set(shuffled[n_test + n_val:])

    mapping: Dict[str, str] = {}
    for g in train_groups:
        mapping[g] = "train"
    for g in val_groups:
        mapping[g] = "val"
    for g in test_groups:
        mapping[g] = "test"

    if len(mapping) != n:
        raise RuntimeError("Split mapping does not cover all groups. Bug in split logic.")
    if (train_groups & val_groups) or (train_groups & test_groups) or (val_groups & test_groups):
        raise RuntimeError("Group overlap across splits. Bug in split logic.")

    return mapping


# ============================================================
# Bag table construction + IO
# ============================================================
def coerce_str_series(s: pd.Series, name: str) -> pd.Series:
    if s.isna().any():
        n_na = int(s.isna().sum())
        raise ValueError(f"{name} contains NaN for {n_na} rows. Fix upstream metadata.")
    return s.astype(str)


def assert_bag_consistency(df: pd.DataFrame) -> None:
    """Each bag_id must map to exactly one (group_id, label)."""
    g_nuniq = df.groupby("bag_id")["group_id"].nunique()
    if (g_nuniq > 1).any():
        bad = g_nuniq[g_nuniq > 1].index.tolist()[:10]
        raise ValueError(
            "Some bag_id map to multiple group_id (mixed bags / leakage risk). "
            f"Examples (up to 10): {bad}"
        )

    y_nuniq = df.groupby("bag_id")["label"].nunique()
    if (y_nuniq > 1).any():
        bad = y_nuniq[y_nuniq > 1].index.tolist()[:10]
        raise ValueError(
            "Some bag_id map to multiple labels (inconsistent supervision). "
            f"Examples (up to 10): {bad}"
        )


@requires_obs_cols(cols=["cell_id"], obs_param="obs")
def make_bags_table(*, obs: pd.DataFrame, bags_cfg: BagsCfg) -> pd.DataFrame:
    required = ["cell_id", bags_cfg.bag_id_col, bags_cfg.label_col, bags_cfg.group_id_col]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(f"Missing required adata.obs columns: {missing}")

    df = pd.DataFrame(
        {
            "cell_id": coerce_str_series(obs["cell_id"], "cell_id"),
            "bag_id": coerce_str_series(obs[bags_cfg.bag_id_col], bags_cfg.bag_id_col),
            "label": coerce_str_series(obs[bags_cfg.label_col], bags_cfg.label_col),
            "group_id": coerce_str_series(obs[bags_cfg.group_id_col], bags_cfg.group_id_col),
        }
    )

    if df["cell_id"].duplicated().any():
        n_dup = int(df["cell_id"].duplicated().sum())
        raise ValueError(f"cell_id must be unique; found {n_dup} duplicates.")

    assert_bag_consistency(df)
    return df


def write_csv_gz(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # pandas supports compression='gzip' without external deps.
    df.to_csv(out_path, index=False, encoding="utf-8", compression="gzip")


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


# ============================================================
# Feature + packing
# ============================================================
def _select_hvg_inplace(adata, *, n_top: int, layer: Optional[str]) -> None:
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(n_top, adata.n_vars),
            flavor="seurat_v3",
            layer=layer,
            subset=True,
            inplace=True,
        )
        return
    except Exception as e:
        logging.warning("HVG seurat_v3 failed (%s). Falling back to seurat.", str(e))

    # CRITICAL: seurat + raw counts layer can expm1 overflow -> force layer=None
    if layer is not None:
        logging.info("Using layer=None for seurat HVG to avoid expm1 overflow on raw counts.")
        layer = None

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(n_top, adata.n_vars),
        flavor="seurat",
        layer=layer,  # now None
        subset=True,
        inplace=True,
    )

def compute_cell_features_pca(
    adata_path: Path,
    *,
    hvg: int,
    pca_dim: int,
    train_mask: np.ndarray,
    seed: int,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Train-only fit pipeline:
      1) HVG selection is fit on TRAIN cells only (optionally using layer='counts')
      2) StandardScaler fit on TRAIN cells only
      3) PCA fit on TRAIN cells only
      4) TRANSFORM applied to ALL cells (train/val/test) to produce X_pca

    Returns:
      - X_pca: (n_cells, pca_dim_eff) float32
      - obs_min: original obs copy (to verify alignment upstream)
    """
    logging.info("Reading AnnData for features: %s", str(adata_path))
    adata = sc.read_h5ad(adata_path)

    # Ensure cell_id exists
    if "cell_id" not in adata.obs.columns:
        adata.obs["cell_id"] = adata.obs_names.astype(str)

    n_cells = int(adata.n_obs)
    train_mask = np.asarray(train_mask).astype(bool)
    if train_mask.ndim != 1 or train_mask.shape[0] != n_cells:
        raise ValueError(f"train_mask must be 1D bool of length n_cells={n_cells}; got {train_mask.shape}")
    n_train = int(train_mask.sum())
    if n_train <= 1:
        raise ValueError(f"train_mask has too few TRAIN cells: {n_train}")

    # --- 1) HVG selection (FIT on train only) ---
    layer = "counts" if "counts" in adata.layers else None
    if layer is None:
        logging.warning("layers['counts'] not found; HVG will use adata.X (likely normalized/log1p).")
    else:
        logging.info("Using layer='%s' for HVG selection (train-only).", layer)

    adata_train = adata[train_mask].copy()

    # If we use raw counts layer, remove log1p annotation to avoid expm1 overflow inside Scanpy
    if layer == "counts" and "log1p" in adata_train.uns:
        logging.info("Removing adata_train.uns['log1p'] for HVG on raw counts layer to avoid expm1 overflow.")
        adata_train.uns.pop("log1p", None)

    _select_hvg_inplace(adata_train, n_top=hvg, layer=layer)
    hvg_genes = adata_train.var_names.astype(str).tolist()
    if len(hvg_genes) < 2:
        raise RuntimeError("HVG selection produced too few genes. Check preprocessing / HVG settings.")
    logging.info("Selected HVGs (train-only): n=%d", len(hvg_genes))

    # Subset FULL data to train-selected HVGs
    adata_hvg = adata[:, hvg_genes].copy()

    # --- 2) Prepare matrices (dense) ---
    X_full = adata_hvg.X
    X_train = X_full[train_mask]

    if sp.issparse(X_train):
        X_train = X_train.toarray()
    X_train = np.asarray(X_train, dtype=np.float32)

    # --- 3) Scale (FIT on train only) ---
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)  # float64 output (sklearn default)

    # --- 4) PCA (FIT on train only) ---
    # PCA constraint: n_components <= min(n_train, n_features)
    n_feat = int(X_train_scaled.shape[1])
    pca_dim_eff = int(max(2, min(pca_dim, n_feat, n_train - 1)))
    if pca_dim_eff != int(pca_dim):
        logging.info("Adjust pca_dim: %d -> %d (n_train=%d, n_feat=%d)", int(pca_dim), pca_dim_eff, n_train, n_feat)

    pca = PCA(
        n_components=pca_dim_eff,
        svd_solver="randomized",
        random_state=int(seed),
    )
    pca.fit(X_train_scaled)

    # --- 5) Transform ALL cells (chunked to limit RAM) ---
    X_pca = np.empty((n_cells, pca_dim_eff), dtype=np.float32)
    batch = 2048

    for start in range(0, n_cells, batch):
        end = min(n_cells, start + batch)
        X_chunk = X_full[start:end]
        if sp.issparse(X_chunk):
            X_chunk = X_chunk.toarray()
        X_chunk = np.asarray(X_chunk, dtype=np.float32)

        X_chunk_scaled = scaler.transform(X_chunk)
        X_pca[start:end] = pca.transform(X_chunk_scaled).astype(np.float32, copy=False)

    # --- 6) Optional: persist feature-fit artifacts for reproducibility ---
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "hvg_genes_train.csv").write_text("\n".join(hvg_genes) + "\n", encoding="utf-8")

        np.savez(
            artifacts_dir / "scaler_train.npz",
            mean_=scaler.mean_.astype(np.float64, copy=False),
            scale_=scaler.scale_.astype(np.float64, copy=False),
        )
        np.savez(
            artifacts_dir / "pca_train.npz",
            components_=pca.components_.astype(np.float64, copy=False),
            mean_=pca.mean_.astype(np.float64, copy=False),
            explained_variance_=pca.explained_variance_.astype(np.float64, copy=False),
            explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float64, copy=False),
        )
        logging.info("Wrote feature-fit artifacts under: %s", str(artifacts_dir))

    obs_min = adata.obs.copy()
    if X_pca.shape[0] != obs_min.shape[0]:
        raise RuntimeError("Feature/obs row mismatch. Bug in feature computation.")
    return X_pca, obs_min

def pack_bags_from_table_and_features(
    *,
    bags_table: pd.DataFrame,
    X_cell: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    """
    Deterministic packing:
      - bags sorted by bag_id
      - within bag, preserve original cell order by appearance in bags_table
    Returns dict ready to be saved to .npz plus helpful metadata.
    """
    # Align X_cell rows with bags_table rows (they must be same order of cells)
    # We enforce by constructing bags_table from obs of the SAME AnnData used to compute features.
    if X_cell.shape[0] != len(bags_table):
        raise ValueError(f"X_cell rows ({X_cell.shape[0]}) != bags_table rows ({len(bags_table)})")

    # Stable label mapping
    labels_sorted = np.sort(bags_table["label"].unique().astype(str))
    label_to_id = {lab: i for i, lab in enumerate(labels_sorted)}

    bag_ids_sorted = np.sort(bags_table["bag_id"].unique().astype(str))

    # Precompute indices per bag, preserving row order
    # Using groupby with sort=False preserves encounter order for idx arrays.
    tmp = bags_table.copy()
    tmp["idx"] = np.arange(tmp.shape[0], dtype=np.int64)
    gb = tmp.groupby("bag_id", sort=False)["idx"].apply(np.array).to_dict()

    feats = []
    bag_ptr = [0]
    bag_ids_out = []
    bag_y = []
    bag_group = []
    bag_label = []
    bag_ncells = []

    for b in bag_ids_sorted:
        idx = gb.get(b)
        if idx is None or len(idx) == 0:
            continue

        # Enforce single label/group per bag (already checked, but keep defensive)
        labs = tmp.iloc[idx]["label"].unique()
        grps = tmp.iloc[idx]["group_id"].unique()
        if len(labs) != 1 or len(grps) != 1:
            raise RuntimeError(f"Inconsistent bag {b}: labels={labs}, groups={grps}")

        xb = X_cell[idx, :]
        feats.append(xb)
        bag_ptr.append(bag_ptr[-1] + xb.shape[0])

        bag_ids_out.append(b)
        bag_label.append(str(labs[0]))
        bag_group.append(str(grps[0]))
        bag_y.append(int(label_to_id[str(labs[0])]))
        bag_ncells.append(int(len(idx)))

    X_concat = np.vstack(feats).astype(np.float32, copy=False)
    packed = {
        "X": X_concat,
        "bag_ptr": np.array(bag_ptr, dtype=np.int64),
        "bag_ids": np.array(bag_ids_out, dtype=object),
        "y": np.array(bag_y, dtype=np.int64),
        "groups": np.array(bag_group, dtype=object),
        "label_names": labels_sorted.astype(object),
        "label_to_id": label_to_id,  # python dict (not directly saved to npz)
        "bag_meta": pd.DataFrame(
            {
                "bag_id": np.array(bag_ids_out, dtype=object),
                "group_id": np.array(bag_group, dtype=object),
                "label": np.array(bag_label, dtype=object),
                "y": np.array(bag_y, dtype=np.int64),
                "n_cells": np.array(bag_ncells, dtype=np.int64),
            }
        ),
    }
    return packed


# ============================================================
# Main runner
# ============================================================
@log_step("build_bags")
@requires_file_exists("processed_h5ad")
def run_build_bags(
    *,
    config_path: Path,
    out_dir: Path,
    processed_h5ad: Path,
    force: bool,
) -> None:
    cfg = load_cfg(config_path)
    seed = get_seed(cfg)
    bags_cfg = get_bags_cfg(cfg)
    split_cfg = get_split_cfg(cfg)
    feat_cfg = get_features_cfg(cfg)

    if split_cfg.method != "group_holdout":
        raise NotImplementedError(f"split.method={split_cfg.method} not implemented.")

    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    # Keep step-specific resolved config snapshot
    shutil.copyfile(config_path, art_dir / "config_build_bags.yaml")

    # Rerun safety
    out_npz = out_dir / "bags.npz"
    out_cell_table = art_dir / "bags_table.csv.gz"
    out_bag_meta = out_dir / "bags_meta.csv"
    out_split_bags = out_dir / "split_bags.csv"
    marker = out_dir / "bags.ok"

    if (
        out_npz.exists()
        and out_cell_table.exists()
        and out_bag_meta.exists()
        and out_split_bags.exists()
        and marker.exists()
        and not force
    ):
        logging.info("Skip: outputs exist (use --force to rebuild). out=%s", str(out_dir))
        return

    # ---- 1) build strict cell-level bags table (obs only, backed read) ----
    adata_backed = sc.read_h5ad(processed_h5ad, backed="r")
    obs = adata_backed.obs.copy()
    df = make_bags_table(obs=obs, bags_cfg=bags_cfg)

    # ---- 2) split mapping (group holdout) ----
    mapping = group_holdout_split(
        df["group_id"].tolist(),
        test_size=split_cfg.test_size,
        val_size=split_cfg.val_size,
        seed=seed,
    )
    df["split"] = df["group_id"].map(mapping)
    if df["split"].isna().any():
        raise RuntimeError("Some rows have missing split assignment. Bug in mapping/apply.")

    # Close backed file early
    try:
        adata_backed.file.close()
    except Exception:
        pass

    # Persist strict cell table (no parquet)
    write_csv_gz(df, out_cell_table)
    logging.info("Wrote cell-level bags table: %s", str(out_cell_table))

    # ---- 3) compute features + pack bags ----
    train_mask = (df["split"].astype(str).to_numpy() == "train")
    X_cell, obs_for_align = compute_cell_features_pca(
        processed_h5ad,
        hvg=feat_cfg.hvg,
        pca_dim=feat_cfg.pca_dim,
        train_mask=train_mask,
        seed=seed,
        artifacts_dir=art_dir,
    )

    # Ensure alignment between df and features by cell_id (enterprise-grade guard)
    # We require same set and same order to avoid silent misalignment.
    cell_id_df = df["cell_id"].astype(str).to_numpy()
    cell_id_obs = obs_for_align["cell_id"].astype(str).to_numpy()
    if len(cell_id_df) != len(cell_id_obs):
        raise RuntimeError("cell_id length mismatch between table and AnnData obs. Bug upstream.")
    if not np.array_equal(cell_id_df, cell_id_obs):
        raise RuntimeError(
            "cell_id order mismatch between bags_table and AnnData obs. "
            "This must be fixed upstream; refusing to continue to avoid silent leakage/misalignment."
        )

    packed = pack_bags_from_table_and_features(bags_table=df, X_cell=X_cell, seed=seed)

    # Bag-level split table
    bag_meta = packed["bag_meta"].copy()
    bag_to_split = (
        df.drop_duplicates("bag_id")[["bag_id", "split"]]
        .set_index("bag_id")["split"]
        .to_dict()
    )
    bag_meta["split"] = bag_meta["bag_id"].map(bag_to_split)
    if bag_meta["split"].isna().any():
        raise RuntimeError("Some bags missing split assignment. Bug in bag_to_split mapping.")

    # ---- 4) write outputs ----
    np.savez_compressed(
        out_npz,
        X=packed["X"],
        bag_ptr=packed["bag_ptr"],
        bag_ids=packed["bag_ids"],
        y=packed["y"],
        groups=packed["groups"],
        label_names=packed["label_names"],
        feat_dim=np.array([packed["X"].shape[1]], dtype=np.int64),
        hvg_n=np.array([feat_cfg.hvg], dtype=np.int64),
        pca_dim=np.array([packed["X"].shape[1]], dtype=np.int64),
    )
    bag_meta.to_csv(out_bag_meta, index=False, encoding="utf-8")
    bag_meta[["bag_id", "split"]].to_csv(out_split_bags, index=False, encoding="utf-8")
    marker.write_text("ok\n", encoding="utf-8")

    # ---- 5) summaries ----
    logging.info("Wrote %s", str(out_npz))
    logging.info("Wrote %s", str(out_bag_meta))
    logging.info("Wrote %s", str(out_split_bags))
    logging.info("Wrote marker %s", str(marker))

    logging.info(
        "cells=%d | bags=%d | groups=%d | feat_dim=%d",
        len(df),
        int(bag_meta["bag_id"].nunique()),
        int(bag_meta["group_id"].nunique()),
        int(packed["X"].shape[1]),
    )
    logging.info("cells per split:\n%s", df["split"].value_counts().to_string())
    logging.info("bags per split:\n%s", bag_meta["split"].value_counts().to_string())
    logging.info(
        "groups per split:\n%s",
        bag_meta.drop_duplicates("group_id")["split"].value_counts().to_string(),
    )

    # Optional: label mapping log
    logging.info("label_to_id=%s", str(packed["label_to_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument(
        "--preprocess_out",
        default=None,
        type=str,
        help="Path to runs/<dataset>/preprocess (dir) or processed.h5ad (file). Optional.",
    )
    parser.add_argument(
        "--processed_h5ad",
        default=None,
        type=str,
        help="Direct path to processed.h5ad. Optional; overrides --preprocess_out.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out)

    setup_logging(out_dir)
    logging.info("Starting build_bags")
    logging.info("config=%s", str(config_path))
    logging.info("out=%s", str(out_dir))

    cfg = load_cfg(config_path)
    processed_h5ad = resolve_processed_h5ad(
        cfg,
        preprocess_out=Path(args.preprocess_out) if args.preprocess_out else None,
        processed_h5ad=Path(args.processed_h5ad) if args.processed_h5ad else None,
    )

    run_build_bags(
        config_path=config_path,
        out_dir=out_dir,
        processed_h5ad=processed_h5ad,
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
