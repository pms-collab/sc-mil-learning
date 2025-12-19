# src/build_bags.py
from __future__ import annotations

import argparse
import logging
import shutil
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd
import yaml

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


def _detect_parquet_engine() -> str:
    """Return available parquet engine name or raise RuntimeError."""
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception as e:
            raise RuntimeError(
                "No Parquet engine found. Install pyarrow (preferred) or fastparquet.\n"
                "Example: pip install pyarrow"
            ) from e


def requires_parquet_engine() -> Callable:
    """
    Fail fast if no parquet engine exists.
    Does NOT pass engine to the function; just validates environment.
    """
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            engine = _detect_parquet_engine()
            logging.info("Parquet engine detected: %s", engine)
            return fn(*args, **kwargs)
        return wrapper
    return deco


def requires_file_exists(param_name: str) -> Callable:
    """
    Validate a Path-like argument exists before function body runs.
    Expects the function to be called with keyword arg `param_name=Path(...)`
    or positional args where that name exists in kwargs (prefer kwargs).
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
    """
    Validate required columns exist in a pandas DataFrame before running function.
    """
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


def resolve_processed_h5ad(out_dir: Path) -> Path:
    return out_dir / "artifacts" / "processed.h5ad"


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
    val_groups = set(shuffled[n_test : n_test + n_val])
    train_groups = set(shuffled[n_test + n_val :])

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
            f"Some bag_id map to multiple group_id (mixed bags / leakage risk). "
            f"Examples (up to 10): {bad}"
        )

    y_nuniq = df.groupby("bag_id")["label"].nunique()
    if (y_nuniq > 1).any():
        bad = y_nuniq[y_nuniq > 1].index.tolist()[:10]
        raise ValueError(
            f"Some bag_id map to multiple labels (inconsistent supervision). "
            f"Examples (up to 10): {bad}"
        )


@requires_obs_cols(
    cols=["cell_id"],  # base requirement, plus dynamic ones checked inside
    obs_param="obs",
)
def make_bags_table(*, obs: pd.DataFrame, bags_cfg: BagsCfg) -> pd.DataFrame:
    # Enforce presence of dataset-specific columns here (input contract)
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


def write_parquet_strict(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = _detect_parquet_engine()  # guaranteed by decorator, but keep defensive
    df.to_parquet(out_path, index=False, engine=engine, compression="snappy")


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
# Main runner
# ============================================================
@log_step("build_bags")
@requires_parquet_engine()
@requires_file_exists("processed_h5ad")
def run_build_bags(
    *,
    config_path: Path,
    out_dir: Path,
    processed_h5ad: Path,
) -> None:
    cfg = load_cfg(config_path)
    seed = get_seed(cfg)
    bags_cfg = get_bags_cfg(cfg)
    split_cfg = get_split_cfg(cfg)

    if split_cfg.method != "group_holdout":
        raise NotImplementedError(f"split.method={split_cfg.method} not implemented.")

    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    # Keep step-specific resolved config snapshot
    shutil.copyfile(config_path, art_dir / "config_build_bags.yaml")

    # Read obs only (backed mode reduces memory pressure)
    adata = sc.read_h5ad(processed_h5ad, backed="r")
    obs = adata.obs.copy()

    df = make_bags_table(obs=obs, bags_cfg=bags_cfg)

    mapping = group_holdout_split(
        df["group_id"].tolist(),
        test_size=split_cfg.test_size,
        val_size=split_cfg.val_size,
        seed=seed,
    )
    df["split"] = df["group_id"].map(mapping)
    if df["split"].isna().any():
        raise RuntimeError("Some rows have missing split assignment. Bug in mapping/apply.")

    out_parquet = art_dir / "bags.parquet"
    write_parquet_strict(df, out_parquet)

    # Summary
    logging.info("Wrote %s", str(out_parquet))
    logging.info(
        "cells=%d | bags=%d | groups=%d",
        len(df), df["bag_id"].nunique(), df["group_id"].nunique()
    )
    logging.info("cells per split:\n%s", df["split"].value_counts().to_string())
    logging.info(
        "groups per split:\n%s",
        df.drop_duplicates("group_id")["split"].value_counts().to_string(),
    )

    # Close backed AnnData
    try:
        adata.file.close()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out)

    setup_logging(out_dir)
    logging.info("Starting build_bags")
    logging.info("config=%s", str(config_path))
    logging.info("out=%s", str(out_dir))

    processed_h5ad = resolve_processed_h5ad(out_dir)

    run_build_bags(
        config_path=config_path,
        out_dir=out_dir,
        processed_h5ad=processed_h5ad,
    )


if __name__ == "__main__":
    main()
