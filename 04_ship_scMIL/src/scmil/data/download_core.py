# src/scmil/data/download_core.py
from __future__ import annotations

import gzip
import logging
import os
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import anndata as ad
import pandas as pd
import yaml
from scipy.io import mmread
from scipy.sparse import csr_matrix


log = logging.getLogger(__name__)


# -------------------------
# Paths / Config
# -------------------------
@dataclass(frozen=True)
class DownloadPaths:
    dataset_dir: Path          # <root>/raw/<dataset>
    download_dir: Path         # <root>/raw/<dataset>/_download
    extracted_dir: Path        # <root>/raw/<dataset>/_download/extracted
    raw_h5ad: Path             # <root>/raw/<dataset>/raw.h5ad
    raw_marker: Path           # <root>/raw/<dataset>/raw.h5ad.ok
    extract_marker: Path       # <root>/raw/<dataset>/_download/.extracted.ok
    log_file: Path             # <root>/raw/<dataset>/_download/logs/download.log
    config_snapshot: Path      # <root>/raw/<dataset>/_download/config_download.yaml


def load_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict.")
    return cfg


def resolve_download_paths(cfg: dict) -> DownloadPaths:
    data = cfg.get("data", {}) or {}
    root = Path(str(data.get("root", "data")))
    dataset = str(data.get("dataset", "")).strip()
    if not dataset:
        raise KeyError("Missing config: data.dataset")

    dataset_dir = root / "raw" / dataset
    download_dir = dataset_dir / "_download"
    extracted_dir = download_dir / "extracted"
    raw_h5ad = dataset_dir / "raw.h5ad"

    return DownloadPaths(
        dataset_dir=dataset_dir,
        download_dir=download_dir,
        extracted_dir=extracted_dir,
        raw_h5ad=raw_h5ad,
        raw_marker=raw_h5ad.with_suffix(raw_h5ad.suffix + ".ok"),
        extract_marker=download_dir / ".extracted.ok",
        log_file=download_dir / "logs" / "download.log",
        config_snapshot=download_dir / "config_download.yaml",
    )


def _get_source_cfg(cfg: dict) -> Tuple[str, Dict[str, object]]:
    """
    Expects:
      cfg["data"]["sources"][<dataset_key>] with:
        - suppl_base_url
        - files {tar, genes, meta, extracted{ctrl_mtx, ctrl_barcodes, stim_mtx, stim_barcodes}}
    """
    data = cfg.get("data", {}) or {}
    dataset_key = str(data.get("dataset", "")).strip()
    if not dataset_key:
        raise KeyError("Missing config: data.dataset")

    sources = data.get("sources", {}) or {}
    if dataset_key not in sources:
        raise KeyError(f"Missing config: data.sources['{dataset_key}']")

    src = sources[dataset_key]
    if not isinstance(src, dict):
        raise ValueError(f"data.sources['{dataset_key}'] must be a mapping(dict).")

    base = str(src.get("suppl_base_url", "")).strip()
    if not base:
        raise KeyError(f"Missing config: data.sources['{dataset_key}'].suppl_base_url")

    files = src.get("files", {})
    if not isinstance(files, dict):
        raise KeyError(f"Missing/invalid config: data.sources['{dataset_key}'].files")

    # required keys
    req = ["tar", "genes", "meta", "extracted"]
    missing = [k for k in req if k not in files]
    if missing:
        raise KeyError(f"Missing keys in files: {missing}")

    ex = files.get("extracted", {})
    if not isinstance(ex, dict):
        raise KeyError("files.extracted must be a dict")

    ex_req = ["ctrl_mtx", "ctrl_barcodes", "stim_mtx", "stim_barcodes"]
    ex_missing = [k for k in ex_req if k not in ex]
    if ex_missing:
        raise KeyError(f"Missing keys in files.extracted: {ex_missing}")

    return base.rstrip("/"), files


# -------------------------
# I/O helpers
# -------------------------
def read_mtx_gz(path: Path) -> csr_matrix:
    with gzip.open(path, "rb") as f:
        m = mmread(f)
    return m.tocsr()


def read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_barcode(bc: str) -> str:
    """
    - Replace trailing -11 -> -1 (observed meta quirk)
    - If no trailing -<digits>, append -1 (typical 10x)
    """
    bc = str(bc).strip()
    bc = re.sub(r"-11$", "-1", bc)
    if not re.search(r"-\d+$", bc):
        bc = bc + "-1"
    return bc


def download_file(url: str, dest: Path, *, force: bool = False, chunk_bytes: int = 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0 and not force:
        log.info("[download] skip (exists): %s", dest.name)
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    log.info("[download] GET %s", url)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as w:
            while True:
                chunk = r.read(chunk_bytes)
                if not chunk:
                    break
                w.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        # cleanup partial
        if tmp.exists():
            tmp.unlink()
        raise

    log.info("[download] wrote: %s (%d bytes)", dest.name, int(dest.stat().st_size))


def safe_extract_tar(tar_path: Path, out_dir: Path, *, force: bool = False, marker: Optional[Path] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = marker or (out_dir / ".extracted.ok")

    if marker.exists() and not force:
        log.info("[extract] skip (marker exists): %s", str(marker))
        return marker

    log.info("[extract] extracting: %s -> %s", tar_path.name, str(out_dir))

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            name = member.name.replace("\\", "/")

            # block absolute paths
            if name.startswith("/") or re.match(r"^[A-Za-z]:/", name):
                raise RuntimeError(f"Unsafe tar member path (absolute): {member.name}")

            # block traversal
            target = (out_dir / name).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Unsafe tar member path (traversal): {member.name}")

        tar.extractall(out_dir)

    marker.write_text("ok\n", encoding="utf-8")
    log.info("[extract] done, marker: %s", str(marker))
    return marker


# -------------------------
# Raw builder
# -------------------------
def build_raw_h5ad(download_dir: Path, extracted_dir: Path, out_path: Path, files: dict) -> Dict[str, object]:
    genes_path = download_dir / files["genes"]
    meta_path = download_dir / files["meta"]

    ex = files.get("extracted", {})
    mtx_ctrl = extracted_dir / ex["ctrl_mtx"]
    bc_ctrl = extracted_dir / ex["ctrl_barcodes"]
    mtx_stim = extracted_dir / ex["stim_mtx"]
    bc_stim = extracted_dir / ex["stim_barcodes"]

    for p in [genes_path, meta_path, mtx_ctrl, bc_ctrl, mtx_stim, bc_stim]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # genes: ENSG <tab> SYMBOL
    genes_df = pd.read_csv(genes_path, sep="\t", header=None, compression="gzip")
    if genes_df.shape[1] >= 2:
        genes_df = genes_df.iloc[:, :2].copy()
        genes_df.columns = ["ensembl_id", "symbol"]
    else:
        genes_df = genes_df.iloc[:, :1].copy()
        genes_df.columns = ["ensembl_id"]
        genes_df["symbol"] = genes_df["ensembl_id"]

    genes_df["ensembl_id"] = genes_df["ensembl_id"].astype(str).str.strip()
    genes_df["symbol"] = genes_df["symbol"].astype(str).str.strip()

    var = pd.DataFrame(index=pd.Index(genes_df["ensembl_id"].values, name="ensembl_id"))
    var["symbol"] = genes_df["symbol"].values

    # enforce unique var index
    if not var.index.is_unique:
        counts: Dict[str, int] = {}
        new_idx: list[str] = []
        for g in var.index.astype(str):
            if g not in counts:
                counts[g] = 0
                new_idx.append(g)
            else:
                counts[g] += 1
                new_idx.append(f"{g}.{counts[g]}")
        var.index = pd.Index(new_idx, name="ensembl_id")

    # matrices: genes x cells -> transpose to cells x genes
    Xgxc_ctrl = read_mtx_gz(mtx_ctrl)
    Xgxc_stim = read_mtx_gz(mtx_stim)

    if Xgxc_ctrl.shape[0] != len(var) or Xgxc_stim.shape[0] != len(var):
        raise ValueError(
            f"Gene dimension mismatch: var={len(var)}, ctrl={Xgxc_ctrl.shape}, stim={Xgxc_stim.shape}"
        )

    X_ctrl = Xgxc_ctrl.T.tocsr()
    X_stim = Xgxc_stim.T.tocsr()

    # barcodes -> obs_names with __ctrl/__stim
    bc_ctrl_raw = [normalize_barcode(x) for x in read_lines_gz(bc_ctrl)]
    bc_stim_raw = [normalize_barcode(x) for x in read_lines_gz(bc_stim)]

    if len(bc_ctrl_raw) != X_ctrl.shape[0] or len(bc_stim_raw) != X_stim.shape[0]:
        raise ValueError(
            f"Barcode count mismatch: ctrl={len(bc_ctrl_raw)} vs cells={X_ctrl.shape[0]}, "
            f"stim={len(bc_stim_raw)} vs cells={X_stim.shape[0]}"
        )

    obs_names_ctrl = [f"{bc}__ctrl" for bc in bc_ctrl_raw]
    obs_names_stim = [f"{bc}__stim" for bc in bc_stim_raw]

    # meta: index barcode, columns ind(donor), stim(ctrl/stim), multiplets...
    meta = pd.read_csv(meta_path, sep="\t", compression="gzip", index_col=0)
    if "ind" not in meta.columns or "stim" not in meta.columns:
        raise RuntimeError(f"Meta columns missing. Found: {list(meta.columns)}")

    meta = meta.copy()
    meta.index = meta.index.astype(str).map(normalize_barcode)
    meta["donor_id"] = meta["ind"].astype(str).str.strip()
    meta["condition"] = meta["stim"].astype(str).str.strip().str.lower()

    bad = set(meta["condition"].unique()) - {"ctrl", "stim"}
    if bad:
        raise RuntimeError(f"Unexpected meta condition values: {sorted(bad)[:20]}")

    # suffix meta index by condition to match obs_names
    meta.index = meta.index + "__" + meta["condition"]
    if not meta.index.is_unique:
        raise RuntimeError("Meta index not unique after suffixing by condition.")

    # AnnData per condition
    ad_ctrl = ad.AnnData(
        X=X_ctrl,
        obs=pd.DataFrame(index=pd.Index(obs_names_ctrl, name="cell_id")),
        var=var.copy(),
    )
    ad_stim = ad.AnnData(
        X=X_stim,
        obs=pd.DataFrame(index=pd.Index(obs_names_stim, name="cell_id")),
        var=var.copy(),
    )

    # concat + join meta
    adata = ad.concat([ad_ctrl, ad_stim], axis=0, join="outer", merge="unique")
    adata.obs = adata.obs.join(meta, how="left")

    if adata.obs["donor_id"].isna().any() or adata.obs["condition"].isna().any():
        md = (int(adata.obs["donor_id"].isna().sum()), int(adata.obs["condition"].isna().sum()))
        raise RuntimeError(
            "Meta join failed for some cells.\n"
            f"Missing donor_id={md[0]}, missing condition={md[1]}\n"
            "Fix: barcode normalization mismatch between barcodes.tsv.gz and meta index."
        )

    # required pipeline cols
    adata.obs["sample_id"] = adata.obs["donor_id"].astype(str) + "__" + adata.obs["condition"].astype(str)
    adata.obs["cell_id"] = adata.obs_names.astype(str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)

    summary = {
        "raw_h5ad": str(out_path),
        "shape": (int(adata.n_obs), int(adata.n_vars)),
        "condition_counts": adata.obs["condition"].value_counts().to_dict(),
        "n_donors": int(adata.obs["donor_id"].nunique()),
        "n_bags": int(adata.obs["sample_id"].nunique()),
        "obs_index_unique": bool(adata.obs_names.is_unique),
    }
    return summary


# -------------------------
# Core entrypoint
# -------------------------
def run_download(
    *,
    config_path: Path,
    force: bool = False,
) -> Dict[str, object]:
    cfg = load_cfg(config_path)
    paths = resolve_download_paths(cfg)

    # Ensure dirs
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.download_dir.mkdir(parents=True, exist_ok=True)
    (paths.download_dir / "logs").mkdir(parents=True, exist_ok=True)

    # snapshot config (source-of-truth for data prep)
    shutil.copyfile(config_path, paths.config_snapshot)

    base, files = _get_source_cfg(cfg)

    # URLs
    urls = {
        files["tar"]: f"{base}/{files['tar']}",
        files["genes"]: f"{base}/{files['genes']}",
        files["meta"]: f"{base}/{files['meta']}",
    }

    # 1) downloads
    for fname, url in urls.items():
        download_file(url, paths.download_dir / fname, force=force)

    # 2) extraction (skip only if required extracted files exist and not force)
    ex = files.get("extracted", {})
    required_extracted = [
        paths.extracted_dir / ex["ctrl_mtx"],
        paths.extracted_dir / ex["ctrl_barcodes"],
        paths.extracted_dir / ex["stim_mtx"],
        paths.extracted_dir / ex["stim_barcodes"],
    ]

    if all(p.exists() for p in required_extracted) and not force:
        log.info("[extract] skip (required extracted files exist)")
    else:
        tar_path = paths.download_dir / files["tar"]
        if not tar_path.exists():
            raise FileNotFoundError(f"Missing tar after download: {tar_path}")
        safe_extract_tar(
            tar_path,
            paths.extracted_dir,
            force=force,
            marker=paths.extract_marker,
        )

    # extraction completeness check (always)
    for p in required_extracted:
        if not p.exists():
            raise FileNotFoundError(f"Extraction incomplete, missing: {p}")

    # 3) raw.h5ad build (skip if exists + marker and not force)
    if paths.raw_h5ad.exists() and paths.raw_marker.exists() and not force:
        log.info("[raw] skip (exists + marker): %s", str(paths.raw_h5ad))
        return {
            "raw_h5ad": str(paths.raw_h5ad),
            "raw_marker": str(paths.raw_marker),
            "download_dir": str(paths.download_dir),
            "extracted_dir": str(paths.extracted_dir),
            "config_snapshot": str(paths.config_snapshot),
            "log_file": str(paths.log_file),
            "skipped": True,
        }

    summary = build_raw_h5ad(paths.download_dir, paths.extracted_dir, paths.raw_h5ad, files)
    paths.raw_marker.write_text("ok\n", encoding="utf-8")

    log.info("[raw] wrote: %s", str(paths.raw_h5ad))
    log.info("[raw] marker: %s", str(paths.raw_marker))
    log.info(
        "[raw] shape=%s | condition=%s | n_donors=%d | n_bags=%d | obs_unique=%s",
        str(summary["shape"]),
        str(summary["condition_counts"]),
        int(summary["n_donors"]),
        int(summary["n_bags"]),
        str(summary["obs_index_unique"]),
    )

    return {
        "raw_h5ad": str(paths.raw_h5ad),
        "raw_marker": str(paths.raw_marker),
        "download_dir": str(paths.download_dir),
        "extracted_dir": str(paths.extracted_dir),
        "config_snapshot": str(paths.config_snapshot),
        "log_file": str(paths.log_file),
        "skipped": False,
        "summary": summary,
    }


# local import for shutil (kept here to avoid unused in some linters)
import shutil  # noqa: E402
