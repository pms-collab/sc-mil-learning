from __future__ import annotations

import argparse
import yaml
import gzip
import os
import re
import tarfile
import urllib.request
from pathlib import Path


import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata as ad


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
    - Replace trailing -11 -> -1 (your observed meta quirk)
    - If no trailing -<digits>, append -1 (typical 10x)
    """
    bc = str(bc).strip()
    bc = re.sub(r"-11$", "-1", bc)
    if not re.search(r"-\d+$", bc):
        bc = bc + "-1"
    return bc


def download_file(url: str, dest: Path, *, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[download] skip (exists): {dest.name}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"[download] GET {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as w:
        chunk = r.read(1024 * 1024)
        while chunk:
            w.write(chunk)
            chunk = r.read(1024 * 1024)

    os.replace(tmp, dest)
    print(f"[download] wrote: {dest.name} ({dest.stat().st_size} bytes)")


def safe_extract_tar(tar_path: Path, out_dir: Path, *, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted.ok"
    if marker.exists() and not force:
        print(f"[extract] skip (marker exists): {marker}")
        return

    print(f"[extract] extracting: {tar_path.name} -> {out_dir}")
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            name = member.name.replace("\\", "/")
            # block absolute paths
            if name.startswith("/") or re.match(r"^[A-Za-z]:/", name):
                raise RuntimeError(f"Unsafe tar member path (absolute): {member.name}")
            # block path traversal
            target = (out_dir / name).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Unsafe tar member path (traversal): {member.name}")

        tar.extractall(out_dir)

    marker.write_text("ok\n", encoding="utf-8")
    print(f"[extract] done, marker: {marker}")


# -------------------------
# Raw builder (your rules)
# -------------------------
def build_raw_h5ad(download_dir: Path, extracted_dir: Path, out_path: Path, files: dict) -> None:
    genes_path = download_dir / files["genes"]
    meta_path  = download_dir / files["meta"]

    ex = files.get("extracted", {})
    mtx_ctrl = extracted_dir / ex["ctrl_mtx"]
    bc_ctrl = extracted_dir / ex["ctrl_barcodes"]
    mtx_stim = extracted_dir / ex["stim_mtx"]
    bc_stim = extracted_dir / ex["stim_barcodes"]

    for p in [genes_path, meta_path, mtx_ctrl, bc_ctrl, mtx_stim, bc_stim]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # 1) genes: ENSG <tab> SYMBOL
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

    # make ensembl_id unique by suffix (keep length)
    if not var.index.is_unique:
        counts: dict[str, int] = {}
        new_idx: list[str] = []
        for g in var.index.astype(str):
            if g not in counts:
                counts[g] = 0
                new_idx.append(g)
            else:
                counts[g] += 1
                new_idx.append(f"{g}.{counts[g]}")
        var.index = pd.Index(new_idx, name="ensembl_id")

    # 2) matrices: genes x cells -> transpose to cells x genes
    Xgxc_ctrl = read_mtx_gz(mtx_ctrl)
    Xgxc_stim = read_mtx_gz(mtx_stim)

    if Xgxc_ctrl.shape[0] != len(var) or Xgxc_stim.shape[0] != len(var):
        raise ValueError(
            f"Gene dimension mismatch: var={len(var)}, "
            f"ctrl={Xgxc_ctrl.shape}, stim={Xgxc_stim.shape}"
        )

    X_ctrl = Xgxc_ctrl.T.tocsr()
    X_stim = Xgxc_stim.T.tocsr()

    # 3) barcodes -> obs_names with __ctrl/__stim
    bc_ctrl_raw = [normalize_barcode(x) for x in read_lines_gz(bc_ctrl)]
    bc_stim_raw = [normalize_barcode(x) for x in read_lines_gz(bc_stim)]

    if len(bc_ctrl_raw) != X_ctrl.shape[0] or len(bc_stim_raw) != X_stim.shape[0]:
        raise ValueError(
            f"Barcode count mismatch: ctrl={len(bc_ctrl_raw)} vs cells={X_ctrl.shape[0]}, "
            f"stim={len(bc_stim_raw)} vs cells={X_stim.shape[0]}"
        )

    obs_names_ctrl = [f"{bc}__ctrl" for bc in bc_ctrl_raw]
    obs_names_stim = [f"{bc}__stim" for bc in bc_stim_raw]

    # 4) meta: index barcode, columns ind(donor), stim(ctrl/stim), multiplets...
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

    # 5) AnnData per condition
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

    # 6) concat + join meta
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
    adata.write_h5ad(out_path)  # pass Path (no str/as_posix)

    print("[raw] wrote:", out_path)
    print("[raw] shape:", adata.shape)
    print("[raw] condition counts:", adata.obs["condition"].value_counts().to_dict())
    print("[raw] n_donors:", adata.obs["donor_id"].nunique())
    print("[raw] n_bags:", adata.obs["sample_id"].nunique())
    print("[raw] obs index unique:", bool(adata.obs_names.is_unique))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    root = Path(cfg["data"].get("root", "data"))
    dataset = str(cfg["data"]["dataset"])
    dl = root / "raw" / dataset / "_download"
    extracted = dl / "extracted"
    out_path = root / "raw" / dataset / "raw.h5ad"

    # URL config
    dataset_key = str(cfg["data"]["dataset"])
    src = cfg["data"]["sources"][dataset_key]
    base = src["suppl_base_url"].rstrip("/")
    files = src["files"]

    urls = {
        files["tar"]:   f"{base}/{files['tar']}",
        files["genes"]: f"{base}/{files['genes']}",
        files["meta"]:  f"{base}/{files['meta']}",
    }

    marker = out_path.with_suffix(out_path.suffix + ".ok")

    # 1) download
    for fname, url in urls.items():
        download_file(url, dl / fname, force=args.force)

    # 2) extract (skip only if the required extracted files exist)
    ex = files.get("extracted", {})
    need = [
        extracted / ex["ctrl_mtx"],
        extracted / ex["ctrl_barcodes"],
        extracted / ex["stim_mtx"],
        extracted / ex["stim_barcodes"],
    ]

    if all(p.exists() for p in need) and not args.force:
        print("[extract] skip (required extracted files exist)")
    else:
        safe_extract_tar(dl / files["tar"], extracted, force=args.force)

    # extraction completeness check (always)
    for p in need:
        if not p.exists():
            raise FileNotFoundError(f"Extraction incomplete, missing: {p}")

    # 3) build raw
    if out_path.exists() and marker.exists() and not args.force:
        print(f"[raw] skip (exists + marker): {out_path}")
        return

    build_raw_h5ad(dl, extracted, out_path, files)
    marker.write_text("ok\n", encoding="utf-8")
    print(f"[raw] marker: {marker}")


if __name__ == "__main__":
    main()
