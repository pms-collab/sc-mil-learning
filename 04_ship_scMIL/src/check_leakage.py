# src/check_leakage.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _load_paths(bags_dir: Path) -> Dict[str, Path]:
    return {
        "bags_npz": bags_dir / "bags.npz",
        "bags_meta": bags_dir / "bags_meta.csv",
        "split_bags": bags_dir / "split_bags.csv",
        "bags_table": bags_dir / "artifacts" / "bags_table.csv.gz",
    }


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _read_split_bags(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"bag_id", "split"}
    if not need.issubset(df.columns):
        _fail(f"split_bags.csv must contain {sorted(need)}; got cols={list(df.columns)}")
    df["bag_id"] = df["bag_id"].astype(str)
    df["split"] = df["split"].astype(str)
    bad = set(df["split"].unique()) - {"train", "val", "test"}
    if bad:
        _fail(f"Invalid split values: {sorted(bad)}")
    if df["bag_id"].duplicated().any():
        _fail("split_bags.csv has duplicated bag_id")
    return df


def _read_bags_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"bag_id", "group_id", "label", "y", "n_cells", "split"}
    if not need.issubset(df.columns):
        _fail(f"bags_meta.csv must contain {sorted(need)}; got cols={list(df.columns)}")
    df["bag_id"] = df["bag_id"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["split"] = df["split"].astype(str)
    if df["bag_id"].duplicated().any():
        _fail("bags_meta.csv has duplicated bag_id")
    return df


def _read_bags_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    need = {"cell_id", "bag_id", "label", "group_id", "split"}
    if not need.issubset(df.columns):
        _fail(f"bags_table.csv.gz must contain {sorted(need)}; got cols={list(df.columns)}")
    # enforce types
    df["cell_id"] = df["cell_id"].astype(str)
    df["bag_id"] = df["bag_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    df["split"] = df["split"].astype(str)
    return df


def _check_group_leakage(meta: pd.DataFrame) -> Tuple[bool, Dict[str, object]]:
    g_by_split = {s: set(meta.loc[meta["split"] == s, "group_id"].unique().tolist()) for s in ["train", "val", "test"]}
    inter_tv = g_by_split["train"] & g_by_split["val"]
    inter_tt = g_by_split["train"] & g_by_split["test"]
    inter_vt = g_by_split["val"] & g_by_split["test"]
    ok = (len(inter_tv) == 0) and (len(inter_tt) == 0) and (len(inter_vt) == 0)
    report = {
        "groups_per_split": {k: len(v) for k, v in g_by_split.items()},
        "group_overlap_train_val": sorted(list(inter_tv))[:20],
        "group_overlap_train_test": sorted(list(inter_tt))[:20],
        "group_overlap_val_test": sorted(list(inter_vt))[:20],
    }
    return ok, report


def _check_cell_leakage(table: pd.DataFrame) -> Tuple[bool, Dict[str, object]]:
    # cell_id must be unique globally
    n_dup = int(table["cell_id"].duplicated().sum())
    if n_dup > 0:
        return False, {"cell_id_duplicates": n_dup}

    # each bag has a single split (by construction)
    s_nuniq = table.groupby("bag_id")["split"].nunique()
    bad_bag = s_nuniq[s_nuniq > 1]
    if len(bad_bag) > 0:
        return False, {"bags_with_multiple_splits": bad_bag.index.tolist()[:20]}

    return True, {"cell_id_duplicates": 0, "bags_with_multiple_splits": 0}


def _check_bag_consistency(table: pd.DataFrame) -> Tuple[bool, Dict[str, object]]:
    g_nuniq = table.groupby("bag_id")["group_id"].nunique()
    y_nuniq = table.groupby("bag_id")["label"].nunique()
    bad_g = g_nuniq[g_nuniq > 1].index.tolist()
    bad_y = y_nuniq[y_nuniq > 1].index.tolist()
    ok = (len(bad_g) == 0) and (len(bad_y) == 0)
    return ok, {
        "bags_mixed_group": bad_g[:20],
        "bags_mixed_label": bad_y[:20],
    }


def _check_expected_bag_id_format(meta: pd.DataFrame) -> Dict[str, object]:
    # expected: bag_id == f"{group_id}__{label}" (your donor_id__condition convention)
    expected = (meta["group_id"].astype(str) + "__" + meta["label"].astype(str)).astype(str)
    bad = (meta["bag_id"].astype(str) != expected)
    return {
        "bag_id_matches_group__label": int((~bad).sum()),
        "bag_id_mismatch_group__label": int(bad.sum()),
        "mismatch_examples": meta.loc[bad, ["bag_id", "group_id", "label"]].head(10).to_dict(orient="records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags_dir", required=True, type=str, help="runs/<dataset>/bags")
    ap.add_argument("--out", default=None, type=str, help="Optional report JSON path")
    args = ap.parse_args()

    bags_dir = Path(args.bags_dir)
    paths = _load_paths(bags_dir)

    for k, p in paths.items():
        if not p.exists():
            _fail(f"Missing {k}: {p}")

    split_df = _read_split_bags(paths["split_bags"])
    meta_df = _read_bags_meta(paths["bags_meta"])
    table_df = _read_bags_table(paths["bags_table"])

    # cross-file alignment
    set_split = set(split_df["bag_id"].tolist())
    set_meta = set(meta_df["bag_id"].tolist())
    if set_split != set_meta:
        _fail(f"bag_id mismatch between split_bags and bags_meta: "
              f"only_in_split={len(set_split-set_meta)}, only_in_meta={len(set_meta-set_split)}")

    # table must contain exactly meta bags
    set_table = set(table_df["bag_id"].unique().tolist())
    if set_table != set_meta:
        _fail(f"bag_id mismatch between bags_table and bags_meta: "
              f"only_in_table={len(set_table-set_meta)}, only_in_meta={len(set_meta-set_table)}")

    ok_group, rep_group = _check_group_leakage(meta_df)
    ok_cell, rep_cell = _check_cell_leakage(table_df)
    ok_bag, rep_bag = _check_bag_consistency(table_df)
    rep_fmt = _check_expected_bag_id_format(meta_df)

    report = {
        "paths": {k: str(v) for k, v in paths.items()},
        "counts": {
            "n_cells": int(len(table_df)),
            "n_bags": int(meta_df["bag_id"].nunique()),
            "n_groups": int(meta_df["group_id"].nunique()),
        },
        "splits": {
            "bags_per_split": meta_df["split"].value_counts().to_dict(),
            "cells_per_split": table_df["split"].value_counts().to_dict(),
            "groups_per_split": rep_group["groups_per_split"],
        },
        "checks": {
            "group_leakage_ok": ok_group,
            "cell_leakage_ok": ok_cell,
            "bag_consistency_ok": ok_bag,
        },
        "details": {
            "group": rep_group,
            "cell": rep_cell,
            "bag": rep_bag,
            "bag_id_format": rep_fmt,
        },
    }

    # Print concise summary
    print("[leakage] n_cells:", report["counts"]["n_cells"])
    print("[leakage] n_bags:", report["counts"]["n_bags"], "| bags_per_split:", report["splits"]["bags_per_split"])
    print("[leakage] n_groups:", report["counts"]["n_groups"], "| groups_per_split:", report["splits"]["groups_per_split"])
    print("[leakage] group_leakage_ok:", ok_group)
    print("[leakage] cell_leakage_ok:", ok_cell)
    print("[leakage] bag_consistency_ok:", ok_bag)
    print("[leakage] bag_id mismatch(group__label):", rep_fmt["bag_id_mismatch_group__label"])

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[leakage] wrote report:", outp)

    if not (ok_group and ok_cell and ok_bag):
        _fail("Leakage check FAILED. See details above / report JSON.")

    print("[leakage] OK")


if __name__ == "__main__":
    main()
