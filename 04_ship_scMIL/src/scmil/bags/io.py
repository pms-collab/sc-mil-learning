# src/scmil/bags/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


SplitName = Union[str, Sequence[str]]


@dataclass(frozen=True)
class BagsData:
    """
    Canonical MIL bags representation.

    X        : (N_cells_total, D) float32
    bag_ptr  : (N_bags+1,) int64, CSR-like pointer into X rows
    bag_ids  : (N_bags,) object/str, unique bag identifier
    y        : (N_bags,) int64, class id per bag
    groups   : (N_bags,) object/str, group id (e.g., donor_id)
    label_names : (n_labels,) object/str mapping class id -> label string

    Invariants:
      - bag_ptr[0] == 0
      - bag_ptr is non-decreasing
      - bag_ptr[-1] == X.shape[0]
      - len(bag_ids) == len(y) == len(groups) == len(bag_ptr)-1
      - bag_ids unique
    """
    X: np.ndarray
    bag_ptr: np.ndarray
    bag_ids: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    label_names: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.X.shape[0])

    @property
    def feat_dim(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0

    @property
    def n_bags(self) -> int:
        return int(self.bag_ids.shape[0])

    @property
    def n_labels(self) -> int:
        return int(self.label_names.shape[0])

    def bag_slice(self, bag_index: int) -> slice:
        i = int(bag_index)
        if i < 0 or i >= self.n_bags:
            raise IndexError(f"bag_index out of range: {i} (n_bags={self.n_bags})")
        s = int(self.bag_ptr[i])
        e = int(self.bag_ptr[i + 1])
        return slice(s, e)

    def bag_size(self, bag_index: int) -> int:
        sl = self.bag_slice(bag_index)
        return int(sl.stop - sl.start)

    def iter_bags(self) -> Iterator[Tuple[int, str, np.ndarray, int]]:
        """
        Yields: (bag_index, bag_id, X_bag_view, y_bag)
        Note: X_bag_view is a view into X (no copy).
        """
        for i in range(self.n_bags):
            sl = self.bag_slice(i)
            yield i, str(self.bag_ids[i]), self.X[sl], int(self.y[i])

    def bag_index_map(self) -> Dict[str, int]:
        return {str(b): int(i) for i, b in enumerate(self.bag_ids.tolist())}


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _require_keys(z: np.lib.npyio.NpzFile, keys: Sequence[str], *, src: Path) -> None:
    missing = [k for k in keys if k not in z.files]
    if missing:
        raise KeyError(f"Missing keys in npz {src}: {missing}. Found: {sorted(z.files)}")


def _coerce_str_array(a: np.ndarray, name: str) -> np.ndarray:
    # object or str arrays expected
    if a.dtype == object:
        out = a.astype(object, copy=False)
    else:
        # allow unicode/bytes
        out = a.astype(str, copy=False).astype(object, copy=False)
    # no NaN-like (np.nan) checks for object arrays; we enforce empty string not allowed optionally
    return out


def _validate_bag_ptr(bag_ptr: np.ndarray, n_cells: int, n_bags: int, *, src: Path) -> None:
    if bag_ptr.ndim != 1:
        raise ValueError(f"bag_ptr must be 1D; got shape={bag_ptr.shape} in {src}")
    if bag_ptr.dtype != np.int64:
        # allow int32 but coerce upstream; here enforce invariant for downstream simplicity
        raise ValueError(f"bag_ptr must be int64; got dtype={bag_ptr.dtype} in {src}")
    if bag_ptr.shape[0] != n_bags + 1:
        raise ValueError(
            f"bag_ptr length must equal n_bags+1. got {bag_ptr.shape[0]} vs {n_bags+1} in {src}"
        )
    if int(bag_ptr[0]) != 0:
        raise ValueError(f"bag_ptr[0] must be 0; got {bag_ptr[0]} in {src}")
    if int(bag_ptr[-1]) != int(n_cells):
        raise ValueError(
            f"bag_ptr[-1] must equal n_cells. got {bag_ptr[-1]} vs n_cells={n_cells} in {src}"
        )
    # monotonic non-decreasing
    if np.any(bag_ptr[1:] < bag_ptr[:-1]):
        bad = np.where(bag_ptr[1:] < bag_ptr[:-1])[0][:10].tolist()
        raise ValueError(f"bag_ptr must be non-decreasing. violations at indices={bad} in {src}")


def _validate_unique_ids(bag_ids: np.ndarray, *, src: Path) -> None:
    s = pd.Series(bag_ids.astype(str))
    if s.isna().any():
        raise ValueError(f"bag_ids contains NaN in {src}")
    dup = s.duplicated()
    if dup.any():
        ex = s[dup].head(10).tolist()
        raise ValueError(f"bag_ids must be unique. duplicates (up to 10): {ex} in {src}")


def load_bags_npz(path: Union[str, Path], *, allow_pickle: bool = True) -> BagsData:
    """
    Load canonical bags from .npz created by build_bags.

    Expected keys:
      - X (float32/float64 ok; coerced to float32)
      - bag_ptr (int64 preferred; coerced to int64)
      - bag_ids (object/str)
      - y (int64 preferred; coerced to int64)
      - groups (object/str)
      - label_names (object/str)

    Strict invariants validated (see BagsData docstring).
    """
    src = _as_path(path)
    if not src.exists():
        raise FileNotFoundError(f"bags npz not found: {src}")

    z = np.load(src, allow_pickle=bool(allow_pickle))
    try:
        _require_keys(z, ["X", "bag_ptr", "bag_ids", "y", "groups", "label_names"], src=src)

        X = z["X"]
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape={X.shape} in {src}")
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)

        bag_ptr = z["bag_ptr"].astype(np.int64, copy=False)
        bag_ids = _coerce_str_array(z["bag_ids"], "bag_ids")
        y = z["y"].astype(np.int64, copy=False)
        groups = _coerce_str_array(z["groups"], "groups")
        label_names = _coerce_str_array(z["label_names"], "label_names")

        n_bags = int(bag_ids.shape[0])
        if y.shape[0] != n_bags or groups.shape[0] != n_bags:
            raise ValueError(
                f"Length mismatch in {src}: len(bag_ids)={n_bags}, len(y)={y.shape[0]}, len(groups)={groups.shape[0]}"
            )

        _validate_unique_ids(bag_ids, src=src)
        _validate_bag_ptr(bag_ptr, n_cells=int(X.shape[0]), n_bags=n_bags, src=src)

        # y range sanity if label_names exist
        if label_names.shape[0] > 0:
            y_min = int(np.min(y)) if y.size else 0
            y_max = int(np.max(y)) if y.size else -1
            if y_min < 0:
                raise ValueError(f"y contains negative class id (min={y_min}) in {src}")
            if y_max >= int(label_names.shape[0]):
                raise ValueError(
                    f"y contains class id >= n_labels (max={y_max}, n_labels={label_names.shape[0]}) in {src}"
                )

        return BagsData(X=X, bag_ptr=bag_ptr, bag_ids=bag_ids, y=y, groups=groups, label_names=label_names)
    finally:
        try:
            z.close()
        except Exception:
            pass


def load_split_csv(path: Union[str, Path]) -> Dict[str, str]:
    """
    Read split_bags.csv (bag_id, split) -> dict mapping bag_id -> split.
    Valid splits: train/val/test.
    """
    p = _as_path(path)
    if not p.exists():
        raise FileNotFoundError(f"split csv not found: {p}")
    df = pd.read_csv(p)
    if not {"bag_id", "split"}.issubset(df.columns):
        raise ValueError(f"split csv must contain columns bag_id, split; got {list(df.columns)}")
    df["bag_id"] = df["bag_id"].astype(str)
    df["split"] = df["split"].astype(str)
    bad = set(df["split"].unique()) - {"train", "val", "test"}
    if bad:
        raise ValueError(f"Invalid split values: {sorted(bad)} in {p}")
    if df["bag_id"].duplicated().any():
        dup = df.loc[df["bag_id"].duplicated(), "bag_id"].head(10).tolist()
        raise ValueError(f"split csv has duplicated bag_id (up to 10): {dup} in {p}")
    return dict(zip(df["bag_id"].tolist(), df["split"].tolist()))


def load_bags_meta_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read bags_meta.csv with required columns:
      bag_id, group_id, label, y, n_cells, split
    Returns typed DataFrame.
    """
    p = _as_path(path)
    if not p.exists():
        raise FileNotFoundError(f"bags_meta csv not found: {p}")
    df = pd.read_csv(p)
    need = {"bag_id", "group_id", "label", "y", "n_cells", "split"}
    if not need.issubset(df.columns):
        raise ValueError(f"bags_meta.csv must contain {sorted(need)}; got {list(df.columns)} in {p}")
    df = df.copy()
    df["bag_id"] = df["bag_id"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["split"] = df["split"].astype(str)
    df["y"] = df["y"].astype(np.int64)
    df["n_cells"] = df["n_cells"].astype(np.int64)
    if df["bag_id"].duplicated().any():
        dup = df.loc[df["bag_id"].duplicated(), "bag_id"].head(10).tolist()
        raise ValueError(f"bags_meta.csv has duplicated bag_id (up to 10): {dup} in {p}")
    bad = set(df["split"].unique()) - {"train", "val", "test"}
    if bad:
        raise ValueError(f"bags_meta.csv has invalid split values: {sorted(bad)} in {p}")
    return df


def make_split_indices(
    bags: BagsData,
    split_map: Dict[str, str],
    *,
    require_all_present: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Map bag_ids in BagsData order -> split indices dict.

    If require_all_present=True, any missing bag_id in split_map raises.
    """
    splits = []
    missing: List[str] = []
    for bid in bags.bag_ids.astype(str):
        if bid not in split_map:
            missing.append(str(bid))
            splits.append(None)
        else:
            splits.append(split_map[bid])

    if missing and require_all_present:
        raise KeyError(f"{len(missing)} bag_id missing in split map. Examples (up to 10): {missing[:10]}")

    split_arr = np.array(splits, dtype=object)
    idx = {
        "train": np.where(split_arr == "train")[0],
        "val": np.where(split_arr == "val")[0],
        "test": np.where(split_arr == "test")[0],
    }
    # If you want to allow empty, change here; current pipeline expects all exist.
    for k in ["train", "val", "test"]:
        if idx[k].size == 0:
            raise ValueError(f"Empty split: {k}")
    return idx


def parse_splits(splits: SplitName, *, allow_train: bool = True) -> Tuple[str, ...]:
    """
    Normalize split selection.
    Accepts:
      - "test"
      - "test,val"
      - ["test", "val"]
    """
    if isinstance(splits, str):
        parts = [x.strip() for x in splits.split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in splits if str(x).strip()]

    if not parts:
        raise ValueError("splits is empty")

    allowed = {"train", "val", "test"} if allow_train else {"val", "test"}
    bad = [x for x in parts if x not in allowed]
    if bad:
        raise ValueError(f"Invalid splits={bad}. Allowed={sorted(allowed)}")

    # preserve user order, remove duplicates
    seen = set()
    out = []
    for x in parts:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return tuple(out)


def validate_bags_consistency(
    bags: BagsData,
    *,
    split_csv: Optional[Union[str, Path]] = None,
    meta_csv: Optional[Union[str, Path]] = None,
    strict_order: bool = False,
) -> Dict[str, object]:
    """
    Cross-check consistency between bags.npz, split_bags.csv, bags_meta.csv.

    - strict_order=False: set equality required; order not required (common case).
    - strict_order=True : bag_ids order must match meta order after sorting by bag_id? (usually not desired).

    Returns a report dict (no printing).
    """
    report: Dict[str, object] = {
        "npz": {"n_bags": bags.n_bags, "n_cells": bags.n_cells, "feat_dim": bags.feat_dim},
        "split": None,
        "meta": None,
        "ok": True,
        "problems": [],
    }

    bag_ids_npz = pd.Index(bags.bag_ids.astype(str))

    if split_csv is not None:
        smap = load_split_csv(split_csv)
        report["split"] = {"path": str(_as_path(split_csv)), "n_rows": int(len(smap))}
        set_split = set(smap.keys())
        set_npz = set(bag_ids_npz.tolist())
        if set_split != set_npz:
            report["ok"] = False
            report["problems"].append(
                {
                    "type": "bag_id_mismatch_split_vs_npz",
                    "only_in_split": int(len(set_split - set_npz)),
                    "only_in_npz": int(len(set_npz - set_split)),
                }
            )

    if meta_csv is not None:
        meta = load_bags_meta_csv(meta_csv)
        report["meta"] = {"path": str(_as_path(meta_csv)), "n_rows": int(len(meta))}
        set_meta = set(meta["bag_id"].tolist())
        set_npz = set(bag_ids_npz.tolist())
        if set_meta != set_npz:
            report["ok"] = False
            report["problems"].append(
                {
                    "type": "bag_id_mismatch_meta_vs_npz",
                    "only_in_meta": int(len(set_meta - set_npz)),
                    "only_in_npz": int(len(set_npz - set_meta)),
                }
            )
        if strict_order:
            # only meaningful if meta is already in the same order as npz (rare)
            if not np.array_equal(meta["bag_id"].astype(str).to_numpy(), bag_ids_npz.to_numpy()):
                report["ok"] = False
                report["problems"].append({"type": "bag_id_order_mismatch_meta_vs_npz"})

    return report


def select_bag_indices_by_split(
    bags: BagsData,
    *,
    split_csv: Union[str, Path],
    splits: SplitName,
    allow_train: bool = True,
) -> np.ndarray:
    """
    Convenience: returns bag indices for the requested splits in NPZ order.
    """
    splits_t = parse_splits(splits, allow_train=allow_train)
    smap = load_split_csv(split_csv)
    split_arr = np.array([smap[str(b)] for b in bags.bag_ids.astype(str)], dtype=object)
    mask = np.isin(split_arr, list(splits_t))
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"No bags found for splits={splits_t}")
    return idx
