# src/scmil/eval/core.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scmil.bags.io import (
    BagsData,
    load_bags_npz,
    load_bags_meta_csv,
    parse_splits,
    select_bag_indices_by_split,
)

# IMPORTANT: 너 train에서 이미 단일 소스로 만들었어야 하는 모듈
# build_model(model_name, in_dim, n_classes, hidden=..., dropout=..., attn_dim=...)
from scmil.models.mil import build_model


SplitName = Union[str, Sequence[str]]


@dataclass(frozen=True)
class EvalOutputs:
    pred_df: pd.DataFrame
    metrics: Dict[str, object]


def _binary_auroc(y_true01: np.ndarray, score_pos: np.ndarray) -> float:
    """
    sklearn 없이 AUROC(이전 네 구현과 동일한 rank-based).
    y_true01: {0,1}
    score_pos: higher => more positive
    """
    y = y_true01.astype(np.int64)
    s = score_pos.astype(np.float64)

    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    # tie correction: average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    for c in np.where(counts > 1)[0]:
        idx = np.where(inv == c)[0]
        ranks[idx] = ranks[idx].mean()

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _ckpt_get(ckpt: dict, key: str, default):
    return ckpt[key] if key in ckpt else default


def load_model_from_ckpt(
    ckpt_path: Path,
    *,
    device: torch.device,
) -> Tuple[nn.Module, str, Dict[str, object]]:
    """
    ckpt 호환 레이어.
    - 예전 train: {"model": state_dict, "model_name", "in_dim", "n_classes", ...}
    - 새 train: {"state_dict": ..., "model_name", "in_dim", "n_classes", "hidden", "dropout", "attn_dim", ...}
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing ckpt: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model_name = str(_ckpt_get(ckpt, "model_name", "meanpool")).lower()
    in_dim = int(_ckpt_get(ckpt, "in_dim", 0))
    n_classes = int(_ckpt_get(ckpt, "n_classes", 0))
    if in_dim <= 0 or n_classes <= 0:
        raise ValueError(
            f"ckpt missing in_dim/n_classes (got in_dim={in_dim}, n_classes={n_classes}). "
            f"Fix train checkpoint dict."
        )

    # 하이퍼파라미터 키가 없으면 defaults로라도 eval이 돌게 한다(호환성)
    hidden = int(_ckpt_get(ckpt, "hidden", 128))
    dropout = float(_ckpt_get(ckpt, "dropout", 0.1))
    attn_dim = int(_ckpt_get(ckpt, "attn_dim", 128))

    state_dict = _ckpt_get(ckpt, "state_dict", None)
    if state_dict is None:
        # legacy key
        state_dict = _ckpt_get(ckpt, "model", None)
    if state_dict is None:
        raise ValueError("ckpt missing state_dict (expected key 'state_dict' or legacy 'model').")

    model = build_model(
        model_name=model_name,
        in_dim=in_dim,
        n_classes=n_classes,
        hidden=hidden,
        dropout=dropout,
        attn_dim=attn_dim,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    info = {
        "model_name": model_name,
        "in_dim": in_dim,
        "n_classes": n_classes,
        "hidden": hidden,
        "dropout": dropout,
        "attn_dim": attn_dim,
        "ckpt_path": str(ckpt_path),
    }
    return model, model_name, info


@torch.no_grad()
def predict_bags(
    *,
    model: nn.Module,
    model_name: str,
    bags: BagsData,
    bag_indices: np.ndarray,
    splits_by_bag: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """
    bag_indices: NPZ order indices
    splits_by_bag: NPZ order split labels (train/val/test)
    """
    X = bags.X
    bag_ptr = bags.bag_ptr
    bag_ids = bags.bag_ids.astype(str)
    y_true = bags.y.astype(np.int64)
    groups = bags.groups.astype(str)

    rows = []
    for bi in bag_indices.tolist():
        s = int(bag_ptr[bi])
        e = int(bag_ptr[bi + 1])
        Xb = torch.from_numpy(X[s:e]).to(device)

        if model_name == "abmil":
            logits, _w = model(Xb)
        else:
            logits = model(Xb)

        prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred = int(np.argmax(prob))

        row = {
            "bag_id": str(bag_ids[bi]),
            "group_id": str(groups[bi]),
            "split": str(splits_by_bag[bi]),
            "y_true": int(y_true[bi]),
            "y_pred": int(pred),
            "n_cells": int(e - s),
        }

        # keep p0..pK-1 for downstream
        for k in range(prob.shape[0]):
            row[f"p{k}"] = float(prob[k])

        rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(pred_df: pd.DataFrame, *, n_classes: int) -> Dict[str, object]:
    if len(pred_df) == 0:
        raise ValueError("pred_df is empty")

    y_true = pred_df["y_true"].to_numpy(dtype=np.int64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.int64)
    acc = float((y_true == y_pred).mean())

    cm = pd.crosstab(
        pred_df["y_true"],
        pred_df["y_pred"],
        rownames=["true"],
        colnames=["pred"],
        dropna=False,
    )
    cm_dict = {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in cm.to_dict().items()}

    metrics: Dict[str, object] = {
        "n_bags": int(len(pred_df)),
        "n_groups": int(pred_df["group_id"].nunique()),
        "acc": acc,
        "confusion": cm_dict,
    }

    # AUROC only for binary where p1 exists and varies
    if n_classes == 2 and "p1" in pred_df.columns:
        p1 = pred_df["p1"].to_numpy(dtype=np.float64)
        if np.isfinite(p1).all() and len(np.unique(p1)) > 1 and set(np.unique(y_true)).issubset({0, 1}):
            metrics["auroc"] = _binary_auroc(y_true, p1)

    # per-group accuracy
    per_group = (
        pred_df.groupby("group_id")
        .apply(lambda x: float((x["y_true"].to_numpy() == x["y_pred"].to_numpy()).mean()))
        .to_dict()
    )
    metrics["acc_by_group"] = {str(k): float(v) for k, v in per_group.items()}

    return metrics


def run_eval(
    *,
    bags_dir: Path,
    ckpt_path: Path,
    out_dir: Path,
    splits: SplitName = "test",
    allow_train: bool = True,
    write_outputs: bool = True,
) -> EvalOutputs:
    """
    End-to-end eval runner (library-friendly).
    - Inputs are explicit paths (no argparse, no global state).
    - Uses split_bags.csv as split source-of-truth, but also requires bags_meta.csv 존재.
    """
    bags_dir = Path(bags_dir)
    out_dir = Path(out_dir)
    ckpt_path = Path(ckpt_path)

    bags_npz = bags_dir / "bags.npz"
    split_csv = bags_dir / "split_bags.csv"
    meta_csv = bags_dir / "bags_meta.csv"

    if not bags_npz.exists():
        raise FileNotFoundError(f"Missing: {bags_npz}")
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing: {split_csv}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing: {meta_csv}")

    # load
    bags = load_bags_npz(bags_npz)

    # load meta mainly for sanity (and for future extension); split is from split_csv
    _meta = load_bags_meta_csv(meta_csv)

    # indices by split (NPZ order)
    splits_t = parse_splits(splits, allow_train=allow_train)
    idx = select_bag_indices_by_split(bags, split_csv=split_csv, splits=splits_t, allow_train=allow_train)

    # also create splits_by_bag for output table
    split_map = pd.read_csv(split_csv).set_index("bag_id")["split"].astype(str).to_dict()
    splits_by_bag = np.array([split_map[str(b)] for b in bags.bag_ids.astype(str)], dtype=object)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, model_info = load_model_from_ckpt(ckpt_path, device=device)

    pred_df = predict_bags(
        model=model,
        model_name=model_name,
        bags=bags,
        bag_indices=idx,
        splits_by_bag=splits_by_bag,
        device=device,
    )

    metrics = compute_metrics(pred_df, n_classes=int(model_info["n_classes"]))
    metrics["eval_splits"] = list(splits_t)
    metrics["model"] = model_info

    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "predictions.csv").write_text(pred_df.to_csv(index=False), encoding="utf-8")
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return EvalOutputs(pred_df=pred_df, metrics=metrics)

