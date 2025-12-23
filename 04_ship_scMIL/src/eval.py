# src/eval.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Models (must match train.py)
# -------------------------
class MeanPoolMIL(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, X_inst: torch.Tensor) -> torch.Tensor:
        H = self.encoder(X_inst)
        z = H.mean(dim=0, keepdim=True)
        logits = self.head(z)
        return logits.squeeze(0)


class ABMIL(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, attn: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(hidden, attn)
        self.attn_U = nn.Linear(hidden, attn)
        self.attn_w = nn.Linear(attn, 1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, X_inst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H = self.encoder(X_inst)
        A = torch.tanh(self.attn_V(H)) * torch.sigmoid(self.attn_U(H))
        a = self.attn_w(A).squeeze(-1)
        w = torch.softmax(a, dim=0)
        z = torch.sum(w.unsqueeze(-1) * H, dim=0, keepdim=True)
        logits = self.head(z).squeeze(0)
        return logits, w


# -------------------------
# Utils
# -------------------------
def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def load_bags_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    return {
        "X": z["X"].astype(np.float32, copy=False),
        "bag_ptr": z["bag_ptr"].astype(np.int64, copy=False),
        "bag_ids": z["bag_ids"].astype(object),
        "y": z["y"].astype(np.int64, copy=False),
        "groups": z["groups"].astype(object),
        "label_names": z["label_names"].astype(object),
    }


def load_bags_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"bag_id", "group_id", "label", "y", "n_cells", "split"}
    if not need.issubset(df.columns):
        raise ValueError(f"bags_meta.csv must contain {sorted(need)}")
    df["bag_id"] = df["bag_id"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    df["split"] = df["split"].astype(str)
    return df


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def binary_auroc(y_true01: np.ndarray, score_pos: np.ndarray) -> float:
    """
    AUROC without sklearn.
    y_true01: 0/1
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

    # rank-based AUC (handling ties by average rank)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    # tie correction
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    for c in np.where(counts > 1)[0]:
        idx = np.where(inv == c)[0]
        ranks[idx] = ranks[idx].mean()

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def run_eval(
    model: nn.Module,
    model_name: str,
    bags: dict,
    meta: pd.DataFrame,
    eval_splits: Tuple[str, ...],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    X = bags["X"]
    bag_ptr = bags["bag_ptr"]
    bag_ids = bags["bag_ids"].astype(str)
    y = bags["y"]
    groups = bags["groups"].astype(str)

    # join split info from meta (single source of truth)
    meta2 = meta.set_index("bag_id").loc[bag_ids].reset_index()
    split = meta2["split"].astype(str).to_numpy()

    mask = np.isin(split, list(eval_splits))
    if mask.sum() == 0:
        raise ValueError(f"No bags found for eval_splits={eval_splits}")

    idxs = np.where(mask)[0]

    rows = []
    for bi in idxs:
        s = int(bag_ptr[bi])
        e = int(bag_ptr[bi + 1])
        Xb = torch.from_numpy(X[s:e]).to(device)

        if model_name == "abmil":
            logits, _w = model(Xb)
        else:
            logits = model(Xb)

        prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred = int(np.argmax(prob))
        rows.append(
            {
                "bag_id": str(bag_ids[bi]),
                "group_id": str(groups[bi]),
                "split": str(split[bi]),
                "y_true": int(y[bi]),
                "y_pred": int(pred),
                "p0": float(prob[0]) if prob.shape[0] > 0 else float("nan"),
                "p1": float(prob[1]) if prob.shape[0] > 1 else float("nan"),
                "n_cells": int(e - s),
            }
        )

    df = pd.DataFrame(rows)

    # metrics
    y_true = df["y_true"].to_numpy(dtype=np.int64)
    y_pred = df["y_pred"].to_numpy(dtype=np.int64)
    acc = accuracy(y_true, y_pred)

    cm = pd.crosstab(df["y_true"], df["y_pred"], rownames=["true"], colnames=["pred"], dropna=False)
    cm_dict = {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in cm.to_dict().items()}

    metrics = {
        "eval_splits": list(eval_splits),
        "n_bags": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "acc": float(acc),
        "confusion": cm_dict,
    }

    # binary AUROC if applicable
    if df[["p0", "p1"]].notna().all().all() and df["p1"].nunique() > 1 and set(np.unique(y_true)).issubset({0, 1}):
        metrics["auroc"] = binary_auroc(y_true, df["p1"].to_numpy(dtype=np.float64))

    # per-donor accuracy (group_id is donor_id)
    per_group = (
        df.groupby("group_id")
        .apply(lambda x: float((x["y_true"].to_numpy() == x["y_pred"].to_numpy()).mean()))
        .to_dict()
    )
    metrics["acc_by_group"] = {str(k): float(v) for k, v in per_group.items()}

    return df, metrics


def build_model_from_ckpt(ckpt: dict) -> Tuple[nn.Module, str]:
    model_name = str(ckpt.get("model_name", "meanpool")).lower()
    in_dim = int(ckpt["in_dim"])
    n_classes = int(ckpt["n_classes"])

    # hidden/attn are not stored; use defaults matching train.py unless you start varying them.
    if model_name == "abmil":
        model = ABMIL(in_dim=in_dim, n_classes=n_classes)
    elif model_name == "meanpool":
        model = MeanPoolMIL(in_dim=in_dim, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model_name in ckpt: {model_name}")

    model.load_state_dict(ckpt["model"])
    return model, model_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags_dir", required=True, help="runs/<dataset>/bags")
    ap.add_argument("--ckpt", required=True, help="runs/<dataset>/train/<exp>/checkpoints/best.pt")
    ap.add_argument("--out", required=True, help="runs/<dataset>/eval/<name>")
    ap.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to evaluate. Default: test. Example: test,val",
    )
    args = ap.parse_args()

    bags_dir = Path(args.bags_dir)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out)
    setup_logging(out_dir)

    bags_npz = bags_dir / "bags.npz"
    meta_csv = bags_dir / "bags_meta.csv"
    if not bags_npz.exists():
        raise FileNotFoundError(f"Missing: {bags_npz}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing: {meta_csv}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing: {ckpt_path}")

    bags = load_bags_npz(bags_npz)
    meta = load_bags_meta(meta_csv)

    # Define "not used in training donors" in the strict sense:
    # evaluate only splits that are not 'train'. Default is test.
    splits = tuple([s.strip() for s in args.splits.split(",") if s.strip()])
    if any(s not in {"train", "val", "test"} for s in splits):
        raise ValueError(f"Invalid splits={splits}. Must be subset of train/val/test.")
    if "train" in splits:
        logging.warning("You included 'train' in eval splits. This violates 'unseen donors' intent.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device=%s", device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model, model_name = build_model_from_ckpt(ckpt)
    model.to(device)
    model.eval()

    df_pred, metrics = run_eval(
        model=model,
        model_name=model_name,
        bags=bags,
        meta=meta,
        eval_splits=splits,
        device=device,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "predictions.csv"
    metrics_json = out_dir / "metrics.json"
    df_pred.to_csv(pred_csv, index=False, encoding="utf-8")
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logging.info("model=%s", model_name)
    logging.info("eval_splits=%s | n_bags=%d | n_groups=%d | acc=%.3f",
                 str(splits), metrics["n_bags"], metrics["n_groups"], metrics["acc"])
    if "auroc" in metrics:
        logging.info("auroc=%.3f", metrics["auroc"])
    logging.info("wrote %s", str(pred_csv))
    logging.info("wrote %s", str(metrics_json))


if __name__ == "__main__":
    main()
