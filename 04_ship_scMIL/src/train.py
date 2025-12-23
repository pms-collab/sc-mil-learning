# src/train.py
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# -------------------------
# Logging
# -------------------------
def setup_logging(out_dir: Path) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


# -------------------------
# Config helpers
# -------------------------
def load_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict.")
    return cfg


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe if no cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Data structures
# -------------------------
@dataclass(frozen=True)
class BagsData:
    X: np.ndarray            # (Ncells, D)
    bag_ptr: np.ndarray      # (Nbags+1,)
    bag_ids: np.ndarray      # (Nbags,)
    y: np.ndarray            # (Nbags,)
    groups: np.ndarray       # (Nbags,)
    label_names: np.ndarray  # (n_labels,)


def load_bags_npz(path: Path) -> BagsData:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32, copy=False)
    bag_ptr = z["bag_ptr"].astype(np.int64, copy=False)
    bag_ids = z["bag_ids"]
    y = z["y"].astype(np.int64, copy=False)
    groups = z["groups"]
    label_names = z["label_names"]
    # basic sanity
    if bag_ptr.ndim != 1 or bag_ptr.shape[0] != (bag_ids.shape[0] + 1):
        raise ValueError("bag_ptr must be 1D and length must equal n_bags+1.")
    if bag_ptr[0] != 0 or bag_ptr[-1] != X.shape[0]:
        raise ValueError("bag_ptr endpoints inconsistent with X shape.")
    if len(y) != len(bag_ids) or len(groups) != len(bag_ids):
        raise ValueError("y/groups length must equal n_bags.")
    return BagsData(X=X, bag_ptr=bag_ptr, bag_ids=bag_ids, y=y, groups=groups, label_names=label_names)


def load_split_csv(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path)
    if "bag_id" not in df.columns or "split" not in df.columns:
        raise ValueError("split_bags.csv must contain columns: bag_id, split")
    return dict(zip(df["bag_id"].astype(str), df["split"].astype(str)))


def make_split_indices(bags: BagsData, split_map: Dict[str, str]) -> Dict[str, np.ndarray]:
    split = []
    for bid in bags.bag_ids.astype(str):
        if bid not in split_map:
            raise KeyError(f"bag_id missing in split map: {bid}")
        split.append(split_map[bid])
    split = np.array(split, dtype=object)

    idx = {
        "train": np.where(split == "train")[0],
        "val": np.where(split == "val")[0],
        "test": np.where(split == "test")[0],
    }
    for k in ["train", "val", "test"]:
        if idx[k].size == 0:
            raise ValueError(f"Empty split: {k}")
    return idx


# -------------------------
# MIL Models
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
        # X_inst: (n_i, D)
        H = self.encoder(X_inst)                 # (n_i, H)
        z = H.mean(dim=0, keepdim=True)          # (1, H)
        logits = self.head(z)                    # (1, C)
        return logits.squeeze(0)                 # (C,)


class ABMIL(nn.Module):
    """
    Attention-based MIL (Ilse et al. style gating).
    """
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
        # X_inst: (n_i, D)
        H = self.encoder(X_inst)  # (n_i, H)

        A = torch.tanh(self.attn_V(H)) * torch.sigmoid(self.attn_U(H))  # (n_i, attn)
        a = self.attn_w(A).squeeze(-1)                                  # (n_i,)
        w = torch.softmax(a, dim=0)                                     # (n_i,)

        z = torch.sum(w.unsqueeze(-1) * H, dim=0, keepdim=True)         # (1, H)
        logits = self.head(z).squeeze(0)                                # (C,)
        return logits, w


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float((pred == y_true).float().mean().item())


@torch.no_grad()
def eval_split(
    model: nn.Module,
    bags: BagsData,
    bag_indices: np.ndarray,
    device: torch.device,
    model_name: str,
) -> Dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0

    for bi in bag_indices:
        s = int(bags.bag_ptr[bi])
        e = int(bags.bag_ptr[bi + 1])
        Xb = torch.from_numpy(bags.X[s:e]).to(device)
        yb = torch.tensor(int(bags.y[bi]), dtype=torch.long, device=device)

        if model_name == "abmil":
            logits, _w = model(Xb)
        else:
            logits = model(Xb)

        loss = F.cross_entropy(logits.unsqueeze(0), yb.unsqueeze(0))
        losses.append(float(loss.item()))

        pred = int(torch.argmax(logits).item())
        correct += int(pred == int(bags.y[bi]))
        total += 1

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(correct / total) if total else float("nan"),
        "n_bags": int(total),
    }


# -------------------------
# Training loop
# -------------------------
def train(
    cfg: dict,
    bags: BagsData,
    split_idx: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    seed = int(cfg.get("seed", 42))
    set_all_seeds(seed)

    mil_cfg = cfg.get("mil", {})
    model_name = str(mil_cfg.get("model", "meanpool")).lower()
    epochs = int(mil_cfg.get("epochs", 5))
    lr = float(mil_cfg.get("lr", 1e-3))
    batch_size = int(mil_cfg.get("batch_size", 1))  # keep for future; we do per-bag step
    hidden = int(mil_cfg.get("hidden", 128))
    dropout = float(mil_cfg.get("dropout", 0.1))
    attn_dim = int(mil_cfg.get("attn_dim", 128))

    n_classes = int(len(np.unique(bags.y)))
    in_dim = int(bags.X.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device=%s", device)
    logging.info("model=%s | in_dim=%d | n_classes=%d", model_name, in_dim, n_classes)

    if model_name == "abmil":
        model: nn.Module = ABMIL(in_dim=in_dim, n_classes=n_classes, hidden=hidden, attn=attn_dim, dropout=dropout)
    elif model_name == "meanpool":
        model = MeanPoolMIL(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)
    else:
        raise ValueError(f"Unknown mil.model: {model_name}")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Artifacts
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, art / "config_train.yaml")  # filled later via global args; see main

    # Training order
    train_bags = split_idx["train"].copy()
    val_bags = split_idx["val"].copy()
    test_bags = split_idx["test"].copy()

    best_val = -1.0
    best_path = out_dir / "checkpoints" / "best.pt"

    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        # shuffle bags
        np.random.shuffle(train_bags)

        losses = []
        correct = 0
        total = 0

        # per-bag update (true MIL); batch_size ignored intentionally for strictness
        for bi in train_bags:
            s = int(bags.bag_ptr[bi])
            e = int(bags.bag_ptr[bi + 1])
            Xb = torch.from_numpy(bags.X[s:e]).to(device)
            yb = torch.tensor(int(bags.y[bi]), dtype=torch.long, device=device)

            opt.zero_grad()

            if model_name == "abmil":
                logits, _w = model(Xb)
            else:
                logits = model(Xb)

            loss = F.cross_entropy(logits.unsqueeze(0), yb.unsqueeze(0))
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            pred = int(torch.argmax(logits).item())
            correct += int(pred == int(bags.y[bi]))
            total += 1

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc = float(correct / total) if total else float("nan")

        val_metrics = eval_split(model, bags, val_bags, device, model_name)
        test_metrics = eval_split(model, bags, test_bags, device, model_name)

        dt = time.time() - t0
        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "sec": dt,
        }
        history.append(row)

        logging.info(
            "ep=%d | train loss=%.4f acc=%.3f | val loss=%.4f acc=%.3f | test acc=%.3f | %.2fs",
            ep, train_loss, train_acc, val_metrics["loss"], val_metrics["acc"], test_metrics["acc"], dt
        )

        # checkpoint on val acc
        if val_metrics["acc"] >= best_val:
            best_val = val_metrics["acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "in_dim": in_dim,
                    "n_classes": n_classes,
                    "seed": seed,
                },
                best_path,
            )

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False, encoding="utf-8")

    # Final eval with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    final_val = eval_split(model, bags, val_bags, device, model_name)
    final_test = eval_split(model, bags, test_bags, device, model_name)

    summary = {
        "best_ckpt": str(best_path),
        "final_val": final_val,
        "final_test": final_test,
        "label_names": [str(x) for x in bags.label_names.tolist()],
        "n_bags": int(len(bags.bag_ids)),
        "n_cells": int(bags.X.shape[0]),
        "splits": {
            "train_bags": int(len(train_bags)),
            "val_bags": int(len(val_bags)),
            "test_bags": int(len(test_bags)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote summary.json and history.csv")
    logging.info("FINAL val acc=%.3f | test acc=%.3f", final_val["acc"], final_test["acc"])


# -------------------------
# Main
# -------------------------
def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    ap.add_argument("--bags_dir", required=True, help="runs/<dataset>/bags")
    ap.add_argument("--out", required=True, help="runs/<dataset>/train/<exp_name>")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    # Repro snapshot
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, art / "config_train.yaml")

    bags_dir = Path(args.bags_dir)
    bags_npz = bags_dir / "bags.npz"
    split_csv = bags_dir / "split_bags.csv"
    if not bags_npz.exists():
        raise FileNotFoundError(f"Missing: {bags_npz}")
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing: {split_csv}")

    bags = load_bags_npz(bags_npz)
    split_map = load_split_csv(split_csv)
    split_idx = make_split_indices(bags, split_map)

    logging.info("bags_dir=%s", str(bags_dir))
    logging.info("n_bags=%d | n_cells=%d | feat_dim=%d", len(bags.bag_ids), bags.X.shape[0], bags.X.shape[1])
    logging.info(
        "split bags: train=%d val=%d test=%d",
        len(split_idx["train"]), len(split_idx["val"]), len(split_idx["test"])
    )

    train(cfg, bags, split_idx, out_dir)


if __name__ == "__main__":
    main()
