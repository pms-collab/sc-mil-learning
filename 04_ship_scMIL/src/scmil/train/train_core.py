# src/scmil/train/core.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scmil.models.mil import build_model
from scmil.utils.seed import set_all_seeds

# ---- Bags I/O (필요하면 별도 모듈로 분리해도 됨) ----
@dataclass(frozen=True)
class BagsData:
    X: np.ndarray
    bag_ptr: np.ndarray
    bag_ids: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    label_names: np.ndarray

from scmil.bags.io import load_bags_npz, load_split_csv, make_split_indices

@torch.no_grad()
def _eval_split(
    model: nn.Module,
    model_name: str,
    bags: BagsData,
    bag_indices: np.ndarray,
    device: torch.device,
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

@dataclass(frozen=True)
class TrainParams:
    model: str
    epochs: int
    lr: float
    hidden: int
    dropout: float
    attn_dim: int

def _get_train_params(cfg: dict) -> TrainParams:
    mil = cfg.get("mil", {}) or {}
    return TrainParams(
        model=str(mil.get("model", "meanpool")).lower(),
        epochs=int(mil.get("epochs", 5)),
        lr=float(mil.get("lr", 1e-3)),
        hidden=int(mil.get("hidden", 128)),
        dropout=float(mil.get("dropout", 0.1)),
        attn_dim=int(mil.get("attn_dim", 128)),
    )

def run_train(cfg: dict, paths, *, out_dir: Path | None = None, force: bool = False) -> dict:
    """
    Train baseline MIL model using bags.npz + split_bags.csv.
    - No argparse, no logging.basicConfig, no config snapshot.
    - Uses `paths` to locate inputs and default output dir.
    """
    seed = int(cfg.get("seed", 42))
    set_all_seeds(seed)

    out_dir = out_dir or paths.train_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pt"

    history_csv = out_dir / "history.csv"
    summary_json = out_dir / "summary.json"

    # idempotent skip
    if (best_ckpt.exists() and history_csv.exists() and summary_json.exists()) and not force:
        logging.info("Skip train: outputs exist (use --force to retrain). out=%s", str(out_dir))
        return json.loads(summary_json.read_text(encoding="utf-8"))

    # inputs
    bags_npz = paths.bags_npz
    split_csv = paths.split_bags
    if not bags_npz.exists():
        raise FileNotFoundError(f"Missing: {bags_npz}")
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing: {split_csv}")

    bags = load_bags_npz(bags_npz)
    split_map = load_split_csv(split_csv)
    split_idx = make_split_indices(bags, split_map)

    tp = _get_train_params(cfg)

    n_classes = int(len(bags.label_names))
    in_dim = int(bags.X.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device=%s", device)
    logging.info("model=%s | in_dim=%d | n_classes=%d", tp.model, in_dim, n_classes)

    model = build_model(
        tp.model,
        in_dim=in_dim,
        n_classes=n_classes,
        hidden=tp.hidden,
        dropout=tp.dropout,
        attn_dim=tp.attn_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=tp.lr)

    train_bags = split_idx["train"].copy()
    val_bags = split_idx["val"].copy()
    test_bags = split_idx["test"].copy()

    best_val = -1.0
    history = []

    for ep in range(1, tp.epochs + 1):
        t0 = time.time()
        model.train()
        np.random.shuffle(train_bags)

        losses = []
        correct = 0
        total = 0

        for bi in train_bags:
            s = int(bags.bag_ptr[bi])
            e = int(bags.bag_ptr[bi + 1])
            Xb = torch.from_numpy(bags.X[s:e]).to(device)
            yb = torch.tensor(int(bags.y[bi]), dtype=torch.long, device=device)

            opt.zero_grad()
            if tp.model == "abmil":
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

        val_metrics = _eval_split(model, tp.model, bags, val_bags, device)
        test_metrics = _eval_split(model, tp.model, bags, test_bags, device)

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

        if val_metrics["acc"] >= best_val:
            best_val = val_metrics["acc"]
            torch.save(
                {
                    # weights
                    "state_dict": model.state_dict(),
                    # model reconstruction
                    "model_name": tp.model,
                    "in_dim": in_dim,
                    "n_classes": n_classes,
                    "hidden": tp.hidden,
                    "dropout": tp.dropout,
                    "attn_dim": tp.attn_dim,
                    # reproducibility
                    "seed": seed,
                    "label_names": [str(x) for x in bags.label_names.tolist()],
                },
                best_ckpt,
            )

    # write history
    pd.DataFrame(history).to_csv(history_csv, index=False, encoding="utf-8")

    # final eval with best checkpoint
    ckpt = torch.load(best_ckpt, map_location=device)
    model2 = build_model(
        ckpt["model_name"],
        in_dim=int(ckpt["in_dim"]),
        n_classes=int(ckpt["n_classes"]),
        hidden=int(ckpt["hidden"]),
        dropout=float(ckpt["dropout"]),
        attn_dim=int(ckpt["attn_dim"]),
    ).to(device)
    model2.load_state_dict(ckpt["state_dict"])
    model2.eval()

    final_val = _eval_split(model2, ckpt["model_name"], bags, val_bags, device)
    final_test = _eval_split(model2, ckpt["model_name"], bags, test_bags, device)

    summary = {
        "best_ckpt": str(best_ckpt),
        "model": {
            "model_name": ckpt["model_name"],
            "in_dim": int(ckpt["in_dim"]),
            "n_classes": int(ckpt["n_classes"]),
            "hidden": int(ckpt["hidden"]),
            "dropout": float(ckpt["dropout"]),
            "attn_dim": int(ckpt["attn_dim"]),
        },
        "seed": int(ckpt["seed"]),
        "final_val": final_val,
        "final_test": final_test,
        "label_names": ckpt["label_names"],
        "counts": {
            "n_bags": int(len(bags.bag_ids)),
            "n_cells": int(bags.X.shape[0]),
        },
        "splits": {
            "train_bags": int(len(train_bags)),
            "val_bags": int(len(val_bags)),
            "test_bags": int(len(test_bags)),
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote %s and %s", str(summary_json), str(history_csv))
    logging.info("FINAL val acc=%.3f | test acc=%.3f", final_val["acc"], final_test["acc"])
    return summary

