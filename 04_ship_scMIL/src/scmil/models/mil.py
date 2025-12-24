# src/scmil/models/mil.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn

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
        return self.head(z).squeeze(0)

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

def build_model(
    model_name: str,
    *,
    in_dim: int,
    n_classes: int,
    hidden: int,
    dropout: float,
    attn_dim: int,
) -> nn.Module:
    m = model_name.lower()
    if m == "meanpool":
        return MeanPoolMIL(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)
    if m == "abmil":
        return ABMIL(in_dim=in_dim, n_classes=n_classes, hidden=hidden, attn=attn_dim, dropout=dropout)
    raise ValueError(f"Unknown model_name: {model_name}")

