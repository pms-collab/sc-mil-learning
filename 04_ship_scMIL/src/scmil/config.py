# src/scmil/config.py
from __future__ import annotations
from pathlib import Path
import yaml

def load_yaml(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return cfg

def get_seed(cfg: dict, default: int = 42) -> int:
    seed = cfg.get("seed", default)
    if not isinstance(seed, int):
        raise ValueError(f"seed must be int; got {type(seed)}")
    return seed

