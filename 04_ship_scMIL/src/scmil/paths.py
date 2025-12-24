# src/scmil/paths.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_root: Path
    dataset: str
    run_dir: Path

    # ---- data ----
    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw" / self.dataset
    @property
    def raw_h5ad(self) -> Path:
        return self.raw_dir / "raw.h5ad"
    @property
    def raw_marker(self) -> Path:
        return self.raw_h5ad.with_suffix(self.raw_h5ad.suffix + ".ok")

    # ---- preprocess ----
    @property
    def preprocess_dir(self) -> Path:
        return self.run_dir / "preprocess"
    @property
    def preprocess_art(self) -> Path:
        return self.preprocess_dir / "artifacts"
    @property
    def processed_h5ad(self) -> Path:
        return self.preprocess_art / "processed.h5ad"

    # ---- bags ----
    @property
    def bags_dir(self) -> Path:
        return self.run_dir / "bags"
    @property
    def bags_art(self) -> Path:
        return self.bags_dir / "artifacts"
    @property
    def bags_npz(self) -> Path:
        return self.bags_dir / "bags.npz"
    @property
    def bags_meta(self) -> Path:
        return self.bags_dir / "bags_meta.csv"
    @property
    def split_bags(self) -> Path:
        return self.bags_dir / "split_bags.csv"
    @property
    def bags_table(self) -> Path:
        return self.bags_art / "bags_table.csv.gz"
    @property
    def bags_marker(self) -> Path:
        return self.bags_dir / "bags.ok"

    # ---- train/eval/leakage ----
    @property
    def train_dir(self) -> Path:
        return self.run_dir / "train" / "baseline"
    @property
    def best_ckpt(self) -> Path:
        return self.train_dir / "checkpoints" / "best.pt"
    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval" / "test"
    @property
    def leak_dir(self) -> Path:
        return self.run_dir / "leakage"
    @property
    def leak_report(self) -> Path:
        return self.leak_dir / "report.json"

def resolve_paths(cfg: dict, *, repo_root: Path, run_dir: Path) -> Paths:
    data_root = Path(cfg.get("data", {}).get("root", "data"))
    dataset = str(cfg.get("data", {}).get("dataset", "")).strip()
    if not dataset:
        raise ValueError("cfg.data.dataset is empty")
    return Paths(
        repo_root=repo_root.resolve(),
        data_root=(repo_root / data_root).resolve(),
        dataset=dataset,
        run_dir=run_dir.resolve(),
    )

