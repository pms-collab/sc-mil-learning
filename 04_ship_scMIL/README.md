# 04_ship_scMIL

End-to-end scRNA-seq MIL pipeline for **GSE96583 batch2** (PBMC, 8 donors, ctrl/stim).
Steps: download → build `raw.h5ad` → preprocess → build MIL bags (group-split by donor) → train baseline → eval → leakage check.

## Quickstart (WSL/Ubuntu)

### Get the code
```bash
# Option A) Clone the repo
git clone https://github.com/pms-collab/sc-mil-learning.git
cd sc-mil-learning/04_ship_scMIL

# Option B) If you already cloned the mono-repo
cd sc-mil-learning/04_ship_scMIL
```

### Environment (conda recommended)
```bash
conda env create -f environment.yml
conda activate scmil
```

### Run (one command)
From the project root('04_ship_scMIL'):
```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh --config configs/base.yaml --rundir runs/gse96583_batch2/wsl_e2e --force
```

## Output (expected)
After a successful run:
- Raw AnnData: data/raw/gse96583_batch2/raw.h5ad
- Processed AnnData: <RunDir>/preprocess/artifacts/processed.h5ad
- Bags: <RunDir>/bags/
  - bags.npz, bags_meta.csv, split_bags.csv, bags.ok
- Checkpoint: <RunDir>/train/baseline/checkpoints/best.pt
- Eval: <RunDir>/eval/test/
  - predictions.csv, metrics.json
- Leakage report: <RunDir>/leakage/report.json
- Logs: <RunDir>/logs/*.log

## Config contract
Required adata.obs columns (exact names):
- donor_id (used as group_id)
- condition (label, e.g. ctrl/stim)
- sample_id (bag identifier; must be unique per donor×condition or equivalent)

## Notes
- scikit-misc is only needed if you want Scanpy HVG selection with flavor="seurat_v3"; otherwise the code falls back.
- Baseline metrics can look artificially high because the dataset is small at the bag level (16 bags total in this setup). Do not over-interpret.
