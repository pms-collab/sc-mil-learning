# 04_ship_scMIL

## Goal
Reproducible scRNA-MIL pipeline with:
- one-command run
- baseline metrics (macro_f1)
- leakage-proof split (donor-level)

## Run (one command)
```bash
# positional args: <CONFIG> <OUT>
bash scripts/run_all.sh configs/base.yaml outputs/run_dev
```

## Outputs (DoD)
After the command finishes with exit code 0:
- `outputs/run_dev/metrics/metrics.csv` exists and has column `macro_f1`
- `outputs/run_dev/leakage_check.txt` exists and contains `group overlap = 0`
- `outputs/run_dev/logs/*.log` exists

## Config contract (must match column names)
Required obs/metadata columns (exact names):
- donor_id
- condition # label: IFN-stimulated vs control
- sample_id # bag_id = donor_id + condition (or equivalent unique sample key)
Required bags.parquet columns:
- cell_id, bag_id, label, group_id, split

## Pipeline steps (executed by `scripts/run_all.sh`)
1) Download/cache dataset (GSE96583)
2) Preprocess: QC → normalize/log1p → HVG → PCA → save features
3) Build bags + group split (`group_id = donor_id`)
4) Train MIL (`meanpool` baseline, then `abmil`)
5) Evaluate → metrics + per-class
6) Leakage check → write `leakage_check.txt` (FAIL => non-zero exit)

## Output layout (expected files)
After a successful run, the following paths must exist:
- `outputs/run_dev/logs/01_download.log`
- `outputs/run_dev/logs/02_preprocess.log`
- `outputs/run_dev/logs/03_build_bags.log`
- `outputs/run_dev/logs/04_train.log`
- `outputs/run_dev/logs/05_eval.log`
- `outputs/run_dev/artifacts/processed.h5ad`
- `outputs/run_dev/artifacts/features.parquet` (or `features.npy`)
- `outputs/run_dev/artifacts/bags.parquet`
- `outputs/run_dev/preds/preds.parquet`
- `outputs/run_dev/metrics/metrics.csv`
- `outputs/run_dev/metrics/per_class.csv` (optional but recommended)
- `outputs/run_dev/leakage_check.txt`

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Currenet state (checkpoint DoD)
- `scripts/run_all.sh` exists
- `configs/base.yaml` exists
- running `bash scripts/run_all.sh configs/base.yaml outputs/run_dev` creates:
  - `outputs/run_dev/metrics/metrics.csv` with column `macro_f1`
  - `outputs/run_dev/leakage_check.txt` containing `group overlap = 0`
  - `outputs/run_dev/logs/*.log`



### Troubleshooting
```md
## Troubleshooting (paste if blocked)
```text
- commit hash:
- command you ran:
- OS (Windows/WSL/Linux):
- python -V:
- pip freeze (first 30 lines):
- last 200 lines of the failing log:
- tree -a -L 4 outputs/run_dev:
- contents of outputs/run_dev/leakage_check.txt:
```

## Data access
- Raw/cache data directory: `data/` (gitignored)
- Expected structure (created by downloader):
  - `data/raw/gse96583/`
  - `data/raw/gse96583/metadata.csv` (must include `cell_id, donor_id, condition, sample_id`)

## Model selection
Set in `configs/base.yaml`:
- `mil.model: meanpool` (baseline, first milestone)
- `mil.model: abmil` (submission model)

## Current state (checkpoint DoD)
- [ ] `scripts/run_all.sh` exists
- [ ] `configs/base.yaml` exists
- [ ] Running the command creates DoD outputs:
```bash
bash scripts/run_all.sh configs/base.yaml outputs/run_dev
```
- `outputs/run_dev/metrics/metrics.csv` has column `macro_f1`
- `outputs/run_dev/leakage_check.txt` contains `group overlap = 0`
- `outputs/run_dev/logs/` contains `*.log`
- 


