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
- outputs/run_dev/metrics/metrics.csv exists and has column macro_f1
- outputs/run_dev/leakage_check.txt exists and contains group overlap = 0
- outputs/run_dev/logs/*.log exists

## Config contract (must match column names)
Required obs/metadata columns (exact names):
- donor_id
- condition # label: IFN-stimulated vs control
- sample_id # bag_id = donor_id + condition (or equivalent unique sample key)
Required bags.parquet columns:
- cell_id, bag_id, label, group_id, split


