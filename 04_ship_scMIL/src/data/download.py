import argparse
from pathlib import Path
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    raw_dir = Path("data/raw/gse96583")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # stub metadata (real implementation will replace)
    meta_path = raw_dir / "metadata.csv"
    if not meta_path.exists():
        with meta_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["cell_id", "donor_id", "condition", "sample_id"])
            w.writeheader()
            # minimal rows so downstream doesn't crash
            w.writerow({"cell_id":"cell_0","donor_id":"donor_0","condition":"control","sample_id":"donor_0_control"})
            w.writerow({"cell_id":"cell_1","donor_id":"donor_1","condition":"stimulated","sample_id":"donor_1_stimulated"})

    print(f"[download] wrote {meta_path}")

if __name__ == "__main__":
    main()

