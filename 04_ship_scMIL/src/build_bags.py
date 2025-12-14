import argparse
from pathlib import Path
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    bags_path = art / "bags.parquet"  # stub: file exists; later replace with real parquet
    with bags_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cell_id","bag_id","label","group_id","split"])
        w.writeheader()
        w.writerow({"cell_id":"cell_0","bag_id":"donor_0_control","label":"control","group_id":"donor_0","split":"train"})
        w.writerow({"cell_id":"cell_1","bag_id":"donor_1_stimulated","label":"stimulated","group_id":"donor_1","split":"test"})

    print(f"[build_bags] wrote {bags_path}")

if __name__ == "__main__":
    main()

