import argparse
from pathlib import Path
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    met = out / "metrics"
    met.mkdir(parents=True, exist_ok=True)

    metrics_csv = met / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["macro_f1"])
        w.writeheader()
        w.writerow({"macro_f1":"0.0000"})  # stub value

    per_class = met / "per_class.csv"
    with per_class.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class","f1"])
        w.writeheader()
        w.writerow({"class":"control","f1":"0.0000"})
        w.writerow({"class":"stimulated","f1":"0.0000"})

    print(f"[eval] wrote {metrics_csv} and {per_class}")

if __name__ == "__main__":
    main()

