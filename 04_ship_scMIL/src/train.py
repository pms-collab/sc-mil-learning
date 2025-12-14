import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    preds = out / "preds"
    preds.mkdir(parents=True, exist_ok=True)

    # placeholder
    (preds / "preds.parquet").write_text("", encoding="utf-8")
    print(f"[train] wrote {preds/'preds.parquet'}")

if __name__ == "__main__":
    main()

