import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    # stub outputs (placeholder files)
    (art / "processed.h5ad").write_bytes(b"")       # placeholder
    (art / "features.parquet").write_text("", encoding="utf-8")  # placeholder

    print(f"[preprocess] wrote {art/'processed.h5ad'} and {art/'features.parquet'}")

if __name__ == "__main__":
    main()

