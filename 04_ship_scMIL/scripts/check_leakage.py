# scripts/check_leakage.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scmil.leakage.leakage_core import run_leakage_check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags_dir", required=True, type=str, help="Path to bags output dir")
    ap.add_argument("--out", default=None, type=str, help="Optional report JSON path")
    ap.add_argument("--non_strict", action="store_true", help="Do not raise on failed checks (still writes report)")
    args = ap.parse_args()

    bags_dir = Path(args.bags_dir)
    out_path = Path(args.out) if args.out else None
    strict = (not args.non_strict)

    report = run_leakage_check(
        bags_dir=bags_dir,
        out_path=out_path,
        strict=strict,
    )

    # concise stdout
    print("[leakage] n_cells:", report["counts"]["n_cells"])
    print("[leakage] n_bags :", report["counts"]["n_bags"], "| bags_per_split:", report["splits"]["bags_per_split"])
    print("[leakage] n_groups:", report["counts"]["n_groups"], "| groups_per_split:", report["splits"]["groups_per_split"])
    print("[leakage] checks:", report["checks"])
    print("[leakage] bag_id mismatch(group__label):", report["details"]["bag_id_format"]["bag_id_mismatch_group__label"])

    if out_path is not None:
        print("[leakage] wrote report:", str(out_path))

    if strict:
        print("[leakage] OK")


if __name__ == "__main__":
    main()

