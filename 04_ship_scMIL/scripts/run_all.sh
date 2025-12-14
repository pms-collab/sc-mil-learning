#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${1:-configs/base.yaml}"
OUT="${2:-outputs/run_dev}"

echo "[run_all] root=${ROOT_DIR}"
echo "[run_all] config=${CONFIG}"
echo "[run_all] out=${OUT}"

mkdir -p "${OUT}/logs" "${OUT}/metrics" "${OUT}/artifacts" "${OUT}/preds"

python -u -m src.data.download   --config "${CONFIG}" --out "${OUT}" > "${OUT}/logs/01_download.log" 2>&1
python -u -m src.preprocess      --config "${CONFIG}" --out "${OUT}" > "${OUT}/logs/02_preprocess.log" 2>&1
python -u -m src.build_bags      --config "${CONFIG}" --out "${OUT}" > "${OUT}/logs/03_build_bags.log" 2>&1
python -u -m src.train           --config "${CONFIG}" --out "${OUT}" > "${OUT}/logs/04_train.log" 2>&1
python -u -m src.eval            --config "${CONFIG}" --out "${OUT}" > "${OUT}/logs/05_eval.log" 2>&1
python -u -m src.check_leakage   --config "${CONFIG}" --out "${OUT}" > "${OUT}/leakage_check.txt" 2>&1

echo "[run_all] DONE"
