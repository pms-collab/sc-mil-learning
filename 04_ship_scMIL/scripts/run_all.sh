#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
OUT="${2:-outputs/run_dev}"

echo "[run_all] config=${CONFIG}"
echo "[run_all] out=${OUT}"

mkdir -p "${OUT}/logs" "${OUT}/metrics" "${OUT}/artifacts" "${OUT}/preds"

python -m src.data.download --config "${CONFIG}" --out "${OUT}" | tee "${OUT}/logs/01_download.log"
python -m src.preprocess   --config "${CONFIG}" --out "${OUT}" | tee "${OUT}/logs/02_preprocess.log"
python -m src.build_bags   --config "${CONFIG}" --out "${OUT}" | tee "${OUT}/logs/03_build_bags.log"
python -m src.train        --config "${CONFIG}" --out "${OUT}" | tee "${OUT}/logs/04_train.log"
python -m src.eval         --config "${CONFIG}" --out "${OUT}" | tee "${OUT}/logs/05_eval.log"
python -m src.check_leakage --config "${CONFIG}" --out "${OUT}" > "${OUT}/leakage_check.txt"

echo "[run_all] DONE"
ls -R "${OUT}" | head -n 200
