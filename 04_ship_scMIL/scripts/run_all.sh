#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${1:-configs/base.yaml}"
RUN_ROOT="${2:-runs/gse96583_batch2/e2e_dev}"
FORCE="${3:-}"

mkdir -p "${RUN_ROOT}/logs"

PRE_OUT="${RUN_ROOT}/preprocess"
BAGS_OUT="${RUN_ROOT}/bags"
TRAIN_OUT="${RUN_ROOT}/train/meanpool_v1"
EVAL_OUT="${RUN_ROOT}/eval/test_only"

force_arg=()
if [[ "${FORCE}" == "force" ]]; then
  force_arg+=(--force)
fi

echo "[run_all] root=${ROOT_DIR}"
echo "[run_all] config=${CONFIG}"
echo "[run_all] run_root=${RUN_ROOT}"

python -u -m src.data.download --config "${CONFIG}" "${force_arg[@]}" > "${RUN_ROOT}/logs/01_download.log" 2>&1
python -u -m src.preprocess   --config "${CONFIG}" --out "${PRE_OUT}" > "${RUN_ROOT}/logs/02_preprocess.log" 2>&1
python -u -m src.build_bags   --config "${CONFIG}" --out "${BAGS_OUT}" --preprocess_out "${PRE_OUT}" "${force_arg[@]}" > "${RUN_ROOT}/logs/03_build_bags.log" 2>&1
python -u -m src.check_leakage --bags_dir "${BAGS_OUT}" --out "${BAGS_OUT}/artifacts/leakage_report.json" > "${RUN_ROOT}/logs/04_check_leakage.log" 2>&1
python -u -m src.train        --config "${CONFIG}" --bags_dir "${BAGS_OUT}" --out "${TRAIN_OUT}" > "${RUN_ROOT}/logs/05_train.log" 2>&1
python -u -m src.eval         --bags_dir "${BAGS_OUT}" --ckpt "${TRAIN_OUT}/checkpoints/best.pt" --out "${EVAL_OUT}" --splits test > "${RUN_ROOT}/logs/06_eval.log" 2>&1

echo "[run_all] DONE: ${RUN_ROOT}"
