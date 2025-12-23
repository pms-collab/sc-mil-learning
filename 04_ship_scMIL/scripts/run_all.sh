#!/usr/bin/env bash
# scripts/run_all.sh
# End-to-end runner for scMIL pipeline (WSL/Linux bash)
# download -> preprocess -> build_bags -> train -> eval -> check_leakage
#
# Logs: <RunDir>/logs/01_*.log
# Force: wipe RunDir outputs (preprocess/bags/train/eval/leakage) and pass --force to steps that support it.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_all.sh --config <configs/base.yaml> --rundir <runs/.../e2e> [--python <python>] [--force]

Args:
  --python   Path to python executable (default: python in current env)
  --config   Path to YAML config (required)
  --rundir   Output run directory (required)
  --force    Remove run outputs under RunDir (preprocess/bags/train/eval/leakage) before running

Example:
  conda activate scmil
  chmod +x scripts/run_all.sh
  ./scripts/run_all.sh --config configs/base.yaml --rundir runs/gse96583_batch2/wsl_e2e --force
EOF
}

PYTHON="python"
CONFIG=""
RUNDIR=""
FORCE=0

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON="$2"; shift 2;;
    --config)
      CONFIG="$2"; shift 2;;
    --rundir)
      RUNDIR="$2"; shift 2;;
    --force)
      FORCE=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2;;
  esac
done

[[ -n "$CONFIG" ]] || { echo "Missing --config" >&2; usage; exit 2; }
[[ -n "$RUNDIR" ]] || { echo "Missing --rundir" >&2; usage; exit 2; }

# ---------- helpers ----------
assert_file() {
  local p="$1"; local msg="${2:-}"
  if [[ ! -f "$p" ]]; then
    if [[ -n "$msg" ]]; then
      echo "$msg" >&2
    fi
    echo "Missing required file: $p" >&2
    exit 1
  fi
}

ensure_dir() {
  mkdir -p "$1"
}

abspath() {
  # prefer realpath if available
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$1"
  else
    # python fallback
    "$PYTHON" - <<'PY' "$1"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
  fi
}

format_cmd() {
  # shell-escaped command string
  printf '%q ' "$@"
}

run_step() {
  local name="$1"
  local log="$2"
  shift 2
  local -a cmd=( "$@" )

  ensure_dir "$(dirname "$log")"
  rm -f "$log"

  local cmdline
  cmdline="$(format_cmd "${cmd[@]}")"

  # Always write command line first (diagnose empty-output)
  echo "[cmd] $cmdline" > "$log"

  echo "==> $name"
  echo "    $cmdline"
  echo "    log: $log"

  if ! "${cmd[@]}" 2>&1 | tee -a "$log" ; then
    echo "Step '$name' failed. See log: $log" >&2
    exit 1
  fi
}

# ---------- resolve repo root / cd ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------- resolve paths ----------
CONFIG_ABS="$(abspath "$CONFIG")"
ensure_dir "$RUNDIR"
RUNDIR_ABS="$(abspath "$RUNDIR")"

LOGDIR="$RUNDIR_ABS/logs"
ensure_dir "$LOGDIR"


# ---------- parse dataset info from YAML via python (no yq dependency) ----------
IFS=$'\t' read -r DATA_ROOT DATASET < <(
  "$PYTHON" - "$CONFIG_ABS" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
d = (cfg.get("data", {}) or {})
root = d.get("root", "data")
ds = str(d.get("dataset", "")).strip()
print(f"{root}\t{ds}")
PY
)


if [[ -z "${DATASET:-}" ]]; then
  echo "Config parse produced empty data.dataset. Fix $CONFIG_ABS" >&2
  exit 1
fi
if [[ -z "${DATA_ROOT:-}" ]]; then
  DATA_ROOT="data"
fi

# ---------- derived artifact paths ----------
RAW_DIR="$REPO_ROOT/$DATA_ROOT/raw/$DATASET"
RAW_H5AD="$RAW_DIR/raw.h5ad"
RAW_MARKER="$RAW_H5AD.ok"

PRE_OUT="$RUNDIR_ABS/preprocess"
PROC_H5AD="$PRE_OUT/artifacts/processed.h5ad"

BAGS_OUT="$RUNDIR_ABS/bags"
BAGS_NPZ="$BAGS_OUT/bags.npz"
BAGS_META="$BAGS_OUT/bags_meta.csv"
SPLIT_BAGS="$BAGS_OUT/split_bags.csv"
BAGS_MARKER="$BAGS_OUT/bags.ok"
BAGS_TABLE="$BAGS_OUT/artifacts/bags_table.csv.gz"

TRAIN_OUT="$RUNDIR_ABS/train/baseline"
BEST_CKPT="$TRAIN_OUT/checkpoints/best.pt"
TRAIN_SUMMARY="$TRAIN_OUT/summary.json"
TRAIN_HISTORY="$TRAIN_OUT/history.csv"

EVAL_OUT="$RUNDIR_ABS/eval/test"
EVAL_PRED="$EVAL_OUT/predictions.csv"
EVAL_METRICS="$EVAL_OUT/metrics.json"

LEAK_DIR="$RUNDIR_ABS/leakage"
LEAK_REPORT="$LEAK_DIR/report.json"

# ---------- force behavior ----------
if [[ "$FORCE" -eq 1 ]]; then
  echo "==> FORCE enabled: removing run outputs under RunDir (preprocess/bags/train/eval/leakage)."
  rm -rf "$PRE_OUT" "$BAGS_OUT" "$RUNDIR_ABS/train" "$RUNDIR_ABS/eval" "$LEAK_DIR"
  ensure_dir "$PRE_OUT" "$BAGS_OUT" "$TRAIN_OUT" "$EVAL_OUT" "$LEAK_DIR"
fi

# ---------- Step 01: download ----------
cmd_download=( "$PYTHON" -u -m src.data.download --config "$CONFIG_ABS" )
if [[ "$FORCE" -eq 1 ]]; then cmd_download+=( --force ); fi
run_step "01_download" "$LOGDIR/01_download.log" "${cmd_download[@]}"

assert_file "$RAW_H5AD"   "download did not produce raw.h5ad as expected."
assert_file "$RAW_MARKER" "download did not produce raw marker (.ok) as expected."

# ---------- Step 02: preprocess ----------
cmd_pre=( "$PYTHON" -u -m src.preprocess --config "$CONFIG_ABS" --out "$PRE_OUT" )
run_step "02_preprocess" "$LOGDIR/02_preprocess.log" "${cmd_pre[@]}"

assert_file "$PROC_H5AD" "preprocess did not produce artifacts/processed.h5ad."

# ---------- Step 03: build_bags ----------
cmd_bags=( "$PYTHON" -u -m src.build_bags --config "$CONFIG_ABS" --out "$BAGS_OUT" --preprocess_out "$PRE_OUT" )
if [[ "$FORCE" -eq 1 ]]; then cmd_bags+=( --force ); fi
run_step "03_build_bags" "$LOGDIR/03_build_bags.log" "${cmd_bags[@]}"

assert_file "$BAGS_NPZ"    "build_bags did not produce bags.npz."
assert_file "$BAGS_META"   "build_bags did not produce bags_meta.csv."
assert_file "$SPLIT_BAGS"  "build_bags did not produce split_bags.csv."
assert_file "$BAGS_MARKER" "build_bags did not produce bags.ok."
assert_file "$BAGS_TABLE"  "build_bags did not produce artifacts/bags_table.csv.gz."

# ---------- Step 04: train ----------
cmd_train=( "$PYTHON" -u -m src.train --config "$CONFIG_ABS" --bags_dir "$BAGS_OUT" --out "$TRAIN_OUT" )
if [[ "$FORCE" -eq 1 ]]; then cmd_train+=( --force ); fi
run_step "04_train" "$LOGDIR/04_train.log" "${cmd_train[@]}"

assert_file "$BEST_CKPT"     "train did not produce checkpoints/best.pt."
assert_file "$TRAIN_SUMMARY" "train did not produce summary.json."
assert_file "$TRAIN_HISTORY" "train did not produce history.csv."

# ---------- Step 05: eval ----------
cmd_eval=( "$PYTHON" -u -m src.eval --bags_dir "$BAGS_OUT" --ckpt "$BEST_CKPT" --out "$EVAL_OUT" --splits test )
run_step "05_eval" "$LOGDIR/05_eval.log" "${cmd_eval[@]}"

assert_file "$EVAL_PRED"    "eval did not produce predictions.csv."
assert_file "$EVAL_METRICS" "eval did not produce metrics.json."

# ---------- Step 06: check_leakage ----------
ensure_dir "$LEAK_DIR"
cmd_leak=( "$PYTHON" -u -m src.check_leakage --bags_dir "$BAGS_OUT" --out "$LEAK_REPORT" )
run_step "06_check_leakage" "$LOGDIR/06_check_leakage.log" "${cmd_leak[@]}"

assert_file "$LEAK_REPORT" "check_leakage did not produce report.json."

echo ""
echo "DONE"
echo "RunDir: $RUNDIR_ABS"
echo "Logs : $LOGDIR"
echo "Raw  : $RAW_H5AD"
echo "Proc : $PROC_H5AD"
echo "Bags : $BAGS_OUT"
echo "Ckpt : $BEST_CKPT"
echo "Eval : $EVAL_OUT"
echo "Leak : $LEAK_REPORT"
