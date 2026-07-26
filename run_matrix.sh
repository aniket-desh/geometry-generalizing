#!/usr/bin/env bash
set -uo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/workspace/ohope}"
CODE_DIR="${CODE_DIR:-$EXPERIMENT_ROOT/code}"
RESULTS_DIR="${RESULTS_DIR:-$EXPERIMENT_ROOT/results}"
CACHE_DIR="${CACHE_DIR:-$EXPERIMENT_ROOT/hf-cache}"
PYTHON_BIN="${PYTHON_BIN:-$EXPERIMENT_ROOT/.venv/bin/python}"

export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export MPLCONFIGDIR="$EXPERIMENT_ROOT/matplotlib"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$RESULTS_DIR" "$CACHE_DIR" "$MPLCONFIGDIR"
cd "$CODE_DIR" || exit 1

models=(
  gpt2
  pythia-160m
  gpt2-medium
  pythia-410m
  gpt2-large
  pythia-1b
)

failures=()
for model in "${models[@]}"; do
  echo
  echo "===== $model: $(date -Is) ====="
  if ! "$PYTHON_BIN" -u run_model.py \
    --model "$model" \
    --output-root "$RESULTS_DIR" \
    --cache-dir "$CACHE_DIR"; then
    failures+=("$model")
    echo "FAILED: $model"
  fi
done

"$PYTHON_BIN" -u render_aggregate.py --results "$RESULTS_DIR" || true
"$PYTHON_BIN" -u summarize_results.py --results "$RESULTS_DIR" || true

if ((${#failures[@]})); then
  echo "failed models: ${failures[*]}"
  exit 1
fi
echo "matrix complete: $(date -Is)"
