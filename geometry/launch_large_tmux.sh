#!/usr/bin/env bash
set -euo pipefail

repo_root="${LARGE_REPO_ROOT:-/home/ubuntu/a/vi-activation-geometry}"
work_root="${LARGE_WORK_ROOT:-/home/ubuntu/a/vi-activation-geometry-large-work}"
venv_python="${LARGE_PYTHON:-/home/ubuntu/a/.venv/bin/python}"
results_root="${LARGE_RESULTS_ROOT:-${work_root}/results}"
log_root="${LARGE_LOG_ROOT:-${work_root}/logs}"
figure_root="${LARGE_FIGURE_ROOT:-${work_root}/figures}"
analysis_slot_root="${LARGE_ANALYSIS_SLOT_ROOT:-${work_root}/analysis-slots}"
session_prefix="${LARGE_SESSION_PREFIX:-vi-priority-large}"
analysis_slots="${LARGE_ANALYSIS_SLOTS:-4}"
operator_workers="${LARGE_OPERATOR_WORKERS:-2}"
timeout_hours="${LARGE_TIMEOUT_HOURS:-24}"
min_free_gb="${LARGE_MIN_FREE_GB:-40}"
compile="${LARGE_COMPILE:-1}"
launch_train="${LARGE_LAUNCH_TRAIN:-1}"
launch_analysis="${LARGE_LAUNCH_ANALYSIS:-0}"
gpu_ids=(0 1 2 3)
shard_count="${#gpu_ids[@]}"
dry_run=0
self_test=0

usage() {
    echo "usage: $0 [--dry-run | --self-test]"
}

for argument in "$@"; do
    case "${argument}" in
        --dry-run) dry_run=1 ;;
        --self-test) self_test=1 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if (( dry_run && self_test )); then
    echo "--dry-run and --self-test are mutually exclusive" >&2
    exit 2
fi
if [[ "${gpu_ids[*]}" != "0 1 2 3" ]]; then
    echo "internal GPU assignment is invalid" >&2
    exit 2
fi
if (( analysis_slots < 1 || analysis_slots > shard_count )); then
    echo "LARGE_ANALYSIS_SLOTS must be between 1 and ${shard_count}" >&2
    exit 2
fi
if (( operator_workers < 1 || operator_workers > shard_count )); then
    echo "LARGE_OPERATOR_WORKERS must be between 1 and ${shard_count}" >&2
    exit 2
fi
case "${compile}" in
    0) compile_flag="" ;;
    1) compile_flag="--compile" ;;
    *) echo "LARGE_COMPILE must be 0 or 1" >&2; exit 2 ;;
esac
for toggle in "${launch_train}" "${launch_analysis}"; do
    case "${toggle}" in
        0|1) ;;
        *) echo "LARGE_LAUNCH_* switches must be 0 or 1" >&2; exit 2 ;;
    esac
done
for path in \
    "${repo_root}" "${work_root}" "${venv_python}" "${results_root}" \
    "${log_root}" "${figure_root}" "${analysis_slot_root}"; do
    if [[ "${path}" == *"'"* ]]; then
        echo "paths containing single quotes are unsupported: ${path}" >&2
        exit 2
    fi
done

if (( self_test )); then
    test -x "${venv_python}"
    cd "${repo_root}"
    "${venv_python}" geometry/priority_large.py --self-test
    "${venv_python}" geometry/priority_pipeline.py --presets large --self-test
    "${venv_python}" geometry/render_priority.py --preset large --self-test
    "${venv_python}" geometry/summarize_priority_evidence.py --self-test
    echo "large launcher self-test passed"
    exit 0
fi

if (( ! dry_run )); then
    command -v tmux >/dev/null
    test -x "${venv_python}"
    test -f "${repo_root}/geometry/priority_large.py"
    mkdir -p \
        "${results_root}" "${log_root}/tmux" "${figure_root}" \
        "${analysis_slot_root}"
fi

validate_gpu() {
    case "$1" in
        0|1|2|3) ;;
        *) echo "refusing invalid GPU assignment: $1" >&2; exit 2 ;;
    esac
}

session_count=0
start_session() {
    local gpu="$1"
    local name="$2"
    local command="$3"
    local session_log="${log_root}/tmux/${name}.log"
    local wrapped
    validate_gpu "${gpu}"
    session_count=$((session_count + 1))
    wrapped="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 bash -lc 'set -o pipefail; cd \"${repo_root}\"; ${command} 2>&1 | tee -a \"${session_log}\"'"
    if (( dry_run )); then
        echo "${name}: physical GPU ${gpu}"
        echo "  ${wrapped}"
        return
    fi
    if tmux has-session -t "${name}" 2>/dev/null; then
        echo "${name}: already exists"
        return
    fi
    tmux new-session -d -s "${name}" "${wrapped}"
    echo "${name}: launched"
}

if (( launch_train )); then
    for gpu in "${gpu_ids[@]}"; do
        start_session "${gpu}" "${session_prefix}-train-gpu${gpu}" \
            "\"${venv_python}\" geometry/priority_large.py --shard-index ${gpu} --shard-count ${shard_count} --output-root \"${results_root}\" --log-root \"${log_root}\" --device cuda --min-free-gb ${min_free_gb} ${compile_flag}"
    done
fi

if (( launch_analysis )); then
    common="--presets large --results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --analysis-slots ${analysis_slots} --analysis-slot-root \"${analysis_slot_root}\" --timeout-hours ${timeout_hours} --min-free-gb ${min_free_gb} --device cuda"
    start_session 0 "${session_prefix}-operator" \
        "\"${venv_python}\" geometry/priority_pipeline.py --stage operator --workers ${operator_workers} ${common}"
    for gpu in "${gpu_ids[@]}"; do
        start_session "${gpu}" "${session_prefix}-causal-gpu${gpu}" \
            "\"${venv_python}\" geometry/priority_pipeline.py --stage causal-shard --shard-index ${gpu} --shard-count ${shard_count} ${common}"
    done
    start_session 0 "${session_prefix}-causal-join" \
        "\"${venv_python}\" geometry/priority_pipeline.py --stage causal-join --shard-count ${shard_count} ${common}"
    start_session 0 "${session_prefix}-finalize" \
        "\"${venv_python}\" geometry/priority_pipeline.py --stage finalize --shard-count ${shard_count} ${common}"
fi

expected_sessions=$((launch_train * 4 + launch_analysis * 7))
if (( session_count != expected_sessions )); then
    echo "internal session plan is incomplete: ${session_count}" >&2
    exit 2
fi
echo "planned ${session_count} large-suite sessions across ${shard_count} restricted GPUs"
