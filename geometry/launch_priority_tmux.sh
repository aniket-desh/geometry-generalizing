#!/usr/bin/env bash
set -euo pipefail

repo_root="${PRIORITY_REPO_ROOT:-/home/ubuntu/a/vi-activation-geometry}"
work_root="${PRIORITY_WORK_ROOT:-/home/ubuntu/a/vi-activation-geometry-work}"
venv_python="${PRIORITY_PYTHON:-${repo_root}/.venv/bin/python}"
results_root="${PRIORITY_RESULTS_ROOT:-${work_root}/results}"
log_root="${PRIORITY_LOG_ROOT:-${work_root}/logs}"
figure_root="${PRIORITY_FIGURE_ROOT:-${work_root}/figures}"
selection_root="${PRIORITY_SELECTION_ROOT:-${work_root}/selected}"
archive_root="${PRIORITY_ARCHIVE_ROOT:-${work_root}/archives}"
session_prefix="${PRIORITY_SESSION_PREFIX:-vi-priority}"
analysis_slots="${PRIORITY_ANALYSIS_SLOTS:-4}"
operator_workers="${PRIORITY_OPERATOR_WORKERS:-2}"
timeout_hours="${PRIORITY_TIMEOUT_HOURS:-24}"
min_free_gb="${PRIORITY_MIN_FREE_GB:-8}"
chunk_mib="${PRIORITY_CHUNK_MIB:-42}"
upload_endpoint="${PRIORITY_UPLOAD_ENDPOINT:-https://temp.sh/upload}"
compile="${PRIORITY_COMPILE:-1}"
launch_key_train="${PRIORITY_LAUNCH_KEY_TRAIN:-1}"
launch_analysis="${PRIORITY_LAUNCH_ANALYSIS:-1}"
launch_scale="${PRIORITY_LAUNCH_SCALE:-0}"
scale_phase="${PRIORITY_SCALE_PHASE:-parallel}"
scale_results_root="${PRIORITY_SCALE_RESULTS_ROOT:-${work_root}/scale-results}"
scale_log_root="${PRIORITY_SCALE_LOG_ROOT:-${work_root}/scale-logs}"
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
    echo "PRIORITY_ANALYSIS_SLOTS must be between 1 and ${shard_count}" >&2
    exit 2
fi
if (( operator_workers < 1 || operator_workers > shard_count )); then
    echo "PRIORITY_OPERATOR_WORKERS must be between 1 and ${shard_count}" >&2
    exit 2
fi
case "${compile}" in
    0) compile_flag="" ;;
    1) compile_flag="--compile" ;;
    *) echo "PRIORITY_COMPILE must be 0 or 1" >&2; exit 2 ;;
esac
case "${scale_phase}" in
    parallel|after-key) ;;
    *) echo "PRIORITY_SCALE_PHASE must be parallel or after-key" >&2; exit 2 ;;
esac
for toggle in "${launch_key_train}" "${launch_analysis}" "${launch_scale}"; do
    case "${toggle}" in
        0|1) ;;
        *) echo "PRIORITY_LAUNCH_* switches must be 0 or 1" >&2; exit 2 ;;
    esac
done
for path in \
    "${repo_root}" "${work_root}" "${venv_python}" "${results_root}" \
    "${log_root}" "${figure_root}" "${selection_root}" "${archive_root}" \
    "${scale_results_root}" "${scale_log_root}"; do
    if [[ "${path}" == *"'"* ]]; then
        echo "paths containing single quotes are unsupported: ${path}" >&2
        exit 2
    fi
done

if (( self_test )); then
    test -x "${venv_python}"
    cd "${repo_root}"
    "${venv_python}" geometry/priority_train.py --self-test
    "${venv_python}" geometry/priority_scale.py --self-test
    "${venv_python}" geometry/priority_pipeline.py --self-test
    "${venv_python}" geometry/pack_priority.py --self-test
    echo "launcher self-test passed"
    exit 0
fi

if (( ! dry_run )); then
    command -v tmux >/dev/null
    test -x "${venv_python}"
    test -f "${repo_root}/geometry/priority_train.py"
    mkdir -p \
        "${results_root}" "${log_root}/tmux" "${figure_root}" \
        "${selection_root}" "${archive_root}"
    if (( launch_scale )); then
        mkdir -p \
            "${scale_results_root}" "${scale_log_root}/tmux"
    fi
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
    local session_log_root="${4:-${log_root}/tmux}"
    local session_log="${session_log_root}/${name}.log"
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

if (( launch_key_train )); then
    for gpu in "${gpu_ids[@]}"; do
        start_session "${gpu}" "${session_prefix}-train-gpu${gpu}" \
            "\"${venv_python}\" geometry/priority_train.py --shard-index ${gpu} --shard-count ${shard_count} --output-root \"${results_root}\" --log-root \"${log_root}\" --device cuda --min-free-gb ${min_free_gb} ${compile_flag}"
    done
fi

if (( launch_scale )); then
    scale_wait=""
    if [[ "${scale_phase}" == "after-key" ]]; then
        scale_wait="--wait-for-key-root \"${log_root}/training\""
    fi
    for gpu in "${gpu_ids[@]}"; do
        start_session "${gpu}" "${session_prefix}-scale-gpu${gpu}" \
            "\"${venv_python}\" geometry/priority_scale.py --shard-index ${gpu} --shard-count ${shard_count} --output-root \"${scale_results_root}\" --log-root \"${scale_log_root}\" --device cuda --min-free-gb ${min_free_gb} --timeout-hours ${timeout_hours} ${scale_wait} ${compile_flag}" \
            "${scale_log_root}/tmux"
    done
fi

common="--results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --analysis-slots ${analysis_slots} --timeout-hours ${timeout_hours} --min-free-gb ${min_free_gb} --device cuda"

if (( launch_analysis )); then
    start_session 0 "${session_prefix}-operator" \
        "\"${venv_python}\" geometry/priority_pipeline.py --stage operator --workers ${operator_workers} ${common}"

    for shard in "${gpu_ids[@]}"; do
        start_session "${shard}" "${session_prefix}-causal-gpu${shard}" \
            "\"${venv_python}\" geometry/priority_pipeline.py --stage causal-shard --shard-index ${shard} --shard-count ${shard_count} ${common}"
    done

    start_session 0 "${session_prefix}-causal-join" \
        "\"${venv_python}\" geometry/priority_pipeline.py --stage causal-join --shard-count ${shard_count} ${common}"

    start_session 0 "${session_prefix}-finalize" \
        "\"${venv_python}\" geometry/priority_pipeline.py --stage finalize --shard-count ${shard_count} ${common}"

    start_session 0 "${session_prefix}-pack" \
        "\"${venv_python}\" geometry/pack_priority.py --results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --selection-root \"${selection_root}\" --output-root \"${archive_root}\" --timeout-hours ${timeout_hours} --min-free-gib ${min_free_gb} --chunk-mib ${chunk_mib} --upload-endpoint \"${upload_endpoint}\""
fi

expected_sessions=$((launch_key_train * 4 + launch_analysis * 8 + launch_scale * 4))
if (( session_count != expected_sessions )); then
    echo "internal session plan is incomplete: ${session_count}" >&2
    exit 2
fi
echo "planned ${session_count} restricted sessions across ${shard_count} GPUs"
