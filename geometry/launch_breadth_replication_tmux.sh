#!/usr/bin/env bash
set -euo pipefail

repo_root="${BREADTH_REPLICATION_REPO_ROOT:-/home/ubuntu/a/vi-activation-geometry}"
work_root="${BREADTH_REPLICATION_WORK_ROOT:-/home/ubuntu/a/vi-activation-geometry-work}"
venv_python="${BREADTH_REPLICATION_PYTHON:-/home/ubuntu/a/.venv/bin/python}"
results_root="${BREADTH_REPLICATION_RESULTS_ROOT:-${work_root}/breadth-replication-results}"
log_root="${BREADTH_REPLICATION_LOG_ROOT:-${work_root}/breadth-replication-logs}"
session_prefix="${BREADTH_REPLICATION_SESSION_PREFIX:-vi-breadth-replicate}"
workers_per_gpu="${BREADTH_REPLICATION_WORKERS_PER_GPU:-2}"
min_free_gb="${BREADTH_REPLICATION_MIN_FREE_GB:-40}"
compile="${BREADTH_REPLICATION_COMPILE:-1}"
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
if (( workers_per_gpu < 1 || workers_per_gpu > 2 )); then
    echo "BREADTH_REPLICATION_WORKERS_PER_GPU must be 1 or 2" >&2
    exit 2
fi
if ! [[ "${min_free_gb}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "BREADTH_REPLICATION_MIN_FREE_GB must be a nonnegative number" >&2
    exit 2
fi
if [[ ! "${session_prefix}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "BREADTH_REPLICATION_SESSION_PREFIX is not a safe tmux name" >&2
    exit 2
fi
case "${compile}" in
    0) compile_flag="" ;;
    1) compile_flag="--compile" ;;
    *) echo "BREADTH_REPLICATION_COMPILE must be 0 or 1" >&2; exit 2 ;;
esac
for path in \
    "${repo_root}" "${work_root}" "${venv_python}" \
    "${results_root}" "${log_root}"; do
    if [[ "${path}" == *"'"* ]]; then
        echo "paths containing single quotes are unsupported: ${path}" >&2
        exit 2
    fi
done

if (( self_test )); then
    test -x "${venv_python}"
    cd "${repo_root}"
    "${venv_python}" geometry/launch_sweep.py --self-test
    echo "restricted breadth replication launcher self-test passed"
    exit 0
fi

if (( ! dry_run )); then
    command -v tmux >/dev/null
    test -x "${venv_python}"
    test -f "${repo_root}/geometry/launch_sweep.py"
    mkdir -p "${results_root}" "${log_root}/tmux"
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
    local session_log="${log_root}/tmux/${name}.log"
    local command
    local wrapped
    validate_gpu "${gpu}"
    session_count=$((session_count + 1))
    command="\"${venv_python}\" geometry/launch_sweep.py --profile breadth-replicate --shard-index ${gpu} --shard-count ${shard_count} --workers ${workers_per_gpu} --output-root \"${results_root}\" --log-root \"${log_root}\" --min-free-gb ${min_free_gb} --device cuda ${compile_flag}"
    wrapped="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 bash -lc 'set -o pipefail; cd \"${repo_root}\"; ${command} 2>&1 | tee -a \"${session_log}\"'"
    if (( dry_run )); then
        echo "${name}: physical GPU ${gpu}, shard ${gpu}/${shard_count}"
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

for gpu in "${gpu_ids[@]}"; do
    start_session "${gpu}" "${session_prefix}-gpu${gpu}"
done

if (( session_count != 4 )); then
    echo "internal session plan is incomplete: ${session_count}" >&2
    exit 2
fi
echo "planned 4 restricted sessions for 8 runs across physical GPUs 0-3"
