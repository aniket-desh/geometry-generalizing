#!/usr/bin/env bash
set -euo pipefail

repo_root="${PRIORITY_REPO_ROOT:-/workspace/geometry/repo}"
venv_python="${PRIORITY_PYTHON:-/workspace/geometry/.venv/bin/python}"
results_root="${PRIORITY_RESULTS_ROOT:-/workspace/geometry-reuse-results}"
log_root="${PRIORITY_LOG_ROOT:-/workspace/geometry-priority-logs}"
figure_root="${PRIORITY_FIGURE_ROOT:-/workspace/geometry-priority-figures}"
selection_root="${PRIORITY_SELECTION_ROOT:-/workspace/geometry-priority-selected}"
archive_root="${PRIORITY_ARCHIVE_ROOT:-/workspace/geometry-priority-stage-archives}"
shard_count="${PRIORITY_CAUSAL_SHARDS:-3}"
analysis_slots="${PRIORITY_ANALYSIS_SLOTS:-2}"
operator_workers="${PRIORITY_OPERATOR_WORKERS:-2}"
timeout_hours="${PRIORITY_TIMEOUT_HOURS:-10}"

mkdir -p "${log_root}" "${figure_root}" "${archive_root}"

start_session() {
    local name="$1"
    local command="$2"
    if tmux has-session -t "${name}" 2>/dev/null; then
        echo "${name}: already exists"
        return
    fi
    tmux new-session -d -s "${name}" \
        "bash -lc 'set -o pipefail; cd \"${repo_root}\"; ${command} 2>&1 | tee -a \"${log_root}/${name}.log\"'"
    echo "${name}: launched"
}

common="--results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --analysis-slots ${analysis_slots} --timeout-hours ${timeout_hours}"

start_session priority-fallback-operator \
    "\"${venv_python}\" geometry/priority_pipeline.py --stage operator --workers ${operator_workers} ${common}"

for shard in $(seq 0 $((shard_count - 1))); do
    start_session "priority-fallback-causal-${shard}" \
        "\"${venv_python}\" geometry/priority_pipeline.py --stage causal-shard --shard-index ${shard} --shard-count ${shard_count} ${common}"
done

start_session priority-fallback-causal-join \
    "\"${venv_python}\" geometry/priority_pipeline.py --stage causal-join --shard-count ${shard_count} ${common}"

start_session priority-fallback-finalize \
    "\"${venv_python}\" geometry/priority_pipeline.py --stage finalize --shard-count ${shard_count} ${common}"

start_session priority-fallback-pack \
    "\"${venv_python}\" geometry/pack_priority.py --results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --selection-root \"${selection_root}\" --output-root \"${archive_root}\" --timeout-hours ${timeout_hours}"
