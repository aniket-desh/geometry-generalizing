#!/usr/bin/env bash
set -euo pipefail

repo_root="${KEY60_REPO_ROOT:-/workspace/geometry/repo}"
venv_python="${KEY60_PYTHON:-/workspace/geometry/.venv/bin/python}"
results_root="${KEY60_RESULTS_ROOT:-/workspace/geometry-reuse-results}"
log_root="${KEY60_LOG_ROOT:-/workspace/geometry-key60-logs}"
figure_root="${KEY60_FIGURE_ROOT:-/workspace/geometry-key60-figures}"
archive_root="${KEY60_ARCHIVE_ROOT:-/workspace/geometry-key60-stage-archives}"
shard_count="${KEY60_CAUSAL_SHARDS:-6}"

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

start_session key60-train \
    "\"${venv_python}\" geometry/launch_reuse.py --profile key60 --workers 6 --steps 60000 --output-root \"${results_root}\" --log-root \"${log_root}\" --min-free-gb 8 --compile"

start_session key60-operator \
    "\"${venv_python}\" geometry/key60_pipeline.py --stage operator --workers 3 --results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --timeout-hours 12"

for shard in $(seq 0 $((shard_count - 1))); do
    start_session "key60-causal-${shard}" \
        "\"${venv_python}\" geometry/key60_pipeline.py --stage causal-shard --shard-index ${shard} --shard-count ${shard_count} --results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --timeout-hours 12"
done

start_session key60-causal-join \
    "\"${venv_python}\" geometry/key60_pipeline.py --stage causal-join --shard-count ${shard_count} --results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --timeout-hours 12"

start_session key60-finalize \
    "\"${venv_python}\" geometry/key60_pipeline.py --stage finalize --shard-count ${shard_count} --results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --timeout-hours 12"

start_session key60-pack \
    "\"${venv_python}\" geometry/pack_key60.py --results-root \"${results_root}\" --log-root \"${log_root}\" --figure-root \"${figure_root}\" --output-root \"${archive_root}\" --timeout-hours 12"
