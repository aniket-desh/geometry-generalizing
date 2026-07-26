from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from launch_sweep import BREADTH_PILOT_TASKS
from render_reuse import SERIES, _save_static, _step_axis, _style_axis, plt


TASK_LABELS = {
    "torus5": "torus 5×5",
    "xor16": "XOR 16",
    "dihedral12": "dihedral 12",
    "path16": "path 16",
    "tree15": "tree 15",
    "broken12": "broken cycle 12",
    "random31": "random 31",
    "cycle24": "cycle 24",
    "cycle31": "cycle 31",
}
LINESTYLES = ("-", "-", "-", "-", "-", "-", "-", (0, (4, 2)), (0, (2, 2)))
ACTION_METRIC_TASKS = frozenset(
    {
        "torus5",
        "xor16",
        "dihedral12",
        "broken12",
        "random31",
        "cycle24",
        "cycle31",
    }
)
GENERATOR_METRIC_TASKS = frozenset(
    {
        "torus5",
        "xor16",
        "dihedral12",
        "broken12",
        "cycle24",
        "cycle31",
    }
)


@dataclass(frozen=True)
class BreadthRun:
    task: str
    path: Path
    config: dict[str, object]
    records: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the nine seed-0 relational breadth runs, validate their "
            "60k endpoints, and render measured-checkpoint trajectories."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/workspace/geometry-breadth-results"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/geometry-breadth-figures"),
    )
    parser.add_argument("--expected-step", type=int, default=60_000)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    return parser.parse_args()


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_records(path: Path) -> list[dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or "step" not in record:
            continue
        records[int(record["step"])] = record
    return [records[step] for step in sorted(records)]


def nested_value(record: dict[str, object], key: str) -> float | None:
    value: object = record
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def discover_complete_runs(
    root: Path,
    *,
    expected_step: int,
) -> tuple[list[BreadthRun], dict[str, str]]:
    by_task: dict[str, BreadthRun] = {}
    pending = {task: "missing run directory" for task in BREADTH_PILOT_TASKS}
    for config_path in sorted(root.glob("*/config.json")):
        config = load_json(config_path)
        if not isinstance(config, dict):
            continue
        task = str(config.get("task", ""))
        if (
            task not in BREADTH_PILOT_TASKS
            or str(config.get("preset")) != "micro"
            or int(config.get("seed", -1)) != 0
        ):
            continue
        if task in by_task:
            raise ValueError(f"duplicate breadth run for {task}")
        run_dir = config_path.parent
        done = load_json(run_dir / "done.json")
        if not isinstance(done, dict):
            pending[task] = "missing done.json"
            continue
        final_step = int(done.get("final_step", 0))
        if final_step < expected_step:
            pending[task] = f"done at {final_step:,}"
            continue
        records = load_records(run_dir / "metrics.jsonl")
        if not records or int(records[-1].get("step", -1)) < expected_step:
            pending[task] = "metrics do not contain the validated endpoint"
            continue
        records = [
            record for record in records if int(record.get("step", -1)) <= expected_step
        ]
        required = ["test_accuracy"]
        if task in ACTION_METRIC_TASKS:
            required.append("node_geometry.action_defect")
        if task in GENERATOR_METRIC_TASKS:
            required.append("node_geometry.generator_error")
        if any(
            not any(nested_value(record, key) is not None for record in records)
            for key in required
        ):
            pending[task] = "required behavior or geometry metric is absent"
            continue
        by_task[task] = BreadthRun(task, run_dir, config, records)
        pending.pop(task, None)
    ordered = [by_task[task] for task in BREADTH_PILOT_TASKS if task in by_task]
    return ordered, pending


def wait_for_runs(
    root: Path,
    *,
    expected_step: int,
    poll_seconds: float,
    timeout_hours: float,
) -> list[BreadthRun]:
    deadline = time.monotonic() + timeout_hours * 3600.0
    last_message = 0.0
    while True:
        runs, pending = discover_complete_runs(root, expected_step=expected_step)
        if not pending:
            return runs
        if time.monotonic() >= deadline:
            detail = ", ".join(f"{task}: {reason}" for task, reason in pending.items())
            raise TimeoutError(f"breadth runs did not complete: {detail}")
        if time.monotonic() - last_message >= 300.0:
            detail = ", ".join(f"{task}: {reason}" for task, reason in pending.items())
            print(f"{now()} waiting for breadth runs; {detail}", flush=True)
            last_message = time.monotonic()
        time.sleep(poll_seconds)


def measured_curve(
    run: BreadthRun,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(int(record["step"]), nested_value(record, key)) for record in run.records]
    pairs = [(step, value) for step, value in pairs if value is not None]
    return (
        np.asarray([step for step, _ in pairs], dtype=float),
        np.asarray([value for _, value in pairs], dtype=float),
    )


def render_breadth(
    runs: list[BreadthRun],
    *,
    output: Path,
    expected_step: int,
) -> dict[str, object]:
    if {run.task for run in runs} != set(BREADTH_PILOT_TASKS):
        raise ValueError("rendering requires the exact nine-task breadth matrix")
    specs = (
        ("test_accuracy", "held-out accuracy", frozenset(BREADTH_PILOT_TASKS)),
        (
            "node_geometry.action_defect",
            "action defect",
            ACTION_METRIC_TASKS,
        ),
        (
            "node_geometry.generator_error",
            "generator error",
            GENERATOR_METRIC_TASKS,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), sharex=True)
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.98, bottom=0.30, wspace=0.36)
    handles: list[Line2D] = []
    for index, run in enumerate(runs):
        color = SERIES[index % len(SERIES)]
        linestyle = LINESTYLES[index]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
                label=TASK_LABELS[run.task],
            )
        )
        for axis, (key, ylabel, applicable_tasks) in zip(axes, specs):
            if run.task not in applicable_tasks:
                continue
            xs, ys = measured_curve(run, key)
            axis.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyle,
                linewidth=1.35,
                alpha=0.92,
            )
            axis.scatter(
                xs,
                ys,
                color=color,
                s=3.0,
                alpha=0.34,
                linewidths=0,
            )
            axis.set_xlabel("step")
            axis.set_ylabel(ylabel)
            axis.set_xlim(0, expected_step)
            _step_axis(axis)
            _style_axis(axis)
    for axis, (_, _, applicable_tasks) in zip(axes, specs):
        not_applicable = [
            TASK_LABELS[task]
            for task in BREADTH_PILOT_TASKS
            if task not in applicable_tasks
        ]
        if not_applicable:
            axis.text(
                0.02,
                0.02,
                "n/a: " + ", ".join(not_applicable),
                transform=axis.transAxes,
                color="#65737e",
                fontsize=6.6,
                ha="left",
                va="bottom",
            )
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        handlelength=2.2,
        columnspacing=1.25,
    )
    output.mkdir(parents=True, exist_ok=True)
    artifacts = _save_static(fig, output / "breadth-trajectories")
    manifest_path = output / "breadth-render-manifest.json"
    done_path = output / "breadth-render-done.json"
    manifest = {
        "created_at": now(),
        "expected_step": expected_step,
        "tasks": [run.task for run in runs],
        "runs": [str(run.path) for run in runs],
        "seed": 0,
        "aggregation": "none; one measured run per task",
        "checkpoint_rendering": "measured evaluation records only; no smoothing",
        "panels": [
            {
                "metric": key,
                "applicable_tasks": [
                    task
                    for task in BREADTH_PILOT_TASKS
                    if task in applicable_tasks
                ],
                "not_applicable_tasks": [
                    task
                    for task in BREADTH_PILOT_TASKS
                    if task not in applicable_tasks
                ],
            }
            for key, _, applicable_tasks in specs
        ],
        "artifacts": [
            *(str(path) for path in artifacts),
            str(manifest_path),
            str(done_path),
        ],
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        done_path,
        {
            "completed_at": now(),
            "expected_step": expected_step,
            "task_count": len(runs),
            "manifest": str(manifest_path),
        },
    )
    return manifest


def write_fixture(root: Path, expected_step: int) -> None:
    for index, task in enumerate(BREADTH_PILOT_TASKS):
        run_dir = root / f"{task}-micro-s0"
        run_dir.mkdir(parents=True)
        config = {
            "task": task,
            "preset": "micro",
            "seed": 0,
            "task_family": task.rstrip("0123456789"),
        }
        (run_dir / "config.json").write_text(json.dumps(config) + "\n")
        (run_dir / "done.json").write_text(
            json.dumps({"final_step": expected_step}) + "\n"
        )
        records = []
        for step in (0, expected_step // 2, expected_step):
            progress = step / expected_step
            records.append(
                {
                    "step": step,
                    "test_accuracy": min(1.0, progress * (1 - 0.04 * index)),
                    "node_geometry": {
                        "action_defect": (
                            1.1 - 0.7 * progress + 0.03 * index
                            if task in ACTION_METRIC_TASKS
                            else None
                        ),
                        "generator_error": (
                            1.2 - 0.8 * progress + 0.02 * index
                            if task in GENERATOR_METRIC_TASKS
                            else None
                        ),
                    },
                }
            )
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )


def run_self_test(destination: Path | None) -> None:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if destination is None:
        temporary = tempfile.TemporaryDirectory(prefix="breadth-render-")
        base = Path(temporary.name)
    else:
        base = destination
        base.mkdir(parents=True, exist_ok=True)
    fixture = base / "fixture"
    output = base / "plots"
    expected_step = 60_000
    write_fixture(fixture, expected_step)
    runs, pending = discover_complete_runs(fixture, expected_step=expected_step)
    if pending:
        raise AssertionError(f"fixture remained pending: {pending}")
    manifest = render_breadth(runs, output=output, expected_step=expected_step)
    expected = {
        "breadth-trajectories.png",
        "breadth-trajectories.pdf",
        "breadth-render-manifest.json",
        "breadth-render-done.json",
    }
    missing = sorted(name for name in expected if not (output / name).exists())
    if missing or manifest["aggregation"] != "none; one measured run per task":
        raise AssertionError(f"breadth self-test failed; missing={missing}")
    print(f"breadth self-test passed: {output}", flush=True)
    if temporary is not None:
        temporary.cleanup()


def main() -> None:
    args = parse_args()
    if args.expected_step < 1:
        raise ValueError("--expected-step must be positive")
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("poll interval and timeout must be positive")
    if args.self_test:
        run_self_test(args.self_test_output)
        return
    if args.wait:
        runs = wait_for_runs(
            args.results_root,
            expected_step=args.expected_step,
            poll_seconds=args.poll_seconds,
            timeout_hours=args.timeout_hours,
        )
    else:
        runs, pending = discover_complete_runs(
            args.results_root, expected_step=args.expected_step
        )
        if pending:
            detail = ", ".join(f"{task}: {reason}" for task, reason in pending.items())
            raise RuntimeError(f"breadth matrix is incomplete; {detail}")
    manifest = render_breadth(
        runs,
        output=args.output,
        expected_step=args.expected_step,
    )
    print(
        f"{now()} wrote {len(manifest['artifacts'])} breadth artifacts "
        f"from {len(runs)} validated runs",
        flush=True,
    )


if __name__ == "__main__":
    main()
