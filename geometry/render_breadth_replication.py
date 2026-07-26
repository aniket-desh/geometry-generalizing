from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from geogen.tasks import make_task
from launch_sweep import BREADTH_REPLICATION_TASKS
from render_reuse import NORD, SERIES, _save_static, _step_axis, _style_axis, plt


TASKS = ("torus5", "cycle31", "dihedral12", "random31")
if set(TASKS) != set(BREADTH_REPLICATION_TASKS):
    raise RuntimeError("renderer tasks disagree with the replication launcher")
SEEDS = (0, 1, 2)
EXPECTED_STEP = 60_000
TASK_LABELS = {
    "torus5": "torus 5×5",
    "cycle31": "cycle 31",
    "dihedral12": "dihedral 12",
    "random31": "random 31",
}
TASK_FAMILIES = {
    "torus5": "torus",
    "cycle31": "cycle",
    "dihedral12": "dihedral",
    "random31": "random_permutation",
}
TASK_COLORS = dict(zip(TASKS, SERIES))
MODEL = {"width": 128, "depth": 2, "heads": 4, "mlp_ratio": 4}
SEMANTIC_PROTOCOL = {
    "preset": "micro",
    "corruption": 0.0,
    "train_fraction": 0.4,
    "batch_size": 4_096,
    "learning_rate": 1e-3,
    "weight_decay": 1.0,
    "aliases": 4,
    "contexts": 16,
    "steps": EXPECTED_STEP,
    "model": MODEL,
}
CADENCE = {
    0: {
        "eval_every": 250,
        "snapshot_every": 500,
        "checkpoint_every": 3_000,
        "keep_checkpoints": 2,
        "dense_checkpoint_every": 500,
    },
    1: {
        "eval_every": 500,
        "snapshot_every": 1_000,
        "checkpoint_every": 15_000,
        "keep_checkpoints": 2,
        "dense_checkpoint_every": 1_000,
    },
    2: {
        "eval_every": 500,
        "snapshot_every": 1_000,
        "checkpoint_every": 15_000,
        "keep_checkpoints": 2,
        "dense_checkpoint_every": 1_000,
    },
}
REQUIRED_FILES = (
    "config.json",
    "done.json",
    "metrics.jsonl",
    "operation_table.npy",
)


class EvidenceError(ValueError):
    pass


class IncompleteMatrix(EvidenceError):
    pass


@dataclass(frozen=True)
class Run:
    task: str
    seed: int
    source: str
    config: dict[str, object]
    records: list[dict[str, object]]
    table_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the preserved seed-0 breadth archive with the exact "
            "seed-1/2 replication and render validated behavior trajectories."
        )
    )
    parser.add_argument("--seed0-archive", type=Path)
    parser.add_argument("--replication-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    return parser.parse_args()


def now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def json_object(data: bytes, *, source: str) -> dict[str, object]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{source} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"{source} is not a JSON object")
    return payload


def metric_records(data: bytes, *, source: str) -> list[dict[str, object]]:
    by_step: dict[int, dict[str, object]] = {}
    try:
        lines = data.decode().splitlines()
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{source} is not UTF-8 JSONL") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            step = int(record["step"])
            accuracy = float(record["test_accuracy"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise EvidenceError(
                f"{source}:{line_number} has no valid step/test_accuracy"
            ) from error
        if not isinstance(record, dict):
            raise EvidenceError(f"{source}:{line_number} is not a JSON object")
        if step < 0 or step > EXPECTED_STEP:
            raise EvidenceError(f"{source}:{line_number} has step {step}")
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise EvidenceError(
                f"{source}:{line_number} has invalid test_accuracy"
            )
        by_step[step] = record
    records = [by_step[step] for step in sorted(by_step)]
    if not records or int(records[-1]["step"]) != EXPECTED_STEP:
        raise IncompleteMatrix(f"{source} has no {EXPECTED_STEP:,}-step endpoint")
    return records


def load_table(data: bytes, *, source: str) -> np.ndarray:
    try:
        table = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise EvidenceError(f"{source} is not a valid NumPy array") from error
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        raise EvidenceError(f"{source} is not a square operation table")
    return np.asarray(table, dtype=np.int64)


def archive_seed0_payloads(archive: Path) -> dict[str, dict[str, bytes]]:
    if not archive.is_file():
        raise IncompleteMatrix(f"missing seed-0 archive: {archive}")
    patterns = {
        task: re.compile(rf"^{re.escape(task)}-micro-s0-[0-9a-f]{{8}}$")
        for task in TASKS
    }
    payloads: dict[str, dict[str, bytes]] = {}
    try:
        with tarfile.open(archive, mode="r|*") as handle:
            for member in handle:
                if not member.isfile():
                    continue
                parts = Path(member.name).parts
                if len(parts) < 2 or parts[-1] not in REQUIRED_FILES:
                    continue
                task = next(
                    (
                        candidate
                        for candidate, pattern in patterns.items()
                        if pattern.fullmatch(parts[-2])
                    ),
                    None,
                )
                if task is None:
                    continue
                run_payloads = payloads.setdefault(task, {})
                filename = parts[-1]
                stored_run_name = run_payloads.setdefault(
                    "__run_name__",
                    parts[-2].encode(),
                ).decode()
                if stored_run_name != parts[-2]:
                    raise EvidenceError(
                        f"multiple seed-0 run directories for {task} in {archive}"
                    )
                if filename in run_payloads:
                    raise EvidenceError(
                        f"duplicate seed-0 {filename} for {task} in {archive}"
                    )
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise EvidenceError(f"cannot read {member.name} from {archive}")
                run_payloads[filename] = extracted.read()
    except (tarfile.TarError, OSError) as error:
        raise EvidenceError(f"cannot read seed-0 archive {archive}") from error
    missing = {
        task: sorted(set(REQUIRED_FILES) - set(payloads.get(task, {})))
        for task in TASKS
        if set(REQUIRED_FILES) - set(payloads.get(task, {}))
    }
    if missing:
        raise IncompleteMatrix(f"seed-0 archive is incomplete: {missing}")
    return payloads


def replication_payloads(root: Path) -> dict[tuple[str, int], dict[str, bytes]]:
    if not root.is_dir():
        raise IncompleteMatrix(f"missing replication root: {root}")
    payloads: dict[tuple[str, int], dict[str, bytes]] = {}
    for config_path in sorted(root.glob("*/config.json")):
        try:
            config_data = config_path.read_bytes()
            config = json_object(config_data, source=str(config_path))
            task = str(config.get("task", ""))
            preset = str(config.get("preset", ""))
            seed = int(config.get("seed", -1))
        except (OSError, TypeError, ValueError) as error:
            raise EvidenceError(f"cannot inspect {config_path}") from error
        if task not in TASKS:
            continue
        if preset != "micro" or seed not in (1, 2):
            raise EvidenceError(
                f"unexpected restricted-breadth identity in {config_path.parent}"
            )
        if str(config.get("run_name", "")) != config_path.parent.name:
            raise EvidenceError(
                f"run_name does not match directory {config_path.parent}"
            )
        identity = (task, seed)
        if identity in payloads:
            raise EvidenceError(f"duplicate restricted-breadth identity {identity}")
        files = {"config.json": config_data}
        for filename in REQUIRED_FILES[1:]:
            path = config_path.parent / filename
            try:
                files[filename] = path.read_bytes()
            except FileNotFoundError as error:
                raise IncompleteMatrix(f"missing {path}") from error
            except OSError as error:
                raise EvidenceError(f"cannot read {path}") from error
        payloads[identity] = files
    expected = {(task, seed) for task in TASKS for seed in (1, 2)}
    missing = sorted(expected - set(payloads))
    if missing:
        raise IncompleteMatrix(f"replication matrix is missing {missing}")
    return payloads


def protocol_payload(config: dict[str, object]) -> dict[str, object]:
    return {
        key: config.get(key)
        for key in SEMANTIC_PROTOCOL
        if key != "model"
    } | {"model": config.get("model")}


def validate_run(
    payloads: dict[str, bytes],
    *,
    task: str,
    seed: int,
    source: str,
) -> Run:
    config = json_object(payloads["config.json"], source=f"{source}/config.json")
    done = json_object(payloads["done.json"], source=f"{source}/done.json")
    records = metric_records(
        payloads["metrics.jsonl"],
        source=f"{source}/metrics.jsonl",
    )
    table = load_table(
        payloads["operation_table.npy"],
        source=f"{source}/operation_table.npy",
    )
    expected_task = make_task(task, seed=seed)
    expected_table = np.asarray(expected_task.table, dtype=np.int64)
    observed_sha = hashlib.sha256(table.tobytes()).hexdigest()
    expected_sha = hashlib.sha256(expected_table.tobytes()).hexdigest()
    try:
        checks = {
            "task": str(config["task"]) == task,
            "preset": str(config["preset"]) == "micro",
            "seed": int(config["seed"]) == seed,
            "split_seed": int(config["split_seed"]) == seed,
            "task_seed": int(config["task_seed"]) == seed,
            "token_seed": int(config["token_seed"]) == seed,
            "run_name": (
                str(config["run_name"])
                == source.rsplit("!", 1)[-1].rsplit("/", 1)[-1]
            ),
            "task_family": str(config["task_family"]) == TASK_FAMILIES[task],
            "task_order": int(config["task_order"]) == expected_task.order,
            "table_config_hash": str(config["task_table_sha256"]) == observed_sha,
            "table_generator_hash": observed_sha == expected_sha,
            "table_contents": np.array_equal(table, expected_table),
            "done_run_name": str(done["run_name"]) == str(config["run_name"]),
            "done_step": int(done["final_step"]) == EXPECTED_STEP,
        }
        for key, expected in SEMANTIC_PROTOCOL.items():
            observed = config.get(key)
            checks[f"protocol_{key}"] = (
                observed == expected
                if not isinstance(expected, float)
                else abs(float(observed) - expected) < 1e-12
            )
        for key, expected in CADENCE[seed].items():
            checks[f"cadence_{key}"] = int(config[key]) == expected
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"{source} has an invalid config or done marker") from error
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise EvidenceError(f"{source} fails validation: {', '.join(failed)}")
    return Run(task, seed, source, config, records, observed_sha)


def load_matrix(seed0_archive: Path, replication_root: Path) -> list[Run]:
    seed0 = archive_seed0_payloads(seed0_archive)
    replication = replication_payloads(replication_root)
    runs = [
        validate_run(
            seed0[task],
            task=task,
            seed=0,
            source=(
                f"{seed0_archive}!"
                f"{seed0[task]['__run_name__'].decode()}"
            ),
        )
        for task in TASKS
    ]
    runs.extend(
        validate_run(
            replication[(task, seed)],
            task=task,
            seed=seed,
            source=str(
                json_object(
                    replication[(task, seed)]["config.json"],
                    source=f"{task}-s{seed}/config.json",
                )["run_name"]
            ),
        )
        for task in TASKS
        for seed in (1, 2)
    )
    identities = {(run.task, run.seed) for run in runs}
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    if len(runs) != len(expected) or identities != expected:
        raise EvidenceError("combined breadth matrix is not exactly 4 tasks × 3 seeds")
    protocol_hashes = {canonical_sha256(protocol_payload(run.config)) for run in runs}
    if protocol_hashes != {canonical_sha256(SEMANTIC_PROTOCOL)}:
        raise EvidenceError("semantic protocol hashes do not match across all 12 runs")
    for task in TASKS:
        hashes = {run.table_sha256 for run in runs if run.task == task}
        expected_count = 3 if task == "random31" else 1
        if len(hashes) != expected_count:
            relation = "distinct" if task == "random31" else "identical"
            raise EvidenceError(f"{task} task hashes are not {relation} across seeds")
    return sorted(runs, key=lambda run: (TASKS.index(run.task), run.seed))


def wait_for_matrix(
    seed0_archive: Path,
    replication_root: Path,
    *,
    poll_seconds: float,
    timeout_hours: float,
) -> list[Run]:
    deadline = time.monotonic() + timeout_hours * 3_600
    last_message = 0.0
    while True:
        try:
            return load_matrix(seed0_archive, replication_root)
        except IncompleteMatrix as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(str(error)) from error
            if time.monotonic() - last_message >= 300:
                print(f"{now()} waiting for breadth replication; {error}", flush=True)
                last_message = time.monotonic()
            time.sleep(poll_seconds)


def curve(run: Run) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([int(record["step"]) for record in run.records], dtype=float),
        np.asarray(
            [float(record["test_accuracy"]) for record in run.records],
            dtype=float,
        ),
    )


def strict_pointwise_median(runs: list[Run]) -> tuple[np.ndarray, np.ndarray]:
    by_seed = {
        run.seed: {
            int(record["step"]): float(record["test_accuracy"])
            for record in run.records
        }
        for run in runs
    }
    common_steps = set.intersection(*(set(values) for values in by_seed.values()))
    if not common_steps or EXPECTED_STEP not in common_steps:
        raise EvidenceError("the three seeds have no complete measured-step overlap")
    steps = np.asarray(sorted(common_steps), dtype=float)
    values = np.asarray(
        [
            [by_seed[seed][int(step)] for step in steps]
            for seed in SEEDS
        ],
        dtype=float,
    )
    return steps, np.median(values, axis=0)


def first_crossing(run: Run, threshold: float = 0.9) -> int | None:
    return next(
        (
            int(record["step"])
            for record in run.records
            if float(record["test_accuracy"]) >= threshold
        ),
        None,
    )


def render_trajectory_layout(
    by_task: dict[str, list[Run]],
    output: Path,
    *,
    mobile: bool,
) -> list[Path]:
    panel_rows, panel_columns = ((len(TASKS), 1) if mobile else (2, 2))
    fig, axes = plt.subplots(
        panel_rows,
        panel_columns,
        figsize=((3.9, 7.8) if mobile else (7.8, 4.25)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_alpha(0)
    if mobile:
        fig.subplots_adjust(
            left=0.18,
            right=0.99,
            top=0.985,
            bottom=0.15,
            hspace=0.28,
        )
    else:
        fig.subplots_adjust(
            left=0.085,
            right=0.995,
            top=0.97,
            bottom=0.22,
            hspace=0.30,
            wspace=0.15,
        )
    for index, (axis, task) in enumerate(zip(axes.flat, TASKS)):
        color = TASK_COLORS[task]
        task_runs = by_task[task]
        for run in task_runs:
            xs, ys = curve(run)
            axis.plot(
                xs,
                ys,
                color=color,
                linewidth=0.75,
                alpha=0.20,
                solid_capstyle="round",
            )
            axis.scatter(xs, ys, color=color, s=2.4, alpha=0.13, linewidths=0)
        xs, ys = strict_pointwise_median(task_runs)
        axis.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0,
            alpha=1.0,
            solid_capstyle="round",
        )
        axis.set_title(
            TASK_LABELS[task],
            loc="left",
            color=NORD["ink"],
            fontsize=8.5,
            pad=2,
        )
        axis.set_xlim(0, EXPECTED_STEP)
        axis.set_ylim(-0.03, 1.03)
        axis.set_yticks((0.0, 0.5, 1.0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        _step_axis(axis)
        _style_axis(axis)
        if mobile:
            if index < len(TASKS) - 1:
                axis.tick_params(bottom=False, labelbottom=False)
                axis.spines["bottom"].set_visible(False)
        else:
            row, column = divmod(index, 2)
            if column:
                axis.tick_params(left=False, labelleft=False)
                axis.spines["left"].set_visible(False)
            if row == 0:
                axis.tick_params(bottom=False, labelbottom=False)
                axis.spines["bottom"].set_visible(False)
    fig.text(
        0.54,
        0.075 if mobile else 0.09,
        "step",
        ha="center",
        va="bottom",
    )
    fig.text(
        0.025 if mobile else 0.015,
        0.56,
        "held-out accuracy",
        ha="left",
        va="center",
        rotation=90,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=NORD["muted"],
                linewidth=0.8,
                alpha=0.25,
                label="individual seeds",
            ),
            Line2D(
                [0],
                [0],
                color=NORD["muted"],
                linewidth=2.0,
                label="median",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=7.8,
        handlelength=2.0,
        columnspacing=1.5,
    )
    stem = (
        "breadth-replication-trajectories-mobile"
        if mobile
        else "breadth-replication-trajectories"
    )
    return _save_static(fig, output / stem)


def render(runs: list[Run], output: Path) -> dict[str, object]:
    by_task = {
        task: [run for run in runs if run.task == task]
        for task in TASKS
    }
    if any(
        [run.seed for run in task_runs] != list(SEEDS)
        for task_runs in by_task.values()
    ):
        raise EvidenceError("rendering requires ordered seeds 0, 1, and 2 per task")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [
        *render_trajectory_layout(by_task, output, mobile=False),
        *render_trajectory_layout(by_task, output, mobile=True),
    ]
    protocol_hash = canonical_sha256(SEMANTIC_PROTOCOL)
    task_summaries = {}
    for task, task_runs in by_task.items():
        endpoint_values = [
            float(run.records[-1]["test_accuracy"]) for run in task_runs
        ]
        task_summaries[task] = {
            "endpoint_accuracy_by_seed": {
                str(run.seed): float(run.records[-1]["test_accuracy"])
                for run in task_runs
            },
            "median_endpoint_accuracy": float(np.median(endpoint_values)),
            "first_90_percent_step_by_seed": {
                str(run.seed): first_crossing(run) for run in task_runs
            },
            "task_table_sha256_by_seed": {
                str(run.seed): run.table_sha256 for run in task_runs
            },
        }
    summary_path = output / "breadth-replication-summary.json"
    manifest_path = output / "breadth-replication-render-manifest.json"
    done_path = output / "breadth-replication-render-done.json"
    summary = {
        "created_at": now(),
        "expected_step": EXPECTED_STEP,
        "task_count": len(TASKS),
        "seed_count": len(SEEDS),
        "run_count": len(runs),
        "tasks": task_summaries,
    }
    atomic_json(summary_path, summary)
    manifest = {
        "created_at": now(),
        "matrix": {
            "tasks": list(TASKS),
            "seeds": list(SEEDS),
            "run_count": len(runs),
            "expected_step": EXPECTED_STEP,
        },
        "validation": {
            "semantic_protocol_sha256": protocol_hash,
            "semantic_protocol": SEMANTIC_PROTOCOL,
            "seed_fields": "seed = split_seed = task_seed = token_seed",
            "task_tables": (
                "generator-exact; deterministic task hashes match across seeds; "
                "random31 hashes match its explicit seeded generators"
            ),
        },
        "aggregation": (
            "faint measured seed trajectories plus the pointwise median at "
            "steps measured in all three seeds"
        ),
        "layouts": {
            "desktop": {
                "stem": "breadth-replication-trajectories",
                "panels": "2x2",
                "data": "the same 12 validated runs",
            },
            "mobile": {
                "stem": "breadth-replication-trajectories-mobile",
                "panels": "4x1",
                "data": "the same 12 validated runs",
            },
        },
        "checkpoint_rendering": "measured records only; no interpolation, smoothing, confidence interval, or extrapolation",
        "runs": [
            {
                "task": run.task,
                "seed": run.seed,
                "source": run.source,
                "task_table_sha256": run.table_sha256,
            }
            for run in runs
        ],
        "artifacts": [
            *(str(path) for path in artifacts),
            str(summary_path),
            str(manifest_path),
            str(done_path),
        ],
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        done_path,
        {
            "completed_at": now(),
            "run_count": len(runs),
            "manifest": str(manifest_path),
            "summary": str(summary_path),
        },
    )
    return manifest


def config_fixture(task: str, seed: int, run_name: str) -> dict[str, object]:
    task_spec = make_task(task, seed=seed)
    table_sha = hashlib.sha256(task_spec.table.tobytes()).hexdigest()
    return {
        **SEMANTIC_PROTOCOL,
        "task": task,
        "seed": seed,
        "split_seed": seed,
        "task_seed": seed,
        "token_seed": seed,
        "run_name": run_name,
        "task_family": TASK_FAMILIES[task],
        "task_order": task_spec.order,
        "task_table_sha256": table_sha,
        **CADENCE[seed],
    }


def fixture_files(task: str, seed: int, run_name: str) -> dict[str, bytes]:
    task_spec = make_task(task, seed=seed)
    config = config_fixture(task, seed, run_name)
    records = []
    baseline = 1.0 / task_spec.order
    ceiling = 0.05 if task == "random31" else 0.94 - 0.04 * seed
    for step in range(0, EXPECTED_STEP + 1, 250 if seed == 0 else 500):
        progress = step / EXPECTED_STEP
        accuracy = baseline + (ceiling - baseline) * progress
        records.append({"step": step, "test_accuracy": accuracy})
    table_buffer = io.BytesIO()
    np.save(table_buffer, task_spec.table)
    return {
        "config.json": (json.dumps(config) + "\n").encode(),
        "done.json": (
            json.dumps({"run_name": run_name, "final_step": EXPECTED_STEP}) + "\n"
        ).encode(),
        "metrics.jsonl": "".join(
            json.dumps(record) + "\n" for record in records
        ).encode(),
        "operation_table.npy": table_buffer.getvalue(),
    }


def write_fixture(seed0_archive: Path, replication_root: Path) -> None:
    with tarfile.open(seed0_archive, mode="w:gz") as archive:
        for task in TASKS:
            run_name = f"{task}-micro-s0-00000000"
            for filename, data in fixture_files(task, 0, run_name).items():
                info = tarfile.TarInfo(
                    f"workspace/geometry-breadth-results/{run_name}/{filename}"
                )
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    for task in TASKS:
        for seed in (1, 2):
            run_name = f"{task}-micro-s{seed}-{seed:08x}"
            run_dir = replication_root / run_name
            run_dir.mkdir(parents=True)
            for filename, data in fixture_files(task, seed, run_name).items():
                (run_dir / filename).write_bytes(data)


def run_self_test(destination: Path | None) -> None:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if destination is None:
        temporary = tempfile.TemporaryDirectory(prefix="breadth-replication-render-")
        base = Path(temporary.name)
    else:
        base = destination
        base.mkdir(parents=True, exist_ok=True)
    archive = base / "seed0.tar.gz"
    replication = base / "replication"
    output = base / "figures"
    write_fixture(archive, replication)
    runs = load_matrix(archive, replication)
    manifest = render(runs, output)
    expected = {
        "breadth-replication-trajectories.png",
        "breadth-replication-trajectories.pdf",
        "breadth-replication-trajectories-mobile.png",
        "breadth-replication-trajectories-mobile.pdf",
        "breadth-replication-summary.json",
        "breadth-replication-render-manifest.json",
        "breadth-replication-render-done.json",
    }
    missing = sorted(name for name in expected if not (output / name).is_file())
    artifact_names = {
        Path(path).name for path in manifest.get("artifacts", ())
    }
    if (
        missing
        or artifact_names != expected
        or manifest["matrix"]["run_count"] != 12
        or manifest.get("layouts", {}).get("mobile", {}).get("panels") != "4x1"
    ):
        raise AssertionError(f"breadth replication self-test failed: {missing}")
    from PIL import Image

    with Image.open(
        output / "breadth-replication-trajectories-mobile.png"
    ) as image:
        if image.getbbox() is None or image.height <= 1.6 * image.width:
            raise AssertionError("breadth mobile figure is not a portrait render")
    tampered = next(replication.glob("torus5-micro-s1-*/config.json"))
    config = json.loads(tampered.read_text())
    config["weight_decay"] = 0.0
    tampered.write_text(json.dumps(config) + "\n")
    try:
        load_matrix(archive, replication)
    except EvidenceError:
        pass
    else:
        raise AssertionError("tampered protocol was not rejected")
    print(f"breadth replication renderer self-test passed: {output}", flush=True)
    if temporary is not None:
        temporary.cleanup()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args.self_test_output)
        return
    if not args.seed0_archive or not args.replication_root or not args.output:
        raise ValueError(
            "--seed0-archive, --replication-root, and --output are required"
        )
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("poll interval and timeout must be positive")
    runs = (
        wait_for_matrix(
            args.seed0_archive,
            args.replication_root,
            poll_seconds=args.poll_seconds,
            timeout_hours=args.timeout_hours,
        )
        if args.wait
        else load_matrix(args.seed0_archive, args.replication_root)
    )
    manifest = render(runs, args.output)
    print(
        f"{now()} wrote {len(manifest['artifacts'])} artifacts from "
        f"{manifest['matrix']['run_count']} validated runs",
        flush=True,
    )


if __name__ == "__main__":
    main()
