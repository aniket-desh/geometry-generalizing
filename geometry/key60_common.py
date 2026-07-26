from __future__ import annotations

import fcntl
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator


FINAL_STEP = 60_000
OPERATOR_STEPS = (10_000, 30_000, 60_000)
KEY_CONDITIONS = (
    ("cycle113", 0.0, "clean"),
    ("cycle113", 0.15, "corrupt15"),
    ("random113", 0.0, "random"),
)
PRESETS = ("grok", "micro")
SEEDS = (0, 1, 2)
KEY_COUNT = len(KEY_CONDITIONS) * len(PRESETS) * len(SEEDS)
CAUSAL_FOLDS = 3
CAUSAL_CONTROLS = {
    "learned_generator",
    "exact_state_swap",
    "target_centroid",
    "scrambled_successor",
    "random_orthogonal",
}


@dataclass(frozen=True)
class KeyRun:
    path: Path
    task: str
    corruption: float
    condition: str
    preset: str
    seed: int

    @property
    def slug(self) -> str:
        return f"{self.condition}-{self.preset}-s{self.seed}"


@dataclass(frozen=True)
class CausalJob:
    run: KeyRun
    step: int

    @property
    def slug(self) -> str:
        return f"{self.run.slug}-p{self.step:06d}"


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def expected_keys() -> set[tuple[str, float, str, int]]:
    return {
        (task, corruption, preset, seed)
        for task, corruption, _ in KEY_CONDITIONS
        for preset in PRESETS
        for seed in SEEDS
    }


def condition_for(task: str, corruption: float) -> str:
    for candidate_task, candidate_corruption, condition in KEY_CONDITIONS:
        if task == candidate_task and abs(corruption - candidate_corruption) < 1e-8:
            return condition
    raise ValueError(f"unexpected key60 condition: {task}, {corruption}")


def validated_specs(manifest_path: Path) -> list[dict[str, object]] | None:
    payload = load_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("profile") != "key60":
        return None
    specs = payload.get("runs")
    if not isinstance(specs, list) or len(specs) != KEY_COUNT:
        return None
    if any(
        not isinstance(spec, dict) or int(spec.get("steps", -1)) != FINAL_STEP
        for spec in specs
    ):
        return None
    observed = {
        (
            str(spec.get("task")),
            float(spec.get("corruption", 0.0)),
            str(spec.get("preset")),
            int(spec.get("seed", -1)),
        )
        for spec in specs
        if isinstance(spec, dict)
    }
    expected = expected_keys()
    if len(observed) != len(specs) or observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"key60 manifest mismatch; missing={missing}, extra={extra}")
    return specs


def training_results_valid(
    results_path: Path,
    *,
    specs: list[dict[str, object]],
) -> bool:
    payload = load_json(results_path)
    if not isinstance(payload, list) or len(payload) != len(specs):
        return False
    expected = expected_keys()
    observed: set[tuple[str, float, str, int]] = set()
    for result in payload:
        if (
            not isinstance(result, dict)
            or int(result.get("returncode", -1)) != 0
            or int(result.get("steps", -1)) != FINAL_STEP
        ):
            return False
        observed.add(
            (
                str(result.get("task")),
                float(result.get("corruption", 0.0)),
                str(result.get("preset")),
                int(result.get("seed", -1)),
            )
        )
    return len(observed) == len(payload) and observed == expected


def _final_behavior_valid(run_dir: Path) -> bool:
    path = run_dir / "metrics.jsonl"
    try:
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    for record in records:
        if not isinstance(record, dict) or int(record.get("step", -1)) != FINAL_STEP:
            continue
        try:
            value = float(record["test_accuracy"])
        except (KeyError, TypeError, ValueError):
            return False
        return math.isfinite(value)
    return False


def discover_runs(
    results_root: Path,
    specs: list[dict[str, object]],
) -> list[KeyRun] | None:
    indexed: dict[tuple[str, float, str, int], list[Path]] = {}
    for config_path in results_root.glob("*/config.json"):
        payload = load_json(config_path)
        if not isinstance(payload, dict):
            continue
        key = (
            str(payload.get("task")),
            float(
                payload.get(
                    "task_corruption_fraction",
                    payload.get("corruption", 0.0),
                )
            ),
            str(payload.get("preset")),
            int(payload.get("seed", -1)),
        )
        indexed.setdefault(key, []).append(config_path.parent)

    runs: list[KeyRun] = []
    for spec in specs:
        key = (
            str(spec["task"]),
            float(spec.get("corruption", 0.0)),
            str(spec["preset"]),
            int(spec["seed"]),
        )
        candidates = indexed.get(key, [])
        if len(candidates) != 1:
            return None
        run_dir = candidates[0]
        config = load_json(run_dir / "config.json")
        done = load_json(run_dir / "done.json")
        if not isinstance(config, dict) or not isinstance(done, dict):
            return None
        if (
            str(done.get("run_name")) != str(config.get("run_name"))
            or int(done.get("final_step", -1)) < FINAL_STEP
            or not _final_behavior_valid(run_dir)
            or any(
                not (run_dir / f"weights-{step:06d}.pt").is_file()
                for step in OPERATOR_STEPS
            )
        ):
            return None
        runs.append(
            KeyRun(
                path=run_dir,
                task=key[0],
                corruption=key[1],
                condition=condition_for(key[0], key[1]),
                preset=key[2],
                seed=key[3],
            )
        )
    return sorted(runs, key=lambda run: (run.preset, run.condition, run.seed))


def ready_runs(
    *,
    results_root: Path,
    manifest_path: Path,
    results_path: Path,
) -> list[KeyRun] | None:
    specs = validated_specs(manifest_path)
    if specs is None or not training_results_valid(results_path, specs=specs):
        return None
    runs = discover_runs(results_root, specs)
    if runs is None or len(runs) != KEY_COUNT:
        return None
    return runs


def wait_for_runs(
    *,
    results_root: Path,
    manifest_path: Path,
    results_path: Path,
    poll_seconds: float,
    timeout_hours: float,
) -> list[KeyRun]:
    if poll_seconds <= 0 or timeout_hours <= 0:
        raise ValueError("poll interval and timeout must be positive")
    deadline = time.monotonic() + timeout_hours * 3600
    last_message = 0.0
    while time.monotonic() < deadline:
        runs = ready_runs(
            results_root=results_root,
            manifest_path=manifest_path,
            results_path=results_path,
        )
        if runs is not None:
            return runs
        if time.monotonic() - last_message >= 300:
            print(
                f"{now()} waiting for strict {KEY_COUNT}-run key60 matrix",
                flush=True,
            )
            last_message = time.monotonic()
        time.sleep(poll_seconds)
    raise TimeoutError("key60 training matrix did not validate before timeout")


def causal_schedule(runs: list[KeyRun]) -> list[CausalJob]:
    jobs = [CausalJob(run, FINAL_STEP) for run in runs]
    jobs.extend(
        CausalJob(run, step)
        for run in runs
        if run.preset == "grok" and run.seed == 0
        for step in (10_000, 30_000)
    )
    return sorted(
        jobs,
        key=lambda job: (
            job.run.preset,
            job.run.condition,
            job.run.seed,
            job.step,
        ),
    )


def operator_prefix(run: KeyRun) -> Path:
    return run.path / "operator_reuse_zz_key60"


def causal_prefix(job: CausalJob) -> Path:
    return job.run.path / f"causal_reuse_zz_key60_{job.step:06d}"


def operator_output_valid(run: KeyRun) -> bool:
    prefix = operator_prefix(run)
    payload = load_json(prefix.with_suffix(".json"))
    config = load_json(run.path / "config.json")
    if not isinstance(payload, dict) or not isinstance(config, dict):
        return False
    metadata = payload.get("metadata")
    records = payload.get("records")
    if (
        not isinstance(metadata, dict)
        or metadata.get("run_name") != config.get("run_name")
        or int(metadata.get("folds", -1)) != 5
        or not isinstance(records, list)
    ):
        return False
    observed_steps = {
        int(record.get("step", -1)) for record in records if isinstance(record, dict)
    }
    if observed_steps != set(OPERATOR_STEPS):
        return False
    for step in OPERATOR_STEPS:
        final_output = [
            record
            for record in records
            if isinstance(record, dict)
            and int(record.get("step", -1)) == step
            and str(record.get("view")) == "output"
        ]
        if not final_output:
            return False
        last_layer = max(int(record.get("layer", -1)) for record in final_output)
        selected = [
            record
            for record in final_output
            if int(record.get("layer", -1)) == last_layer
        ]
        if len(selected) != 1:
            return False
        try:
            geometry = float(selected[0]["joint_cv_error"])
            usable = float(selected[0]["usable_reuse_gain_bits"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(geometry) or not math.isfinite(usable):
            return False
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            prefix.with_suffix(".jsonl"),
            prefix.with_suffix(".csv"),
        )
    )


def causal_output_valid(job: CausalJob) -> bool:
    prefix = causal_prefix(job)
    payload = load_json(prefix.with_suffix(".json"))
    config = load_json(job.run.path / "config.json")
    if not isinstance(payload, dict) or not isinstance(config, dict):
        return False
    metadata = payload.get("metadata")
    records = payload.get("records")
    checkpoint = f"weights-{job.step:06d}.pt"
    if (
        not isinstance(metadata, dict)
        or metadata.get("run_name") != config.get("run_name")
        or int(metadata.get("folds", -1)) != CAUSAL_FOLDS
        or metadata.get("checkpoints") != [checkpoint]
        or not isinstance(records, list)
        or not records
    ):
        return False
    sites = metadata.get("patch_sites")
    if not isinstance(sites, list) or not sites:
        return False
    expected_sites = {
        (str(site.get("position")), int(site.get("layer", -1)))
        for site in sites
        if isinstance(site, dict)
    }
    if len(expected_sites) != len(sites):
        return False
    groups: dict[tuple[int, str, int], set[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            return False
        try:
            step = int(record.get("step", -1))
            key = (
                int(record.get("fold", -1)),
                str(record.get("position")),
                int(record.get("layer", -1)),
            )
        except (TypeError, ValueError):
            return False
        if step != job.step or record.get("checkpoint") != checkpoint:
            return False
        groups.setdefault(key, set()).add(str(record.get("control")))
    if set(range(CAUSAL_FOLDS)) != {key[0] for key in groups}:
        return False
    if expected_sites != {(key[1], key[2]) for key in groups}:
        return False
    if any(controls != CAUSAL_CONTROLS for controls in groups.values()):
        return False
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            prefix.with_suffix(".jsonl"),
            prefix.with_suffix(".csv"),
        )
    )


def wait_for_marker(
    path: Path,
    *,
    poll_seconds: float,
    timeout_hours: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_hours * 3600
    last_message = 0.0
    while time.monotonic() < deadline:
        payload = load_json(path)
        if isinstance(payload, dict) and payload.get("status") == "complete":
            return payload
        if time.monotonic() - last_message >= 300:
            print(f"{now()} waiting for marker {path}", flush=True)
            last_message = time.monotonic()
        time.sleep(poll_seconds)
    raise TimeoutError(f"marker did not complete before timeout: {path}")


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
