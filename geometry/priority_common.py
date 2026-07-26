from __future__ import annotations

import fcntl
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from key60_common import (
    CAUSAL_CONTROLS,
    CAUSAL_FOLDS,
    KEY_CONDITIONS,
    PRESETS,
    SEEDS,
    CausalJob,
    KeyRun,
    load_json,
    now,
)


HORIZONS = {
    "clean": 60_000,
    "corrupt15": 30_000,
    "random": 30_000,
}
BASE_OPERATOR_STEPS = (10_000, 30_000)


def horizon_for(run: KeyRun) -> int:
    return HORIZONS[run.condition]


def operator_steps_for(run: KeyRun) -> tuple[int, ...]:
    return (*BASE_OPERATOR_STEPS, 60_000) if run.condition == "clean" else BASE_OPERATOR_STEPS


def causal_schedule(runs: list[KeyRun]) -> list[CausalJob]:
    return sorted(
        (CausalJob(run, horizon_for(run)) for run in runs),
        key=lambda job: (
            job.run.preset,
            job.run.condition,
            job.run.seed,
        ),
    )


def operator_prefix(run: KeyRun) -> Path:
    return run.path / "operator_reuse_zz_priority"


def causal_prefix(job: CausalJob) -> Path:
    return job.run.path / f"causal_reuse_zz_priority_{job.step:06d}"


def _condition_for(task: str, corruption: float) -> str | None:
    for expected_task, expected_corruption, condition in KEY_CONDITIONS:
        if task == expected_task and abs(corruption - expected_corruption) < 1e-8:
            return condition
    return None


def _behavior_at(run_dir: Path, step: int) -> float | None:
    try:
        lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    except OSError:
        return None
    value = None
    for line in lines:
        try:
            record = json.loads(line)
            if int(record.get("step", -1)) == step:
                candidate = float(record["test_accuracy"])
                if math.isfinite(candidate):
                    value = candidate
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return value


def _identity_valid(
    config: dict[str, object],
    *,
    presets: tuple[str, ...],
) -> bool:
    try:
        seed = int(config["seed"])
        preset = str(config.get("preset"))
        expected_batch_size = 2_048 if preset == "medium" else 4_096
        return (
            preset in presets
            and int(config.get("split_seed", seed)) == seed
            and int(config.get("task_seed", seed)) == seed
            and int(config.get("token_seed", -1)) == 100_000 + seed
            and int(config.get("aliases", -1)) == 4
            and int(config.get("contexts", -1)) == 16
            and int(config.get("batch_size", -1)) == expected_batch_size
            and abs(float(config.get("train_fraction", -1.0)) - 0.3) < 1e-8
            and abs(float(config.get("weight_decay", -1.0)) - 1.0) < 1e-8
        )
    except (KeyError, TypeError, ValueError):
        return False


def expected_specs(
    presets: tuple[str, ...] = PRESETS,
) -> list[tuple[str, float, str, str, int]]:
    return [
        (task, corruption, condition, preset, seed)
        for task, corruption, condition in KEY_CONDITIONS
        for preset in presets
        for seed in SEEDS
    ]


def _indexed_runs(
    results_root: Path,
    presets: tuple[str, ...] = PRESETS,
) -> dict[tuple[str, str, int], list[KeyRun]]:
    indexed: dict[tuple[str, str, int], list[KeyRun]] = {}
    for config_path in results_root.glob("*/config.json"):
        config = load_json(config_path)
        if not isinstance(config, dict) or not _identity_valid(config, presets=presets):
            continue
        task = str(config.get("task"))
        corruption = float(
            config.get(
                "task_corruption_fraction",
                config.get("corruption", 0.0),
            )
        )
        condition = _condition_for(task, corruption)
        preset = str(config.get("preset"))
        seed = int(config.get("seed", -1))
        if condition is None or preset not in presets or seed not in SEEDS:
            continue
        run = KeyRun(
            path=config_path.parent,
            task=task,
            corruption=corruption,
            condition=condition,
            preset=preset,
            seed=seed,
        )
        indexed.setdefault((condition, preset, seed), []).append(run)
    return indexed


def run_ready(run: KeyRun) -> bool:
    horizon = horizon_for(run)
    return (
        _behavior_at(run.path, horizon) is not None
        and all(
            (run.path / f"weights-{step:06d}.pt").is_file()
            for step in operator_steps_for(run)
        )
    )


def find_run(
    results_root: Path,
    *,
    condition: str,
    preset: str,
    seed: int,
    presets: tuple[str, ...] = PRESETS,
) -> KeyRun | None:
    candidates = _indexed_runs(results_root, presets).get((condition, preset, seed), [])
    if len(candidates) != 1 or not run_ready(candidates[0]):
        return None
    return candidates[0]


def wait_for_run(
    results_root: Path,
    *,
    condition: str,
    preset: str,
    seed: int,
    poll_seconds: float,
    timeout_hours: float,
    presets: tuple[str, ...] = PRESETS,
) -> KeyRun:
    deadline = time.monotonic() + timeout_hours * 3600
    previous_signature = None
    while time.monotonic() < deadline:
        run = find_run(
            results_root,
            condition=condition,
            preset=preset,
            seed=seed,
            presets=presets,
        )
        if run is not None:
            signature = _checkpoint_signature([run])
            if signature == previous_signature:
                return run
            previous_signature = signature
        else:
            previous_signature = None
        time.sleep(poll_seconds)
    raise TimeoutError(f"priority run did not validate: {condition}-{preset}-s{seed}")


def discover_runs(
    results_root: Path,
    presets: tuple[str, ...] = PRESETS,
) -> list[KeyRun] | None:
    indexed = _indexed_runs(results_root, presets)

    runs: list[KeyRun] = []
    for _, _, condition in KEY_CONDITIONS:
        for preset in presets:
            for seed in SEEDS:
                candidates = indexed.get((condition, preset, seed), [])
                if len(candidates) != 1:
                    return None
                run = candidates[0]
                if not run_ready(run):
                    return None
                runs.append(run)
    expected_count = len(KEY_CONDITIONS) * len(presets) * len(SEEDS)
    if len(runs) != expected_count:
        return None
    return sorted(runs, key=lambda run: (run.preset, run.condition, run.seed))


def _checkpoint_signature(runs: list[KeyRun]) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for run in runs:
        for step in operator_steps_for(run):
            path = run.path / f"weights-{step:06d}.pt"
            signature.append((str(path), step, path.stat().st_size))
    return tuple(signature)


def wait_for_runs(
    *,
    results_root: Path,
    poll_seconds: float,
    timeout_hours: float,
    presets: tuple[str, ...] = PRESETS,
) -> list[KeyRun]:
    deadline = time.monotonic() + timeout_hours * 3600
    previous_signature = None
    last_message = 0.0
    while time.monotonic() < deadline:
        runs = discover_runs(results_root, presets)
        if runs is not None:
            signature = _checkpoint_signature(runs)
            if signature == previous_signature:
                return runs
            previous_signature = signature
        else:
            previous_signature = None
        if time.monotonic() - last_message >= 300:
            print(
                f"{now()} waiting for mixed endpoints: clean 60k, controls 30k",
                flush=True,
            )
            last_message = time.monotonic()
        time.sleep(poll_seconds)
    raise TimeoutError("priority training matrix did not validate before timeout")


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
        or metadata.get("projection_fit") != "inductive_state_alias_fold"
        or not isinstance(records, list)
    ):
        return False
    expected_steps = set(operator_steps_for(run))
    observed_steps = {
        int(record.get("step", -1)) for record in records if isinstance(record, dict)
    }
    if observed_steps != expected_steps:
        return False
    for step in expected_steps:
        selected = [
            record
            for record in records
            if isinstance(record, dict)
            and int(record.get("step", -1)) == step
            and str(record.get("view")) == "output"
        ]
        if not selected:
            return False
        final_layer = max(int(record.get("layer", -1)) for record in selected)
        selected = [
            record for record in selected if int(record.get("layer", -1)) == final_layer
        ]
        try:
            geometry = float(selected[-1]["joint_cv_error"])
            usable = float(selected[-1]["usable_reuse_gain_bits"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(geometry) or not math.isfinite(usable):
            return False
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (prefix.with_suffix(".jsonl"), prefix.with_suffix(".csv"))
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
    groups: dict[tuple[int, str, int], set[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            return False
        key = (
            int(record.get("fold", -1)),
            str(record.get("position")),
            int(record.get("layer", -1)),
        )
        if (
            int(record.get("step", -1)) != job.step
            or record.get("checkpoint") != checkpoint
        ):
            return False
        groups.setdefault(key, set()).add(str(record.get("control")))
    return (
        set(range(CAUSAL_FOLDS)) == {key[0] for key in groups}
        and expected_sites == {(key[1], key[2]) for key in groups}
        and all(controls == CAUSAL_CONTROLS for controls in groups.values())
        and all(
            path.is_file() and path.stat().st_size > 0
            for path in (prefix.with_suffix(".jsonl"), prefix.with_suffix(".csv"))
        )
    )


@contextmanager
def analysis_slot(
    root: Path,
    *,
    count: int,
    poll_seconds: float,
    timeout_hours: float,
) -> Iterator[None]:
    if count < 1:
        raise ValueError("analysis slot count must be positive")
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_hours * 3600
    handles = [(root / f"slot-{index}.lock").open("a+") for index in range(count)]
    try:
        while time.monotonic() < deadline:
            for handle in handles:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                return
            time.sleep(poll_seconds)
        raise TimeoutError("no priority analysis slot became available")
    finally:
        for handle in handles:
            handle.close()
