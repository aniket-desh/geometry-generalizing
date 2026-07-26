from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from key60_common import (
    CAUSAL_CONTROLS,
    CAUSAL_FOLDS,
    KEY_CONDITIONS,
    SEEDS,
    KeyRun,
)
from priority_common import (
    BATCH_SIZE_BY_PRESET,
    HORIZONS,
    MIN_QUALIFIED_CAUSAL_EXAMPLES,
    causal_evidence_metric,
    operator_steps_for,
)


SUITE_PRESETS = {
    "core": ("grok", "micro"),
    "scale": ("small", "medium"),
    "large": ("large",),
    "capacity": ("small", "medium", "large"),
}
NEGATIVE_CAUSAL_CONTROLS = (
    "scrambled_successor",
    "random_orthogonal",
)
REFERENCE_CAUSAL_CONTROLS = (
    "exact_state_swap",
    "target_centroid",
)
BASELINE_CAUSAL_CONTROLS = (
    "source",
    "natural_shift",
)
OPERATOR_STEM = "operator_reuse_zz_priority"


class EvidenceError(ValueError):
    pass


def _finite(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise EvidenceError(f"{label} is not finite")
    return result


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise EvidenceError(f"missing {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read valid JSON from {path}") from error


def _condition(task: str, corruption: float) -> str | None:
    for expected_task, expected_corruption, name in KEY_CONDITIONS:
        if task == expected_task and abs(corruption - expected_corruption) < 1e-8:
            return name
    return None


def _config_identity(config: dict[str, object]) -> tuple[str, str, int] | None:
    try:
        task = str(config["task"])
        corruption = float(
            config.get(
                "task_corruption_fraction",
                config.get("corruption", 0.0),
            )
        )
        condition = _condition(task, corruption)
        if condition is None:
            return None
        return condition, str(config["preset"]), int(config["seed"])
    except (KeyError, TypeError, ValueError):
        return None


def _candidate_suites(results_root: Path) -> list[str]:
    identities: set[tuple[str, str, int]] = set()
    for config_path in results_root.glob("*/config.json"):
        payload = _load_json(config_path)
        if isinstance(payload, dict):
            identity = _config_identity(payload)
            if identity is not None:
                identities.add(identity)
    candidates = []
    for suite, presets in SUITE_PRESETS.items():
        expected = {
            (condition, preset, seed)
            for _, _, condition in KEY_CONDITIONS
            for preset in presets
            for seed in SEEDS
        }
        if expected.issubset(identities):
            candidates.append(suite)
    return candidates


def _select_suite(results_root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    candidates = _candidate_suites(results_root)
    if len(candidates) != 1:
        raise EvidenceError(
            "automatic suite detection requires exactly one complete suite; "
            f"found {candidates or 'none'}"
        )
    return candidates[0]


def _validate_config(
    config: dict[str, object],
    *,
    run_dir: Path,
    condition: str,
    preset: str,
    seed: int,
) -> None:
    horizon = HORIZONS[condition]
    expected_batch = BATCH_SIZE_BY_PRESET[preset]
    expected_task = next(
        task for task, _, name in KEY_CONDITIONS if name == condition
    )
    expected_corruption = next(
        corruption for _, corruption, name in KEY_CONDITIONS if name == condition
    )
    try:
        observed_corruption = float(
            config.get(
                "task_corruption_fraction",
                config.get("corruption", 0.0),
            )
        )
        checks = {
            "run_name": str(config["run_name"]) == run_dir.name,
            "task": str(config["task"]) == expected_task,
            "corruption": abs(observed_corruption - expected_corruption) < 1e-8,
            "preset": str(config["preset"]) == preset,
            "seed": int(config["seed"]) == seed,
            "split_seed": int(config.get("split_seed", seed)) == seed,
            "task_seed": int(config.get("task_seed", seed)) == seed,
            "token_seed": int(config["token_seed"]) == 100_000 + seed,
            "steps": int(config["steps"]) == horizon,
            "task_order": int(config["task_order"]) == 113,
            "aliases": int(config["aliases"]) == 4,
            "contexts": int(config["contexts"]) == 16,
            "batch_size": int(config["batch_size"]) == expected_batch,
            "train_fraction": abs(float(config["train_fraction"]) - 0.3) < 1e-8,
            "weight_decay": abs(float(config["weight_decay"]) - 1.0) < 1e-8,
        }
        if preset == "large":
            checks.update(
                {
                    "eval_every": int(config["eval_every"]) == 1_000,
                    "snapshot_every": int(config["snapshot_every"]) == 5_000,
                    "dense_checkpoint_every": (
                        int(config["dense_checkpoint_every"]) == 10_000
                    ),
                    "checkpoint_every": int(config["checkpoint_every"]) == 30_000,
                    "keep_checkpoints": int(config["keep_checkpoints"]) == 2,
                }
            )
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"invalid priority config in {run_dir}") from error
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise EvidenceError(
            f"{run_dir.name} fails priority identity fields: {', '.join(failed)}"
        )

    model = config.get("model")
    if not isinstance(model, dict) or int(model.get("depth", -1)) < 1:
        raise EvidenceError(f"{run_dir.name} has no valid model depth")
    if preset == "large" and (
        int(model.get("width", -1)) != 768
        or int(model.get("depth", -1)) != 8
    ):
        raise EvidenceError(f"{run_dir.name} is not the large 768x8 preset")


def _discover_runs(results_root: Path, suite: str) -> list[KeyRun]:
    presets = SUITE_PRESETS[suite]
    indexed: dict[tuple[str, str, int], list[tuple[Path, dict[str, object]]]] = {}
    for config_path in results_root.glob("*/config.json"):
        payload = _load_json(config_path)
        if not isinstance(payload, dict):
            raise EvidenceError(f"{config_path} is not a JSON object")
        identity = _config_identity(payload)
        if (
            identity is not None
            and identity[1] in presets
            and identity[2] in SEEDS
        ):
            indexed.setdefault(identity, []).append((config_path.parent, payload))

    expected = {
        (condition, preset, seed)
        for _, _, condition in KEY_CONDITIONS
        for preset in presets
        for seed in SEEDS
    }
    missing = sorted(identity for identity in expected if identity not in indexed)
    duplicate = sorted(
        identity for identity in expected if len(indexed.get(identity, [])) != 1
    )
    expected_count = len(expected)
    if missing or duplicate:
        raise EvidenceError(
            f"priority matrix is not the exact {expected_count}-run identity set; "
            f"missing={missing}, nonunique={duplicate}"
        )

    runs: list[KeyRun] = []
    for condition, preset, seed in sorted(expected):
        run_dir, config = indexed[(condition, preset, seed)][0]
        _validate_config(
            config,
            run_dir=run_dir,
            condition=condition,
            preset=preset,
            seed=seed,
        )
        horizon = HORIZONS[condition]
        done = _load_json(run_dir / "done.json")
        if (
            not isinstance(done, dict)
            or done.get("run_name") != config.get("run_name")
            or int(done.get("final_step", -1)) != horizon
        ):
            raise EvidenceError(
                f"{run_dir.name} has no exact {horizon}-step completion marker"
            )
        missing_weights = [
            step
            for step in operator_steps_for(
                KeyRun(
                    path=run_dir,
                    task=str(config["task"]),
                    corruption=float(
                        config.get(
                            "task_corruption_fraction",
                            config.get("corruption", 0.0),
                        )
                    ),
                    condition=condition,
                    preset=preset,
                    seed=seed,
                )
            )
            if not (run_dir / f"weights-{step:06d}.pt").is_file()
        ]
        if missing_weights:
            raise EvidenceError(
                f"{run_dir.name} is missing operator checkpoints {missing_weights}"
            )
        runs.append(
            KeyRun(
                path=run_dir,
                task=str(config["task"]),
                corruption=float(
                    config.get(
                        "task_corruption_fraction",
                        config.get("corruption", 0.0),
                    )
                ),
                condition=condition,
                preset=preset,
                seed=seed,
            )
        )
    if len(runs) != expected_count:
        raise EvidenceError(f"expected {expected_count} runs, found {len(runs)}")
    return sorted(runs, key=lambda run: (run.preset, run.condition, run.seed))


def _load_behavior(run: KeyRun) -> dict[str, object]:
    horizon = HORIZONS[run.condition]
    path = run.path / "metrics.jsonl"
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise EvidenceError(f"cannot read {path}") from error
    by_step: dict[int, float] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            step = int(record["step"])
            value = _finite(
                record["test_accuracy"],
                label=f"{run.slug} test_accuracy at line {line_number}",
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise EvidenceError(
                f"invalid behavior record at {path}:{line_number}"
            ) from error
        if not 0 <= step <= horizon:
            raise EvidenceError(
                f"{run.slug} behavior step {step} lies outside horizon {horizon}"
            )
        if not 0.0 <= value <= 1.0:
            raise EvidenceError(
                f"{run.slug} test accuracy {value} lies outside [0, 1]"
            )
        previous = by_step.get(step)
        if previous is not None and not math.isclose(previous, value, abs_tol=1e-12):
            raise EvidenceError(
                f"{run.slug} has conflicting held-out accuracy at step {step}"
            )
        by_step[step] = value
    if not by_step or 0 not in by_step or horizon not in by_step:
        raise EvidenceError(
            f"{run.slug} requires measured behavior at steps 0 and {horizon}"
        )

    ordered = sorted(by_step.items())
    crossing = next((step for step, value in ordered if value >= 0.9), None)
    peak_value = max(value for _, value in ordered)
    peak_step = next(step for step, value in ordered if value == peak_value)
    post_peak = [value for step, value in ordered if step >= peak_step]
    final_value = by_step[horizon]
    return {
        "measured_evaluations": len(ordered),
        "first_90_percent_step": crossing,
        "peak_step": peak_step,
        "peak_held_out_accuracy": peak_value,
        "final_step": horizon,
        "final_held_out_accuracy": final_value,
        "peak_to_final_drop": peak_value - final_value,
        "post_peak_min_accuracy": min(post_peak),
        "post_peak_max_drawdown": peak_value - min(post_peak),
        "post_peak_standard_deviation": float(np.std(post_peak)),
    }


def _load_split(run: KeyRun, config: dict[str, object]) -> dict[str, object]:
    try:
        table = np.asarray(np.load(run.path / "operation_table.npy"), dtype=np.int64)
        train_mask = np.asarray(np.load(run.path / "train_mask.npy"), dtype=bool)
    except (OSError, ValueError) as error:
        raise EvidenceError(f"cannot load table split for {run.slug}") from error
    order = int(config["task_order"])
    if table.shape != (order, order) or train_mask.shape != table.shape:
        raise EvidenceError(f"{run.slug} has an invalid operation-table split shape")
    if table.min() < 0 or table.max() >= order:
        raise EvidenceError(f"{run.slug} operation table contains invalid labels")
    expected_table_hash = config.get("task_table_sha256")
    observed_table_hash = hashlib.sha256(table.tobytes()).hexdigest()
    if expected_table_hash != observed_table_hash:
        raise EvidenceError(f"{run.slug} operation-table SHA-256 does not match config")

    held_out = ~train_mask
    held_out_count = int(held_out.sum())
    if held_out_count == 0:
        raise EvidenceError(f"{run.slug} has no held-out table entries")
    left = np.arange(order, dtype=np.int64)[:, None]
    right = np.arange(order, dtype=np.int64)[None, :]
    canonical = (left + right) % order
    correct = int(np.count_nonzero((table == canonical) & held_out))
    return {
        "train_cells": int(train_mask.sum()),
        "held_out_cells": held_out_count,
        "held_out_mask_sha256": hashlib.sha256(
            np.asarray(held_out, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "canonical_rule_correct": correct,
        "canonical_rule_accuracy": correct / held_out_count,
        "operation_table_sha256": observed_table_hash,
    }


def _expected_successor(order: int) -> tuple[list[int], str]:
    successor = (np.arange(order, dtype=np.int64) + 1) % order
    digest = hashlib.sha256(
        np.asarray(successor, dtype="<i8").tobytes()
    ).hexdigest()
    return successor.tolist(), digest


def _validate_successor(
    metadata: dict[str, object],
    *,
    run: KeyRun,
    order: int,
    source: Path,
) -> str:
    vector, digest = _expected_successor(order)
    checks = {
        "run_name": metadata.get("run_name") == run.path.name,
        "successor_mode": metadata.get("successor_mode")
        == "latent_label_plus_one",
        "successor_preregistered": metadata.get("successor_preregistered") is True,
        "successor_vector": metadata.get("successor_vector") == vector,
        "successor_sha256": metadata.get("successor_sha256") == digest,
        "generator_relation": metadata.get("generator_relation") is None,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise EvidenceError(
            f"{source} fails preregistered successor fields: {', '.join(failed)}"
        )
    return digest


def _analysis_dir(
    run: KeyRun,
    *,
    results_root: Path,
    analysis_root: Path | None,
) -> Path:
    if analysis_root is None or analysis_root.resolve() == results_root.resolve():
        return run.path
    candidates = (
        analysis_root / run.path.name,
        analysis_root / run.slug,
    )
    existing = [
        path
        for path in candidates
        if (path / f"{OPERATOR_STEM}.json").is_file()
    ]
    if len(existing) != 1:
        raise EvidenceError(
            f"analysis root must contain exactly one directory for {run.slug}; "
            f"checked {[str(path) for path in candidates]}"
        )
    return existing[0]


def _require_sidecars(prefix: Path) -> None:
    missing = [
        path.name
        for path in (prefix.with_suffix(".jsonl"), prefix.with_suffix(".csv"))
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise EvidenceError(f"{prefix} is missing nonempty sidecars {missing}")


def _operator_summary(
    run: KeyRun,
    *,
    analysis_dir: Path,
    config: dict[str, object],
) -> tuple[dict[str, object], str]:
    prefix = analysis_dir / OPERATOR_STEM
    payload = _load_json(prefix.with_suffix(".json"))
    _require_sidecars(prefix)
    if not isinstance(payload, dict):
        raise EvidenceError(f"{prefix}.json is not an object")
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(records, list):
        raise EvidenceError(f"{prefix}.json lacks metadata or records")
    if (
        int(metadata.get("folds", -1)) != 5
        or metadata.get("projection_fit") != "inductive_state_alias_fold"
    ):
        raise EvidenceError(f"{prefix}.json has the wrong inductive operator protocol")
    digest = _validate_successor(
        metadata,
        run=run,
        order=int(config["task_order"]),
        source=prefix.with_suffix(".json"),
    )

    expected_steps = operator_steps_for(run)
    observed_steps = {
        int(record.get("step", -1))
        for record in records
        if isinstance(record, dict)
    }
    if observed_steps != set(expected_steps):
        raise EvidenceError(
            f"{run.slug} operator steps are {sorted(observed_steps)}, "
            f"expected {list(expected_steps)}"
        )
    depth = int(config["model"]["depth"])
    checkpoints = []
    for step in expected_steps:
        selected = [
            record
            for record in records
            if isinstance(record, dict)
            and int(record.get("step", -1)) == step
            and record.get("view") == "output"
            and int(record.get("layer", -1)) == depth
        ]
        if len(selected) != 1:
            raise EvidenceError(
                f"{run.slug} needs one output-final operator record at step {step}"
            )
        record = selected[0]
        checkpoint = {
            "step": step,
            "joint_cv_error": _finite(
                record.get("joint_cv_error"),
                label=f"{run.slug} joint_cv_error at {step}",
            ),
            "alias_held_out_usable_gain_bits": _finite(
                record.get("usable_reuse_gain_bits"),
                label=f"{run.slug} usable_reuse_gain_bits at {step}",
            ),
            "lookup_reuse_gain_bits": _finite(
                record.get("lookup_reuse_gain_bits"),
                label=f"{run.slug} lookup_reuse_gain_bits at {step}",
            ),
            "shared_vs_null_gain_bits": _finite(
                record.get("shared_vs_null_gain_bits"),
                label=f"{run.slug} shared_vs_null_gain_bits at {step}",
            ),
        }
        checkpoints.append(checkpoint)
    return {
        "folds": 5,
        "projection_fit": "inductive_state_alias_fold",
        "checkpoints": checkpoints,
        "endpoint": checkpoints[-1],
    }, digest


def _causal_metric(record: dict[str, object], *, label: str) -> tuple[float, str]:
    try:
        return causal_evidence_metric(record)
    except ValueError as error:
        raise EvidenceError(f"{label}: {error}") from error


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise EvidenceError("cannot take a finite median of an empty series")
    return float(np.median(array))


def _causal_summary(
    run: KeyRun,
    *,
    analysis_dir: Path,
    config: dict[str, object],
) -> tuple[dict[str, object], str]:
    horizon = HORIZONS[run.condition]
    stem = f"causal_reuse_zz_priority_{horizon:06d}"
    prefix = analysis_dir / stem
    payload = _load_json(prefix.with_suffix(".json"))
    _require_sidecars(prefix)
    if not isinstance(payload, dict):
        raise EvidenceError(f"{prefix}.json is not an object")
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(records, list):
        raise EvidenceError(f"{prefix}.json lacks metadata or records")
    checkpoint = f"weights-{horizon:06d}.pt"
    depth = int(config["model"]["depth"])
    expected_sites = (("node", 0), ("output", depth))
    observed_sites = {
        (str(site.get("position")), int(site.get("layer", -1)))
        for site in metadata.get("patch_sites", [])
        if isinstance(site, dict)
    }
    if (
        int(metadata.get("folds", -1)) != CAUSAL_FOLDS
        or metadata.get("checkpoints") != [checkpoint]
        or observed_sites != set(expected_sites)
    ):
        raise EvidenceError(f"{prefix}.json has the wrong endpoint causal protocol")
    declared_controls = metadata.get("controls")
    if (
        not isinstance(declared_controls, dict)
        or set(declared_controls) != CAUSAL_CONTROLS
    ):
        raise EvidenceError(f"{prefix}.json does not declare the exact causal controls")
    digest = _validate_successor(
        metadata,
        run=run,
        order=int(config["task_order"]),
        source=prefix.with_suffix(".json"),
    )

    grouped: dict[tuple[str, int, str], dict[int, tuple[float, str]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise EvidenceError(f"{prefix}.json contains a non-object record")
        if (
            int(record.get("step", -1)) != horizon
            or record.get("checkpoint") != checkpoint
        ):
            raise EvidenceError(f"{prefix}.json contains a record at the wrong checkpoint")
        position = str(record.get("position"))
        layer = int(record.get("layer", -1))
        control = str(record.get("control"))
        fold = int(record.get("fold", -1))
        if (position, layer) not in expected_sites:
            raise EvidenceError(f"{prefix}.json contains an unexpected patch site")
        if fold not in range(CAUSAL_FOLDS):
            raise EvidenceError(f"{prefix}.json contains an unexpected causal fold")
        if control not in CAUSAL_CONTROLS:
            if control not in BASELINE_CAUSAL_CONTROLS:
                raise EvidenceError(f"{prefix}.json contains an unexpected control")
            continue
        value = _causal_metric(
            record,
            label=f"{run.slug} {position}/{layer} {control} fold {fold}",
        )
        key = (position, layer, control)
        if fold in grouped.setdefault(key, {}):
            raise EvidenceError(f"{run.slug} has duplicate causal fold {key + (fold,)}")
        grouped[key][fold] = value

    site_summaries: dict[str, object] = {}
    expected_folds = set(range(CAUSAL_FOLDS))
    for position, layer in expected_sites:
        controls: dict[str, object] = {}
        for control in sorted(CAUSAL_CONTROLS):
            fold_values = grouped.get((position, layer, control), {})
            if set(fold_values) != expected_folds:
                raise EvidenceError(
                    f"{run.slug} {position}/{layer} {control} has folds "
                    f"{sorted(fold_values)}, expected {sorted(expected_folds)}"
                )
            ordered = [fold_values[fold] for fold in range(CAUSAL_FOLDS)]
            values = [value for value, _ in ordered]
            metric_keys = [key for _, key in ordered]
            controls[control] = {
                "fold_values": values,
                "fold_metric_keys": metric_keys,
                "median_success": _median(values),
            }

        learned = float(controls["learned_generator"]["median_success"])
        negative_medians = {
            control: float(controls[control]["median_success"])
            for control in NEGATIVE_CAUSAL_CONTROLS
        }
        reference_medians = {
            control: float(controls[control]["median_success"])
            for control in REFERENCE_CAUSAL_CONTROLS
        }
        all_noncanonical = {**negative_medians, **reference_medians}
        site_summaries[f"{position}_{layer if position == 'node' else 'final'}"] = {
            "position": position,
            "layer": layer,
            "canonical_cycle_median_success": learned,
            "controls": controls,
            "negative_control_max_median": max(negative_medians.values()),
            "negative_control_argmax": max(
                negative_medians, key=negative_medians.get
            ),
            "negative_control_max_fold": max(
                value
                for control in NEGATIVE_CAUSAL_CONTROLS
                for value in controls[control]["fold_values"]
            ),
            "reference_control_max_median": max(reference_medians.values()),
            "all_noncanonical_control_max_median": max(all_noncanonical.values()),
            "canonical_minus_negative_control_max": (
                learned - max(negative_medians.values())
            ),
        }
    return {
        "step": horizon,
        "folds": CAUSAL_FOLDS,
        "sites": site_summaries,
    }, digest


def _run_summary(
    run: KeyRun,
    *,
    results_root: Path,
    analysis_root: Path | None,
) -> tuple[dict[str, object], set[str]]:
    config = _load_json(run.path / "config.json")
    if not isinstance(config, dict):
        raise EvidenceError(f"{run.path}/config.json is not an object")
    analysis_dir = _analysis_dir(
        run,
        results_root=results_root,
        analysis_root=analysis_root,
    )
    behavior = _load_behavior(run)
    split = _load_split(run, config)
    operator, operator_hash = _operator_summary(
        run,
        analysis_dir=analysis_dir,
        config=config,
    )
    causal, causal_hash = _causal_summary(
        run,
        analysis_dir=analysis_dir,
        config=config,
    )
    return {
        "run": run.slug,
        "run_name": run.path.name,
        "condition": run.condition,
        "preset": run.preset,
        "seed": run.seed,
        "horizon": HORIZONS[run.condition],
        "behavior": behavior,
        "held_out_split": split,
        "operator": operator,
        "causal": causal,
    }, {operator_hash, causal_hash}


def _optional_median(values: Iterable[object]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(finite)) if finite else None


def _group_summaries(
    runs: list[dict[str, object]],
    presets: tuple[str, ...],
) -> list[dict[str, object]]:
    summaries = []
    for preset in presets:
        for _, _, condition in KEY_CONDITIONS:
            selected = [
                run
                for run in runs
                if run["preset"] == preset and run["condition"] == condition
            ]
            if len(selected) != len(SEEDS):
                raise EvidenceError(
                    f"{condition}/{preset} does not contain exactly three seeds"
                )
            crossings = [
                run["behavior"]["first_90_percent_step"] for run in selected
            ]
            summaries.append(
                {
                    "condition": condition,
                    "preset": preset,
                    "seeds": len(selected),
                    "first_90_percent_crossing_rate": (
                        sum(value is not None for value in crossings) / len(selected)
                    ),
                    "median_first_90_percent_step_among_crossers": _optional_median(
                        crossings
                    ),
                    "median_peak_held_out_accuracy": _optional_median(
                        run["behavior"]["peak_held_out_accuracy"] for run in selected
                    ),
                    "median_final_held_out_accuracy": _optional_median(
                        run["behavior"]["final_held_out_accuracy"] for run in selected
                    ),
                    "median_post_peak_max_drawdown": _optional_median(
                        run["behavior"]["post_peak_max_drawdown"] for run in selected
                    ),
                    "median_canonical_rule_accuracy": _optional_median(
                        run["held_out_split"]["canonical_rule_accuracy"]
                        for run in selected
                    ),
                    "median_endpoint_joint_cv_error": _optional_median(
                        run["operator"]["endpoint"]["joint_cv_error"]
                        for run in selected
                    ),
                    "median_endpoint_alias_held_out_usable_gain_bits": _optional_median(
                        run["operator"]["endpoint"][
                            "alias_held_out_usable_gain_bits"
                        ]
                        for run in selected
                    ),
                    "median_node0_causal_success": _optional_median(
                        run["causal"]["sites"]["node_0"][
                            "canonical_cycle_median_success"
                        ]
                        for run in selected
                    ),
                    "median_output_final_causal_success": _optional_median(
                        run["causal"]["sites"]["output_final"][
                            "canonical_cycle_median_success"
                        ]
                        for run in selected
                    ),
                    "max_output_final_negative_control_median": max(
                        run["causal"]["sites"]["output_final"][
                            "negative_control_max_median"
                        ]
                        for run in selected
                    ),
                }
            )
    return summaries


def summarize(
    *,
    results_root: Path,
    analysis_root: Path | None = None,
    suite: str = "auto",
) -> dict[str, object]:
    results_root = results_root.resolve()
    if not results_root.is_dir():
        raise EvidenceError(f"results root is not a directory: {results_root}")
    if analysis_root is not None:
        analysis_root = analysis_root.resolve()
        if not analysis_root.is_dir():
            raise EvidenceError(
                f"analysis root is not a directory: {analysis_root}"
            )
    selected_suite = _select_suite(results_root, suite)
    presets = SUITE_PRESETS[selected_suite]
    discovered = _discover_runs(results_root, selected_suite)
    run_summaries: list[dict[str, object]] = []
    successor_hashes: set[str] = set()
    for run in discovered:
        run_summary, run_hashes = _run_summary(
            run,
            results_root=results_root,
            analysis_root=analysis_root,
        )
        run_summaries.append(run_summary)
        successor_hashes.update(run_hashes)
    expected_vector, expected_hash = _expected_successor(113)
    if successor_hashes != {expected_hash}:
        raise EvidenceError(
            f"analysis successor hashes are {sorted(successor_hashes)}, "
            f"expected only {expected_hash}"
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": selected_suite,
        "results_root": str(results_root),
        "analysis_root": str(analysis_root or results_root),
        "validation": {
            "status": "complete",
            "exact_run_count": len(run_summaries),
            "presets": list(presets),
            "seeds": list(SEEDS),
            "conditions": [name for _, _, name in KEY_CONDITIONS],
            "horizons": HORIZONS,
            "successor": {
                "status": "preregistered",
                "mode": "latent_label_plus_one",
                "definition": "latent label k maps to (k + 1) mod 113",
                "vector": expected_vector,
                "sha256_little_endian_int64": expected_hash,
                "validated_operator_outputs": len(run_summaries),
                "validated_causal_outputs": len(run_summaries),
            },
            "causal_metric": {
                "minimum_qualified_examples": MIN_QUALIFIED_CAUSAL_EXAMPLES,
                "qualified_metric": "qualified_desired_accuracy",
                "fallback_metric": "desired_accuracy",
                "probability_recovery_used": False,
            },
        },
        "definitions": {
            "first_90_percent_step": (
                "earliest measured evaluation with held-out accuracy at least 0.90; "
                "no interpolation"
            ),
            "post_peak_max_drawdown": (
                "peak held-out accuracy minus the minimum measured accuracy at or "
                "after the earliest measured peak"
            ),
            "post_peak_standard_deviation": (
                "population standard deviation of measured held-out accuracy at "
                "or after the earliest measured peak"
            ),
            "canonical_rule_accuracy": (
                "fraction of this run's held-out operation-table labels exactly "
                "equal to (left + relation) mod 113; this is label agreement, "
                "not model accuracy"
            ),
            "alias_held_out_usable_gain_bits": (
                "minimum of lookup and zero-prediction code lengths minus the "
                "shared canonical-successor code length, evaluated on held-out aliases"
            ),
            "causal_success": (
                "median across three folds; use qualified_desired_accuracy only "
                f"with at least {MIN_QUALIFIED_CAUSAL_EXAMPLES} qualified examples, "
                "otherwise use bounded desired_accuracy across all evaluated "
                "examples; probability_recovery is excluded"
            ),
            "negative_control_max": (
                "maximum of scrambled-successor and norm-matched random-orthogonal "
                "control success; exact-state and target-centroid references are "
                "reported separately"
            ),
        },
        "runs": run_summaries,
        "groups": _group_summaries(run_summaries, presets),
    }


def _percent(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def _number(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _step(value: object) -> str:
    return "—" if value is None else f"{int(value) // 1000}k"


def markdown(summary: dict[str, object]) -> str:
    successor = summary["validation"]["successor"]
    lines = [
        f"# Priority evidence summary: {summary['suite']}",
        "",
        (
            f"Validated {summary['validation']['exact_run_count']} exact mixed-horizon "
            f"runs. The preregistered canonical successor SHA-256 is "
            f"`{successor['sha256_little_endian_int64']}`."
        ),
        "",
        (
            "First 90% is the first measured crossing. Drawdown is the largest "
            "measured decline after the earliest peak. Canonical agreement is a "
            "property of held-out labels, not model accuracy. Causal controls are "
            "the stronger of scrambled-successor and random-orthogonal medians."
        ),
        "",
        "## Per-run evidence",
        "",
        (
            "| run | 90% | peak | final | drawdown | canonical | joint CV | "
            "alias gain | node 0 | node ctrl | output | output ctrl |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        behavior = run["behavior"]
        split = run["held_out_split"]
        operator = run["operator"]["endpoint"]
        sites = run["causal"]["sites"]
        node = sites["node_0"]
        output = sites["output_final"]
        lines.append(
            "| {run} | {crossing} | {peak} | {final} | {drawdown} | "
            "{canonical} | {joint} | {gain} | {node} | {node_control} | "
            "{output} | {output_control} |".format(
                run=run["run"],
                crossing=_step(behavior["first_90_percent_step"]),
                peak=_percent(behavior["peak_held_out_accuracy"]),
                final=_percent(behavior["final_held_out_accuracy"]),
                drawdown=_percent(behavior["post_peak_max_drawdown"]),
                canonical=_percent(split["canonical_rule_accuracy"]),
                joint=_number(operator["joint_cv_error"]),
                gain=f"{operator['alias_held_out_usable_gain_bits'] / 1000.0:.2f}k",
                node=_percent(node["canonical_cycle_median_success"]),
                node_control=_percent(node["negative_control_max_median"]),
                output=_percent(output["canonical_cycle_median_success"]),
                output_control=_percent(output["negative_control_max_median"]),
            )
        )
    lines.extend(
        [
            "",
            "## Three-seed medians",
            "",
            (
                "| condition / preset | crossed | first 90% | final | drawdown | "
                "joint CV | alias gain | node 0 | output | worst output ctrl |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["groups"]:
        lines.append(
            "| {condition} / {preset} | {crossed} | {crossing} | {final} | "
            "{drawdown} | {joint} | {gain} | {node} | {output} | {control} |".format(
                condition=group["condition"],
                preset=group["preset"],
                crossed=_percent(group["first_90_percent_crossing_rate"]),
                crossing=_step(
                    group["median_first_90_percent_step_among_crossers"]
                ),
                final=_percent(group["median_final_held_out_accuracy"]),
                drawdown=_percent(group["median_post_peak_max_drawdown"]),
                joint=_number(group["median_endpoint_joint_cv_error"]),
                gain=(
                    f"{group['median_endpoint_alias_held_out_usable_gain_bits'] / 1000.0:.2f}k"
                ),
                node=_percent(group["median_node0_causal_success"]),
                output=_percent(group["median_output_final_causal_success"]),
                control=_percent(
                    group["max_output_final_negative_control_median"]
                ),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_summary(
    summary: dict[str, object],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    _atomic_text(json_path, json.dumps(summary, indent=2) + "\n")
    _atomic_text(markdown_path, markdown(summary))


def _synthetic_fixture(root: Path, *, suite: str = "core") -> None:
    presets = SUITE_PRESETS[suite]
    order = 113
    canonical = (
        np.arange(order, dtype=np.int64)[:, None]
        + np.arange(order, dtype=np.int64)[None, :]
    ) % order
    train_mask = (
        np.indices((order, order), dtype=np.int64).sum(axis=0) % 10
    ) < 3
    successor, successor_hash = _expected_successor(order)
    for task, corruption, condition in KEY_CONDITIONS:
        for preset in presets:
            for seed in SEEDS:
                horizon = HORIZONS[condition]
                run_name = f"{condition}-{preset}-s{seed}-synthetic"
                run_dir = root / run_name
                run_dir.mkdir(parents=True)
                table = canonical.copy()
                if condition == "corrupt15":
                    table[~train_mask] = (table[~train_mask] + 1) % order
                elif condition == "random":
                    table = (
                        7 * np.arange(order, dtype=np.int64)[:, None]
                        + 11 * np.arange(order, dtype=np.int64)[None, :]
                        + 1
                    ) % order
                depth = {
                    "grok": 1,
                    "micro": 2,
                    "small": 4,
                    "medium": 6,
                    "large": 8,
                }[preset]
                config = {
                    "run_name": run_name,
                    "task": task,
                    "task_corruption_fraction": corruption,
                    "task_order": order,
                    "preset": preset,
                    "seed": seed,
                    "split_seed": seed,
                    "task_seed": seed,
                    "token_seed": 100_000 + seed,
                    "steps": horizon,
                    "aliases": 4,
                    "contexts": 16,
                    "batch_size": BATCH_SIZE_BY_PRESET[preset],
                    "eval_every": 1_000,
                    "snapshot_every": 5_000,
                    "dense_checkpoint_every": 10_000,
                    "checkpoint_every": 30_000,
                    "keep_checkpoints": 2,
                    "train_fraction": 0.3,
                    "weight_decay": 1.0,
                    "model": {
                        "width": 768 if preset == "large" else 128,
                        "depth": depth,
                    },
                    "task_table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
                }
                _atomic_text(
                    run_dir / "config.json",
                    json.dumps(config, indent=2) + "\n",
                )
                _atomic_text(
                    run_dir / "done.json",
                    json.dumps(
                        {"run_name": run_name, "final_step": horizon}, indent=2
                    )
                    + "\n",
                )
                np.save(run_dir / "operation_table.npy", table)
                np.save(run_dir / "train_mask.npy", train_mask)
                behavior_steps = [0, 10_000, 20_000, 30_000]
                if condition == "clean":
                    behavior_steps.extend((40_000, 50_000, 60_000))
                behavior_values = {
                    step: value
                    for step, value in zip(
                        behavior_steps,
                        (0.10, 0.91, 0.95, 0.80, 0.84, 0.82, 0.81),
                    )
                }
                _atomic_text(
                    run_dir / "metrics.jsonl",
                    "".join(
                        json.dumps(
                            {"step": step, "test_accuracy": behavior_values[step]}
                        )
                        + "\n"
                        for step in behavior_steps
                    ),
                )
                successor_metadata = {
                    "run_name": run_name,
                    "successor_mode": "latent_label_plus_one",
                    "successor_preregistered": True,
                    "successor_vector": successor,
                    "successor_sha256": successor_hash,
                    "generator_relation": None,
                }
                operator_records = []
                run = KeyRun(
                    path=run_dir,
                    task=task,
                    corruption=corruption,
                    condition=condition,
                    preset=preset,
                    seed=seed,
                )
                for step in operator_steps_for(run):
                    (run_dir / f"weights-{step:06d}.pt").write_bytes(b"synthetic")
                    operator_records.append(
                        {
                            "step": step,
                            "view": "output",
                            "layer": depth,
                            "joint_cv_error": 0.25,
                            "usable_reuse_gain_bits": 1_500.0,
                            "lookup_reuse_gain_bits": 2_000.0,
                            "shared_vs_null_gain_bits": 1_500.0,
                        }
                    )
                operator_prefix = run_dir / OPERATOR_STEM
                _atomic_text(
                    operator_prefix.with_suffix(".json"),
                    json.dumps(
                        {
                            "metadata": {
                                **successor_metadata,
                                "folds": 5,
                                "projection_fit": "inductive_state_alias_fold",
                            },
                            "records": operator_records,
                        },
                        indent=2,
                    )
                    + "\n",
                )
                operator_prefix.with_suffix(".jsonl").write_text("{}\n")
                operator_prefix.with_suffix(".csv").write_text("step\n")

                causal_records = []
                causal_values = {
                    "learned_generator": 0.80,
                    "exact_state_swap": 1.00,
                    "target_centroid": 0.95,
                    "scrambled_successor": 0.10,
                    "random_orthogonal": 0.20,
                }
                checkpoint = f"weights-{horizon:06d}.pt"
                for fold in range(CAUSAL_FOLDS):
                    for position, layer in (("node", 0), ("output", depth)):
                        for control, value in causal_values.items():
                            causal_records.append(
                                {
                                    "step": horizon,
                                    "checkpoint": checkpoint,
                                    "fold": fold,
                                    "position": position,
                                    "layer": layer,
                                    "control": control,
                                    "qualified_examples": 128,
                                    "qualified_desired_accuracy": min(
                                        1.0,
                                        value + 0.01 * fold,
                                    ),
                                    "desired_accuracy": min(
                                        1.0,
                                        value + 0.01 * fold,
                                    ),
                                    "probability_recovery": 8.0,
                                }
                            )
                causal_prefix = (
                    run_dir / f"causal_reuse_zz_priority_{horizon:06d}"
                )
                _atomic_text(
                    causal_prefix.with_suffix(".json"),
                    json.dumps(
                        {
                            "metadata": {
                                **successor_metadata,
                                "folds": CAUSAL_FOLDS,
                                "checkpoints": [checkpoint],
                                "patch_sites": [
                                    {"position": "node", "layer": 0},
                                    {"position": "output", "layer": depth},
                                ],
                                "controls": {
                                    control: "synthetic"
                                    for control in causal_values
                                },
                            },
                            "records": causal_records,
                        },
                        indent=2,
                    )
                    + "\n",
                )
                causal_prefix.with_suffix(".jsonl").write_text("{}\n")
                causal_prefix.with_suffix(".csv").write_text("step\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(
        prefix="priority-evidence-summary-"
    ) as temporary:
        root = Path(temporary)
        _synthetic_fixture(root)
        summary = summarize(results_root=root, suite="core")
        if summary["validation"]["exact_run_count"] != 18:
            raise AssertionError("synthetic exact run count was not validated")
        sample = summary["runs"][0]
        behavior = sample["behavior"]
        if (
            behavior["first_90_percent_step"] != 10_000
            or behavior["peak_step"] != 20_000
            or not math.isclose(behavior["post_peak_max_drawdown"], 0.15)
        ):
            raise AssertionError("synthetic behavior summary is incorrect")
        if not math.isclose(
            sample["operator"]["endpoint"]["alias_held_out_usable_gain_bits"],
            1_500.0,
        ):
            raise AssertionError("synthetic operator summary is incorrect")
        output = sample["causal"]["sites"]["output_final"]
        if (
            not math.isclose(output["canonical_cycle_median_success"], 0.81)
            or not math.isclose(output["negative_control_max_median"], 0.21)
        ):
            raise AssertionError("synthetic causal summary is incorrect")
        rendered = markdown(summary)
        if "18 exact mixed-horizon runs" not in rendered:
            raise AssertionError("synthetic Markdown summary is incomplete")
    with tempfile.TemporaryDirectory(
        prefix="priority-scale-evidence-summary-"
    ) as temporary:
        root = Path(temporary)
        _synthetic_fixture(root, suite="scale")
        summary = summarize(results_root=root, suite="auto")
        if (
            summary["suite"] != "scale"
            or summary["validation"]["presets"] != ["small", "medium"]
            or summary["validation"]["exact_run_count"] != 18
        ):
            raise AssertionError("synthetic scale suite was not validated")
    with tempfile.TemporaryDirectory(
        prefix="priority-large-evidence-summary-"
    ) as temporary:
        root = Path(temporary)
        _synthetic_fixture(root, suite="large")
        summary = summarize(results_root=root, suite="auto")
        if (
            summary["suite"] != "large"
            or summary["validation"]["presets"] != ["large"]
            or summary["validation"]["exact_run_count"] != 9
        ):
            raise AssertionError("synthetic large suite was not validated")
    print(
        "self-test passed for core, scale, and large: exact suite identities and "
        "horizons, behavior crossing and drawdown, canonical split agreement, "
        "preregistered successor hash, operator reuse, causal sites, and controls"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize one exact core, scale, large, or capacity suite."
        )
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        help=(
            "Optional separate root containing one analysis directory per run "
            "name or condition-preset-seed slug."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=("auto", *SUITE_PRESETS),
        default="auto",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.results_root is None:
        raise SystemExit("--results-root is required")
    json_path = args.output_json or (
        args.results_root / "priority-evidence-summary.json"
    )
    markdown_path = args.output_markdown or (
        args.results_root / "priority-evidence-summary.md"
    )
    summary = summarize(
        results_root=args.results_root,
        analysis_root=args.analysis_root,
        suite=args.suite,
    )
    write_summary(
        summary,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    print(f"wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
