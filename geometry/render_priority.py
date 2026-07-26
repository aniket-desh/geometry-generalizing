from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from key60_common import KEY_CONDITIONS, PRESETS, SEEDS, CausalJob, KeyRun, atomic_json, load_json
from priority_common import (
    HORIZONS,
    causal_output_valid,
    causal_prefix,
    causal_schedule,
    horizon_for,
    operator_output_valid,
    operator_prefix,
    operator_steps_for,
)
from render_key60 import (
    CONTROL_LABEL,
    CONTROL_ORDER,
    NORD,
    _finite,
    _median_curve,
    _records,
    _save,
    _step_axis,
    _style_axis,
)


CONDITION_STYLE = {
    "clean": ("clean · 60k", NORD["frost_dark"]),
    "corrupt15": ("15% corrupted · 30k", NORD["orange"]),
    "random": ("random table · 30k", NORD["purple"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render explicit mixed-endpoint Nord spaghetti plots."
    )
    parser.add_argument("--run", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preset", action="append", dest="presets")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    return parser.parse_args()


def _load_runs(paths: Iterable[Path], presets: tuple[str, ...]) -> list[KeyRun]:
    runs = []
    for path in paths:
        config = load_json(path / "config.json")
        if not isinstance(config, dict):
            raise ValueError(f"missing config: {path}")
        task = str(config.get("task"))
        corruption = float(config.get("task_corruption_fraction", config.get("corruption", 0.0)))
        condition = next(
            (
                name
                for expected_task, expected_corruption, name in KEY_CONDITIONS
                if task == expected_task and abs(corruption - expected_corruption) < 1e-8
            ),
            None,
        )
        if condition is None:
            raise ValueError(f"unexpected priority run: {path}")
        runs.append(
            KeyRun(
                path=path,
                task=task,
                corruption=corruption,
                condition=condition,
                preset=str(config.get("preset")),
                seed=int(config.get("seed", -1)),
            )
        )
    observed = {(run.condition, run.preset, run.seed) for run in runs}
    expected = {
        (condition, preset, seed)
        for _, _, condition in KEY_CONDITIONS
        for preset in presets
        for seed in SEEDS
    }
    expected_count = len(KEY_CONDITIONS) * len(presets) * len(SEEDS)
    if len(runs) != expected_count or observed != expected:
        raise ValueError(
            f"expected exact {expected_count}-run priority matrix; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return sorted(runs, key=lambda run: (run.preset, run.condition, run.seed))


def _behavior_curve(run: KeyRun) -> tuple[np.ndarray, np.ndarray]:
    horizon = horizon_for(run)
    pairs = []
    try:
        lines = (run.path / "metrics.jsonl").read_text().splitlines()
    except OSError as error:
        raise ValueError(f"missing behavior data for {run.slug}") from error
    for line in lines:
        try:
            record = json.loads(line)
            step = int(record["step"])
            value = _finite(record.get("test_accuracy"))
            if step <= horizon and value is not None:
                pairs.append((step, value))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    deduplicated = {step: value for step, value in pairs}
    if horizon not in deduplicated:
        raise ValueError(f"missing {horizon} behavior endpoint for {run.slug}")
    steps = sorted(deduplicated)
    return np.asarray(steps, dtype=float), np.asarray([deduplicated[step] for step in steps])


def _operator_curve(run: KeyRun, key: str) -> tuple[np.ndarray, np.ndarray]:
    records = _records(operator_prefix(run).with_suffix(".json"))
    pairs = []
    for step in operator_steps_for(run):
        candidates = [
            record
            for record in records
            if str(record.get("view")) == "output"
            and int(record.get("step", -1)) == step
        ]
        if not candidates:
            raise ValueError(f"missing operator step {step} for {run.slug}")
        layer = max(int(record.get("layer", -1)) for record in candidates)
        value = _finite(
            next(
                record
                for record in reversed(candidates)
                if int(record.get("layer", -1)) == layer
            ).get(key)
        )
        if value is None:
            raise ValueError(f"missing {key} at {step} for {run.slug}")
        pairs.append((step, value / 1000.0 if key.endswith("_bits") else value))
    return (
        np.asarray([step for step, _ in pairs], dtype=float),
        np.asarray([value for _, value in pairs], dtype=float),
    )


def _causal_value(run: KeyRun, control: str = "learned_generator") -> float:
    step = horizon_for(run)
    records = _records(causal_prefix(CausalJob(run, step)).with_suffix(".json"))
    candidates = [
        record
        for record in records
        if str(record.get("control")) == control
        and str(record.get("position")) == "output"
        and int(record.get("step", -1)) == step
    ]
    if not candidates:
        raise ValueError(f"missing causal {control} at {step} for {run.slug}")
    layer = max(int(record.get("layer", -1)) for record in candidates)
    values = []
    for record in candidates:
        if int(record.get("layer", -1)) != layer:
            continue
        value = next(
            (
                candidate
                for key in ("qualified_desired_accuracy", "probability_recovery", "desired_accuracy")
                if (candidate := _finite(record.get(key))) is not None
            ),
            None,
        )
        if value is not None:
            values.append(value)
    if not values:
        raise ValueError(f"causal metric absent for {run.slug}")
    return float(np.median(values))


def _causal_curve(run: KeyRun) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([horizon_for(run)], dtype=float), np.asarray([_causal_value(run)])


def render_spaghetti(
    runs: list[KeyRun],
    output: Path,
    presets: tuple[str, ...],
) -> list[Path]:
    specs: tuple[tuple[str, str, Callable[[KeyRun], tuple[np.ndarray, np.ndarray]]], ...] = (
        ("held-out accuracy", "accuracy", _behavior_curve),
        ("generator error", "error", lambda run: _operator_curve(run, "joint_cv_error")),
        (
            "usable shared-rule gain",
            "kbit",
            lambda run: _operator_curve(run, "usable_reuse_gain_bits"),
        ),
        ("causal endpoint", "success", _causal_curve),
    )
    fig, axes = plt.subplots(
        len(presets),
        len(specs),
        figsize=(12.7, 5.25),
        constrained_layout=True,
        squeeze=False,
        sharex="col",
    )
    fig.patch.set_alpha(0)
    for row, preset in enumerate(presets):
        for column, (title, ylabel, curve_fn) in enumerate(specs):
            axis = axes[row, column]
            for _, _, condition in KEY_CONDITIONS:
                label, color = CONDITION_STYLE[condition]
                curves = []
                for run in [
                    item
                    for item in runs
                    if item.preset == preset and item.condition == condition
                ]:
                    xs, ys = curve_fn(run)
                    curves.append((xs, ys))
                    axis.plot(xs, ys, color=color, alpha=0.18, linewidth=0.9)
                    axis.scatter(xs, ys, color=color, alpha=0.18, s=7, linewidths=0)
                median_x, median_y = _median_curve(curves)
                axis.plot(median_x, median_y, color=color, linewidth=2.4, label=label)
                axis.scatter(median_x, median_y, color=color, s=16, linewidths=0, zorder=3)
            if row == 0:
                axis.set_title(title, fontweight="normal")
            axis.set_ylabel(f"{preset}\n{ylabel}" if column == 0 else ylabel)
            if row == len(presets) - 1:
                axis.set_xlabel("step")
            axis.set_xlim(-1_500, 61_500)
            _step_axis(axis)
            _style_axis(axis)
            if column in {0, 3}:
                axis.set_ylim(-0.03, 1.03)
                axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            if column == 2:
                axis.axhline(0, color=NORD["muted"], alpha=0.25, linewidth=0.7)
    handles = [
        Line2D([0], [0], color=color, linewidth=2.4, label=label)
        for label, color in CONDITION_STYLE.values()
    ]
    handles.extend(
        (
            Line2D(
                [0],
                [0],
                color=NORD["muted"],
                alpha=0.22,
                linewidth=0.9,
                marker="o",
                markersize=3,
                label="seeds",
            ),
            Line2D(
                [0],
                [0],
                color=NORD["ink"],
                linewidth=2.4,
                marker="o",
                markersize=4,
                label="median",
            ),
        )
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    return _save(fig, output / "priority-spaghetti")


def render_causal_controls(
    runs: list[KeyRun],
    output: Path,
    presets: tuple[str, ...],
) -> list[Path]:
    fig, axes = plt.subplots(
        len(presets),
        len(KEY_CONDITIONS),
        figsize=(9.2, 5.2),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_alpha(0)
    x = np.arange(len(CONTROL_ORDER))
    colors = [NORD["green"], NORD["frost_dark"], NORD["cyan"], NORD["orange"], NORD["purple"]]
    for row, preset in enumerate(presets):
        for column, (_, _, condition) in enumerate(KEY_CONDITIONS):
            axis = axes[row, column]
            seed_curves = []
            for run in [
                item
                for item in runs
                if item.preset == preset and item.condition == condition
            ]:
                values = np.asarray([_causal_value(run, control) for control in CONTROL_ORDER])
                seed_curves.append(values)
                axis.plot(x, values, color=NORD["muted"], alpha=0.18, linewidth=0.85)
                axis.scatter(x, values, color=colors, alpha=0.22, s=12, linewidths=0)
            median = np.median(np.stack(seed_curves), axis=0)
            axis.plot(x, median, color=NORD["ink"], linewidth=2.2)
            axis.scatter(x, median, color=colors, s=25, linewidths=0, zorder=3)
            if row == 0:
                axis.set_title(CONDITION_STYLE[condition][0], fontweight="normal")
            if column == 0:
                axis.set_ylabel(f"{preset}\nsuccess")
            axis.set_xticks(x, [CONTROL_LABEL[name] for name in CONTROL_ORDER], rotation=24)
            axis.set_ylim(-0.03, 1.03)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            _style_axis(axis)
    fig.legend(
        handles=[
            Line2D([0], [0], color=NORD["muted"], alpha=0.2, linewidth=0.85, label="seeds"),
            Line2D([0], [0], color=NORD["ink"], linewidth=2.2, label="median"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    return _save(fig, output / "priority-causal-controls")


def render(
    runs: list[KeyRun],
    output: Path,
    presets: tuple[str, ...],
) -> dict[str, object]:
    invalid_operator = [run.slug for run in runs if not operator_output_valid(run)]
    invalid_causal = [job.slug for job in causal_schedule(runs) if not causal_output_valid(job)]
    gates = {
        "behavior": all(int(_behavior_curve(run)[0][-1]) == horizon_for(run) for run in runs),
        "geometry": not invalid_operator,
        "usable_mdl": not invalid_operator,
        "causal": not invalid_causal,
    }
    if not all(gates.values()):
        raise ValueError(f"priority evidence incomplete: {gates}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [
        *render_spaghetti(runs, output, presets),
        *render_causal_controls(runs, output, presets),
    ]
    summary = []
    for preset in presets:
        for _, _, condition in KEY_CONDITIONS:
            selected = [run for run in runs if run.preset == preset and run.condition == condition]
            summary.append(
                {
                    "preset": preset,
                    "condition": condition,
                    "endpoint_step": HORIZONS[condition],
                    "seed_count": len(selected),
                    "test_accuracy_median": float(np.median([_behavior_curve(run)[1][-1] for run in selected])),
                    "generator_error_median": float(np.median([_operator_curve(run, "joint_cv_error")[1][-1] for run in selected])),
                    "usable_mdl_kbit_median": float(np.median([_operator_curve(run, "usable_reuse_gain_bits")[1][-1] for run in selected])),
                    "causal_success_median": float(np.median([_causal_value(run) for run in selected])),
                }
            )
    summary_path = output / "priority-endpoint-summary.json"
    atomic_json(summary_path, summary)
    artifacts.append(summary_path)
    manifest_path = output / "priority-render-manifest.json"
    manifest = {
        "status": "complete",
        "run_count": len(runs),
        "horizons": HORIZONS,
        "mixed_endpoints_explicit": True,
        "conditions": [condition for _, _, condition in KEY_CONDITIONS],
        "presets": list(presets),
        "seeds": list(SEEDS),
        "operator_steps": {
            condition: list(operator_steps_for(next(run for run in runs if run.condition == condition)))
            for condition in HORIZONS
        },
        "causal_schedule": {
            "clean": "60000",
            "corrupt15": "30000",
            "random": "30000",
            "patch_site": "output residual stream, final layer",
            "folds": 3,
            "controls": list(CONTROL_ORDER),
        },
        "gates": gates,
        "plot_summary": "faded seeds and bold pointwise medians; no intervals or smoothing",
        "artifacts": [str(path) for path in (*artifacts, manifest_path)],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _synthetic_run(
    root: Path,
    task: str,
    corruption: float,
    condition: str,
    preset: str,
    seed: int,
) -> KeyRun:
    run = KeyRun(root / f"{condition}-{preset}-s{seed}", task, corruption, condition, preset, seed)
    run.path.mkdir()
    atomic_json(
        run.path / "config.json",
        {
            "run_name": run.path.name,
            "task": task,
            "task_corruption_fraction": corruption,
            "preset": preset,
            "seed": seed,
        },
    )
    horizon = horizon_for(run)
    steps = sorted({0, 10_000, 30_000, horizon})
    (run.path / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"step": step, "test_accuracy": min(1.0, step / horizon + seed * 0.01)}) + "\n"
            for step in steps
        )
    )
    prefix = operator_prefix(run)
    atomic_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "run_name": run.path.name,
                "folds": 5,
                "projection_fit": "inductive_state_alias_fold",
            },
            "records": [
                {
                    "step": step,
                    "view": "output",
                    "layer": 1,
                    "joint_cv_error": 1.0 - step / 70_000,
                    "usable_reuse_gain_bits": step / 12,
                }
                for step in operator_steps_for(run)
            ],
        },
    )
    prefix.with_suffix(".jsonl").write_text("{}\n")
    prefix.with_suffix(".csv").write_text("step\n")
    job = CausalJob(run, horizon)
    prefix = causal_prefix(job)
    checkpoint = f"weights-{horizon:06d}.pt"
    atomic_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "run_name": run.path.name,
                "folds": 3,
                "checkpoints": [checkpoint],
                "patch_sites": [{"position": "output", "layer": 1}],
            },
            "records": [
                {
                    "step": horizon,
                    "checkpoint": checkpoint,
                    "fold": fold,
                    "position": "output",
                    "layer": 1,
                    "control": control,
                    "qualified_desired_accuracy": (
                        min(1.0, horizon / 60_000 + seed * 0.01)
                        if control == "learned_generator"
                        else 0.03
                    ),
                }
                for fold in range(3)
                for control in CONTROL_ORDER
            ],
        },
    )
    prefix.with_suffix(".jsonl").write_text("{}\n")
    prefix.with_suffix(".csv").write_text("step\n")
    return run


def self_test(
    output: Path | None,
    presets: tuple[str, ...],
) -> None:
    context = tempfile.TemporaryDirectory(prefix="render-priority-") if output is None else None
    root = Path(context.name) if context is not None else output
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    inputs.mkdir()
    runs = [
        _synthetic_run(inputs, task, corruption, condition, preset, seed)
        for task, corruption, condition in KEY_CONDITIONS
        for preset in presets
        for seed in SEEDS
    ]
    figures = root / "figures"
    manifest = render(runs, figures, presets)
    expected = {
        "priority-spaghetti.png",
        "priority-spaghetti.pdf",
        "priority-causal-controls.png",
        "priority-causal-controls.pdf",
        "priority-endpoint-summary.json",
        "priority-render-manifest.json",
    }
    if manifest.get("status") != "complete" or not expected.issubset({path.name for path in figures.iterdir()}):
        raise AssertionError("priority renderer missed an artifact")
    print(f"self-test passed: {figures}")
    if context is not None:
        context.cleanup()


def main() -> None:
    args = parse_args()
    presets = tuple(args.presets) if args.presets else PRESETS
    if len(presets) != 2 or len(set(presets)) != len(presets):
        raise ValueError("--preset must select exactly two distinct presets")
    if args.self_test:
        self_test(args.self_test_output, presets)
        return
    if not args.run or args.output is None:
        raise ValueError("--run and --output are required")
    render(_load_runs(args.run, presets), args.output, presets)


if __name__ == "__main__":
    main()
