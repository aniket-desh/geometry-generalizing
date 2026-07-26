from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

from key60_common import (
    FINAL_STEP,
    KEY_CONDITIONS,
    KEY_COUNT,
    OPERATOR_STEPS,
    PRESETS,
    SEEDS,
    CausalJob,
    KeyRun,
    atomic_json,
    causal_output_valid,
    causal_prefix,
    causal_schedule,
    load_json,
    operator_output_valid,
    operator_prefix,
)


NORD = {
    "ink": "#2E3440",
    "muted": "#4C566A",
    "frost_dark": "#5E81AC",
    "frost": "#81A1C1",
    "cyan": "#88C0D0",
    "red": "#BF616A",
    "orange": "#D08770",
    "yellow": "#EBCB8B",
    "green": "#A3BE8C",
    "purple": "#B48EAD",
}
CONDITION_STYLE = {
    "clean": ("clean", NORD["frost_dark"]),
    "corrupt15": ("15% corrupted", NORD["orange"]),
    "random": ("random table", NORD["purple"]),
}
CONTROL_ORDER = (
    "learned_generator",
    "exact_state_swap",
    "target_centroid",
    "scrambled_successor",
    "random_orthogonal",
)
CONTROL_LABEL = {
    "learned_generator": "learned",
    "exact_state_swap": "exact",
    "target_centroid": "target",
    "scrambled_successor": "scrambled",
    "random_orthogonal": "random",
}


def _font_family() -> str:
    for candidate in ("Alegreya", "Vollkorn"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            return candidate
        except ValueError:
            continue
    return "DejaVu Serif"


plt.rcParams.update(
    {
        "font.family": _font_family(),
        "font.size": 9.0,
        "axes.labelcolor": NORD["ink"],
        "axes.edgecolor": NORD["muted"],
        "xtick.color": NORD["muted"],
        "ytick.color": NORD["muted"],
        "text.color": NORD["ink"],
        "svg.fonttype": "none",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the strict key60 matrix as Nord spaghetti plots."
    )
    parser.add_argument("--run", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    return parser.parse_args()


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _records(path: Path) -> list[dict[str, object]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [record for record in payload["records"] if isinstance(record, dict)]
    return []


def _load_runs(paths: Iterable[Path]) -> list[KeyRun]:
    runs: list[KeyRun] = []
    for path in paths:
        config = load_json(path / "config.json")
        if not isinstance(config, dict):
            raise ValueError(f"missing config: {path}")
        task = str(config.get("task"))
        corruption = float(
            config.get(
                "task_corruption_fraction",
                config.get("corruption", 0.0),
            )
        )
        condition = None
        for expected_task, expected_corruption, name in KEY_CONDITIONS:
            if task == expected_task and abs(corruption - expected_corruption) < 1e-8:
                condition = name
                break
        if condition is None:
            raise ValueError(f"unexpected key60 run: {path}")
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
        for preset in PRESETS
        for seed in SEEDS
    }
    if len(runs) != KEY_COUNT or len(observed) != len(runs) or observed != expected:
        raise ValueError(
            f"expected the exact {KEY_COUNT}-run key60 matrix; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return sorted(runs, key=lambda run: (run.preset, run.condition, run.seed))


def _behavior_curve(run: KeyRun) -> tuple[np.ndarray, np.ndarray]:
    records = []
    try:
        for line in (run.path / "metrics.jsonl").read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid behavior data for {run.slug}") from error
    pairs = [
        (int(record["step"]), _finite(record.get("test_accuracy")))
        for record in records
        if "step" in record and int(record["step"]) <= FINAL_STEP
    ]
    pairs = [(step, value) for step, value in pairs if value is not None]
    if not pairs or pairs[-1][0] != FINAL_STEP:
        raise ValueError(f"missing final behavior for {run.slug}")
    return (
        np.asarray([step for step, _ in pairs], dtype=float),
        np.asarray([value for _, value in pairs], dtype=float),
    )


def _operator_curve(
    run: KeyRun,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    records = _records(operator_prefix(run).with_suffix(".json"))
    candidates = [
        record
        for record in records
        if str(record.get("view")) == "output"
        and int(record.get("step", -1)) in OPERATOR_STEPS
        and _finite(record.get("layer")) is not None
    ]
    pairs: list[tuple[int, float]] = []
    for step in OPERATOR_STEPS:
        at_step = [record for record in candidates if int(record["step"]) == step]
        if not at_step:
            raise ValueError(f"missing operator step {step} for {run.slug}")
        layer = max(int(record["layer"]) for record in at_step)
        selected = [record for record in at_step if int(record["layer"]) == layer]
        value = _finite(selected[-1].get(key))
        if value is None:
            raise ValueError(f"missing {key} at {step} for {run.slug}")
        pairs.append((step, value / 1000.0 if key.endswith("_bits") else value))
    return (
        np.asarray([step for step, _ in pairs], dtype=float),
        np.asarray([value for _, value in pairs], dtype=float),
    )


def _causal_value(
    run: KeyRun,
    step: int,
    *,
    control: str = "learned_generator",
) -> float:
    job = CausalJob(run, step)
    records = _records(causal_prefix(job).with_suffix(".json"))
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
    values: list[float] = []
    for record in candidates:
        if int(record.get("layer", -1)) != layer:
            continue
        value = None
        for key in (
            "qualified_desired_accuracy",
            "probability_recovery",
            "desired_accuracy",
        ):
            value = _finite(record.get(key))
            if value is not None:
                break
        if value is not None:
            values.append(value)
    if not values:
        raise ValueError(f"causal metric absent for {run.slug} at {step}")
    return float(np.median(values))


def _causal_curve(run: KeyRun) -> tuple[np.ndarray, np.ndarray]:
    steps = [FINAL_STEP]
    if run.preset == "grok" and run.seed == 0:
        steps = [10_000, 30_000, FINAL_STEP]
    return (
        np.asarray(steps, dtype=float),
        np.asarray([_causal_value(run, step) for step in steps], dtype=float),
    )


def _median_curve(
    curves: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    values: dict[float, list[float]] = defaultdict(list)
    for xs, ys in curves:
        for x, y in zip(xs, ys):
            if math.isfinite(float(x)) and math.isfinite(float(y)):
                values[float(x)].append(float(y))
    xs = np.asarray(sorted(values), dtype=float)
    ys = np.asarray([np.median(values[x]) for x in xs], dtype=float)
    return xs, ys


def _style_axis(axis: plt.Axes) -> None:
    axis.patch.set_alpha(0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(NORD["muted"])
    axis.tick_params(length=3, width=0.7, labelsize=8)
    axis.grid(False)


def _step_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"
            )
        )
    )


def _save(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(
        png,
        dpi=240,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        pdf,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    return [png, pdf]


def render_spaghetti(runs: list[KeyRun], output: Path) -> list[Path]:
    specs: tuple[
        tuple[str, str, Callable[[KeyRun], tuple[np.ndarray, np.ndarray]]],
        ...,
    ] = (
        ("held-out accuracy", "accuracy", _behavior_curve),
        (
            "generator error",
            "error",
            lambda run: _operator_curve(run, "joint_cv_error"),
        ),
        (
            "usable shared-rule gain",
            "kbit",
            lambda run: _operator_curve(run, "usable_reuse_gain_bits"),
        ),
        ("causal shift success", "success", _causal_curve),
    )
    fig, axes = plt.subplots(
        len(PRESETS),
        len(specs),
        figsize=(12.7, 5.25),
        constrained_layout=True,
        squeeze=False,
        sharex="col",
    )
    fig.patch.set_alpha(0)
    for row, preset in enumerate(PRESETS):
        for column, (title, ylabel, curve_fn) in enumerate(specs):
            axis = axes[row, column]
            for _, _, condition in KEY_CONDITIONS:
                label, color = CONDITION_STYLE[condition]
                condition_runs = [
                    run
                    for run in runs
                    if run.preset == preset and run.condition == condition
                ]
                curves = []
                for run in condition_runs:
                    xs, ys = curve_fn(run)
                    curves.append((xs, ys))
                    axis.plot(
                        xs,
                        ys,
                        color=color,
                        alpha=0.18,
                        linewidth=0.9,
                    )
                    axis.scatter(
                        xs,
                        ys,
                        color=color,
                        alpha=0.18,
                        s=7,
                        linewidths=0,
                    )
                median_x, median_y = _median_curve(curves)
                axis.plot(
                    median_x,
                    median_y,
                    color=color,
                    linewidth=2.4,
                    label=label,
                )
                axis.scatter(
                    median_x,
                    median_y,
                    color=color,
                    s=16,
                    linewidths=0,
                    zorder=3,
                )
            if row == 0:
                axis.set_title(title, fontweight="normal")
            if column == 0:
                axis.set_ylabel(f"{preset}\n{ylabel}")
            else:
                axis.set_ylabel(ylabel)
            if row == len(PRESETS) - 1:
                axis.set_xlabel("step")
            _step_axis(axis)
            _style_axis(axis)
            if column in {0, 3}:
                axis.set_ylim(-0.03, 1.03)
                axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            if column == 2:
                axis.axhline(
                    0,
                    color=NORD["muted"],
                    alpha=0.25,
                    linewidth=0.7,
                )
    condition_handles = [
        Line2D([0], [0], color=color, linewidth=2.4, label=label)
        for label, color in CONDITION_STYLE.values()
    ]
    summary_handles = [
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
    ]
    fig.legend(
        handles=condition_handles + summary_handles,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        handlelength=2.2,
    )
    return _save(fig, output / "key60-spaghetti")


def render_causal_controls(runs: list[KeyRun], output: Path) -> list[Path]:
    fig, axes = plt.subplots(
        len(PRESETS),
        len(KEY_CONDITIONS),
        figsize=(9.2, 5.2),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_alpha(0)
    x = np.arange(len(CONTROL_ORDER))
    control_colors = [
        NORD["green"],
        NORD["frost_dark"],
        NORD["cyan"],
        NORD["orange"],
        NORD["purple"],
    ]
    for row, preset in enumerate(PRESETS):
        for column, (_, _, condition) in enumerate(KEY_CONDITIONS):
            axis = axes[row, column]
            condition_runs = [
                run
                for run in runs
                if run.preset == preset and run.condition == condition
            ]
            seed_curves = []
            for run in condition_runs:
                values = np.asarray(
                    [
                        _causal_value(run, FINAL_STEP, control=control)
                        for control in CONTROL_ORDER
                    ]
                )
                seed_curves.append(values)
                axis.plot(
                    x,
                    values,
                    color=NORD["muted"],
                    alpha=0.18,
                    linewidth=0.85,
                )
                axis.scatter(
                    x,
                    values,
                    color=control_colors,
                    alpha=0.22,
                    s=12,
                    linewidths=0,
                )
            medians = np.median(np.stack(seed_curves), axis=0)
            axis.plot(x, medians, color=NORD["ink"], linewidth=2.2)
            axis.scatter(
                x,
                medians,
                color=control_colors,
                s=25,
                linewidths=0,
                zorder=3,
            )
            if row == 0:
                axis.set_title(CONDITION_STYLE[condition][0], fontweight="normal")
            if column == 0:
                axis.set_ylabel(f"{preset}\nsuccess")
            axis.set_xticks(x, [CONTROL_LABEL[name] for name in CONTROL_ORDER])
            axis.tick_params(axis="x", rotation=24)
            axis.set_ylim(-0.03, 1.03)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            _style_axis(axis)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=NORD["muted"],
                alpha=0.2,
                linewidth=0.85,
                label="seeds",
            ),
            Line2D(
                [0],
                [0],
                color=NORD["ink"],
                linewidth=2.2,
                label="median",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handlelength=2.2,
    )
    return _save(fig, output / "key60-causal-controls")


def render(runs: list[KeyRun], output: Path) -> dict[str, object]:
    invalid_operator = [run.slug for run in runs if not operator_output_valid(run)]
    invalid_causal = [
        job.slug for job in causal_schedule(runs) if not causal_output_valid(job)
    ]
    gates = {
        "behavior": all(_behavior_curve(run)[0][-1] == FINAL_STEP for run in runs),
        "geometry": not invalid_operator,
        "usable_mdl": not invalid_operator,
        "causal": not invalid_causal,
    }
    if not all(gates.values()):
        raise ValueError(
            f"key60 evidence incomplete; gates={gates}, "
            f"operator={invalid_operator}, causal={invalid_causal}"
        )
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [
        *render_spaghetti(runs, output),
        *render_causal_controls(runs, output),
    ]
    endpoint_summary: list[dict[str, object]] = []
    for preset in PRESETS:
        for _, _, condition in KEY_CONDITIONS:
            selected = [
                run
                for run in runs
                if run.preset == preset and run.condition == condition
            ]
            endpoint_summary.append(
                {
                    "preset": preset,
                    "condition": condition,
                    "seed_count": len(selected),
                    "test_accuracy_median": float(
                        np.median([_behavior_curve(run)[1][-1] for run in selected])
                    ),
                    "generator_error_median": float(
                        np.median(
                            [
                                _operator_curve(run, "joint_cv_error")[1][-1]
                                for run in selected
                            ]
                        )
                    ),
                    "usable_mdl_kbit_median": float(
                        np.median(
                            [
                                _operator_curve(
                                    run,
                                    "usable_reuse_gain_bits",
                                )[
                                    1
                                ][-1]
                                for run in selected
                            ]
                        )
                    ),
                    "causal_success_median": float(
                        np.median([_causal_value(run, FINAL_STEP) for run in selected])
                    ),
                }
            )
    summary_path = output / "key60-endpoint-summary.json"
    atomic_json(summary_path, endpoint_summary)
    artifacts.append(summary_path)
    manifest = {
        "status": "complete",
        "run_count": len(runs),
        "final_step": FINAL_STEP,
        "conditions": [condition for _, _, condition in KEY_CONDITIONS],
        "presets": list(PRESETS),
        "seeds": list(SEEDS),
        "operator_steps": list(OPERATOR_STEPS),
        "causal_schedule": {
            "endpoint": "all 18 runs at 60000",
            "pre_post": ("grok seed 0, all three conditions, at 10000 and 30000"),
            "patch_site": "output residual stream, final layer",
            "folds": 3,
            "controls": list(CONTROL_ORDER),
        },
        "gates": gates,
        "plot_summary": (
            "faded individual seeds with pointwise bold medians; "
            "no confidence intervals, interpolation, or smoothing"
        ),
        "artifacts": [str(path) for path in artifacts],
    }
    manifest_path = output / "key60-render-manifest.json"
    manifest["artifacts"].append(str(manifest_path))
    atomic_json(manifest_path, manifest)
    return manifest


def _synthetic_run(
    root: Path,
    *,
    task: str,
    corruption: float,
    condition: str,
    preset: str,
    seed: int,
) -> KeyRun:
    run_dir = root / f"{condition}-{preset}-s{seed}"
    run_dir.mkdir()
    atomic_json(
        run_dir / "config.json",
        {
            "run_name": run_dir.name,
            "task": task,
            "task_corruption_fraction": corruption,
            "preset": preset,
            "seed": seed,
        },
    )
    metric_lines = [
        json.dumps(
            {
                "step": step,
                "test_accuracy": min(1.0, step / 60_000 + seed * 0.01),
            }
        )
        for step in (0, 10_000, 30_000, 60_000)
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(metric_lines) + "\n")
    operator_records = [
        {
            "step": step,
            "checkpoint": f"weights-{step:06d}.pt",
            "view": "output",
            "layer": 1,
            "joint_cv_error": max(0.02, 1.0 - step / 65_000),
            "usable_reuse_gain_bits": step / 12,
        }
        for step in OPERATOR_STEPS
    ]
    prefix = operator_prefix(KeyRun(run_dir, task, corruption, condition, preset, seed))
    atomic_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {"run_name": run_dir.name, "folds": 5},
            "records": operator_records,
        },
    )
    prefix.with_suffix(".jsonl").write_text("{}\n")
    prefix.with_suffix(".csv").write_text("step\n60000\n")
    return KeyRun(run_dir, task, corruption, condition, preset, seed)


def _write_synthetic_causal(job: CausalJob) -> None:
    prefix = causal_prefix(job)
    checkpoint = f"weights-{job.step:06d}.pt"
    records = []
    for fold in range(3):
        for control in CONTROL_ORDER:
            value = (
                min(1.0, job.step / FINAL_STEP)
                if control == "learned_generator"
                else 0.03
            )
            records.append(
                {
                    "step": job.step,
                    "checkpoint": checkpoint,
                    "fold": fold,
                    "position": "output",
                    "layer": 1,
                    "control": control,
                    "qualified_desired_accuracy": value,
                }
            )
    atomic_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "run_name": job.run.path.name,
                "folds": 3,
                "checkpoints": [checkpoint],
                "patch_sites": [{"position": "output", "layer": 1}],
            },
            "records": records,
        },
    )
    prefix.with_suffix(".jsonl").write_text("{}\n")
    prefix.with_suffix(".csv").write_text("step\n60000\n")


def self_test(output: Path | None) -> None:
    import tempfile

    context = (
        tempfile.TemporaryDirectory(prefix="render-key60-") if output is None else None
    )
    root = Path(context.name) if context is not None else output
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    inputs.mkdir()
    runs = [
        _synthetic_run(
            inputs,
            task=task,
            corruption=corruption,
            condition=condition,
            preset=preset,
            seed=seed,
        )
        for task, corruption, condition in KEY_CONDITIONS
        for preset in PRESETS
        for seed in SEEDS
    ]
    for job in causal_schedule(runs):
        _write_synthetic_causal(job)
    figures = root / "figures"
    manifest = render(runs, figures)
    expected = {
        "key60-spaghetti.png",
        "key60-spaghetti.pdf",
        "key60-causal-controls.png",
        "key60-causal-controls.pdf",
        "key60-endpoint-summary.json",
        "key60-render-manifest.json",
    }
    observed = {path.name for path in figures.iterdir()}
    if manifest.get("status") != "complete" or not expected.issubset(observed):
        raise AssertionError("key60 render self-test did not emit every artifact")
    print(f"self-test passed: {figures}")
    if context is not None:
        context.cleanup()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.self_test_output)
        return
    if not args.run or args.output is None:
        raise ValueError("--run and --output are required")
    runs = _load_runs(args.run)
    render(runs, args.output)


if __name__ == "__main__":
    main()
