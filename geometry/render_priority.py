from __future__ import annotations

import argparse
import hashlib
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
    MIN_QUALIFIED_CAUSAL_EXAMPLES,
    causal_evidence_metric,
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
    _records,
    _step_axis,
    _style_axis,
)


CONDITION_STYLE = {
    "clean": ("clean · 60k", NORD["frost_dark"]),
    "corrupt15": ("15% corrupted · 30k", NORD["orange"]),
    "random": ("random table · 30k", NORD["purple"]),
}
PRIORITY_CONTROL_LABEL = {
    **CONTROL_LABEL,
    "learned_generator": "canonical cycle",
    "exact_state_swap": "exact state",
    "target_centroid": "target centroid",
    "scrambled_successor": "scrambled",
    "random_orthogonal": "random orthogonal",
}
SUITE_PRESETS = {
    frozenset(("grok", "micro")): "core",
    frozenset(("small", "medium")): "scale",
    frozenset(("large",)): "large",
    frozenset(("small", "medium", "large")): "capacity",
}
PRESET_LABEL = {
    "grok": "grok · 128×1",
    "micro": "micro · 128×2",
    "small": "small · 256×4",
    "medium": "medium · 512×6",
    "large": "large · 768×8",
}
PRESET_DEPTH = {
    "grok": 1,
    "micro": 2,
    "small": 4,
    "medium": 6,
    "large": 8,
}
CAUSAL_SITES = (
    ("node", "node 0"),
    ("output", "output final"),
)


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


def _suite_for(presets: tuple[str, ...]) -> str:
    suite = SUITE_PRESETS.get(frozenset(presets))
    if suite is None:
        raise ValueError(
            "priority figures require core (grok,micro), scale (small,medium), "
            "large, or capacity (small,medium,large)"
        )
    return suite


def _pointwise_median(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if len(curves) != len(SEEDS):
        raise ValueError(
            f"expected {len(SEEDS)} seed trajectories, received {len(curves)}"
        )
    reference = curves[0][0]
    if any(
        xs.shape != reference.shape or not np.array_equal(xs, reference)
        for xs, _ in curves[1:]
    ):
        raise ValueError("seed trajectories do not share the same measured steps")
    values = np.stack([ys for _, ys in curves])
    if values.shape[1] != reference.shape[0] or not np.isfinite(values).all():
        raise ValueError("seed trajectories contain incomplete values")
    return reference.copy(), np.median(values, axis=0)


def _save_figure(
    fig: plt.Figure,
    base: Path,
    *,
    title: str,
    caption: str,
) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(
        png,
        dpi=240,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Title": title,
            "Description": caption,
            "Software": "geometry/render_priority.py",
        },
    )
    fig.savefig(
        pdf,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Title": title,
            "Subject": caption,
            "Keywords": (
                "activation geometry; pointwise median; mixed horizon; "
                "preregistered latent cycle"
            ),
            "Creator": "geometry/render_priority.py",
        },
    )
    plt.close(fig)
    return [png, pdf]


def _caption_payload(
    suite: str,
    presets: tuple[str, ...],
) -> dict[str, object]:
    trajectory_name = f"{suite}-trajectories"
    sites_name = f"{suite}-causal-sites"
    controls_name = f"{suite}-output-final-controls"
    common = (
        "The cross-condition probe is the preregistered latent-label cycle "
        "k → (k + 1) mod n; its orientation was fixed before activations were "
        "inspected, and arbitrary surface token IDs never choose it."
    )
    seed_policy = (
        "Faint lines show the three measured seeds and the bold line is their "
        "pointwise median; there is no confidence interval."
    )
    trajectory_policy = (
        f"{seed_policy} Straight segments only join measured evaluations or "
        "checkpoints, with no smoothing."
    )
    paired_policy = (
        f"{seed_policy} Straight segments connect paired measurements within "
        "each seed, with no smoothing."
    )
    causal_metric_policy = (
        "Each fold reports target accuracy on the qualified subset only when "
        f"that subset contains at least {MIN_QUALIFIED_CAUSAL_EXAMPLES} examples; "
        "otherwise it reports absolute target accuracy across all evaluated "
        "examples. Probability-recovery ratios are excluded because a near-zero "
        "natural-shift gain makes them unstable."
    )
    return {
        "suite": suite,
        "presets": list(presets),
        "probe": {
            "status": "preregistered",
            "successor_mode": "latent_label_plus_one",
            "definition": "latent label k maps to (k + 1) mod n",
            "orientation": "fixed before activation inspection",
        },
        "causal_metric": {
            "minimum_qualified_examples": MIN_QUALIFIED_CAUSAL_EXAMPLES,
            "qualified_metric": "qualified_desired_accuracy",
            "fallback_metric": "desired_accuracy",
            "probability_recovery_used": False,
            "definition": causal_metric_policy,
        },
        "mixed_horizons": {
            "clean": 60_000,
            "corrupt15": 30_000,
            "random": 30_000,
            "extrapolation": False,
        },
        "figures": {
            trajectory_name: {
                "title": f"{suite.capitalize()} behavior, geometry, and compression trajectories",
                "caption": (
                    f"{trajectory_policy} Clean runs end at 60k steps; 15%-corrupted "
                    "and random-table controls end at 30k, with no values "
                    "extended beyond those endpoints. Canonical-cycle error "
                    "and alias-held-out reuse gain are evaluated on the output "
                    "residual stream after the final block. The reuse code "
                    "holds out aliases, not a nested state-and-alias split, so "
                    f"it is the weaker compression diagnostic. {common}"
                ),
            },
            sites_name: {
                "title": f"{suite.capitalize()} canonical-cycle intervention sites",
                "caption": (
                    "Canonical-cycle intervention success at the input-state "
                    "activation before the first transformer block (node 0) "
                    "and at the output residual stream after the final block "
                    f"(output final). {causal_metric_policy} {paired_policy} {common}"
                ),
            },
            controls_name: {
                "title": f"{suite.capitalize()} output-final intervention controls",
                "caption": (
                    "Canonical-cycle and matched control interventions at the "
                    "output residual stream after the final transformer block "
                    "(output final). Exact-state and target-centroid controls "
                    "are label-informed references; scrambled-successor and "
                    "random-orthogonal controls test specificity. "
                    f"{causal_metric_policy} {paired_policy} {common}"
                ),
            },
        },
    }


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


def _causal_layer(run: KeyRun, position: str) -> int:
    if position == "node":
        return 0
    if position != "output":
        raise ValueError(f"unsupported causal position: {position}")
    config = load_json(run.path / "config.json")
    model = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model, dict):
        raise ValueError(f"missing model depth for {run.slug}")
    try:
        layer = int(model["depth"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid model depth for {run.slug}") from error
    if layer < 1:
        raise ValueError(f"invalid output-final layer for {run.slug}")
    return layer


def _causal_value(
    run: KeyRun,
    control: str = "learned_generator",
    *,
    position: str = "output",
) -> float:
    step = horizon_for(run)
    records = _records(causal_prefix(CausalJob(run, step)).with_suffix(".json"))
    layer = _causal_layer(run, position)
    candidates = [
        record
        for record in records
        if str(record.get("control")) == control
        and str(record.get("position")) == position
        and int(record.get("layer", -1)) == layer
        and int(record.get("step", -1)) == step
    ]
    if not candidates:
        raise ValueError(
            f"missing causal {control} at {position} layer {layer}, "
            f"step {step} for {run.slug}"
        )
    values: list[float] = []
    for record in candidates:
        try:
            value, _ = causal_evidence_metric(record)
        except ValueError as error:
            raise ValueError(
                f"invalid causal metric for {run.slug} at {position} "
                f"layer {layer}, fold {record.get('fold')}: {error}"
            ) from error
        values.append(value)
    return float(np.median(values))


def _causal_site_curve(run: KeyRun) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.arange(len(CAUSAL_SITES), dtype=float),
        np.asarray(
            [
                _causal_value(run, position=position)
                for position, _ in CAUSAL_SITES
            ],
            dtype=float,
        ),
    )


def render_trajectories(
    runs: list[KeyRun],
    output: Path,
    presets: tuple[str, ...],
    suite: str,
    captions: dict[str, object],
) -> list[Path]:
    specs: tuple[tuple[str, str, Callable[[KeyRun], tuple[np.ndarray, np.ndarray]]], ...] = (
        ("held-out accuracy", "accuracy", _behavior_curve),
        (
            "canonical-cycle error",
            "relative error",
            lambda run: _operator_curve(run, "joint_cv_error"),
        ),
        (
            "alias-held-out reuse gain",
            "gain (kbit)",
            lambda run: _operator_curve(run, "usable_reuse_gain_bits"),
        ),
    )
    fig, axes = plt.subplots(
        len(presets),
        len(specs),
        figsize=(10.1, 2.35 * len(presets) + 0.55),
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
                median_x, median_y = _pointwise_median(curves)
                axis.plot(median_x, median_y, color=color, linewidth=2.4, label=label)
                axis.scatter(median_x, median_y, color=color, s=16, linewidths=0, zorder=3)
            if row == 0:
                axis.set_title(title, fontweight="normal")
            axis.set_ylabel(
                f"{PRESET_LABEL[preset]}\n{ylabel}" if column == 0 else ylabel
            )
            if row == len(presets) - 1:
                axis.set_xlabel("step")
            axis.set_xlim(-1_500, 61_500)
            _step_axis(axis)
            _style_axis(axis)
            if column == 0:
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
    figure = captions["figures"][f"{suite}-trajectories"]
    return _save_figure(
        fig,
        output / f"{suite}-trajectories",
        title=str(figure["title"]),
        caption=str(figure["caption"]),
    )


def render_causal_sites(
    runs: list[KeyRun],
    output: Path,
    presets: tuple[str, ...],
    suite: str,
    captions: dict[str, object],
) -> list[Path]:
    fig, axes = plt.subplots(
        len(presets),
        len(KEY_CONDITIONS),
        figsize=(9.2, 2.35 * len(presets) + 0.55),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_alpha(0)
    x = np.arange(len(CAUSAL_SITES))
    for row, preset in enumerate(presets):
        for column, (_, _, condition) in enumerate(KEY_CONDITIONS):
            axis = axes[row, column]
            _, color = CONDITION_STYLE[condition]
            seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
            for run in [
                item
                for item in runs
                if item.preset == preset and item.condition == condition
            ]:
                xs, values = _causal_site_curve(run)
                seed_curves.append((xs, values))
                axis.plot(xs, values, color=color, alpha=0.18, linewidth=0.9)
                axis.scatter(
                    xs,
                    values,
                    color=color,
                    alpha=0.18,
                    s=12,
                    linewidths=0,
                )
            median_x, median = _pointwise_median(seed_curves)
            axis.plot(median_x, median, color=color, linewidth=2.4)
            axis.scatter(
                median_x,
                median,
                color=color,
                s=25,
                linewidths=0,
                zorder=3,
            )
            if row == 0:
                axis.set_title(CONDITION_STYLE[condition][0], fontweight="normal")
            if column == 0:
                axis.set_ylabel(f"{PRESET_LABEL[preset]}\nsuccess")
            axis.set_xticks(
                x,
                [label for _, label in CAUSAL_SITES],
            )
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
    figure = captions["figures"][f"{suite}-causal-sites"]
    return _save_figure(
        fig,
        output / f"{suite}-causal-sites",
        title=str(figure["title"]),
        caption=str(figure["caption"]),
    )


def render_causal_controls(
    runs: list[KeyRun],
    output: Path,
    presets: tuple[str, ...],
    suite: str,
    captions: dict[str, object],
) -> list[Path]:
    fig, axes = plt.subplots(
        len(presets),
        len(KEY_CONDITIONS),
        figsize=(9.2, 2.35 * len(presets) + 0.55),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_alpha(0)
    x = np.arange(len(CONTROL_ORDER))
    for row, preset in enumerate(presets):
        for column, (_, _, condition) in enumerate(KEY_CONDITIONS):
            axis = axes[row, column]
            _, color = CONDITION_STYLE[condition]
            seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
            for run in [
                item
                for item in runs
                if item.preset == preset and item.condition == condition
            ]:
                values = np.asarray(
                    [
                        _causal_value(run, control, position="output")
                        for control in CONTROL_ORDER
                    ]
                )
                seed_curves.append((x, values))
                axis.plot(x, values, color=color, alpha=0.18, linewidth=0.85)
                axis.scatter(
                    x,
                    values,
                    color=color,
                    alpha=0.18,
                    s=12,
                    linewidths=0,
                )
            median_x, median = _pointwise_median(seed_curves)
            axis.plot(median_x, median, color=color, linewidth=2.2)
            axis.scatter(
                median_x,
                median,
                color=color,
                s=25,
                linewidths=0,
                zorder=3,
            )
            if row == 0:
                axis.set_title(CONDITION_STYLE[condition][0], fontweight="normal")
            if column == 0:
                axis.set_ylabel(f"{PRESET_LABEL[preset]}\nsuccess")
            axis.set_xticks(
                x,
                [PRIORITY_CONTROL_LABEL[name] for name in CONTROL_ORDER],
                rotation=24,
            )
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
    )
    figure = captions["figures"][f"{suite}-output-final-controls"]
    return _save_figure(
        fig,
        output / f"{suite}-output-final-controls",
        title=str(figure["title"]),
        caption=str(figure["caption"]),
    )


def render(
    runs: list[KeyRun],
    output: Path,
    presets: tuple[str, ...],
) -> dict[str, object]:
    suite = _suite_for(presets)
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
    captions = _caption_payload(suite, presets)
    artifacts = [
        *render_trajectories(runs, output, presets, suite, captions),
        *render_causal_sites(runs, output, presets, suite, captions),
        *render_causal_controls(runs, output, presets, suite, captions),
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
                    "test_accuracy_median": float(
                        np.median(
                            [_behavior_curve(run)[1][-1] for run in selected]
                        )
                    ),
                    "canonical_cycle_error_median": float(
                        np.median(
                            [
                                _operator_curve(run, "joint_cv_error")[1][-1]
                                for run in selected
                            ]
                        )
                    ),
                    "alias_heldout_reuse_gain_kbit_median": float(
                        np.median(
                            [
                                _operator_curve(
                                    run,
                                    "usable_reuse_gain_bits",
                                )[1][-1]
                                for run in selected
                            ]
                        )
                    ),
                    "canonical_cycle_node0_causal_success_median": float(
                        np.median(
                            [
                                _causal_value(run, position="node")
                                for run in selected
                            ]
                        )
                    ),
                    "canonical_cycle_output_final_causal_success_median": float(
                        np.median(
                            [
                                _causal_value(run, position="output")
                                for run in selected
                            ]
                        )
                    ),
                }
            )
    summary_path = output / f"{suite}-endpoint-summary.json"
    atomic_json(summary_path, summary)
    artifacts.append(summary_path)
    captions_path = output / f"{suite}-figure-captions.json"
    atomic_json(captions_path, captions)
    artifacts.append(captions_path)
    manifest_path = output / "priority-render-manifest.json"
    manifest = {
        "status": "complete",
        "suite": suite,
        "run_count": len(runs),
        "horizons": HORIZONS,
        "mixed_endpoints_explicit": True,
        "horizon_extrapolation": False,
        "conditions": [condition for _, _, condition in KEY_CONDITIONS],
        "presets": list(presets),
        "preset_labels": {preset: PRESET_LABEL[preset] for preset in presets},
        "seeds": list(SEEDS),
        "operator_steps": {
            condition: list(
                operator_steps_for(
                    next(run for run in runs if run.condition == condition)
                )
            )
            for condition in HORIZONS
        },
        "operator_probe": {
            "successor_mode": "latent_label_plus_one",
            "registration": "preregistered before activation inspection",
            "view": "output",
            "layer": "model.depth",
            "projection_fit": "inductive_state_alias_fold",
            "reuse_gain_scope": (
                "lookup-vs-shared code scored on held-out aliases only; "
                "not nested state-and-alias MDL"
            ),
        },
        "causal_schedule": {
            "clean": "60000",
            "corrupt15": "30000",
            "random": "30000",
            "probe": "preregistered canonical latent-label k -> k + 1 cycle",
            "successor_mode": "latent_label_plus_one",
            "patch_sites": {
                "node0": {
                    "position": "node",
                    "layer": 0,
                    "description": "input-state activation before the first block",
                },
                "output_final": {
                    "position": "output",
                    "layer": "model.depth",
                    "description": "output residual stream after the final block",
                },
            },
            "folds": 3,
            "controls": list(CONTROL_ORDER),
            "metric_policy": {
                "minimum_qualified_examples": MIN_QUALIFIED_CAUSAL_EXAMPLES,
                "qualified_metric": "qualified_desired_accuracy",
                "fallback_metric": "desired_accuracy",
                "probability_recovery_used": False,
            },
        },
        "gates": gates,
        "render_policy": {
            "seed_trajectories": "faint",
            "aggregate": "bold pointwise median",
            "confidence_intervals": False,
            "smoothing": False,
            "segments": "connect measured points only",
        },
        "captions": captions["figures"],
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
    depth = PRESET_DEPTH[preset]
    order = 17
    successor = (np.arange(order, dtype=np.int64) + 1) % order
    successor_metadata = {
        "successor_mode": "latent_label_plus_one",
        "successor_preregistered": True,
        "successor_vector": successor.tolist(),
        "successor_sha256": hashlib.sha256(
            np.asarray(successor, dtype="<i8").tobytes()
        ).hexdigest(),
        "generator_relation": None,
    }
    atomic_json(
        run.path / "config.json",
        {
            "run_name": run.path.name,
            "task": task,
            "task_corruption_fraction": corruption,
            "task_order": order,
            "preset": preset,
            "seed": seed,
            "model": {"depth": depth},
        },
    )
    horizon = horizon_for(run)
    steps = sorted({0, 10_000, 30_000, horizon})
    behavior_endpoint = {
        "clean": 0.94,
        "corrupt15": 0.62,
        "random": 0.08,
    }[condition]
    (run.path / "metrics.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "step": step,
                    "test_accuracy": float(
                        np.clip(
                            behavior_endpoint * (step / horizon) ** 1.35
                            + (seed - 1) * 0.015 * (step / max(horizon, 1)),
                            0.0,
                            1.0,
                        )
                    ),
                }
            )
            + "\n"
            for step in steps
        )
    )
    prefix = operator_prefix(run)
    error_endpoint = {
        "clean": 0.18,
        "corrupt15": 0.55,
        "random": 0.92,
    }[condition]
    gain_endpoint = {
        "clean": 5_200.0,
        "corrupt15": 550.0,
        "random": -450.0,
    }[condition]
    atomic_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "run_name": run.path.name,
                "folds": 5,
                "projection_fit": "inductive_state_alias_fold",
                **successor_metadata,
            },
            "records": [
                {
                    "step": step,
                    "view": "output",
                    "layer": depth,
                    "joint_cv_error": float(
                        0.96
                        + (error_endpoint - 0.96) * (step / horizon)
                        + (seed - 1) * 0.015
                    ),
                    "usable_reuse_gain_bits": float(
                        gain_endpoint * (step / horizon)
                        + (seed - 1) * 180.0
                    ),
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
    sites = (("node", 0), ("output", depth))
    canonical_base = {
        "clean": {"node": 0.71, "output": 0.88},
        "corrupt15": {"node": 0.31, "output": 0.44},
        "random": {"node": 0.04, "output": 0.05},
    }[condition]
    control_base = {
        "exact_state_swap": 0.93,
        "target_centroid": 0.84,
        "scrambled_successor": 0.04,
        "random_orthogonal": 0.03,
    }
    qualified_examples = 0 if condition == "random" else 128
    atomic_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "run_name": run.path.name,
                "folds": 3,
                "checkpoints": [checkpoint],
                "patch_sites": [
                    {"position": position, "layer": layer}
                    for position, layer in sites
                ],
                "successor_mode": "latent_label_plus_one",
                **successor_metadata,
            },
            "records": [
                {
                    "step": horizon,
                    "checkpoint": checkpoint,
                    "fold": fold,
                    "position": position,
                    "layer": layer,
                    "control": control,
                    "qualified_examples": qualified_examples,
                    "qualified_desired_accuracy": (
                        None
                        if condition == "random"
                        else float(
                            np.clip(
                                (
                                    canonical_base[position]
                                    if control == "learned_generator"
                                    else control_base[control]
                                )
                                + (seed - 1) * 0.02
                                + (fold - 1) * 0.006,
                                0.0,
                                1.0,
                            )
                        )
                    ),
                    "desired_accuracy": float(
                        np.clip(
                            (
                                canonical_base[position]
                                if control == "learned_generator"
                                else control_base[control]
                            )
                            + (seed - 1) * 0.02
                            + (fold - 1) * 0.006,
                            0.0,
                            1.0,
                        )
                    ),
                    "probability_recovery": 4.0 if condition == "random" else 0.5,
                }
                for fold in range(3)
                for position, layer in sites
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
    suite = _suite_for(presets)
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
        f"{suite}-trajectories.png",
        f"{suite}-trajectories.pdf",
        f"{suite}-causal-sites.png",
        f"{suite}-causal-sites.pdf",
        f"{suite}-output-final-controls.png",
        f"{suite}-output-final-controls.pdf",
        f"{suite}-endpoint-summary.json",
        f"{suite}-figure-captions.json",
        "priority-render-manifest.json",
    }
    observed = {path.name for path in figures.iterdir()}
    if (
        manifest.get("status") != "complete"
        or manifest.get("suite") != suite
        or not expected.issubset(observed)
    ):
        raise AssertionError("priority renderer missed an artifact")
    if manifest.get("horizon_extrapolation") is not False:
        raise AssertionError("mixed-horizon renderer permits extrapolation")
    policy = manifest.get("render_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("aggregate") != "bold pointwise median"
        or policy.get("confidence_intervals") is not False
        or policy.get("smoothing") is not False
    ):
        raise AssertionError("spaghetti render policy is not explicit")
    causal_policy = manifest.get("causal_schedule", {}).get("metric_policy")
    if (
        not isinstance(causal_policy, dict)
        or causal_policy.get("minimum_qualified_examples")
        != MIN_QUALIFIED_CAUSAL_EXAMPLES
        or causal_policy.get("fallback_metric") != "desired_accuracy"
        or causal_policy.get("probability_recovery_used") is not False
    ):
        raise AssertionError("causal evidence policy is not explicit")
    captions = load_json(figures / f"{suite}-figure-captions.json")
    if (
        not isinstance(captions, dict)
        or captions.get("probe", {}).get("status") != "preregistered"
        or "node 0" not in json.dumps(captions)
        or "output final" not in json.dumps(captions)
    ):
        raise AssertionError("figure captions omit probe or intervention sites")
    summary = load_json(figures / f"{suite}-endpoint-summary.json")
    if (
        not isinstance(summary, list)
        or {int(record["endpoint_step"]) for record in summary} != {30_000, 60_000}
        or any(
            "canonical_cycle_node0_causal_success_median" not in record
            or "canonical_cycle_output_final_causal_success_median" not in record
            or "alias_heldout_reuse_gain_kbit_median" not in record
            or "usable_mdl_kbit_median" in record
            for record in summary
        )
    ):
        raise AssertionError(
            "endpoint summary obscures horizons, causal sites, or reuse scope"
        )
    random_records = [
        record
        for record in summary
        if isinstance(record, dict) and record.get("condition") == "random"
    ]
    if any(
        not 0.0
        <= float(record["canonical_cycle_output_final_causal_success_median"])
        <= 1.0
        for record in random_records
    ):
        raise AssertionError("zero-support causal controls used an unstable ratio")
    operator_probe = manifest.get("operator_probe")
    if (
        not isinstance(operator_probe, dict)
        or "held-out aliases only" not in str(operator_probe.get("reuse_gain_scope"))
    ):
        raise AssertionError("renderer overstates the alias-held-out reuse code")
    from PIL import Image

    with Image.open(figures / f"{suite}-trajectories.png") as image:
        if "preregistered latent-label cycle" not in image.info.get("Description", ""):
            raise AssertionError("PNG metadata omitted the figure caption")
    try:
        _pointwise_median(
            [
                (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
                (np.asarray([0.0]), np.asarray([0.0])),
                (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
            ]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("pointwise median accepted mismatched seed grids")
    print(f"self-test passed: {figures}")
    if context is not None:
        context.cleanup()


def main() -> None:
    args = parse_args()
    presets = tuple(args.presets) if args.presets else PRESETS
    if len(presets) != len(set(presets)):
        raise ValueError("--preset values must be distinct")
    _suite_for(presets)
    if args.self_test:
        self_test(args.self_test_output, presets)
        return
    if not args.run or args.output is None:
        raise ValueError("--run and --output are required")
    render(_load_runs(args.run, presets), args.output, presets)


if __name__ == "__main__":
    main()
