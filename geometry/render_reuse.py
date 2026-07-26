from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter


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
SERIES = (
    NORD["frost_dark"],
    NORD["cyan"],
    NORD["green"],
    NORD["yellow"],
    NORD["orange"],
    NORD["red"],
    NORD["purple"],
)


def _font_family() -> str:
    for candidate in ("Alegreya", "Vollkorn"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            return candidate
        except ValueError:
            pass
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


@dataclass
class RunData:
    path: Path
    config: dict[str, object]
    metrics: list[dict[str, object]]
    operator: list[dict[str, object]] = field(default_factory=list)
    causal: list[dict[str, object]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.config.get("run_name", self.path.name))

    @property
    def seed(self) -> int:
        return int(self.config.get("seed", 0))

    @property
    def preset(self) -> str:
        return str(self.config.get("preset", ""))

    @property
    def task(self) -> str:
        return str(self.config.get("task", ""))

    @property
    def corruption(self) -> float:
        return float(
            self.config.get(
                "task_corruption_fraction",
                self.config.get("corruption", 0.0),
            )
        )

    @property
    def condition(self) -> str:
        family = str(self.config.get("task_family", ""))
        if family == "random" or self.task.startswith("random"):
            return "random"
        if self.corruption <= 0:
            return "clean"
        return "corrupted"

    @property
    def activation_paths(self) -> list[Path]:
        return sorted(self.path.glob("activations-*.npz"), key=_step_from_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the reusable-geometry experiment suite."
    )
    parser.add_argument(
        "--results",
        type=Path,
        action="append",
        help="Result root or run directory; repeat for multiple roots.",
    )
    parser.add_argument(
        "--output", type=Path, help="Directory for figures and the manifest."
    )
    parser.add_argument("--task", default="cycle113", help="Task name, or all.")
    parser.add_argument("--preset", default="grok", help="Model preset, or all.")
    parser.add_argument(
        "--main-condition",
        default="clean",
        help="Condition used for checkpoint-aligned traces.",
    )
    parser.add_argument(
        "--operator-view",
        choices=("node", "output"),
        default="node",
        help="Activation view for operator metrics.",
    )
    parser.add_argument(
        "--operator-layer",
        default="last",
        help="Operator layer index, or last.",
    )
    parser.add_argument(
        "--causal-position", choices=("node", "output"), default="output"
    )
    parser.add_argument(
        "--causal-layer", default="last", help="Causal layer index, or last."
    )
    parser.add_argument(
        "--animate-run",
        default="auto",
        help="Run directory/name substring, auto, or none.",
    )
    parser.add_argument(
        "--animation-view", choices=("node", "output"), default="node"
    )
    parser.add_argument("--animation-layer", default="last")
    parser.add_argument("--max-animation-frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Render a synthetic suite and verify every expected artifact.",
    )
    parser.add_argument(
        "--self-test-output",
        type=Path,
        help="Keep synthetic self-test inputs and figures in this directory.",
    )
    return parser.parse_args()


def _step_from_path(path: Path) -> int:
    try:
        return int(path.stem.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        if path.suffix == ".csv":
            with path.open(newline="") as handle:
                return list(csv.DictReader(handle))
        if path.suffix == ".jsonl":
            records = []
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
            return records
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, csv.Error):
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    return []


def _analysis_payload(path: Path) -> tuple[str | None, list[dict[str, object]]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, []
    if not isinstance(payload, dict):
        return None, []
    metadata = payload.get("metadata", {})
    run_name = metadata.get("run_name") if isinstance(metadata, dict) else None
    records = payload.get("records", [])
    return (
        str(run_name) if run_name is not None else None,
        records if isinstance(records, list) else [],
    )


def discover_runs(roots: Iterable[Path]) -> list[RunData]:
    runs: list[RunData] = []
    seen_configs: set[Path] = set()
    for root in roots:
        candidates = [root / "config.json"] if root.is_dir() else []
        if root.is_dir():
            candidates.extend(root.rglob("config.json"))
        for config_path in sorted(candidates):
            resolved = config_path.resolve()
            if resolved in seen_configs or not config_path.exists():
                continue
            seen_configs.add(resolved)
            run_dir = config_path.parent
            try:
                config = json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            run = RunData(
                path=run_dir,
                config=config,
                metrics=_records(run_dir / "metrics.jsonl"),
            )
            operator_paths = sorted(run_dir.glob("operator_reuse*.json"))
            causal_paths = sorted(run_dir.glob("causal_reuse*.json"))
            if operator_paths:
                for path in operator_paths:
                    _, records = _analysis_payload(path)
                    run.operator.extend(records)
            if not run.operator:
                for fallback in (
                    run_dir / "operator_reuse.jsonl",
                    run_dir / "operator_reuse.csv",
                ):
                    run.operator.extend(_records(fallback))
                    if run.operator:
                        break
            if causal_paths:
                for path in causal_paths:
                    _, records = _analysis_payload(path)
                    run.causal.extend(records)
            if not run.causal:
                for fallback in (
                    run_dir / "causal_reuse.jsonl",
                    run_dir / "causal_reuse.csv",
                ):
                    run.causal.extend(_records(fallback))
                    if run.causal:
                        break
            runs.append(run)

    by_name = {run.name: run for run in runs}
    run_paths = {run.path.resolve() for run in runs}
    for root in roots:
        if not root.is_dir():
            continue
        for stem, attribute in (
            ("operator_reuse*.json", "operator"),
            ("causal_reuse*.json", "causal"),
        ):
            for path in root.rglob(stem):
                if path.parent.resolve() in run_paths:
                    continue
                run_name, records = _analysis_payload(path)
                if run_name in by_name:
                    getattr(by_name[run_name], attribute).extend(records)
    return sorted(runs, key=lambda run: (run.task, run.preset, run.seed, run.name))


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _preferred_reuse_gain(record: dict[str, object]) -> float | None:
    for key in (
        "usable_reuse_gain_bits",
        "lookup_reuse_gain_bits",
        "reuse_gain_bits",
    ):
        value = _finite(record.get(key))
        if value is not None:
            return value
    return None


def _layer_value(layer: str, records: list[dict[str, object]]) -> int:
    available = sorted(
        {
            int(record["layer"])
            for record in records
            if _finite(record.get("layer")) is not None
        }
    )
    if not available:
        raise ValueError("analysis records contain no layers")
    return available[-1] if layer == "last" else int(layer)


def select_operator(
    run: RunData, *, view: str, layer: str
) -> list[dict[str, object]]:
    candidates = [
        record
        for record in run.operator
        if str(record.get("view")) == view
        and _finite(record.get("step")) is not None
        and _finite(record.get("layer")) is not None
    ]
    if not candidates:
        return []
    selected_layer = _layer_value(layer, candidates)
    selected: dict[int, dict[str, object]] = {}
    for record in candidates:
        if int(record.get("layer", -1)) == selected_layer:
            selected[int(record["step"])] = record
    return [selected[step] for step in sorted(selected)]


def select_causal(
    run: RunData,
    *,
    position: str,
    layer: str,
    final_only: bool = True,
) -> list[dict[str, object]]:
    candidates = [
        record
        for record in run.causal
        if str(record.get("position")) == position
        and _finite(record.get("step")) is not None
        and _finite(record.get("layer")) is not None
    ]
    if not candidates:
        return []
    if final_only:
        final_step = max(int(record["step"]) for record in candidates)
        candidates = [
            record for record in candidates if int(record["step"]) == final_step
        ]
    selected_layer = _layer_value(layer, candidates)
    return [
        record
        for record in candidates
        if int(record.get("layer", -1)) == selected_layer
    ]


def matching_runs(
    runs: Iterable[RunData],
    *,
    task: str,
    preset: str,
    condition: str | None = None,
) -> list[RunData]:
    return [
        run
        for run in runs
        if (task == "all" or run.task == task)
        and (preset == "all" or run.preset == preset)
        and (condition is None or run.condition == condition)
    ]


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


def _save_static(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=240, transparent=True, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return [png, pdf]


def _median_curve(curves: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    values: dict[float, list[float]] = defaultdict(list)
    for xs, ys in curves:
        for x, y in zip(xs, ys):
            if math.isfinite(float(x)) and math.isfinite(float(y)):
                values[float(x)].append(float(y))
    xs = np.asarray(sorted(values))
    ys = np.asarray([np.median(values[x]) for x in xs])
    return xs, ys


def _metric_curve(
    run: RunData,
    key: str,
    *,
    operator_view: str,
    operator_layer: str,
) -> tuple[np.ndarray, np.ndarray]:
    records = (
        select_operator(run, view=operator_view, layer=operator_layer)
        if key
        in {
            "joint_cv_error",
            "usable_reuse_gain_bits",
            "lookup_reuse_gain_bits",
            "reuse_gain_bits",
        }
        else run.metrics
    )
    pairs = [
        (
            int(record["step"]),
            (
                _preferred_reuse_gain(record)
                if key
                in {
                    "usable_reuse_gain_bits",
                    "lookup_reuse_gain_bits",
                    "reuse_gain_bits",
                }
                else _finite(record.get(key))
            ),
        )
        for record in records
        if "step" in record
    ]
    pairs = [(step, value) for step, value in pairs if value is not None]
    if not pairs:
        return np.asarray([]), np.asarray([])
    pairs.sort()
    xs = np.asarray([pair[0] for pair in pairs], dtype=float)
    ys = np.asarray([pair[1] for pair in pairs], dtype=float)
    if key in {
        "usable_reuse_gain_bits",
        "lookup_reuse_gain_bits",
        "reuse_gain_bits",
    }:
        ys /= 1000.0
    return xs, ys


def _causal_curve(
    run: RunData, *, position: str, layer: str
) -> tuple[np.ndarray, np.ndarray]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for record in select_causal(
        run, position=position, layer=layer, final_only=False
    ):
        if str(record.get("control")) != "learned_generator":
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
            by_step[int(record["step"])].append(value)
    if not by_step:
        return np.asarray([]), np.asarray([])
    xs = np.asarray(sorted(by_step), dtype=float)
    ys = np.asarray([np.median(by_step[int(step)]) for step in xs])
    return xs, ys


def render_spaghetti(
    runs: list[RunData],
    *,
    output: Path,
    operator_view: str,
    operator_layer: str,
    causal_position: str,
    causal_layer: str,
) -> list[Path]:
    specs: list[tuple[str, str, str]] = [
        ("test_accuracy", "held-out accuracy", NORD["frost_dark"]),
    ]
    if any(run.operator for run in runs):
        specs.extend(
            [
                ("joint_cv_error", "generator error", NORD["red"]),
                (
                    "usable_reuse_gain_bits",
                    "usable shared-rule gain (kbit)",
                    NORD["green"],
                ),
            ]
        )
    if any(run.causal for run in runs):
        specs.append(
            (
                "causal_shift_success",
                "causal shift success",
                NORD["purple"],
            )
        )
    fig, axes = plt.subplots(
        1,
        len(specs),
        figsize=(3.4 * len(specs), 2.75),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
    )
    fig.patch.set_alpha(0)
    legend_handles = [
        Line2D([0], [0], color=NORD["muted"], alpha=0.24, linewidth=1.0),
        Line2D([0], [0], color=NORD["frost_dark"], linewidth=2.5),
    ]
    for axis, (key, label, color) in zip(axes[0], specs):
        curves = []
        for run in runs:
            if key == "causal_shift_success":
                xs, ys = _causal_curve(
                    run, position=causal_position, layer=causal_layer
                )
            else:
                xs, ys = _metric_curve(
                    run,
                    key,
                    operator_view=operator_view,
                    operator_layer=operator_layer,
                )
            if xs.size:
                curves.append((xs, ys))
                axis.plot(xs, ys, color=color, alpha=0.16, linewidth=0.85)
        median_x, median_y = _median_curve(curves)
        if median_x.size:
            axis.plot(median_x, median_y, color=color, linewidth=2.5)
        axis.set_xlabel("step")
        axis.set_ylabel(label)
        _step_axis(axis)
        _style_axis(axis)
        if key == "test_accuracy":
            axis.set_ylim(-0.03, 1.03)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        if key in {
            "usable_reuse_gain_bits",
            "lookup_reuse_gain_bits",
            "reuse_gain_bits",
        }:
            axis.axhline(0, color=NORD["muted"], alpha=0.25, linewidth=0.7)
        if key == "causal_shift_success":
            axis.set_ylim(-0.03, 1.03)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0, 0].legend(
        legend_handles,
        ("seeds", "median"),
        loc="lower right",
        frameon=False,
        fontsize=8,
        handlelength=2.3,
    )
    return _save_static(fig, output / "training-spaghetti")


def _accuracy_at(run: RunData, step: int, record: dict[str, object]) -> float | None:
    direct = _finite(record.get("test_accuracy"))
    if direct is not None:
        return direct
    candidates = [
        (abs(int(metric["step"]) - step), _finite(metric.get("test_accuracy")))
        for metric in run.metrics
        if "step" in metric
    ]
    candidates = [candidate for candidate in candidates if candidate[1] is not None]
    return min(candidates)[1] if candidates else None


def render_generalization_reuse(
    runs: list[RunData],
    *,
    output: Path,
    operator_view: str,
    operator_layer: str,
) -> list[Path]:
    curves: list[list[tuple[float, float, float, int]]] = []
    for run in runs:
        points = []
        for record in select_operator(
            run, view=operator_view, layer=operator_layer
        ):
            step = int(record["step"])
            accuracy = _accuracy_at(run, step, record)
            error = _finite(record.get("joint_cv_error"))
            reuse = _preferred_reuse_gain(record)
            if accuracy is not None and error is not None and reuse is not None:
                points.append((accuracy, error, reuse / 1000.0, step))
        if points:
            curves.append(sorted(points, key=lambda point: point[-1]))
    if not curves:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    fig.patch.set_alpha(0)
    for axis, value_index, ylabel, color in (
        (axes[0], 1, "generator error", NORD["red"]),
        (axes[1], 2, "usable shared-rule gain (kbit)", NORD["green"]),
    ):
        by_step: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for curve in curves:
            xs = np.asarray([point[0] for point in curve])
            ys = np.asarray([point[value_index] for point in curve])
            axis.plot(xs, ys, color=color, alpha=0.16, linewidth=0.9)
            axis.scatter(xs, ys, s=8, color=color, alpha=0.15, linewidths=0)
            for point in curve:
                by_step[point[-1]].append((point[0], point[value_index]))
        medians = [
            (
                np.median([value[0] for value in by_step[step]]),
                np.median([value[1] for value in by_step[step]]),
            )
            for step in sorted(by_step)
        ]
        axis.plot(
            [point[0] for point in medians],
            [point[1] for point in medians],
            color=color,
            linewidth=2.5,
        )
        axis.set_xlabel("held-out accuracy")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.03, 1.03)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        if value_index == 2:
            axis.axhline(0, color=NORD["muted"], alpha=0.25, linewidth=0.7)
        _style_axis(axis)
    axes[0].legend(
        [
            Line2D([0], [0], color=NORD["muted"], alpha=0.24, linewidth=1.0),
            Line2D([0], [0], color=NORD["frost_dark"], linewidth=2.5),
        ],
        ("seeds", "median"),
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    return _save_static(fig, output / "generalization-and-reuse")


def _final_behavior(run: RunData, key: str) -> float | None:
    records = [
        record for record in run.metrics if _finite(record.get(key)) is not None
    ]
    if not records:
        return None
    record = max(records, key=lambda item: int(item["step"]))
    return _finite(record.get(key))


def _final_operator(
    run: RunData, key: str, *, view: str, layer: str
) -> float | None:
    records = select_operator(run, view=view, layer=layer)
    if not records:
        return None
    record = max(records, key=lambda item: int(item["step"]))
    value = (
        _preferred_reuse_gain(record)
        if key
        in {
            "usable_reuse_gain_bits",
            "lookup_reuse_gain_bits",
            "reuse_gain_bits",
        }
        else _finite(record.get(key))
    )
    if value is not None and key in {
        "usable_reuse_gain_bits",
        "lookup_reuse_gain_bits",
        "reuse_gain_bits",
    }:
        value /= 1000.0
    return value


def _final_causal(
    run: RunData, *, position: str, layer: str
) -> float | None:
    values = []
    for record in select_causal(run, position=position, layer=layer):
        if str(record.get("control")) != "learned_generator":
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
    return float(np.median(values)) if values else None


def render_corruption_phase(
    runs: list[RunData],
    *,
    output: Path,
    operator_view: str,
    operator_layer: str,
) -> list[Path]:
    structured = [
        run
        for run in runs
        if run.condition != "random"
        and (
            str(run.config.get("task_family", "")).endswith("cycle")
            or run.task.startswith("cycle")
        )
    ]
    specs: list[tuple[str, str, str, Callable[[RunData], float | None]]] = [
        (
            "accuracy",
            "held-out accuracy",
            NORD["frost_dark"],
            lambda run: _final_behavior(run, "test_accuracy"),
        )
    ]
    if any(run.operator for run in structured):
        specs.extend(
            [
                (
                    "error",
                    "generator error",
                    NORD["red"],
                    lambda run: _final_operator(
                        run,
                        "joint_cv_error",
                        view=operator_view,
                        layer=operator_layer,
                    ),
                ),
                (
                    "reuse",
                    "usable shared-rule gain (kbit)",
                    NORD["green"],
                    lambda run: _final_operator(
                        run,
                        "usable_reuse_gain_bits",
                        view=operator_view,
                        layer=operator_layer,
                    ),
                ),
            ]
        )
    if not structured:
        return []

    fig, axes = plt.subplots(
        1,
        len(specs),
        figsize=(3.4 * len(specs), 2.8),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_alpha(0)
    for axis, (key, ylabel, color, getter) in zip(axes[0], specs):
        by_seed: dict[int, dict[float, float]] = defaultdict(dict)
        by_corruption: dict[float, list[float]] = defaultdict(list)
        for run in structured:
            value = getter(run)
            if value is None:
                continue
            by_seed[run.seed][run.corruption] = value
            by_corruption[run.corruption].append(value)
        for seed_values in by_seed.values():
            xs = np.asarray(sorted(seed_values))
            ys = np.asarray([seed_values[x] for x in xs])
            axis.plot(xs, ys, color=color, alpha=0.16, linewidth=0.85)
            axis.scatter(xs, ys, color=color, alpha=0.16, s=9, linewidths=0)
        xs = np.asarray(sorted(by_corruption))
        ys = np.asarray([np.median(by_corruption[x]) for x in xs])
        if xs.size:
            axis.plot(xs, ys, color=color, linewidth=2.5)
            axis.scatter(xs, ys, color=color, s=18, linewidths=0, zorder=3)
        axis.set_xlabel("corrupted transitions")
        axis.set_ylabel(ylabel)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.xaxis.set_major_locator(MaxNLocator(5))
        if key == "accuracy":
            axis.set_ylim(-0.03, 1.03)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        if key == "reuse":
            axis.axhline(0, color=NORD["muted"], alpha=0.25, linewidth=0.7)
        _style_axis(axis)
    axes[0, 0].legend(
        [
            Line2D([0], [0], color=NORD["muted"], alpha=0.24, linewidth=1.0),
            Line2D([0], [0], color=NORD["frost_dark"], linewidth=2.5),
        ],
        ("seeds", "median"),
        loc="best",
        frameon=False,
        fontsize=8,
    )
    return _save_static(fig, output / "corruption-phase")


def _condition_key(run: RunData) -> tuple[int, float]:
    return (1, 0.0) if run.condition == "random" else (0, run.corruption)


def _condition_label(key: tuple[int, float]) -> str:
    if key[0]:
        return "random"
    if key[1] == 0:
        return "clean"
    return f"{100 * key[1]:g}%"


def render_condition_summary(
    runs: list[RunData],
    *,
    output: Path,
    operator_view: str,
    operator_layer: str,
    causal_position: str,
    causal_layer: str,
) -> list[Path]:
    conditions = sorted({_condition_key(run) for run in runs})
    if not conditions:
        return []
    specs: list[tuple[str, str, str, Callable[[RunData], float | None]]] = [
        (
            "accuracy",
            "held-out accuracy",
            NORD["frost_dark"],
            lambda run: _final_behavior(run, "test_accuracy"),
        )
    ]
    if any(run.operator for run in runs):
        specs.extend(
            [
                (
                    "error",
                    "generator error",
                    NORD["red"],
                    lambda run: _final_operator(
                        run,
                        "joint_cv_error",
                        view=operator_view,
                        layer=operator_layer,
                    ),
                ),
                (
                    "reuse",
                    "shared rule vs best alternative (kbit)",
                    NORD["green"],
                    lambda run: _final_operator(
                        run,
                        "usable_reuse_gain_bits",
                        view=operator_view,
                        layer=operator_layer,
                    ),
                ),
            ]
        )
    causal_conditions = {
        _condition_key(run)
        for run in runs
        if _final_causal(
            run, position=causal_position, layer=causal_layer
        )
        is not None
    }
    if len(causal_conditions) >= 2:
        specs.append(
            (
                "causal",
                "causal shift success",
                NORD["purple"],
                lambda run: _final_causal(
                    run, position=causal_position, layer=causal_layer
                ),
            )
        )

    fig, axes = plt.subplots(
        1,
        len(specs),
        figsize=(3.4 * len(specs), 2.9),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
    )
    fig.patch.set_alpha(0)
    x = np.arange(len(conditions))
    labels = [_condition_label(key) for key in conditions]
    for axis, (metric, ylabel, color, getter) in zip(axes[0], specs):
        by_seed: dict[int, dict[tuple[int, float], list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        by_condition: dict[tuple[int, float], list[float]] = defaultdict(list)
        for run in runs:
            value = getter(run)
            if value is None:
                continue
            key = _condition_key(run)
            by_seed[run.seed][key].append(value)
            by_condition[key].append(value)
        for seed_values in by_seed.values():
            points = [
                (
                    index,
                    float(np.median(seed_values[key])),
                )
                for index, key in enumerate(conditions)
                if key in seed_values
            ]
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                alpha=0.15,
                linewidth=0.8,
            )
            axis.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                alpha=0.17,
                s=9,
                linewidths=0,
            )
        median_y = [
            (
                float(np.median(by_condition[key]))
                if by_condition.get(key)
                else float("nan")
            )
            for key in conditions
        ]
        axis.plot(x, median_y, color=color, linewidth=2.5)
        axis.scatter(x, median_y, color=color, s=18, linewidths=0, zorder=3)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylabel(ylabel)
        if metric in {"accuracy", "causal"}:
            axis.set_ylim(-0.03, 1.03)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        if metric == "reuse":
            axis.axhline(0, color=NORD["muted"], alpha=0.25, linewidth=0.7)
        _style_axis(axis)
    axes[0, 0].legend(
        [
            Line2D([0], [0], color=NORD["muted"], alpha=0.24, linewidth=1.0),
            Line2D([0], [0], color=NORD["frost_dark"], linewidth=2.5),
        ],
        ("seeds", "median"),
        loc="best",
        frameon=False,
        fontsize=8,
    )
    return _save_static(fig, output / "condition-summary")


def render_causal_controls(
    runs: list[RunData],
    *,
    output: Path,
    position: str,
    layer: str,
) -> list[Path]:
    controls = (
        "learned_generator",
        "exact_state_swap",
        "target_centroid",
        "scrambled_successor",
        "random_orthogonal",
    )
    labels = ("generator", "exact state", "centroid", "scrambled", "random")
    samples: list[dict[str, float]] = []
    for run in runs:
        grouped: dict[int, dict[str, float]] = defaultdict(dict)
        for record in select_causal(run, position=position, layer=layer):
            value = _finite(record.get("probability_recovery"))
            if value is not None and str(record.get("control")) in controls:
                grouped[int(record.get("fold", 0))][str(record["control"])] = value
        samples.extend(grouped.values())
    if not samples:
        return []

    fig, axis = plt.subplots(figsize=(5.8, 3.0), constrained_layout=True)
    fig.patch.set_alpha(0)
    x = np.arange(len(controls))
    for sample in samples:
        available = [
            (index, sample[control])
            for index, control in enumerate(controls)
            if control in sample
        ]
        axis.plot(
            [value[0] for value in available],
            [value[1] for value in available],
            color=NORD["muted"],
            alpha=0.14,
            linewidth=0.75,
        )
        axis.scatter(
            [value[0] for value in available],
            [value[1] for value in available],
            color=[SERIES[value[0]] for value in available],
            alpha=0.18,
            s=12,
            linewidths=0,
        )
    medians = np.asarray(
        [
            np.median([sample[control] for sample in samples if control in sample])
            for control in controls
        ]
    )
    axis.plot(x, medians, color=NORD["ink"], linewidth=2.1, zorder=3)
    axis.scatter(x, medians, color=SERIES[: len(controls)], s=30, zorder=4)
    axis.axhline(0, color=NORD["muted"], alpha=0.18, linewidth=0.7)
    axis.axhline(
        1, color=NORD["muted"], alpha=0.35, linewidth=0.8, linestyle=(0, (3, 3))
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("shift recovery")
    axis.margins(x=0.08)
    _style_axis(axis)
    axis.legend(
        [
            Line2D([0], [0], color=NORD["muted"], alpha=0.24, linewidth=1.0),
            Line2D([0], [0], color=NORD["ink"], linewidth=2.1),
        ],
        ("folds", "median"),
        loc="best",
        frameon=False,
        fontsize=8,
    )
    return _save_static(fig, output / "causal-controls")


def _choose_animation_run(
    runs: list[RunData],
    *,
    request: str,
    task: str,
    preset: str,
) -> RunData | None:
    if request == "none":
        return None
    if request != "auto":
        request_path = Path(request)
        for run in runs:
            if (
                request_path == run.path
                or request_path.resolve() == run.path.resolve()
                or request in run.name
                or request == run.path.name
            ):
                return run
        raise ValueError(f"no run matches --animate-run {request!r}")
    candidates = matching_runs(
        runs, task=task, preset=preset, condition="clean"
    )
    candidates = [run for run in candidates if len(run.activation_paths) >= 2]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda run: (
            _final_behavior(run, "test_accuracy") or -1.0,
            len(run.activation_paths),
        ),
    )


def _resolve_layer(layer: str, count: int) -> int:
    selected = count - 1 if layer == "last" else int(layer)
    if selected < 0:
        selected += count
    if not 0 <= selected < count:
        raise ValueError(f"activation layer {selected} is outside 0..{count - 1}")
    return selected


def _activation_frame(
    path: Path,
    *,
    view: str,
    layer: str,
    depth: int,
) -> np.ndarray:
    with np.load(path) as payload:
        values = np.asarray(payload[view], dtype=np.float64)
    if values.ndim == 2:
        return values[:, None, :]
    if values.ndim == 4:
        return values[_resolve_layer(layer, values.shape[0])]
    if values.ndim == 3 and values.shape[0] == depth + 1:
        return values[_resolve_layer(layer, values.shape[0])][:, None, :]
    if values.ndim == 3:
        return values
    raise ValueError(f"{path} contains unsupported {view} shape {values.shape}")


def _animation_frames(
    run: RunData,
    *,
    view: str,
    layer: str,
    max_frames: int,
) -> tuple[list[np.ndarray], list[int]]:
    paths = run.activation_paths
    if max_frames > 0 and len(paths) > max_frames:
        indices = np.unique(
            np.linspace(0, len(paths) - 1, max_frames, dtype=int)
        )
        paths = [paths[index] for index in indices]
    depth = int(
        (
            run.config.get("model", {})
            if isinstance(run.config.get("model"), dict)
            else {}
        ).get("depth", 0)
    )
    loaded = []
    for path in paths:
        try:
            frame = _activation_frame(
                path, view=view, layer=layer, depth=depth
            )
        except (OSError, ValueError, KeyError):
            continue
        loaded.append((path, frame))
    if len(loaded) < 2:
        raise ValueError("fewer than two readable activation frames")
    shape = loaded[-1][1].shape[:2]
    loaded = [item for item in loaded if item[1].shape[:2] == shape]
    if len(loaded) < 2:
        raise ValueError("fewer than two shape-compatible activation frames")
    paths = [item[0] for item in loaded]
    raw = [item[1] for item in loaded]
    final_centroids = raw[-1].mean(axis=1)
    final_center = final_centroids.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(final_centroids - final_center, full_matrices=False)
    if vt.shape[0] < 2:
        raise ValueError("activation width is too small for a two-dimensional movie")
    basis = vt[:2].T
    projected = [
        (frame - frame.mean(axis=(0, 1), keepdims=True)) @ basis for frame in raw
    ]
    scale = float(
        np.quantile(np.abs(np.concatenate([frame.reshape(-1, 2) for frame in projected])), 0.995)
    )
    projected = [frame / max(scale, 1e-12) for frame in projected]
    steps = [_step_from_path(path) for path in paths]
    return projected, steps


def _successor(run: RunData, order: int) -> np.ndarray:
    table_path = run.path / "operation_table.npy"
    if table_path.exists():
        table = np.load(table_path)
        if table.shape[0] == order:
            relation = 1 if table.shape[1] > 1 else 0
            return np.asarray(table[:, relation], dtype=int)
    return (np.arange(order) + 1) % order


def _enable_bundled_ffmpeg() -> None:
    try:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass


def render_geometry_movie(
    run: RunData,
    *,
    output: Path,
    view: str,
    layer: str,
    max_frames: int,
    fps: int,
) -> list[Path]:
    frames, steps = _animation_frames(
        run, view=view, layer=layer, max_frames=max_frames
    )
    order, aliases = frames[-1].shape[:2]
    successor = _successor(run, order)
    phase_map = LinearSegmentedColormap.from_list("nord-cycle", SERIES, N=order)
    state_colors = phase_map(np.linspace(0, 1, order, endpoint=False))
    point_colors = np.repeat(state_colors, aliases, axis=0)

    fig, axis = plt.subplots(figsize=(5.0, 4.8), dpi=180)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.patch.set_alpha(0)
    axis.patch.set_alpha(0)
    axis.set_aspect("equal")
    axis.set_xlim(-1.12, 1.12)
    axis.set_ylim(-1.12, 1.12)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)
    edges = LineCollection([], colors=NORD["muted"], alpha=0.20, linewidths=0.65)
    axis.add_collection(edges)
    points = axis.scatter([], [], s=9, alpha=0.20, linewidths=0)
    centroids = axis.scatter(
        [], [], s=27, linewidths=0.35, edgecolors=NORD["ink"], zorder=3
    )
    step_label = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        va="top",
        color=NORD["muted"],
        fontsize=9,
    )

    def update(index: int):
        frame = frames[index]
        centers = frame.mean(axis=1)
        edges.set_segments(
            np.stack((centers, centers[successor]), axis=1)
        )
        points.set_offsets(frame.reshape(-1, 2))
        points.set_color(point_colors)
        centroids.set_offsets(centers)
        centroids.set_color(state_colors)
        step_label.set_text(f"step {steps[index]:,}")
        return edges, points, centroids, step_label

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / max(fps, 1),
        blit=True,
        repeat=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    _enable_bundled_ffmpeg()
    movie_path = output / "geometry-orbit.mp4"
    try:
        animation.save(
            movie_path,
            writer=FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=1900,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            ),
            dpi=180,
        )
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError):
        movie_path.unlink(missing_ok=True)
        movie_path = output / "geometry-orbit.gif"
        animation.save(movie_path, writer=PillowWriter(fps=fps), dpi=130)
    update(len(frames) - 1)
    poster = _save_static(fig, output / "geometry-orbit-poster")
    return [movie_path, *poster]


def render_all(
    *,
    roots: list[Path],
    output: Path,
    task: str,
    preset: str,
    main_condition: str,
    operator_view: str,
    operator_layer: str,
    causal_position: str,
    causal_layer: str,
    animate_run: str,
    animation_view: str,
    animation_layer: str,
    max_animation_frames: int,
    fps: int,
) -> dict[str, object]:
    runs = discover_runs(roots)
    selected = matching_runs(
        runs, task=task, preset=preset, condition=main_condition
    )
    exact_phase_runs = matching_runs(runs, task=task, preset=preset)
    phase_orders = {
        int(run.config.get("task_order", -1)) for run in exact_phase_runs
    }
    phase_runs = [
        run
        for run in runs
        if (preset == "all" or run.preset == preset)
        and (
            task == "all"
            or run.task == task
            or (
                run.condition == "random"
                and int(run.config.get("task_order", -2)) in phase_orders
            )
        )
    ]
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    if selected:
        artifacts.extend(
            render_spaghetti(
                selected,
                output=output,
                operator_view=operator_view,
                operator_layer=operator_layer,
                causal_position=causal_position,
                causal_layer=causal_layer,
            )
        )
        artifacts.extend(
            render_generalization_reuse(
                selected,
                output=output,
                operator_view=operator_view,
                operator_layer=operator_layer,
            )
        )
        artifacts.extend(
            render_causal_controls(
                selected,
                output=output,
                position=causal_position,
                layer=causal_layer,
            )
        )
    artifacts.extend(
        render_corruption_phase(
            phase_runs,
            output=output,
            operator_view=operator_view,
            operator_layer=operator_layer,
        )
    )
    artifacts.extend(
        render_condition_summary(
            phase_runs,
            output=output,
            operator_view=operator_view,
            operator_layer=operator_layer,
            causal_position=causal_position,
            causal_layer=causal_layer,
        )
    )
    movie_run = _choose_animation_run(
        runs, request=animate_run, task=task, preset=preset
    )
    if movie_run is not None:
        try:
            artifacts.extend(
                render_geometry_movie(
                    movie_run,
                    output=output,
                    view=animation_view,
                    layer=animation_layer,
                    max_frames=max_animation_frames,
                    fps=fps,
                )
            )
        except (OSError, ValueError, KeyError):
            movie_run = None
    manifest: dict[str, object] = {
        "roots": [str(root) for root in roots],
        "run_count": len(runs),
        "selected_run_count": len(selected),
        "task": task,
        "preset": preset,
        "main_condition": main_condition,
        "operator_slice": {"view": operator_view, "layer": operator_layer},
        "causal_slice": {"position": causal_position, "layer": causal_layer},
        "animation_run": movie_run.name if movie_run is not None else None,
        "artifacts": [str(path) for path in artifacts],
    }
    manifest_path = output / "render-manifest.json"
    manifest["artifacts"].append(str(manifest_path))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _cycle_table(order: int) -> np.ndarray:
    left = np.arange(order)[:, None]
    right = np.arange(order)[None, :]
    return (left + right) % order


def _write_fixture(root: Path) -> None:
    rng = np.random.default_rng(17)
    steps = (0, 5_000, 10_000, 20_000, 40_000)
    order, aliases, width = 17, 4, 10
    angles = 2 * np.pi * np.arange(order) / order
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    embedding, _ = np.linalg.qr(rng.normal(size=(width, 2)))
    for seed in range(3):
        for task, corruption in (
            ("cycle113", 0.0),
            ("cycle113", 0.15),
            ("cycle113", 0.60),
            ("random113", 0.0),
        ):
            random_control = task.startswith("random")
            clean = task == "cycle113" and corruption == 0
            run_dir = root / f"{task}-grok-s{seed}-c{corruption:g}"
            run_dir.mkdir(parents=True)
            config = {
                "run_name": run_dir.name,
                "task": task,
                "task_family": (
                    "random_permutation"
                    if random_control
                    else ("cycle" if corruption == 0 else "broken_cycle")
                ),
                "task_order": order,
                "task_corruption_fraction": corruption,
                "preset": "grok",
                "seed": seed,
                "aliases": aliases,
                "model": {"width": width, "depth": 1, "heads": 2},
            }
            (run_dir / "config.json").write_text(json.dumps(config) + "\n")
            np.save(run_dir / "operation_table.npy", _cycle_table(order))
            metric_records = []
            operator_records = []
            threshold = (
                80_000
                if random_control
                else 12_000 + 22_000 * corruption + seed * 800
            )
            for step in steps:
                accuracy = 1 / (1 + np.exp(-(step - threshold) / 2_700))
                metric_records.append(
                    {
                        "step": step,
                        "train_accuracy": min(1.0, accuracy + 0.35),
                        "test_accuracy": float(accuracy),
                    }
                )
                for layer in (0, 1):
                    usable_gain = float(
                        55_000 * accuracy
                        - 35_000 * corruption
                        - 8_000
                        - 28_000 * random_control
                    )
                    operator_records.append(
                        {
                            "step": step,
                            "view": "node",
                            "layer": layer,
                            "test_accuracy": float(accuracy),
                            "joint_cv_error": float(
                                1.25
                                - 0.95 * accuracy
                                + 0.45 * corruption
                                + 0.35 * random_control
                            ),
                            "usable_reuse_gain_bits": usable_gain,
                            "lookup_reuse_gain_bits": usable_gain + 5_000,
                            "reuse_gain_bits": usable_gain + 10_000,
                        }
                    )
                if clean and seed == 0:
                    blend = float(accuracy)
                    random_points = rng.normal(size=(order, aliases, 2))
                    target = circle[:, None, :] + 0.08 * rng.normal(
                        size=(order, aliases, 2)
                    )
                    phase_points = (1 - blend) * random_points + blend * target
                    high_dimensional = phase_points @ embedding.T
                    layers = np.stack(
                        (0.75 * high_dimensional, high_dimensional), axis=0
                    )
                    np.savez_compressed(
                        run_dir / f"activations-{step:06d}.npz",
                        node=layers.astype(np.float32),
                        output=layers.astype(np.float32),
                    )
            metrics_text = "".join(
                json.dumps(record) + "\n" for record in metric_records
            )
            if random_control and seed == 0:
                metrics_text += '{"step":'
            (run_dir / "metrics.jsonl").write_text(metrics_text)
            (run_dir / "operator_reuse.json").write_text(
                json.dumps(
                    {
                        "metadata": {"run_name": run_dir.name},
                        "records": operator_records,
                    }
                )
                + "\n"
            )
            if random_control and seed == 0:
                (run_dir / "operator_reuse-partial.json").write_text(
                    '{"metadata":'
                )
            if clean:
                causal_records = []
                recoveries = {
                    "learned_generator": 0.76,
                    "exact_state_swap": 1.0,
                    "target_centroid": 0.88,
                    "scrambled_successor": 0.06,
                    "random_orthogonal": 0.02,
                }
                for step in steps:
                    progress = 1 / (
                        1 + np.exp(-(step - threshold) / 2_700)
                    )
                    for fold in range(3):
                        for control, recovery in recoveries.items():
                            causal_records.append(
                                {
                                    "step": step,
                                    "fold": fold,
                                    "position": "output",
                                    "layer": 1,
                                    "control": control,
                                    "qualified_desired_accuracy": float(
                                        recovery * progress
                                        + 0.025 * rng.normal()
                                    ),
                                    "probability_recovery": float(
                                        recovery * progress
                                        + 0.025 * rng.normal()
                                    ),
                                }
                            )
                (run_dir / "causal_reuse.json").write_text(
                    json.dumps(
                        {
                            "metadata": {"run_name": run_dir.name},
                            "records": causal_records,
                        }
                    )
                    + "\n"
                )


def run_self_test(destination: Path | None) -> None:
    if _preferred_reuse_gain(
        {
            "usable_reuse_gain_bits": 1.0,
            "lookup_reuse_gain_bits": 2.0,
            "reuse_gain_bits": 3.0,
        }
    ) != 1.0:
        raise AssertionError("usable reuse gain was not preferred")
    if _preferred_reuse_gain(
        {"lookup_reuse_gain_bits": 2.0, "reuse_gain_bits": 3.0}
    ) != 2.0:
        raise AssertionError("lookup reuse gain fallback failed")
    if _preferred_reuse_gain({"reuse_gain_bits": 3.0}) != 3.0:
        raise AssertionError("legacy reuse gain fallback failed")
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if destination is None:
        temporary = tempfile.TemporaryDirectory(prefix="reuse-render-")
        base = Path(temporary.name)
    else:
        base = destination
        base.mkdir(parents=True, exist_ok=True)
    fixture = base / "fixture"
    plots = base / "plots"
    _write_fixture(fixture)
    manifest = render_all(
        roots=[fixture],
        output=plots,
        task="cycle113",
        preset="grok",
        main_condition="clean",
        operator_view="node",
        operator_layer="last",
        causal_position="output",
        causal_layer="last",
        animate_run="auto",
        animation_view="node",
        animation_layer="last",
        max_animation_frames=8,
        fps=4,
    )
    expected = {
        "training-spaghetti.png",
        "training-spaghetti.pdf",
        "generalization-and-reuse.png",
        "generalization-and-reuse.pdf",
        "corruption-phase.png",
        "corruption-phase.pdf",
        "condition-summary.png",
        "condition-summary.pdf",
        "causal-controls.png",
        "causal-controls.pdf",
        "geometry-orbit-poster.png",
        "geometry-orbit-poster.pdf",
        "render-manifest.json",
    }
    missing = [name for name in expected if not (plots / name).exists()]
    movies = list(plots.glob("geometry-orbit.mp4")) + list(
        plots.glob("geometry-orbit.gif")
    )
    if missing or not movies:
        raise AssertionError(f"missing render outputs: {missing}, movies={movies}")
    print(
        f"self-test passed: {len(manifest['artifacts'])} artifacts in {plots}",
        flush=True,
    )
    if temporary is not None:
        temporary.cleanup()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args.self_test_output)
        return
    if not args.results or args.output is None:
        raise SystemExit("--results and --output are required")
    manifest = render_all(
        roots=args.results,
        output=args.output,
        task=args.task,
        preset=args.preset,
        main_condition=args.main_condition,
        operator_view=args.operator_view,
        operator_layer=args.operator_layer,
        causal_position=args.causal_position,
        causal_layer=args.causal_layer,
        animate_run=args.animate_run,
        animation_view=args.animation_view,
        animation_layer=args.animation_layer,
        max_animation_frames=args.max_animation_frames,
        fps=args.fps,
    )
    print(
        f"wrote {len(manifest['artifacts'])} artifacts from "
        f"{manifest['selected_run_count']} selected runs to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
