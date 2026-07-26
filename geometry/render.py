from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


NORD = {
    "ink": "#2E3440",
    "muted": "#4C566A",
    "blue": "#5E81AC",
    "cyan": "#88C0D0",
    "red": "#BF616A",
    "orange": "#D08770",
    "yellow": "#EBCB8B",
    "green": "#A3BE8C",
    "purple": "#B48EAD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="cycle7")
    parser.add_argument("--preset", default="small")
    parser.add_argument("--animate-run")
    return parser.parse_args()


def nested(record: dict[str, object], key: str) -> float:
    value: object = record
    for part in key.split("."):
        value = value[part]  # type: ignore[index]
    return float(value)


def load_series(
    root: Path, task: str, preset: str
) -> list[list[dict[str, object]]]:
    series = []
    for config_path in sorted(root.glob("*/config.json")):
        config = json.loads(config_path.read_text())
        if config["task"] != task or config["preset"] != preset:
            continue
        metrics_path = config_path.parent / "metrics.jsonl"
        if metrics_path.exists():
            records = [
                json.loads(line)
                for line in metrics_path.read_text().splitlines()
                if line.strip()
            ]
            records.sort(key=lambda record: int(record["step"]))
            series.append(records)
    return series


def render_spaghetti(
    series: list[list[dict[str, object]]], output: Path
) -> None:
    if not series:
        raise ValueError("no matching runs")
    metrics = (
        ("test_accuracy", "held-out accuracy", NORD["blue"]),
        ("node_geometry.cyclic_defect", "cyclic defect", NORD["red"]),
        ("node_geometry.generator_error", "shift error", NORD["green"]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8), constrained_layout=True)
    fig.patch.set_alpha(0)
    by_step: list[dict[int, list[float]]] = [defaultdict(list) for _ in metrics]
    for records in series:
        steps = np.asarray([int(record["step"]) for record in records])
        for metric_index, (key, _, color) in enumerate(metrics):
            values = []
            valid_steps = []
            for step, record in zip(steps, records):
                try:
                    value = nested(record, key)
                except (KeyError, TypeError):
                    continue
                if np.isfinite(value):
                    values.append(value)
                    valid_steps.append(step)
                    by_step[metric_index][int(step)].append(value)
            axes[metric_index].plot(
                valid_steps, values, color=color, alpha=0.18, linewidth=0.8
            )
    for axis, step_values, (_, label, color) in zip(axes, by_step, metrics):
        steps = sorted(step_values)
        medians = [np.median(step_values[step]) for step in steps]
        axis.plot(steps, medians, color=color, linewidth=2.2, label="median")
        axis.set_xlabel("step")
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(NORD["muted"])
        axis.tick_params(colors=NORD["muted"], labelsize=8)
        axis.grid(False)
    axes[0].set_ylim(-0.03, 1.03)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, transparent=True)
    fig.savefig(output.with_suffix(".svg"), transparent=True)
    plt.close(fig)


def pca_frame(activations: np.ndarray) -> np.ndarray:
    centered = activations - activations.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * singular[:2]


def align_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    aligned = [frames[0]]
    for frame in frames[1:]:
        u, _, vt = np.linalg.svd(frame.T @ aligned[-1], full_matrices=False)
        aligned.append(frame @ (u @ vt))
    scale = max(np.abs(frame).max() for frame in aligned)
    return [frame / max(scale, 1e-12) for frame in aligned]


def render_animation(run_dir: Path, output: Path) -> None:
    snapshots = sorted(run_dir.glob("activations-*.npz"))
    if not snapshots:
        raise ValueError(f"no activation snapshots in {run_dir}")
    steps = [int(path.stem.split("-")[-1]) for path in snapshots]
    frames = align_frames(
        [pca_frame(np.load(path)["node"][-1]) for path in snapshots]
    )
    order = frames[0].shape[0]
    colors = [
        NORD["blue"],
        NORD["cyan"],
        NORD["green"],
        NORD["yellow"],
        NORD["orange"],
        NORD["red"],
        NORD["purple"],
    ]

    fig, axis = plt.subplots(figsize=(4.2, 4.2))
    fig.patch.set_alpha(0)
    axis.set_aspect("equal")
    axis.set_xlim(-1.15, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)
    line, = axis.plot([], [], color=NORD["muted"], alpha=0.22, linewidth=1.0)
    scatter = axis.scatter([], [], s=42)
    step_label = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        va="top",
        color=NORD["muted"],
    )

    def update(index: int):
        frame = frames[index]
        closed = np.vstack([frame, frame[0]])
        line.set_data(closed[:, 0], closed[:, 1])
        scatter.set_offsets(frame)
        scatter.set_color([colors[node % len(colors)] for node in range(order)])
        step_label.set_text(f"step {steps[index]:,}")
        return line, scatter, step_label

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=90,
        blit=True,
        repeat=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output,
        writer=FFMpegWriter(fps=12, codec="h264", bitrate=1800),
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    series = load_series(args.results, args.task, args.preset)
    render_spaghetti(series, args.output)
    if args.animate_run:
        render_animation(
            args.results / args.animate_run,
            args.output.with_suffix(".mp4"),
        )


if __name__ == "__main__":
    main()
