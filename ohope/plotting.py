from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import rankdata

from .config import NORD, SCORE_COLORS, SCORE_LABELS


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": NORD["white"],
            "savefig.facecolor": NORD["white"],
            "axes.facecolor": NORD["white"],
            "axes.edgecolor": NORD["dark"],
            "axes.labelcolor": NORD["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.color": NORD["muted"],
            "ytick.color": NORD["muted"],
            "text.color": NORD["ink"],
            "font.family": "serif",
            "font.serif": ["Alegreya", "Georgia", "DejaVu Serif"],
            "font.size": 10.0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "mathtext.fontset": "dejavuserif",
        }
    )


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor=NORD["white"])
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor=NORD["white"])
    plt.close(fig)


def plot_geometry(geometry_path: Path, output: Path) -> None:
    apply_style()
    with np.load(geometry_path) as data:
        eig = np.asarray(data["normalized_eigenvalues"])
        q = np.asarray(data["q_normalized"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.45))
    x = np.linspace(0.0, 1.0, eig.size)
    axes[0].plot(x, eig, color=NORD["blue"], lw=1.9)
    axes[0].axhline(1.0, color=NORD["pale"], lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("spectral percentile")
    axes[0].set_ylabel("normalized eigenvalue")

    axes[1].hist(q, bins=45, color=NORD["purple"], alpha=0.82, edgecolor="none")
    axes[1].axvline(1.0, color=NORD["pale"], lw=0.9)
    axes[1].set_xlabel(r"visibility ratio $q_i$")
    axes[1].set_ylabel("neurons")

    fig.tight_layout(w_pad=2.0)
    _save(fig, output)


def plot_rank_disagreement(scores_path: Path, output: Path) -> None:
    apply_style()
    with np.load(scores_path) as data:
        hope = np.asarray(data["hope"])
        ohope = np.asarray(data["ohope"])
    n = hope.size
    hope_rank = rankdata(hope) / n
    ohope_rank = rankdata(ohope) / n
    disagreement = np.abs(hope_rank - ohope_rank)

    fig, ax = plt.subplots(figsize=(4.45, 4.15))
    order = np.argsort(disagreement)
    point_colors = np.where(
        hope_rank[order] >= ohope_rank[order],
        NORD["purple"],
        NORD["red"],
    )
    alpha = 0.16 + 0.68 * disagreement[order]
    rgba = np.asarray([mcolors.to_rgba(color) for color in point_colors])
    rgba[:, 3] = alpha
    ax.scatter(
        hope_rank[order],
        ohope_rank[order],
        s=11,
        c=rgba,
        linewidths=0,
    )
    ax.plot([0, 1], [0, 1], color=NORD["pale"], lw=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("HOPE percentile")
    ax.set_ylabel("O-HOPE percentile")
    _save(fig, output)


def _spaghetti(
    ax,
    frame: pd.DataFrame,
    *,
    y: str,
    ylabel: str,
) -> None:
    method_order = ("random", "outgoing_norm", "activation_rms", "hope", "ohope", "fisher")
    for method in method_order:
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        color = SCORE_COLORS[method]
        run_curves = []
        for _seed, run in subset.groupby("seed"):
            run = run.sort_values("fraction_retained")
            x = run["fraction_retained"].to_numpy()
            values = run[y].to_numpy()
            run_curves.append(values)
            ax.plot(x, values, color=color, alpha=0.18, lw=0.9)
        median = np.median(np.stack(run_curves), axis=0)
        ax.plot(
            x,
            median,
            color=color,
            lw=2.25,
            marker="o",
            ms=3.0,
            label=SCORE_LABELS[method],
        )
    ax.set_xlabel("retained final-MLP width")
    ax.set_ylabel(ylabel)


def plot_pruning_spaghetti(curves_path: Path, output: Path) -> None:
    apply_style()
    frame = pd.read_csv(curves_path)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.45))
    _spaghetti(axes[0], frame, y="perplexity", ylabel="perplexity")
    _spaghetti(axes[1], frame, y="kl", ylabel="output KL")
    _spaghetti(axes[2], frame, y="top_agreement", ylabel="top-token agreement")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.invert_xaxis()
    fig.tight_layout(w_pad=1.8)
    _save(fig, output)


def plot_correlation_runs(correlation_path: Path, output: Path) -> None:
    apply_style()
    rows = json.loads(correlation_path.read_text())
    frame = pd.DataFrame(rows)
    methods = [
        method
        for method in ("outgoing_norm", "activation_rms", "hope", "ohope", "fisher")
        if method in set(frame["method"])
    ]
    x = np.arange(len(methods), dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    run_keys = list(frame.groupby(["sample_seed", "calibration_seed"]).groups)
    for sample_seed, calibration_seed in run_keys:
        run = frame[
            (frame["sample_seed"] == sample_seed)
            & (frame["calibration_seed"] == calibration_seed)
        ].set_index("method")
        values = np.asarray([run.loc[method, "spearman_kl"] for method in methods])
        ax.plot(x, values, color=NORD["muted"], alpha=0.14, lw=0.85)
        ax.scatter(
            x,
            values,
            c=[SCORE_COLORS[method] for method in methods],
            alpha=0.22,
            s=16,
            linewidths=0,
        )

    medians = [
        float(frame.loc[frame["method"] == method, "spearman_kl"].median())
        for method in methods
    ]
    ax.plot(x, medians, color=NORD["ink"], lw=2.0, alpha=0.9)
    ax.scatter(
        x,
        medians,
        c=[SCORE_COLORS[method] for method in methods],
        s=42,
        linewidths=0.7,
        edgecolors=NORD["ink"],
        zorder=3,
    )
    ax.axhline(0.0, color=NORD["pale"], lw=0.9)
    ax.set_xticks(x, [SCORE_LABELS[method] for method in methods], rotation=18, ha="right")
    ax.set_ylabel("Spearman correlation with ablation KL")
    fig.tight_layout()
    _save(fig, output)


def animate_pruning_orders(
    scores_path: Path,
    output: Path,
    *,
    max_fraction: float = 0.60,
    frames: int = 61,
    fps: int = 12,
) -> None:
    """Animate which neurons each geometry regards as cheapest to forget."""

    apply_style()
    with np.load(scores_path) as data:
        hope = np.asarray(data["hope"])
        ohope = np.asarray(data["ohope"])
    n = hope.size
    hope_rank = rankdata(hope) / n
    ohope_rank = rankdata(ohope) / n
    order_hope = np.argsort(hope)
    order_ohope = np.argsort(ohope)

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.55))
    base = dict(
        x=hope_rank,
        y=ohope_rank,
        s=8,
        color=NORD["pale"],
        alpha=0.75,
        linewidths=0,
    )
    for ax in axes:
        ax.scatter(**base)
        ax.plot([0, 1], [0, 1], color=NORD["pale"], lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("HOPE percentile")
    axes[0].set_ylabel("O-HOPE percentile")
    axes[0].text(0.04, 0.95, "HOPE order", transform=axes[0].transAxes, va="top")
    axes[1].text(0.04, 0.95, "O-HOPE order", transform=axes[1].transAxes, va="top")
    left = axes[0].scatter([], [], s=11, color=NORD["purple"], alpha=0.72, linewidths=0)
    right = axes[1].scatter([], [], s=11, color=NORD["red"], alpha=0.72, linewidths=0)
    fraction_text = fig.text(0.5, 0.01, "", ha="center", color=NORD["muted"])

    def update(frame_index: int):
        fraction = max_fraction * frame_index / max(frames - 1, 1)
        count = int(round(fraction * n))
        left_idx = order_hope[:count]
        right_idx = order_ohope[:count]
        left.set_offsets(np.column_stack([hope_rank[left_idx], ohope_rank[left_idx]]))
        right.set_offsets(np.column_stack([hope_rank[right_idx], ohope_rank[right_idx]]))
        fraction_text.set_text(f"{fraction:.0%} removed")
        return left, right, fraction_text

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    animation.save(output, writer=writer, dpi=160)
    update(frames - 1)
    fig.savefig(output.with_name(output.stem + "-poster.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_model_artifacts(model_dir: Path) -> None:
    figure_dir = model_dir / "figures"
    plot_geometry(model_dir / "geometry.npz", figure_dir / "geometry.png")
    plot_rank_disagreement(model_dir / "scores_seed_0.npz", figure_dir / "rank-disagreement.png")
    if (model_dir / "pruning_curves.csv").exists():
        plot_pruning_spaghetti(
            model_dir / "pruning_curves.csv",
            figure_dir / "pruning-spaghetti.png",
        )
    if (model_dir / "ablation_correlations.json").exists():
        plot_correlation_runs(
            model_dir / "ablation_correlations.json",
            figure_dir / "ablation-correlations.png",
        )
    animate_pruning_orders(
        model_dir / "scores_seed_0.npz",
        figure_dir / "forgetting-orders.mp4",
    )
