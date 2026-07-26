from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap


NORD = (
    "#5E81AC",
    "#88C0D0",
    "#A3BE8C",
    "#EBCB8B",
    "#D08770",
    "#BF616A",
    "#B48EAD",
)
INK = "#2E3440"
MUTED = "#4C566A"


@dataclass(frozen=True)
class MeasuredFrame:
    step: int
    source: Path
    activations: np.ndarray
    train_accuracy: float | None
    test_accuracy: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render measured activation centroids under one PCA basis fitted "
            "at the run's final saved checkpoint."
        )
    )
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--view", choices=("node", "output"), default="node")
    parser.add_argument(
        "--layer",
        default="0",
        help="Stored activation layer index, or last.",
    )
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Render a synthetic measured-checkpoint fixture and audit its manifest.",
    )
    parser.add_argument("--self-test-output", type=Path)
    return parser.parse_args()


def _step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("-", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"cannot parse checkpoint step from {path}") from exc


def _records(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[int(record["step"])] = record
    return records


def _optional_accuracy(
    record: dict[str, object] | None, key: str
) -> float | None:
    if record is None:
        return None
    value = float(record[key])
    return value if np.isfinite(value) else None


def _resolve_layer(request: str, count: int) -> int:
    layer = count - 1 if request == "last" else int(request)
    if layer < 0:
        layer += count
    if not 0 <= layer < count:
        raise ValueError(f"layer {layer} is outside 0..{count - 1}")
    return layer


def _load_centroids(
    path: Path,
    *,
    view: str,
    layer_request: str,
) -> tuple[np.ndarray, int]:
    with np.load(path) as payload:
        if view not in payload:
            raise ValueError(f"{path} does not contain {view!r} activations")
        stored = np.asarray(payload[view], dtype=np.float64)
    if stored.ndim != 3:
        raise ValueError(
            f"{path} has shape {stored.shape}; expected [layer, state, width] "
            "centroids"
        )
    layer = _resolve_layer(layer_request, stored.shape[0])
    values = stored[layer]
    if values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError(f"{path} has too few state centroids for a 2D projection")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite activations")
    return values, layer


def _is_clean_cycle(config: dict[str, object]) -> bool:
    corruption = float(
        config.get(
            "task_corruption_fraction",
            config.get("corruption", 0.0),
        )
    )
    return str(config.get("task_family")) == "cycle" and corruption == 0.0


def _transition_window(
    *,
    steps: list[int],
    metrics: dict[int, dict[str, object]],
) -> tuple[int, int, int | None, int | None]:
    if len(steps) < 2:
        raise ValueError("at least two measured activation checkpoints are required")
    spacing = int(np.median(np.diff(steps)))
    ordered_metrics = [metrics[step] for step in sorted(metrics)]
    memorization = next(
        (
            int(record["step"])
            for record in ordered_metrics
            if float(record.get("train_accuracy", 0.0)) >= 0.99
        ),
        None,
    )
    generalization = next(
        (
            int(record["step"])
            for record in ordered_metrics
            if float(record.get("test_accuracy", 0.0)) >= 0.90
        ),
        None,
    )
    if memorization is None or generalization is None or generalization <= memorization:
        return steps[0], steps[-1], memorization, generalization
    start_target = max(steps[0], memorization - 2 * spacing)
    transition_gap = generalization - memorization
    end_target = min(
        steps[-1],
        generalization + max(transition_gap, 4 * spacing),
    )
    start = max(step for step in steps if step <= start_target)
    end = min(step for step in steps if step >= end_target)
    return start, end, memorization, generalization


def load_measured_frames(
    run: Path,
    *,
    view: str,
    layer_request: str,
    start_step: int | None,
    end_step: int | None,
) -> tuple[
    dict[str, object],
    list[MeasuredFrame],
    np.ndarray,
    int,
    Path,
    int | None,
    int | None,
]:
    config_path = run / "config.json"
    if not config_path.exists():
        raise ValueError(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    if not _is_clean_cycle(config):
        raise ValueError(
            "the measured-geometry movie requires an uncorrupted cycle run"
        )

    paths = sorted(run.glob("activations-*.npz"), key=_step)
    if len(paths) < 2:
        raise ValueError(f"{run} contains fewer than two activation snapshots")
    steps = [_step(path) for path in paths]
    if len(steps) != len(set(steps)):
        raise ValueError(f"{run} contains duplicate activation steps")
    metrics = _records(run / "metrics.jsonl")
    auto_start, auto_end, memorization, generalization = _transition_window(
        steps=steps,
        metrics=metrics,
    )
    selected_start = auto_start if start_step is None else start_step
    selected_end = auto_end if end_step is None else end_step
    if selected_start > selected_end:
        raise ValueError("--start-step must not exceed --end-step")
    selected_paths = [
        path
        for path in paths
        if selected_start <= _step(path) <= selected_end
    ]
    if len(selected_paths) < 2:
        raise ValueError("the selected interval contains fewer than two checkpoints")

    final_activations, layer = _load_centroids(
        paths[-1],
        view=view,
        layer_request=layer_request,
    )
    final_centered = final_activations - final_activations.mean(
        axis=0, keepdims=True
    )
    _, singular, vt = np.linalg.svd(final_centered, full_matrices=False)
    if len(singular) < 2 or singular[1] <= singular[0] * 1e-9:
        raise ValueError("the final checkpoint has no stable two-dimensional PCA plane")
    basis = vt[:2].T

    frames: list[MeasuredFrame] = []
    shape = final_activations.shape
    for path in selected_paths:
        activations, selected_layer = _load_centroids(
            path,
            view=view,
            layer_request=layer_request,
        )
        if selected_layer != layer or activations.shape != shape:
            raise ValueError(f"{path} is incompatible with the final checkpoint")
        record = metrics.get(_step(path))
        frames.append(
            MeasuredFrame(
                step=_step(path),
                source=path,
                activations=activations,
                train_accuracy=_optional_accuracy(record, "train_accuracy"),
                test_accuracy=_optional_accuracy(record, "test_accuracy"),
            )
        )
    return (
        config,
        frames,
        basis,
        layer,
        paths[-1],
        memorization,
        generalization,
    )


def project_frames(
    frames: list[MeasuredFrame], basis: np.ndarray
) -> tuple[list[np.ndarray], float]:
    projected = [
        (frame.activations - frame.activations.mean(axis=0, keepdims=True))
        @ basis
        for frame in frames
    ]
    limit = max(
        float(np.max(np.abs(np.concatenate(projected, axis=0)))) * 1.08,
        1e-12,
    )
    return [values / limit for values in projected], limit


def projection_diagnostics(
    final_activations: np.ndarray,
    basis: np.ndarray,
) -> dict[str, float | int]:
    centered = final_activations - final_activations.mean(axis=0, keepdims=True)
    displayed = centered @ basis
    total_energy = float(np.sum(centered**2))
    displayed_energy = float(np.sum(displayed**2))
    phase = 2 * np.pi * np.arange(len(displayed)) / len(displayed)
    denominator = float(
        np.sum((displayed - displayed.mean(axis=0, keepdims=True)) ** 2)
    )
    harmonic_scores: list[tuple[float, int]] = []
    for frequency in range(1, len(displayed) // 2 + 1):
        design = np.column_stack(
            (
                np.cos(frequency * phase),
                np.sin(frequency * phase),
                np.ones(len(phase)),
            )
        )
        prediction = design @ np.linalg.lstsq(
            design, displayed, rcond=None
        )[0]
        score = 1.0 - float(np.sum((displayed - prediction) ** 2)) / max(
            denominator, 1e-12
        )
        harmonic_scores.append((score, frequency))
    harmonic_score, harmonic_frequency = max(harmonic_scores)
    return {
        "final_pca_fraction_of_centered_state_centroid_energy": (
            displayed_energy / max(total_energy, 1e-12)
        ),
        "best_cyclic_harmonic_frequency_in_displayed_plane": harmonic_frequency,
        "best_cyclic_harmonic_fraction_of_displayed_variance": harmonic_score,
    }


def _site_description(view: str, layer: int, depth: int) -> str:
    if view == "node" and layer == 0:
        return (
            "input-state token embedding plus its fixed position embedding, "
            "before the first transformer block; this site has not yet mixed "
            "in the relation token or nuisance context"
        )
    if layer == depth:
        return (
            f"{view}-grouped residual-stream centroids after all {depth} "
            "transformer blocks and the final layer normalization"
        )
    return (
        f"{view}-grouped residual-stream centroids after {layer} transformer "
        f"block{'s' if layer != 1 else ''}"
    )


def render_gif(
    run: Path,
    output: Path,
    *,
    view: str = "node",
    layer_request: str = "0",
    start_step: int | None = None,
    end_step: int | None = None,
    fps: float = 8.0,
) -> Path:
    if fps <= 0:
        raise ValueError("--fps must be positive")
    (
        config,
        measured,
        basis,
        layer,
        basis_source,
        memorization,
        generalization,
    ) = load_measured_frames(
        run,
        view=view,
        layer_request=layer_request,
        start_step=start_step,
        end_step=end_step,
    )
    projected, projection_limit = project_frames(measured, basis)
    final_activations, _ = _load_centroids(
        basis_source,
        view=view,
        layer_request=layer_request,
    )
    diagnostics = projection_diagnostics(final_activations, basis)
    order = projected[0].shape[0]
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("config does not contain model dimensions")
    try:
        depth = int(model["depth"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("config does not contain a valid model depth") from error
    phase_map = LinearSegmentedColormap.from_list("nord-cycle", NORD, N=order)
    colors = phase_map(np.linspace(0, 1, order, endpoint=False))

    fig, axis = plt.subplots(figsize=(5.6, 5.25), dpi=140)
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.975)
    axis.set_aspect("equal")
    axis.set_xlim(-1.03, 1.03)
    axis.set_ylim(-1.03, 1.03)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)
    points = axis.scatter(
        [],
        [],
        s=max(10, min(28, 1_700 / order)),
        alpha=0.82,
        linewidths=0,
    )
    label = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        va="top",
        color=MUTED,
        fontsize=9,
    )

    hold = max(1, round(fps * 0.7))
    frame_indices = list(range(len(projected))) + [len(projected) - 1] * hold

    def update(frame_index: int):
        frame = measured[frame_index]
        points.set_offsets(projected[frame_index])
        points.set_color(colors)
        accuracy = (
            ""
            if frame.test_accuracy is None
            else f"  ·  held-out {frame.test_accuracy:.1%}"
        )
        label.set_text(f"step {frame.step:,}{accuracy}")
        return points, label

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1_000 / fps,
        blit=True,
        repeat=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=140)
    plt.close(fig)

    manifest = {
        "artifact": str(output),
        "run": str(run),
        "run_name": str(config.get("run_name", run.name)),
        "task": str(config.get("task")),
        "task_family": str(config.get("task_family")),
        "preset": str(config.get("preset")),
        "seed": int(config.get("seed", 0)),
        "view": view,
        "layer": layer,
        "site_description": _site_description(view, layer, depth),
        "point_unit": (
            "one activation centroid per latent state, averaged by the training "
            "evaluator across operation-table partners and evaluation contexts"
        ),
        "frame_steps": [frame.step for frame in measured],
        "frame_sources": [str(frame.source) for frame in measured],
        "basis_source": str(basis_source),
        "basis_step": _step(basis_source),
        "projection": (
            "unsupervised two-component PCA fitted once to centered state "
            "centroids at the run's final saved checkpoint"
        ),
        "projection_diagnostic_population": (
            "the selected layer's centered latent-state centroids; the reported "
            "fraction excludes within-state activation variance and activations "
            "outside this centroid set"
        ),
        "projection_diagnostics": diagnostics,
        "alignment": (
            "the final PCA basis is applied unchanged to every frame; each "
            "checkpoint is centered to remove translation; no per-frame "
            "rotation, reflection, or Procrustes alignment is used"
        ),
        "temporal_rendering": (
            "saved checkpoints only; no temporal interpolation or synthesized "
            "point positions; the final measured frame is repeated briefly"
        ),
        "scale": (
            "one fixed scale from the maximum absolute projected coordinate "
            "across all displayed measured checkpoints"
        ),
        "projection_limit_before_normalization": projection_limit,
        "color": (
            "fixed latent-state index mapped continuously through the Nord "
            "palette; color does not determine the projection"
        ),
        "memorization_step_train_accuracy_ge_0p99": memorization,
        "generalization_step_test_accuracy_ge_0p90": generalization,
        "no_edges": True,
        "no_interpolation": True,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _write_fixture(run: Path) -> None:
    rng = np.random.default_rng(29)
    run.mkdir(parents=True)
    order, width, layers = 17, 12, 2
    steps = (0, 500, 1_000, 1_500, 2_000)
    phase = 2 * np.pi * np.arange(order) / order
    circle = np.column_stack((np.cos(phase), np.sin(phase)))
    embedding, _ = np.linalg.qr(rng.normal(size=(width, 2)))
    config = {
        "run_name": "cycle17-fixture",
        "task": "cycle17",
        "task_family": "cycle",
        "task_corruption_fraction": 0.0,
        "preset": "fixture",
        "seed": 29,
        "model": {"depth": layers - 1, "width": width},
    }
    (run / "config.json").write_text(json.dumps(config) + "\n")
    metrics = []
    for index, step in enumerate(steps):
        progress = index / (len(steps) - 1)
        measured = (
            (1 - progress) * rng.normal(size=(order, 2))
            + progress * circle
        ) @ embedding.T
        stored = np.stack((measured, 0.8 * measured), axis=0)
        np.savez_compressed(
            run / f"activations-{step:06d}.npz",
            node=stored.astype(np.float32),
            output=stored.astype(np.float32),
        )
        metrics.append(
            {
                "step": step,
                "train_accuracy": min(1.0, 0.6 + 0.4 * index),
                "test_accuracy": min(1.0, 0.02 + 0.49 * index),
            }
        )
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metrics)
    )


def self_test(destination: Path | None) -> None:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if destination is None:
        temporary = tempfile.TemporaryDirectory(prefix="measured-geometry-gif-")
        root = Path(temporary.name)
    else:
        root = destination
        root.mkdir(parents=True, exist_ok=True)
    run = root / "fixture"
    output = root / "measured-geometry.gif"
    _write_fixture(run)
    manifest_path = render_gif(
        run,
        output,
        view="node",
        layer_request="0",
        start_step=0,
        end_step=2_000,
        fps=4,
    )
    manifest = json.loads(manifest_path.read_text())
    if not output.exists() or output.stat().st_size == 0:
        raise AssertionError("self-test did not create a GIF")
    if manifest["frame_steps"] != [0, 500, 1_000, 1_500, 2_000]:
        raise AssertionError("manifest does not preserve measured checkpoint steps")
    if manifest["basis_step"] != 2_000:
        raise AssertionError("projection was not fitted at the final checkpoint")
    diagnostics = manifest["projection_diagnostics"]
    if not (
        0.0
        < diagnostics["final_pca_fraction_of_centered_state_centroid_energy"]
        <= 1.0
    ):
        raise AssertionError("invalid final PCA explained-variance diagnostic")
    if "input-state token embedding" not in manifest["site_description"]:
        raise AssertionError("manifest obscures the layer-0 embedding site")
    if "excludes within-state activation variance" not in manifest[
        "projection_diagnostic_population"
    ]:
        raise AssertionError("manifest obscures the PCA diagnostic population")
    if not manifest["no_interpolation"] or not manifest["no_edges"]:
        raise AssertionError("manifest does not prohibit interpolation and edges")
    from PIL import Image

    with Image.open(output) as image:
        if image.n_frames < len(manifest["frame_steps"]):
            raise AssertionError("GIF dropped measured checkpoint frames")
        if image.info.get("loop") != 0:
            raise AssertionError("GIF is not configured to loop indefinitely")
    print(f"self-test passed: {output} and {manifest_path}")
    if temporary is not None:
        temporary.cleanup()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.self_test_output)
        return
    if args.run is None or args.output is None:
        raise SystemExit("--run and --output are required")
    manifest = render_gif(
        args.run,
        args.output,
        view=args.view,
        layer_request=args.layer,
        start_step=args.start_step,
        end_step=args.end_step,
        fps=args.fps,
    )
    print(f"wrote {args.output} and {manifest}")


if __name__ == "__main__":
    main()
