from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from pythia_geometry import DOMAINS, safe_name


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
WEEKDAYS = next(domain for domain in DOMAINS if domain.name == "weekdays")
DEFAULT_REVISIONS = (
    "step0",
    "step4",
    "step64",
    "step1000",
    "step4000",
    "step16000",
    "step64000",
    "step143000",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--revisions", nargs="+", default=DEFAULT_REVISIONS)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gif",
        "--video",
        dest="gif",
        type=Path,
        required=True,
        help="Looping GIF destination; --video remains as a compatibility alias.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.25,
        help="Measured checkpoints shown per second.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def revision_step(revision: str) -> int:
    return int(revision.removeprefix("step"))


def weekday_prompts() -> list[str]:
    return [
        template.format(value=value)
        for value in WEEKDAYS.values
        for template in WEEKDAYS.activation_templates
    ]


@torch.no_grad()
def extract_samples(
    *,
    model_name: str,
    revision: str,
    layer: int,
    cache_dir: Path,
    device: torch.device,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    prompts = weekday_prompts()
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    outputs = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if not 0 <= layer < len(outputs.hidden_states):
        raise ValueError(
            f"layer {layer} is outside 0..{len(outputs.hidden_states) - 1}"
        )
    last_positions = encoded.attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(len(prompts), device=device)
    selected = outputs.hidden_states[layer][
        batch_indices, last_positions
    ].float()
    samples = selected.cpu().numpy().reshape(
        len(WEEKDAYS.values),
        len(WEEKDAYS.activation_templates),
        -1,
    )

    del outputs
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return samples


def ensure_samples(args: argparse.Namespace) -> list[Path]:
    model_dir = args.output_dir / safe_name(args.model)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for revision in args.revisions:
        path = model_dir / f"{revision}-layer{args.layer}.npz"
        paths.append(path)
        if path.exists():
            print(f"{revision}: already extracted", flush=True)
            continue
        samples = extract_samples(
            model_name=args.model,
            revision=revision,
            layer=args.layer,
            cache_dir=args.cache_dir,
            device=torch.device(args.device),
        )
        np.savez_compressed(
            path,
            samples=samples.astype(np.float32),
            prompts=np.asarray(weekday_prompts()),
        )
        print(f"{revision}: wrote {path}", flush=True)
    return paths


def remove_template_offsets(samples: np.ndarray) -> np.ndarray:
    centered = samples - samples.mean(axis=0, keepdims=True)
    return centered.reshape(-1, centered.shape[-1])


def mature_phase_projection(frames: list[np.ndarray]) -> list[np.ndarray]:
    final = frames[-1]
    phase = 2 * np.pi * np.arange(len(WEEKDAYS.values)) / len(
        WEEKDAYS.values
    )
    targets = np.column_stack((np.cos(phase), np.sin(phase)))
    targets = np.repeat(targets, len(WEEKDAYS.activation_templates), axis=0)
    gram = final @ final.T
    ridge = 1e-3 * np.trace(gram) / max(len(gram), 1)
    decoder = final.T @ np.linalg.solve(
        gram + ridge * np.eye(len(gram)),
        targets,
    )
    projected = [frame @ decoder for frame in frames]
    final_radius = np.sqrt(np.mean(np.sum(projected[-1] ** 2, axis=1)))
    return [frame / max(final_radius, 1e-12) for frame in projected]


def render_gif(
    sample_paths: list[Path],
    revisions: tuple[str, ...] | list[str],
    output: Path,
    *,
    fps: float,
) -> None:
    raw = [np.load(path)["samples"] for path in sample_paths]
    high_dimensional = [remove_template_offsets(frame) for frame in raw]
    frames = mature_phase_projection(high_dimensional)
    steps = [revision_step(revision) for revision in revisions]

    coordinates = np.concatenate(frames)
    limit = max(1.25, float(np.quantile(np.abs(coordinates), 0.995)) * 1.08)
    groups = np.repeat(
        np.arange(len(WEEKDAYS.values)),
        len(WEEKDAYS.activation_templates),
    )

    fig, axis = plt.subplots(figsize=(5.8, 5.4))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.11)
    axis.set_facecolor("white")
    axis.set_aspect("equal")
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)

    points = axis.scatter([], [], s=16, alpha=0.28)
    centroids = axis.scatter(
        [],
        [],
        s=58,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    step_label = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        va="top",
        color=MUTED,
        fontsize=10,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            markersize=5,
        )
        for color in NORD
    ]
    axis.legend(
        legend_handles,
        [value[:3] for value in WEEKDAYS.values],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=7,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
        fontsize=8,
        labelcolor=INK,
    )

    def update(index: int):
        frame = frames[index]
        centers = np.stack(
            [frame[groups == group].mean(axis=0) for group in range(7)]
        )
        points.set_offsets(frame)
        points.set_color([NORD[group] for group in groups])
        centroids.set_offsets(centers)
        centroids.set_color(NORD)
        step_label.set_text(f"step {steps[index]:,}")
        return points, centroids, step_label

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=True,
        repeat=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output,
        writer=PillowWriter(fps=fps),
        dpi=160,
    )
    update(len(frames) - 1)
    fig.savefig(
        output.with_name(f"{output.stem}-poster.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    sample_paths = ensure_samples(args)
    render_gif(sample_paths, args.revisions, args.gif, fps=args.fps)
    print(f"wrote {args.gif}", flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
