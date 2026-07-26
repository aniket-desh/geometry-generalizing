from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


# Feature clusters reported for Figure 1 of Engels et al. (2024):
# https://github.com/JoshEngels/MultiDimensionalFeatures
DAY_FEATURES = (2592, 4445, 4663, 4733, 6531, 8179, 9566, 20927, 24185)
YEAR_FEATURES = (1052, 2753, 4427, 6382, 8314, 9576, 9606, 13551, 19734, 20349)
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DAY_LOOKUP = {
    alias: index
    for index, aliases in enumerate(
        (
            ("monday", "mondays", "mon"),
            ("tuesday", "tuesdays", "tues"),
            ("wednesday", "wednesdays", "wed"),
            ("thursday", "thursdays", "thurs"),
            ("friday", "fridays", "fri"),
            ("saturday", "saturdays", "sat"),
            ("sunday", "sundays", "sun"),
        )
    )
    for alias in aliases
}
NORD = (
    "#5E81AC",
    "#88C0D0",
    "#A3BE8C",
    "#EBCB8B",
    "#D08770",
    "#BF616A",
    "#B48EAD",
)
NORD_GREY = "#D8DEE9"
NORD_INK = "#2E3440"
YEAR_CMAP = LinearSegmentedColormap.from_list(
    "nord-years",
    ("#B48EAD", "#5E81AC", "#88C0D0", "#A3BE8C", "#EBCB8B"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-points", type=int, default=20_000)
    parser.add_argument("--target-days", type=int, default=450)
    parser.add_argument("--target-years", type=int, default=900)
    return parser.parse_args()


def reconstruct_cluster(
    feature_acts: torch.Tensor,
    feature_indices: tuple[int, ...],
    decoder: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.as_tensor(
        feature_indices,
        device=feature_acts.device,
        dtype=torch.long,
    )
    selected = feature_acts.index_select(1, indices)
    active = selected.abs().amax(dim=1) > 1e-6
    reconstruction = selected[active] @ decoder.index_select(0, indices)
    return reconstruction, active


def decoded_tokens(tokenizer, tokens: torch.Tensor) -> list[str]:
    return [
        tokenizer.decode([int(token)], clean_up_tokenization_spaces=False)
        for token in tokens
    ]


def day_index(token: str) -> int:
    return DAY_LOOKUP.get(token.lower().strip(), -1)


def year_value(token: str) -> int:
    stripped = token.strip()
    if stripped.isdigit():
        value = int(stripped)
        if 1900 <= value <= 1999:
            return value
    return -1


@torch.no_grad()
def collect(args: argparse.Namespace) -> dict[str, np.ndarray]:
    from datasets import load_dataset
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
        cache_dir=str(args.cache_dir),
    )
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.7.hook_resid_pre",
        device=str(device),
    )[0]
    sae.eval()
    hook_name = "blocks.7.hook_resid_pre"
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    day_vectors: list[np.ndarray] = []
    day_tokens: list[str] = []
    year_vectors: list[np.ndarray] = []
    year_tokens: list[str] = []
    tokens_seen = 0
    day_targets = 0
    year_targets = 0
    documents = iter(dataset)

    while tokens_seen < args.max_tokens:
        tokenized = []
        while len(tokenized) < args.batch_size:
            row = next(documents)
            tokens = model.to_tokens(row["text"])[0, : args.context_length]
            if len(tokens) == args.context_length:
                tokenized.append(tokens)
        batch = torch.stack(tokenized).to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_name)
        residual = cache[hook_name].reshape(-1, model.cfg.d_model)
        feature_acts = sae.encode(residual)
        flat_tokens = batch.reshape(-1)

        for name, features, vector_chunks, token_chunks in (
            ("days", DAY_FEATURES, day_vectors, day_tokens),
            ("years", YEAR_FEATURES, year_vectors, year_tokens),
        ):
            reconstruction, active = reconstruct_cluster(
                feature_acts,
                features,
                sae.W_dec,
            )
            strings = decoded_tokens(model.tokenizer, flat_tokens[active])
            vector_chunks.append(reconstruction.float().cpu().numpy())
            token_chunks.extend(strings)
            if name == "days":
                day_targets += sum(day_index(token) >= 0 for token in strings)
            else:
                year_targets += sum(year_value(token) >= 0 for token in strings)

        tokens_seen += flat_tokens.numel()
        print(
            f"tokens={tokens_seen:,} "
            f"weekday_points={day_targets:,} "
            f"year_points={year_targets:,}",
            flush=True,
        )
        enough_points = (
            sum(len(chunk) for chunk in day_vectors) >= 2_000
            and sum(len(chunk) for chunk in year_vectors) >= 2_000
        )
        enough_targets = (
            day_targets >= args.target_days
            and year_targets >= args.target_years
        )
        if enough_points and enough_targets:
            break

    days = np.concatenate(day_vectors)[: args.max_points]
    years = np.concatenate(year_vectors)[: args.max_points]
    return {
        "day_reconstructions": days,
        "day_tokens": np.asarray(day_tokens[: len(days)]),
        "year_reconstructions": years,
        "year_tokens": np.asarray(year_tokens[: len(years)]),
        "tokens_seen": np.asarray(tokens_seen),
    }


def render(payload: dict[str, np.ndarray], output: Path) -> None:
    day_reconstructions = payload["day_reconstructions"]
    day_tokens = payload["day_tokens"]
    year_reconstructions = payload["year_reconstructions"]
    year_tokens = payload["year_tokens"]

    day_projection = PCA(n_components=4).fit_transform(day_reconstructions)
    year_projection = PCA(n_components=5).fit_transform(year_reconstructions)
    day_groups = np.asarray([day_index(str(token)) for token in day_tokens])
    years = np.asarray([year_value(str(token)) for token in year_tokens])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.7),
        constrained_layout=True,
    )
    fig.patch.set_alpha(0)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
        axis.set_aspect("equal", adjustable="datalim")

    axes[0].scatter(
        day_projection[day_groups < 0, 1],
        day_projection[day_groups < 0, 2],
        s=7,
        color=NORD_GREY,
        alpha=0.25,
        linewidths=0,
        rasterized=True,
    )
    for index, color in enumerate(NORD):
        selected = day_groups == index
        axes[0].scatter(
            day_projection[selected, 1],
            day_projection[selected, 2],
            s=10,
            color=color,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
    axes[0].set_title("weekdays", color=NORD_INK, fontsize=12)
    axes[0].legend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color,
                markersize=5,
            )
            for color in NORD
        ],
        [name[:3] for name in DAY_NAMES],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=7,
        frameon=False,
        fontsize=8,
        handletextpad=0.2,
        columnspacing=0.7,
        labelcolor=NORD_INK,
    )

    axes[1].scatter(
        year_projection[years < 0, 2],
        year_projection[years < 0, 3],
        s=7,
        color=NORD_GREY,
        alpha=0.25,
        linewidths=0,
        rasterized=True,
    )
    selected_years = years >= 0
    axes[1].scatter(
        year_projection[selected_years, 2],
        year_projection[selected_years, 3],
        s=9,
        c=years[selected_years],
        cmap=YEAR_CMAP,
        norm=Normalize(1900, 1999),
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    axes[1].set_title("twentieth-century years", color=NORD_INK, fontsize=12)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=YEAR_CMAP,
            norm=Normalize(1900, 1999),
        ),
        ax=axes[1],
        orientation="horizontal",
        fraction=0.045,
        pad=0.02,
        aspect=25,
    )
    colorbar.set_ticks((1900, 1950, 1999))
    colorbar.ax.tick_params(
        colors=NORD_INK,
        labelsize=8,
        length=0,
        pad=2,
    )
    colorbar.outline.set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, transparent=True, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), transparent=True, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = args.output.with_suffix(".npz")
    if data_path.exists():
        payload = dict(np.load(data_path))
        print(f"loaded {data_path}", flush=True)
    else:
        payload = collect(args)
        np.savez_compressed(data_path, **payload)
        print(f"wrote {data_path}", flush=True)
    render(payload, args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
