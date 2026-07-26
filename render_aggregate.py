#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ohope.config import MODEL_SPECS, NORD
from ohope.plotting import _save, _spaghetti, apply_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available = [
        spec
        for spec in MODEL_SPECS.values()
        if (args.results / spec.slug / "pruning_curves.csv").exists()
    ]
    if not available:
        print("no completed pruning curves found")
        return

    apply_style()
    metrics = (
        ("perplexity", "perplexity"),
        ("kl", "output KL"),
        ("top_agreement", "top-token agreement"),
    )
    fig, axes = plt.subplots(
        len(available),
        len(metrics),
        figsize=(10.8, 2.65 * len(available)),
        squeeze=False,
    )
    for row, spec in enumerate(available):
        frame = pd.read_csv(args.results / spec.slug / "pruning_curves.csv")
        for column, (metric, label) in enumerate(metrics):
            ax = axes[row, column]
            _spaghetti(ax, frame, y=metric, ylabel=label if column == 0 else "")
            ax.invert_xaxis()
            if row < len(available) - 1:
                ax.set_xlabel("")
            if column == 0:
                ax.text(
                    -0.24,
                    0.5,
                    spec.key,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    color=NORD["ink"],
                )
    axes[0, -1].legend(loc="best")
    fig.tight_layout(h_pad=1.4, w_pad=1.5)
    output = args.results / "figures" / "aggregate-pruning.png"
    _save(fig, output)
    print(output)


if __name__ == "__main__":
    main()
