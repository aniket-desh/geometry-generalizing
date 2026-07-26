#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ohope.config import MODEL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pruning_rows = []
    ablation_rows = []
    geometry_rows = []

    for spec in MODEL_SPECS.values():
        model_dir = args.results / spec.slug
        if not (model_dir / "done.json").exists():
            continue

        curves = pd.read_csv(model_dir / "pruning_curves.csv")
        medians = (
            curves.groupby(["method", "fraction_pruned"], as_index=False)
            .median(numeric_only=True)
        )
        medians.insert(0, "model", spec.key)
        pruning_rows.append(medians)

        correlations = pd.DataFrame(
            json.loads((model_dir / "ablation_correlations.json").read_text())
        )
        correlation_columns = [
            column for column in correlations if column.startswith("spearman_")
        ]
        ablation = correlations.groupby("method", as_index=False)[
            correlation_columns
        ].median()
        ablation.insert(0, "model", spec.key)
        ablation_rows.append(ablation)

        with np.load(model_dir / "geometry.npz") as geometry:
            eig = np.asarray(geometry["normalized_eigenvalues"])
            q = np.asarray(geometry["q_normalized"])
            geometry_rows.append(
                {
                    "model": spec.key,
                    "stable_condition_number": float(
                        geometry["stable_condition_number"]
                    ),
                    "eigenvalue_p05": float(np.quantile(eig, 0.05)),
                    "eigenvalue_median": float(np.median(eig)),
                    "eigenvalue_p95": float(np.quantile(eig, 0.95)),
                    "q_p10": float(np.quantile(q, 0.10)),
                    "q_median": float(np.median(q)),
                    "q_p90": float(np.quantile(q, 0.90)),
                }
            )

    if not pruning_rows:
        raise SystemExit("no completed models found")
    summary_dir = args.results / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(pruning_rows, ignore_index=True).to_csv(
        summary_dir / "pruning_medians.csv",
        index=False,
    )
    pd.concat(ablation_rows, ignore_index=True).to_csv(
        summary_dir / "ablation_medians.csv",
        index=False,
    )
    pd.DataFrame(geometry_rows).to_csv(
        summary_dir / "geometry.csv",
        index=False,
    )
    print(summary_dir)


if __name__ == "__main__":
    main()
