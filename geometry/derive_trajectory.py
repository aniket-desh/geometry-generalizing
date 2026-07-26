from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geogen.metrics import cyclic_gram_defect, fourier_energy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--layer", type=int, default=0)
    return parser.parse_args()


def mode_summary(activations: np.ndarray) -> tuple[float, float]:
    _, spectrum = fourier_energy(activations)
    probabilities = np.asarray(spectrum, dtype=np.float64)
    probabilities = probabilities[probabilities > 1e-15]
    if not probabilities.size:
        return 0.0, 0.0
    mode_rank = float(
        np.exp(-np.sum(probabilities * np.log(probabilities)))
    )
    top_three = float(
        np.sort(probabilities)[-min(3, probabilities.size) :].sum()
    )
    return mode_rank, top_three


def main() -> None:
    args = parse_args()
    run_count = 0
    record_count = 0
    for config_path in sorted(args.results.glob("*/config.json")):
        run_dir = config_path.parent
        config = json.loads(config_path.read_text())
        if config["task"] != args.task:
            continue
        behavior = {
            int(record["step"]): record
            for record in (
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
                if line.strip()
            )
        }
        records: list[dict[str, object]] = []
        for snapshot_path in sorted(run_dir.glob("activations-*.npz")):
            step = int(snapshot_path.stem.rsplit("-", 1)[-1])
            if step not in behavior:
                continue
            nodes = np.load(snapshot_path)["node"]
            activations = nodes[args.layer]
            mode_rank, top_three = mode_summary(activations)
            records.append(
                {
                    "step": step,
                    "test_accuracy": behavior[step]["test_accuracy"],
                    "train_accuracy": behavior[step]["train_accuracy"],
                    "cyclic_defect": cyclic_gram_defect(activations),
                    "fourier_mode_rank": mode_rank,
                    "top_fourier_energy": top_three,
                }
            )
            record_count += 1
        if records:
            (run_dir / "trajectory.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            run_count += 1
    print(f"wrote {record_count} records across {run_count} runs")


if __name__ == "__main__":
    main()
