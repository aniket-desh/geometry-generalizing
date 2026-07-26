from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def value_at(record: dict[str, object], key: str) -> float | None:
    value: object = record
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def first_step(
    records: list[dict[str, object]], key: str, threshold: float
) -> int | None:
    for record in records:
        value = value_at(record, key)
        if value is not None and value >= threshold:
            return int(record["step"])
    return None


def minimum_step(records: list[dict[str, object]], key: str) -> int | None:
    candidates = [
        (value, int(record["step"]))
        for record in records
        if (value := value_at(record, key)) is not None
    ]
    return min(candidates)[1] if candidates else None


def load_runs(root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for config_path in sorted(root.glob("*/config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text())
        records = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
        if not records:
            continue
        records.sort(key=lambda record: int(record["step"]))
        final = records[-1]
        runs.append(
            {
                "run": run_dir.name,
                "task": config["task"],
                "family": config["task_family"],
                "preset": config["preset"],
                "seed": config["seed"],
                "parameter_count": config["parameter_count"],
                "final_step": final["step"],
                "final_train_accuracy": final["train_accuracy"],
                "final_test_accuracy": final["test_accuracy"],
                "generalization_step_50": first_step(
                    records, "test_accuracy", 0.5
                ),
                "generalization_step_90": first_step(
                    records, "test_accuracy", 0.9
                ),
                "minimum_node_cyclic_defect_step": minimum_step(
                    records, "node_geometry.cyclic_defect"
                ),
                "minimum_output_cyclic_defect_step": minimum_step(
                    records, "output_geometry.cyclic_defect"
                ),
                "records": records,
            }
        )
    return runs


def aggregate(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        groups[(str(run["task"]), str(run["preset"]))].append(run)
    summaries: list[dict[str, object]] = []
    for (task, preset), group in sorted(groups.items()):
        final_accuracy = np.asarray(
            [float(run["final_test_accuracy"]) for run in group]
        )
        generalized = [
            int(run["generalization_step_90"])
            for run in group
            if run["generalization_step_90"] is not None
        ]
        summaries.append(
            {
                "task": task,
                "preset": preset,
                "runs": len(group),
                "median_final_test_accuracy": float(np.median(final_accuracy)),
                "min_final_test_accuracy": float(np.min(final_accuracy)),
                "max_final_test_accuracy": float(np.max(final_accuracy)),
                "generalized_runs": len(generalized),
                "median_generalization_step_90": (
                    float(np.median(generalized)) if generalized else None
                ),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    output = args.output or args.results / "summary.json"
    runs = load_runs(args.results)
    rows = [
        {key: value for key, value in run.items() if key != "records"}
        for run in runs
    ]
    payload = {
        "run_count": len(runs),
        "groups": aggregate(runs),
        "runs": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {output} and {csv_path}")


if __name__ == "__main__":
    main()
