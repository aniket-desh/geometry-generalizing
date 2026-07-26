from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geogen.metrics import geometry_summary
from geogen.tasks import make_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_count = 0
    snapshot_count = 0
    for config_path in sorted(args.results.glob("*/config.json")):
        run_dir = config_path.parent
        config = json.loads(config_path.read_text())
        split_seed = int(config.get("split_seed", config["seed"]))
        task = make_task(config["task"], seed=split_seed)
        records: list[dict[str, object]] = []
        for snapshot_path in sorted(run_dir.glob("activations-*.npz")):
            step = int(snapshot_path.stem.rsplit("-", 1)[-1])
            snapshot = np.load(snapshot_path)
            record: dict[str, object] = {"step": step}
            for key in ("node", "output"):
                if key not in snapshot:
                    continue
                record[f"{key}_layers"] = [
                    geometry_summary(layer, task)
                    for layer in snapshot[key]
                ]
            records.append(record)
            snapshot_count += 1
        if records:
            output = run_dir / "layer_metrics.jsonl"
            output.write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            run_count += 1
    print(
        f"wrote layer metrics for {run_count} runs "
        f"from {snapshot_count} snapshots"
    )


if __name__ == "__main__":
    main()
