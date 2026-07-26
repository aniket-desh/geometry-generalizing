from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def revision_step(revision: str) -> int:
    if not revision.startswith("step"):
        raise ValueError(f"unexpected revision {revision!r}")
    return int(revision.removeprefix("step"))


def finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def best_layer(
    layers: list[dict[str, object]],
    key: str,
    *,
    maximize: bool,
) -> tuple[int | None, float | None]:
    candidates = [
        (index, value)
        for index, layer in enumerate(layers)
        if (value := finite(layer.get(key))) is not None
    ]
    if not candidates:
        return None, None
    chooser = max if maximize else min
    return chooser(candidates, key=lambda pair: pair[1])


def load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.glob("*/*.json")):
        if metrics_path.name.endswith(".done.json"):
            continue
        model = metrics_path.parent.name.replace("--", "/", 1)
        revision = metrics_path.stem
        payload = json.loads(metrics_path.read_text())
        for domain, metrics in payload.items():
            layers = metrics["layers"]
            cyclic_layer, cyclic_defect = best_layer(
                layers, "cyclic_defect", maximize=False
            )
            generator_layer, generator_error = best_layer(
                layers, "generator_error", maximize=False
            )
            fourier_layer, fundamental_energy = best_layer(
                layers, "fundamental_energy", maximize=True
            )
            behavior = metrics["behavior"]
            rows.append(
                {
                    "model": model,
                    "revision": revision,
                    "step": revision_step(revision),
                    "domain": domain,
                    "order": metrics["order"],
                    "candidate_accuracy": behavior["candidate_accuracy"],
                    "candidate_probability": behavior[
                        "candidate_probability"
                    ],
                    "full_probability": behavior["full_probability"],
                    "best_cyclic_layer": cyclic_layer,
                    "cyclic_defect": cyclic_defect,
                    "best_generator_layer": generator_layer,
                    "generator_error": generator_error,
                    "best_fourier_layer": fourier_layer,
                    "fundamental_energy": fundamental_energy,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output = args.output or args.results / "summary.json"
    rows = load_rows(args.results)
    output.write_text(
        json.dumps({"row_count": len(rows), "rows": rows}, indent=2) + "\n"
    )
    csv_path = output.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {output} and {csv_path}")


if __name__ == "__main__":
    main()
