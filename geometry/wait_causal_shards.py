from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path


SUCCESS = {"complete", "skipped_valid"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all causal endpoint shards and write one atomic handoff "
            "marker for rendering and artifact packing."
        )
    )
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    return parser.parse_args()


def load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def validate(args: argparse.Namespace) -> dict[str, object] | None:
    runs: dict[str, str] = {}
    shards: list[dict[str, object]] = []
    for index in range(args.shard_count):
        manifest_path = args.log_root / f"shard-{index}-manifest.json"
        results_path = args.log_root / f"shard-{index}-results.json"
        manifest = load_json(manifest_path)
        results = load_json(results_path)
        if not isinstance(manifest, dict) or not isinstance(results, list):
            return None
        if (
            int(manifest.get("final_step", -1)) != args.final_step
            or int(manifest.get("shard_index", -1)) != index
            or int(manifest.get("shard_count", -1)) != args.shard_count
            or int(manifest.get("run_count", -1)) != len(results)
        ):
            raise ValueError(f"shard {index} manifest does not match its results")
        expected_names = set(manifest.get("runs", []))
        observed_names = {
            str(result.get("run"))
            for result in results
            if isinstance(result, dict)
        }
        if expected_names != observed_names or len(observed_names) != len(results):
            raise ValueError(f"shard {index} contains missing or duplicate runs")
        for result in results:
            assert isinstance(result, dict)
            if result.get("status") not in SUCCESS:
                if result.get("status") == "failed":
                    raise RuntimeError(f"causal shard {index} contains a failure")
                return None
            output = Path(str(result.get("output", "")))
            if not output.with_suffix(".json").exists():
                return None
            name = str(result["run"])
            if name in runs:
                raise ValueError(f"run {name} appears in multiple shards")
            runs[name] = str(output)
        shards.append(
            {
                "index": index,
                "manifest": str(manifest_path),
                "results": str(results_path),
                "run_count": len(results),
            }
        )
    if len(runs) != args.expected_runs:
        return None
    return {
        "completed_at": datetime.now(UTC).isoformat(),
        "final_step": args.final_step,
        "expected_runs": args.expected_runs,
        "shard_count": args.shard_count,
        "shards": shards,
        "outputs": runs,
    }


def main() -> None:
    args = parse_args()
    if min(
        args.final_step,
        args.expected_runs,
        args.shard_count,
        args.poll_seconds,
        args.timeout_hours,
    ) <= 0:
        raise ValueError("steps, counts, polling, and timeout must be positive")
    deadline = time.monotonic() + args.timeout_hours * 3600
    while time.monotonic() < deadline:
        payload = validate(args)
        if payload is not None:
            args.marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.marker.with_suffix(args.marker.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(args.marker)
            print(f"validated {args.expected_runs} causal endpoint outputs")
            return
        time.sleep(args.poll_seconds)
    raise TimeoutError("causal shards did not complete before timeout")


if __name__ == "__main__":
    main()
