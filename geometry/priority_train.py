from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from key60_common import KEY_CONDITIONS, KEY_COUNT, PRESETS, SEEDS, atomic_json, now
from launch_reuse import Run, run_one
from priority_common import HORIZONS


SHARD_COUNT = 4


def schedule() -> list[Run]:
    return [
        Run(
            task=task,
            corruption=corruption,
            condition=condition,
            preset=preset,
            seed=seed,
            steps=HORIZONS[condition],
            dense_checkpoint_every=1_000,
        )
        for task, corruption, condition in KEY_CONDITIONS
        for preset in PRESETS
        for seed in SEEDS
    ]


def partition(runs: list[Run], shard_count: int = SHARD_COUNT) -> list[list[Run]]:
    if shard_count != SHARD_COUNT:
        raise ValueError(f"priority training requires exactly {SHARD_COUNT} shards")
    shards: list[list[Run]] = [[] for _ in range(shard_count)]
    work = [0] * shard_count
    ordered = sorted(
        runs,
        key=lambda run: (-run.steps, run.preset, run.condition, run.seed),
    )
    for run in ordered:
        shard = min(range(shard_count), key=lambda index: (work[index], index))
        shards[shard].append(run)
        work[shard] += run.steps
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one disjoint shard of the mixed-horizon priority matrix."
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    runs = schedule()
    identities = {(run.condition, run.preset, run.seed) for run in runs}
    expected = {
        (condition, preset, seed)
        for _, _, condition in KEY_CONDITIONS
        for preset in PRESETS
        for seed in SEEDS
    }
    if len(runs) != KEY_COUNT or identities != expected:
        raise AssertionError("priority schedule is not the exact 18-run matrix")
    if any(run.steps != HORIZONS[run.condition] for run in runs):
        raise AssertionError("priority schedule has an incorrect horizon")

    shards = partition(runs)
    flattened = [run for shard in shards for run in shard]
    flattened_identities = [
        (run.condition, run.preset, run.seed) for run in flattened
    ]
    if len(flattened_identities) != len(set(flattened_identities)):
        raise AssertionError("priority shards contain a duplicate identity")
    if set(flattened_identities) != expected:
        raise AssertionError("priority shards do not cover the exact matrix")
    work = [sum(run.steps for run in shard) for shard in shards]
    if len(set(work)) != 1:
        raise AssertionError(f"priority shards are imbalanced: {work}")
    print(
        "self-test passed: 18 unique mixed-horizon runs partition into "
        f"four disjoint {work[0]:,}-step shards"
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.shard_count != SHARD_COUNT:
        raise ValueError(f"--shard-count must be {SHARD_COUNT}")
    if args.shard_index is None or not 0 <= args.shard_index < SHARD_COUNT:
        raise ValueError(f"--shard-index must be between 0 and {SHARD_COUNT - 1}")
    if args.output_root is None or args.log_root is None:
        raise ValueError("--output-root and --log-root are required")
    if args.min_free_gb <= 0:
        raise ValueError("--min-free-gb must be positive")

    selected = partition(schedule(), args.shard_count)[args.shard_index]
    shard_root = args.log_root / "training" / f"shard-{args.shard_index}"
    manifest = {
        "profile": "priority-mixed",
        "created_at": now(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "device": args.device,
        "compile": args.compile,
        "run_count": len(selected),
        "scheduled_steps": sum(run.steps for run in selected),
        "runs": [asdict(run) for run in selected],
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    atomic_json(shard_root / "manifest.json", manifest)
    results: list[dict[str, object]] = []
    for offset, run in enumerate(selected, start=1):
        result = run_one(
            run,
            output_root=args.output_root,
            log_root=shard_root,
            compile_model=args.compile,
            device=args.device,
            min_free_gb=args.min_free_gb,
        )
        results.append(result)
        atomic_json(shard_root / "results.json", results)
        print(
            f"{now()} shard {args.shard_index} [{offset}/{len(selected)}] "
            f"{run.slug}: rc={result['returncode']}",
            flush=True,
        )
        if result["returncode"] != 0:
            raise SystemExit(f"priority training failed: {run.slug}")
    atomic_json(
        shard_root / "complete.json",
        {
            "status": "complete",
            "completed_at": now(),
            "shard_index": args.shard_index,
            "runs": [run.slug for run in selected],
        },
    )


if __name__ == "__main__":
    main()
