from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from key60_common import KEY_CONDITIONS, SEEDS, atomic_json, now
from launch_reuse import Run, run_one
from priority_common import HORIZONS
from priority_train import SHARD_COUNT


LARGE_PRESET = "large"
LARGE_BATCH_SIZE = 1_024
LARGE_COUNT = len(KEY_CONDITIONS) * len(SEEDS)
SHARD_IDENTITIES = (
    (("clean", 0), ("corrupt15", 0)),
    (("clean", 1), ("corrupt15", 1)),
    (("clean", 2), ("corrupt15", 2)),
    (("random", 0), ("random", 1), ("random", 2)),
)


def schedule() -> list[Run]:
    return [
        Run(
            task=task,
            corruption=corruption,
            condition=condition,
            preset=LARGE_PRESET,
            seed=seed,
            steps=HORIZONS[condition],
            dense_checkpoint_every=10_000,
            batch_size=LARGE_BATCH_SIZE,
            eval_every=1_000,
            snapshot_every=5_000,
            checkpoint_every=30_000,
            keep_checkpoints=2,
        )
        for task, corruption, condition in KEY_CONDITIONS
        for seed in SEEDS
    ]


def partition(runs: list[Run]) -> list[list[Run]]:
    indexed = {(run.condition, run.seed): run for run in runs}
    expected = {
        (condition, seed) for _, _, condition in KEY_CONDITIONS for seed in SEEDS
    }
    if len(indexed) != len(runs) or set(indexed) != expected:
        raise ValueError("large schedule is not the exact nine-run identity set")
    return [
        [indexed[identity] for identity in identities]
        for identities in SHARD_IDENTITIES
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one explicit H100 shard of the exact large-model extension."
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    runs = schedule()
    expected = {
        (condition, LARGE_PRESET, seed)
        for _, _, condition in KEY_CONDITIONS
        for seed in SEEDS
    }
    observed = {(run.condition, run.preset, run.seed) for run in runs}
    if len(runs) != LARGE_COUNT or observed != expected:
        raise AssertionError("large extension is not the exact nine-run matrix")
    if any(
        run.steps != HORIZONS[run.condition]
        or run.batch_size != LARGE_BATCH_SIZE
        or run.eval_every != 1_000
        or run.snapshot_every != 5_000
        or run.dense_checkpoint_every != 10_000
        or run.checkpoint_every != 30_000
        or run.keep_checkpoints != 2
        for run in runs
    ):
        raise AssertionError("large extension lost its fixed training protocol")
    shards = partition(runs)
    flattened = [
        (run.condition, run.preset, run.seed) for shard in shards for run in shard
    ]
    if len(flattened) != len(set(flattened)) or set(flattened) != expected:
        raise AssertionError("large shards overlap or omit an identity")
    work = [sum(run.steps for run in shard) for shard in shards]
    if work != [90_000] * SHARD_COUNT:
        raise AssertionError(f"large shards are imbalanced: {work}")
    print(
        "self-test passed: 9 exact large runs partition into four explicit "
        "disjoint 90,000-step H100 shards"
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

    selected = partition(schedule())[args.shard_index]
    shard_root = args.log_root / "training" / f"shard-{args.shard_index}"
    manifest = {
        "profile": "large-extension",
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
            f"{now()} large shard {args.shard_index} "
            f"[{offset}/{len(selected)}] {run.slug}: rc={result['returncode']}",
            flush=True,
        )
        if result["returncode"] != 0:
            raise SystemExit(f"large training failed: {run.slug}")
    atomic_json(
        shard_root / "complete.json",
        {
            "status": "complete",
            "completed_at": now(),
            "shard_index": args.shard_index,
            "scheduled_steps": sum(run.steps for run in selected),
            "runs": [run.slug for run in selected],
        },
    )


if __name__ == "__main__":
    main()
