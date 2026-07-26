from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from key60_common import atomic_json, load_json, now
from key60_common import KEY_CONDITIONS, SEEDS
from launch_reuse import Run, run_one
from priority_common import HORIZONS
from priority_train import SHARD_COUNT, partition


SCALE_PRESETS = ("small", "medium")
SCALE_COUNT = len(KEY_CONDITIONS) * len(SCALE_PRESETS) * len(SEEDS)


def schedule() -> list[Run]:
    return [
        Run(
            task=task,
            corruption=corruption,
            condition=condition,
            preset=preset,
            seed=seed,
            steps=HORIZONS[condition],
            dense_checkpoint_every=10_000,
            batch_size=2_048 if preset == "medium" else 4_096,
            eval_every=1_000,
            snapshot_every=5_000,
            checkpoint_every=30_000,
            keep_checkpoints=2,
        )
        for task, corruption, condition in KEY_CONDITIONS
        for preset in SCALE_PRESETS
        for seed in SEEDS
    ]


def wait_for_key_training(
    root: Path,
    *,
    poll_seconds: float,
    timeout_hours: float,
) -> None:
    deadline = time.monotonic() + timeout_hours * 3600
    markers = [root / f"shard-{index}" / "complete.json" for index in range(SHARD_COUNT)]
    while time.monotonic() < deadline:
        if all(
            isinstance(payload := load_json(marker), dict)
            and payload.get("status") == "complete"
            for marker in markers
        ):
            return
        time.sleep(poll_seconds)
    raise TimeoutError("key training did not complete before the scale phase timeout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one restricted shard of the optional small/medium scale phase."
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--wait-for-key-root", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    runs = schedule()
    expected = {
        (condition, preset, seed)
        for _, _, condition in KEY_CONDITIONS
        for preset in SCALE_PRESETS
        for seed in SEEDS
    }
    observed = {(run.condition, run.preset, run.seed) for run in runs}
    if len(runs) != SCALE_COUNT or observed != expected:
        raise AssertionError("scale phase is not the exact matched 18-run matrix")
    if any(run.steps != HORIZONS[run.condition] for run in runs):
        raise AssertionError("scale phase has an incorrect mixed horizon")
    if any(
        run.eval_every != 1_000
        or run.snapshot_every != 5_000
        or run.dense_checkpoint_every != 10_000
        for run in runs
    ):
        raise AssertionError("scale phase lost its sparse artifact schedule")
    shards = partition(runs)
    flattened = [
        (run.condition, run.preset, run.seed) for shard in shards for run in shard
    ]
    if len(flattened) != len(set(flattened)) or set(flattened) != expected:
        raise AssertionError("scale shards overlap or omit an identity")
    work = [sum(run.steps for run in shard) for shard in shards]
    if len(set(work)) != 1:
        raise AssertionError(f"scale shards are imbalanced: {work}")
    print(
        "self-test passed: 18 matched small/medium runs partition into "
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
    if min(args.min_free_gb, args.poll_seconds, args.timeout_hours) <= 0:
        raise ValueError("disk, poll, and timeout guards must be positive")

    selected = partition(schedule(), args.shard_count)[args.shard_index]
    manifest = {
        "profile": "optional-scale",
        "created_at": now(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "wait_for_key_root": (
            str(args.wait_for_key_root) if args.wait_for_key_root is not None else None
        ),
        "output_root": str(args.output_root),
        "runs": [asdict(run) for run in selected],
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if args.wait_for_key_root is not None:
        wait_for_key_training(
            args.wait_for_key_root,
            poll_seconds=args.poll_seconds,
            timeout_hours=args.timeout_hours,
        )

    shard_root = args.log_root / "training" / f"shard-{args.shard_index}"
    shard_root.mkdir(parents=True, exist_ok=True)
    atomic_json(shard_root / "manifest.json", manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
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
            f"{now()} scale shard {args.shard_index} "
            f"[{offset}/{len(selected)}] {run.slug}: rc={result['returncode']}",
            flush=True,
        )
        if result["returncode"] != 0:
            raise SystemExit(f"scale training failed: {run.slug}")
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
