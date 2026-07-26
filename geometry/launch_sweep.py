from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


BREADTH_PILOT_TASKS = (
    "torus5",
    "xor16",
    "dihedral12",
    "path16",
    "tree15",
    "broken12",
    "random31",
    "cycle24",
    "cycle31",
)

BREADTH_REPLICATION_TASKS = (
    "torus5",
    "cycle31",
    "random31",
    "dihedral12",
)


@dataclass(frozen=True)
class Run:
    task: str
    preset: str
    seed: int
    steps: int
    batch_size: int
    train_fraction: float = 0.4
    weight_decay: float = 1.0
    aliases: int = 4
    contexts: int = 16
    eval_contexts: int = 8
    eval_every: int = 100
    snapshot_every: int = 500
    checkpoint_every: int = 10_000
    keep_checkpoints: int = 0
    dense_checkpoint_every: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=(
            "pilot",
            "anchor",
            "breadth-pilot",
            "breadth-extend",
            "breadth-confirm",
            "breadth-replicate",
            "breadth-diagnostics",
            "breadth",
            "scale",
        ),
        default="pilot",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-root",
        type=Path,
    )
    parser.add_argument(
        "--log-root",
        type=Path,
    )
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def matrix(profile: str) -> list[Run]:
    if profile == "pilot":
        return [
            Run(task, "small", seed, 12_000, 2048)
            for task in ("cycle7", "cycle12", "torus4", "random16")
            for seed in (0, 1)
        ]
    if profile == "anchor":
        return [
            Run(
                "cycle113",
                "grok",
                seed,
                150_000,
                4096,
                train_fraction=0.3,
                aliases=1,
                contexts=1,
                eval_contexts=1,
                eval_every=250,
                snapshot_every=500,
            )
            for seed in range(4)
        ]
    if profile in {"breadth-pilot", "breadth-extend"}:
        steps = 12_000 if profile == "breadth-pilot" else 60_000
        return [
            Run(
                task,
                "micro",
                0,
                steps,
                4096,
                eval_every=250,
                snapshot_every=500,
                checkpoint_every=3_000,
                keep_checkpoints=2,
                dense_checkpoint_every=500,
            )
            for task in BREADTH_PILOT_TASKS
        ]
    if profile == "breadth-confirm":
        return [
            Run(
                task,
                "micro",
                seed,
                60_000,
                4096,
                eval_every=250,
                snapshot_every=500,
                checkpoint_every=3_000,
                keep_checkpoints=2,
                dense_checkpoint_every=500,
            )
            for task in ("cycle24", "cycle31", "random31")
            for seed in (1, 2)
        ]
    if profile == "breadth-replicate":
        return [
            Run(
                task,
                "micro",
                seed,
                60_000,
                4096,
                eval_every=500,
                snapshot_every=1_000,
                checkpoint_every=15_000,
                keep_checkpoints=2,
                dense_checkpoint_every=1_000,
            )
            for task in BREADTH_REPLICATION_TASKS
            for seed in (1, 2)
        ]
    if profile == "breadth-diagnostics":
        return [
            Run(
                task,
                preset,
                0,
                60_000,
                4096,
                eval_every=250,
                snapshot_every=500,
                checkpoint_every=3_000,
                keep_checkpoints=2,
                dense_checkpoint_every=500,
            )
            for task, preset in (
                ("cycle12", "micro"),
                ("dihedral12", "small"),
            )
        ]
    if profile == "breadth":
        tasks = (
            "cycle7",
            "cycle12",
            "cycle24",
            "cycle31",
            "torus4",
            "torus5",
            "xor16",
            "dihedral12",
            "path16",
            "tree15",
            "broken12",
            "random16",
            "random31",
        )
        return [
            Run(task, preset, seed, 40_000, 4096)
            for task in tasks
            for preset in ("micro", "small")
            for seed in range(6)
        ]
    if profile == "scale":
        return [
            Run(task, preset, seed, 60_000, batch)
            for task in (
                "cycle7",
                "cycle12",
                "cycle31",
                "torus5",
                "xor16",
                "dihedral12",
                "broken12",
                "random31",
            )
            for preset, batch in (
                ("small", 4096),
                ("medium", 4096),
                ("large", 2048),
            )
            for seed in range(8)
        ]
    raise ValueError(profile)


def command_for(
    run: Run,
    *,
    output_root: Path,
    compile_model: bool,
    device: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        "--task",
        run.task,
        "--preset",
        run.preset,
        "--seed",
        str(run.seed),
        "--split-seed",
        str(run.seed),
        "--task-seed",
        str(run.seed),
        "--token-seed",
        str(run.seed),
        "--steps",
        str(run.steps),
        "--batch-size",
        str(run.batch_size),
        "--train-fraction",
        str(run.train_fraction),
        "--weight-decay",
        str(run.weight_decay),
        "--aliases",
        str(run.aliases),
        "--contexts",
        str(run.contexts),
        "--eval-contexts",
        str(run.eval_contexts),
        "--eval-every",
        str(run.eval_every),
        "--snapshot-every",
        str(run.snapshot_every),
        "--checkpoint-every",
        str(run.checkpoint_every),
        "--keep-checkpoints",
        str(run.keep_checkpoints),
        "--dense-checkpoint-every",
        str(run.dense_checkpoint_every),
        "--dense-checkpoint-dtype",
        "float16",
        "--output-root",
        str(output_root),
        "--device",
        device,
        "--resume",
    ]
    if compile_model:
        command.append("--compile")
    return command


def run_one(
    run: Run,
    *,
    output_root: Path,
    log_root: Path,
    compile_model: bool,
    device: str,
    min_free_gb: float,
) -> dict[str, object]:
    slug = f"{run.task}-{run.preset}-s{run.seed}"
    log_path = log_root / f"{slug}.log"
    available = shutil.disk_usage(output_root).free / (1024**3)
    if available < min_free_gb:
        return {
            **asdict(run),
            "returncode": 75,
            "elapsed_seconds": 0.0,
            "error": (
                f"disk guard: {available:.1f} GiB free is below "
                f"{min_free_gb:.1f} GiB"
            ),
            "log": str(log_path),
        }
    command = command_for(
        run,
        output_root=output_root,
        compile_model=compile_model,
        device=device,
    )
    started = time.time()
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return {
        **asdict(run),
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - started,
        "log": str(log_path),
    }


def select_shard(runs: list[Run], index: int, count: int) -> list[Run]:
    if count < 1:
        raise ValueError("--shard-count must be positive")
    if index < 0 or index >= count:
        raise ValueError("--shard-index must lie in [0, shard-count)")
    return runs[index::count]


def self_test() -> None:
    runs = matrix("breadth-replicate")
    expected = {
        (task, seed)
        for task in ("torus5", "cycle31", "dihedral12", "random31")
        for seed in (1, 2)
    }
    identities = [(run.task, run.seed) for run in runs]
    if len(identities) != len(set(identities)):
        raise AssertionError("breadth replication identities are not unique")
    if set(identities) != expected:
        raise AssertionError("breadth replication coverage is incomplete")
    for run in runs:
        if (
            run.preset != "micro"
            or run.steps != 60_000
            or run.batch_size != 4_096
            or run.train_fraction != 0.4
            or run.weight_decay != 1.0
            or run.aliases != 4
            or run.contexts != 16
            or run.eval_contexts != 8
            or run.eval_every != 500
            or run.snapshot_every != 1_000
            or run.checkpoint_every != 15_000
            or run.keep_checkpoints != 2
            or run.dense_checkpoint_every != 1_000
        ):
            raise AssertionError(f"unexpected replication protocol: {run}")
        command = command_for(
            run,
            output_root=Path("/tmp/breadth-replication-self-test"),
            compile_model=True,
            device="cuda",
        )
        for option in ("--seed", "--split-seed", "--task-seed", "--token-seed"):
            if command.count(option) != 1:
                raise AssertionError(f"{option} is missing or duplicated")
            if command[command.index(option) + 1] != str(run.seed):
                raise AssertionError(f"{option} does not match the run seed")
    shards = [select_shard(runs, index, 4) for index in range(4)]
    flattened = [run for shard in shards for run in shard]
    if len(flattened) != 8 or set(flattened) != set(runs):
        raise AssertionError("four-shard coverage is incomplete or duplicated")
    if any(len(shard) != 2 for shard in shards):
        raise AssertionError("four-shard schedule is imbalanced")
    print("breadth-replicate self-test passed: 8 unique runs across 4 shards")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    default_output_root = {
        "breadth-pilot": "/workspace/geometry-breadth-results",
        "breadth-extend": "/workspace/geometry-breadth-results",
        "breadth-confirm": "/workspace/geometry-breadth-results",
        "breadth-replicate": "/workspace/geometry-breadth-results",
        "breadth-diagnostics": (
            "/workspace/geometry-breadth-diagnostic-results"
        ),
    }.get(args.profile, "/workspace/geometry-results")
    output_root = args.output_root or Path(default_output_root)
    log_root = args.log_root or Path(
        {
            "breadth-pilot": "/workspace/geometry-breadth-logs",
            "breadth-extend": "/workspace/geometry-breadth-extend-logs",
            "breadth-confirm": "/workspace/geometry-breadth-confirm-logs",
            "breadth-replicate": (
                "/workspace/geometry-breadth-replication-logs"
            ),
            "breadth-diagnostics": (
                "/workspace/geometry-breadth-diagnostic-logs"
            ),
        }.get(args.profile, "/workspace/geometry-logs")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    all_runs = matrix(args.profile)
    runs = select_shard(all_runs, args.shard_index, args.shard_count)
    if not runs:
        raise ValueError("selected shard contains no runs")
    manifest = {
        "profile": args.profile,
        "workers": args.workers,
        "compile": args.compile,
        "device": args.device,
        "output_root": str(output_root),
        "log_root": str(log_root),
        "min_free_gb": args.min_free_gb,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_run_count": len(all_runs),
        "runs": [asdict(run) for run in runs],
    }
    if args.profile == "breadth-diagnostics":
        manifest["comparison_notes"] = {
            "broken12_legacy_metadata": (
                "Existing breadth-run configs report "
                "task_corruption_fraction=0.0 for broken12. The saved table "
                "contains 24 exceptions among 144 cells: nominal 0.15, actual "
                "1/6. Compare operation tables, not the legacy field."
            )
        }
    manifest_name = (
        f"{args.profile}-manifest.json"
        if args.shard_count == 1
        else (
            f"{args.profile}-manifest-"
            f"{args.shard_index:02d}-of-{args.shard_count:02d}.json"
        )
    )
    (log_root / manifest_name).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    print(
        f"launching {len(runs)} runs with {args.workers} workers",
        flush=True,
    )
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_one,
                run,
                output_root=output_root,
                log_root=log_root,
                compile_model=args.compile,
                device=args.device,
                min_free_gb=args.min_free_gb,
            )
            for run in runs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"finished {result['task']}-{result['preset']}-s{result['seed']} "
                f"rc={result['returncode']} in {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    results_name = (
        f"{args.profile}-results.json"
        if args.shard_count == 1
        else (
            f"{args.profile}-results-"
            f"{args.shard_index:02d}-of-{args.shard_count:02d}.json"
        )
    )
    (log_root / results_name).write_text(
        json.dumps(results, indent=2) + "\n"
    )
    failures = [result for result in results if result["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} runs failed")


if __name__ == "__main__":
    main()
