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
        choices=("pilot", "anchor", "breadth-pilot", "breadth", "scale"),
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
    parser.add_argument("--dry-run", action="store_true")
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
    if profile == "breadth-pilot":
        tasks = (
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
        return [
            Run(
                task,
                "micro",
                0,
                12_000,
                4096,
                eval_every=250,
                snapshot_every=500,
                checkpoint_every=3_000,
                keep_checkpoints=2,
                dense_checkpoint_every=500,
            )
            for task in tasks
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
    command = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        "--task",
        run.task,
        "--preset",
        run.preset,
        "--seed",
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
    started = time.time()
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return {
        **asdict(run),
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - started,
        "log": str(log_path),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root or Path(
        "/workspace/geometry-breadth-results"
        if args.profile == "breadth-pilot"
        else "/workspace/geometry-results"
    )
    log_root = args.log_root or Path(
        "/workspace/geometry-breadth-logs"
        if args.profile == "breadth-pilot"
        else "/workspace/geometry-logs"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    runs = matrix(args.profile)
    manifest = {
        "profile": args.profile,
        "workers": args.workers,
        "compile": args.compile,
        "device": args.device,
        "output_root": str(output_root),
        "log_root": str(log_root),
        "min_free_gb": args.min_free_gb,
        "runs": [asdict(run) for run in runs],
    }
    (log_root / f"{args.profile}-manifest.json").write_text(
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
    (log_root / f"{args.profile}-results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    failures = [result for result in results if result["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} runs failed")


if __name__ == "__main__":
    main()
