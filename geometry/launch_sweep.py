from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("pilot", "anchor", "breadth", "scale"),
        default="pilot",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-root", type=Path, default=Path("/workspace/geometry-results")
    )
    parser.add_argument(
        "--log-root", type=Path, default=Path("/workspace/geometry-logs")
    )
    parser.add_argument("--compile", action="store_true")
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
) -> dict[str, object]:
    slug = f"{run.task}-{run.preset}-s{run.seed}"
    log_path = log_root / f"{slug}.log"
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
        "10000",
        "--output-root",
        str(output_root),
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
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    runs = matrix(args.profile)
    manifest = {
        "profile": args.profile,
        "workers": args.workers,
        "compile": args.compile,
        "runs": [asdict(run) for run in runs],
    }
    (args.log_root / f"{args.profile}-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
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
                output_root=args.output_root,
                log_root=args.log_root,
                compile_model=args.compile,
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
    (args.log_root / f"{args.profile}-results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    failures = [result for result in results if result["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} runs failed")


if __name__ == "__main__":
    main()
