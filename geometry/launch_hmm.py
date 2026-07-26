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
    aliases: int = 16
    sequence_steps: int = 8
    train_fraction: float = 0.6
    eval_every: int = 100
    snapshot_every: int = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("breadth", "scale"),
        default="breadth",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/geometry/hmm-results"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/workspace/geometry/logs"),
    )
    return parser.parse_args()


def matrix(profile: str) -> list[Run]:
    if profile == "breadth":
        tasks = (
            "cycle12",
            "cycle24",
            "torus4",
            "torus5",
            "xor16",
            "dihedral12",
            "path16",
            "tree15",
            "broken12",
            "random16",
        )
        return [
            Run(task, "micro", seed, 20_000, 1024)
            for task in tasks
            for seed in range(3)
        ]
    if profile == "scale":
        tasks = (
            "cycle7",
            "cycle12",
            "torus4",
            "xor16",
            "dihedral12",
            "broken12",
            "random16",
        )
        return [
            Run(task, preset, seed, 20_000, batch)
            for task in tasks
            for preset, batch in (("small", 512), ("medium", 256))
            for seed in range(2)
        ]
    raise ValueError(profile)


def run_one(
    run: Run,
    *,
    output_root: Path,
    log_root: Path,
) -> dict[str, object]:
    slug = f"hmm-{run.task}-{run.preset}-s{run.seed}"
    log_path = log_root / f"{slug}.log"
    command = [
        sys.executable,
        str(Path(__file__).with_name("hmm_train.py")),
        "--task",
        run.task,
        "--preset",
        run.preset,
        "--seed",
        str(run.seed),
        "--split-seed",
        "0",
        "--steps",
        str(run.steps),
        "--sequence-steps",
        str(run.sequence_steps),
        "--batch-size",
        str(run.batch_size),
        "--train-fraction",
        str(run.train_fraction),
        "--aliases",
        str(run.aliases),
        "--eval-aliases",
        str(run.aliases),
        "--target-mode",
        "state",
        "--eval-every",
        str(run.eval_every),
        "--snapshot-every",
        str(run.snapshot_every),
        "--checkpoint-every",
        str(run.steps),
        "--output-root",
        str(output_root),
    ]
    started = time.time()
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
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
        "runs": [asdict(run) for run in runs],
    }
    (args.log_root / f"hmm-{args.profile}-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"launching {len(runs)} HMM runs with {args.workers} workers",
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
            )
            for run in runs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"finished {result['task']}-{result['preset']}-"
                f"s{result['seed']} rc={result['returncode']} "
                f"in {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    (args.log_root / f"hmm-{args.profile}-results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    failures = [result for result in results if result["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} HMM runs failed")


if __name__ == "__main__":
    main()
