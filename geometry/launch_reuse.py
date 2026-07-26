from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path


CONDITIONS = (
    ("cycle113", 0.0, "clean"),
    ("cycle113", 0.15, "corrupt15"),
    ("cycle113", 0.60, "corrupt60"),
    ("random113", 0.0, "random"),
    ("cycle113", 0.02, "corrupt02"),
    ("cycle113", 0.05, "corrupt05"),
    ("cycle113", 0.30, "corrupt30"),
    ("cycle113", 1.0, "corrupt100"),
)


@dataclass(frozen=True)
class Run:
    task: str
    corruption: float
    condition: str
    preset: str
    seed: int
    steps: int
    dense_checkpoint_every: int
    batch_size: int = 4096
    train_fraction: float = 0.3
    weight_decay: float = 1.0
    aliases: int = 4
    contexts: int = 16
    eval_contexts: int = 16
    eval_every: int = 250
    snapshot_every: int = 500
    checkpoint_every: int = 25_000
    keep_checkpoints: int = 2

    @property
    def slug(self) -> str:
        return f"{self.condition}-{self.preset}-s{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("pilot", "priority", "confirmation"),
        default="pilot",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--seeds", help="Comma-separated seeds or inclusive ranges, e.g. 1-3,7"
    )
    parser.add_argument("--presets", help="Comma-separated preset names")
    parser.add_argument("--conditions", help="Comma-separated condition names")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/geometry-reuse-results"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/workspace/geometry-reuse-logs"),
    )
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def matrix(profile: str) -> list[Run]:
    if profile == "pilot":
        runs = [
            Run(task, corruption, condition, "grok", 0, 60_000, 500)
            for task, corruption, condition in CONDITIONS
        ]
        runs.append(Run("cycle113", 0.0, "clean", "micro", 0, 60_000, 500))
        return runs
    if profile == "priority":
        grok_conditions = CONDITIONS[:4]
        micro_conditions = tuple(
            condition
            for condition in CONDITIONS
            if condition[2] in {"clean", "corrupt15", "random"}
        )
        return [
            Run(task, corruption, condition, "grok", seed, 150_000, 1_000)
            for seed in range(1, 4)
            for task, corruption, condition in grok_conditions
        ] + [
            Run(task, corruption, condition, "micro", seed, 150_000, 1_000)
            for seed in range(1, 4)
            for task, corruption, condition in micro_conditions
        ]
    if profile == "confirmation":
        return [
            Run(task, corruption, condition, preset, seed, 150_000, 1_000)
            for seed in range(4)
            for preset in ("grok", "micro")
            for task, corruption, condition in CONDITIONS
        ]
    raise ValueError(profile)


def parse_seeds(value: str) -> set[int]:
    seeds: set[int] = set()
    for part in value.split(","):
        bounds = part.strip().split("-", maxsplit=1)
        if len(bounds) == 1:
            seeds.add(int(bounds[0]))
            continue
        start, stop = (int(bound) for bound in bounds)
        if start > stop:
            raise ValueError(f"descending seed range {part!r}")
        seeds.update(range(start, stop + 1))
    return seeds


def parse_names(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


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
        "--corruption",
        str(run.corruption),
        "--preset",
        run.preset,
        "--seed",
        str(run.seed),
        "--split-seed",
        str(run.seed),
        "--task-seed",
        str(run.seed),
        "--token-seed",
        str(100_000 + run.seed),
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


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def run_one(
    run: Run,
    *,
    output_root: Path,
    log_root: Path,
    compile_model: bool,
    device: str,
    min_free_gb: float,
) -> dict[str, object]:
    available = free_gb(output_root)
    if available < min_free_gb:
        return {
            **asdict(run),
            "returncode": 75,
            "elapsed_seconds": 0.0,
            "error": (
                f"disk guard: {available:.1f} GiB free is below "
                f"{min_free_gb:.1f} GiB"
            ),
        }
    command = command_for(
        run,
        output_root=output_root,
        compile_model=compile_model,
        device=device,
    )
    log_path = log_root / f"{run.slug}-to{run.steps}.log"
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
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    profile_log_root = args.log_root / args.profile
    profile_log_root.mkdir(parents=True, exist_ok=True)
    runs = matrix(args.profile)
    if args.steps is not None:
        runs = [replace(run, steps=args.steps) for run in runs]
    if args.seeds:
        selected_seeds = parse_seeds(args.seeds)
        runs = [run for run in runs if run.seed in selected_seeds]
    if args.presets:
        selected_presets = parse_names(args.presets)
        runs = [run for run in runs if run.preset in selected_presets]
    if args.conditions:
        selected_conditions = parse_names(args.conditions)
        runs = [run for run in runs if run.condition in selected_conditions]
    if args.limit is not None:
        runs = runs[: args.limit]
    if not runs:
        raise ValueError("filters selected no runs")
    manifest = {
        "profile": args.profile,
        "workers": args.workers,
        "compile": args.compile,
        "device": args.device,
        "min_free_gb": args.min_free_gb,
        "runs": [asdict(run) for run in runs],
    }
    manifest_path = profile_log_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    print(
        f"launching {len(runs)} {args.profile} runs with "
        f"{args.workers} workers ({free_gb(args.output_root):.1f} GiB free)",
        flush=True,
    )
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_one,
                run,
                output_root=args.output_root,
                log_root=profile_log_root,
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
                f"finished {result['condition']}-{result['preset']}-"
                f"s{result['seed']} rc={result['returncode']} "
                f"in {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    (profile_log_root / "results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    failures = [result for result in results if result["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} runs failed")


if __name__ == "__main__":
    main()
