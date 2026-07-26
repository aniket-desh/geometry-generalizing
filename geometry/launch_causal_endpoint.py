from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from finalize_reuse import (
    CONDITIONS,
    Run,
    atomic_json,
    discover_runs,
    expected_specs,
    load_json,
    now,
    training_results_complete,
)


EXPECTED_CONTROLS = {
    "learned_generator",
    "exact_state_swap",
    "target_centroid",
    "scrambled_successor",
    "random_orthogonal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the validated 64-run confirmation matrix, then run one "
            "deterministic shard of final-checkpoint causal analyses."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/workspace/geometry-reuse-results"),
    )
    parser.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=Path(
            "/workspace/geometry-reuse-logs/confirmation/manifest.json"
        ),
    )
    parser.add_argument(
        "--confirmation-results",
        type=Path,
        default=Path(
            "/workspace/geometry-reuse-logs/confirmation/results.json"
        ),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/workspace/geometry-reuse-logs/causal-final"),
    )
    parser.add_argument("--final-step", type=int, default=150_000)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--max-log-mb", type=float, default=16.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def ready_runs(
    *,
    results_root: Path,
    manifest_path: Path,
    results_path: Path,
    final_step: int,
) -> list[Run] | None:
    specs = expected_specs(manifest_path, final_step=final_step)
    if specs is None:
        return None
    if not training_results_complete(
        results_path,
        expected_count=len(specs),
        final_step=final_step,
    ):
        return None
    return discover_runs(results_root, specs, final_step=final_step)


def wait_for_runs(args: argparse.Namespace) -> list[Run]:
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    last_message = 0.0
    while time.monotonic() < deadline:
        runs = ready_runs(
            results_root=args.results_root,
            manifest_path=args.confirmation_manifest,
            results_path=args.confirmation_results,
            final_step=args.final_step,
        )
        if runs is not None:
            if len(runs) != 64:
                raise ValueError(f"validated matrix has {len(runs)} runs, not 64")
            return runs
        if time.monotonic() - last_message >= 300.0:
            print(
                f"{now()} waiting for the validated 64-run "
                f"{args.final_step}-step confirmation matrix",
                flush=True,
            )
            last_message = time.monotonic()
        time.sleep(args.poll_seconds)
    raise TimeoutError("confirmation matrix did not validate before timeout")


def shard_runs(
    runs: list[Run], *, shard_index: int, shard_count: int
) -> list[Run]:
    if shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index must lie within the shard count")
    ordered = sorted(
        runs,
        key=lambda run: (
            run.preset,
            run.seed,
            run.condition,
            run.task,
            run.corruption,
            str(run.path),
        ),
    )
    return [
        run
        for index, run in enumerate(ordered)
        if index % shard_count == shard_index
    ]


def causal_output_valid(prefix: Path, run: Run, final_step: int) -> bool:
    payload = load_json(prefix.with_suffix(".json"))
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(records, list) or not records:
        return False
    config = load_json(run.path / "config.json")
    if not isinstance(config, dict):
        return False
    if metadata.get("run_name") != config.get("run_name"):
        return False
    checkpoint = f"weights-{final_step:06d}.pt"
    checkpoints = metadata.get("checkpoints")
    if not isinstance(checkpoints, list) or checkpoint not in checkpoints:
        return False
    patch_sites = metadata.get("patch_sites")
    if not isinstance(patch_sites, list) or not patch_sites:
        return False
    try:
        expected_sites = {
            (str(site.get("position")), int(site.get("layer", -1)))
            for site in patch_sites
            if isinstance(site, dict)
        }
        fold_count = int(metadata.get("folds", 0))
    except (TypeError, ValueError):
        return False
    if len(expected_sites) != len(patch_sites):
        return False

    groups: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    for record in records:
        if not isinstance(record, dict):
            return False
        try:
            step = int(record.get("step", -1))
            key = (
                int(record.get("fold", -1)),
                str(record.get("position")),
                int(record.get("layer", -1)),
            )
        except (TypeError, ValueError):
            return False
        if step != final_step or record.get("checkpoint") != checkpoint:
            return False
        groups[key].add(str(record.get("control")))
    folds = set(range(fold_count))
    if folds != {key[0] for key in groups}:
        return False
    if expected_sites != {(key[1], key[2]) for key in groups}:
        return False
    if any(controls != EXPECTED_CONTROLS for controls in groups.values()):
        return False
    return all(
        path.exists() and path.stat().st_size > 0
        for path in (
            prefix.with_suffix(".jsonl"),
            prefix.with_suffix(".csv"),
        )
    )


def command_for(
    *,
    script: Path,
    run: Run,
    prefix: Path,
    final_step: int,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(script),
        "--run-dir",
        str(run.path),
        "--output-prefix",
        str(prefix),
        "--checkpoint-glob",
        f"weights-{final_step:06d}.pt",
        "--steps",
        str(final_step),
        "--folds",
        "3",
        "--max-dimension",
        "16",
        "--batch-size",
        "4096",
        "--device",
        device,
    ]


def run_with_capped_log(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path,
    max_bytes: int,
) -> int:
    written = 0
    truncated = False
    with log_path.open("w") as log:
        header = f"{now()} START {' '.join(command)}\n"
        log.write(header)
        written += len(header.encode())
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            encoded = line.encode()
            if written + len(encoded) <= max_bytes:
                log.write(line)
                written += len(encoded)
            elif not truncated:
                marker = (
                    f"\n{now()} LOG LIMIT REACHED; further child output "
                    "discarded\n"
                )
                log.write(marker)
                written += len(marker.encode())
                truncated = True
        returncode = process.wait()
        log.write(f"{now()} EXIT {returncode}\n")
    return returncode


def run_one(
    *,
    run: Run,
    script: Path,
    log_root: Path,
    final_step: int,
    device: str,
    min_free_gb: float,
    max_log_mb: float,
    dry_run: bool,
) -> dict[str, object]:
    prefix = run.path / "causal_reuse_zz_final"
    if causal_output_valid(prefix, run, final_step):
        return {
            "run": run.slug,
            "status": "skipped_valid",
            "output": str(prefix),
        }
    available = min(free_gb(run.path), free_gb(log_root))
    if available < min_free_gb:
        return {
            "run": run.slug,
            "status": "failed",
            "returncode": 75,
            "error": (
                f"disk guard: {available:.1f} GiB free is below "
                f"{min_free_gb:.1f} GiB"
            ),
        }
    command = command_for(
        script=script,
        run=run,
        prefix=prefix,
        final_step=final_step,
        device=device,
    )
    if dry_run:
        return {
            "run": run.slug,
            "status": "dry_run",
            "command": command,
            "output": str(prefix),
        }

    log_path = log_root / f"{run.slug}.log"
    started = time.monotonic()
    returncode = run_with_capped_log(
        command,
        log_path=log_path,
        cwd=script.parent,
        max_bytes=max(1, math.ceil(max_log_mb * 1024**2)),
    )
    valid = causal_output_valid(prefix, run, final_step)
    return {
        "run": run.slug,
        "status": "complete" if returncode == 0 and valid else "failed",
        "returncode": returncode,
        "validated_output": valid,
        "elapsed_seconds": time.monotonic() - started,
        "free_gb": min(free_gb(run.path), free_gb(log_root)),
        "log": str(log_path),
        "output": str(prefix),
    }


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="causal-endpoint-") as temporary:
        root = Path(temporary)
        results_root = root / "results"
        log_root = root / "logs"
        results_root.mkdir()
        log_root.mkdir()
        specs: list[dict[str, object]] = []
        result_records: list[dict[str, object]] = []
        for task, corruption in CONDITIONS:
            for preset in ("grok", "micro"):
                for seed in range(4):
                    condition = CONDITIONS[(task, corruption)]
                    spec = {
                        "task": task,
                        "corruption": corruption,
                        "condition": condition,
                        "preset": preset,
                        "seed": seed,
                        "steps": 150_000,
                    }
                    specs.append(spec)
                    result_records.append({**spec, "returncode": 0})
                    run_dir = (
                        results_root / f"{condition}-{preset}-s{seed}"
                    )
                    run_dir.mkdir()
                    (run_dir / "config.json").write_text(
                        json.dumps(
                            {
                                "run_name": run_dir.name,
                                "task": task,
                                "task_corruption_fraction": corruption,
                                "preset": preset,
                                "seed": seed,
                            }
                        )
                        + "\n"
                    )
                    (run_dir / "done.json").write_text(
                        json.dumps({"final_step": 150_000}) + "\n"
                    )
                    (run_dir / "weights-150000.pt").touch()
        manifest_path = root / "manifest.json"
        results_path = root / "training-results.json"
        manifest_path.write_text(
            json.dumps({"profile": "confirmation", "runs": specs}) + "\n"
        )
        results_path.write_text(json.dumps(result_records) + "\n")
        runs = ready_runs(
            results_root=results_root,
            manifest_path=manifest_path,
            results_path=results_path,
            final_step=150_000,
        )
        if runs is None or len(runs) != 64:
            raise AssertionError("synthetic confirmation matrix did not validate")
        shards = [
            shard_runs(runs, shard_index=index, shard_count=4)
            for index in range(4)
        ]
        if [len(shard) for shard in shards] != [16, 16, 16, 16]:
            raise AssertionError("synthetic 64-run matrix did not shard evenly")
        paths = [run.path for shard in shards for run in shard]
        if len(paths) != len(set(paths)) or set(paths) != {
            run.path for run in runs
        }:
            raise AssertionError("shards overlap or omit runs")

        sample = shards[0][0]
        prefix = sample.path / "causal_reuse_zz_final"
        checkpoint = "weights-150000.pt"
        sites = [{"position": "output", "layer": 1}]
        records = [
            {
                "step": 150_000,
                "checkpoint": checkpoint,
                "fold": fold,
                "position": "output",
                "layer": 1,
                "control": control,
            }
            for fold in range(3)
            for control in sorted(EXPECTED_CONTROLS)
        ]
        prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "run_name": sample.path.name,
                        "checkpoints": [checkpoint],
                        "patch_sites": sites,
                        "folds": 3,
                    },
                    "records": records,
                }
            )
            + "\n"
        )
        prefix.with_suffix(".jsonl").write_text("{}\n")
        prefix.with_suffix(".csv").write_text("step\n150000\n")
        if not causal_output_valid(prefix, sample, 150_000):
            raise AssertionError("complete synthetic causal output was rejected")
        records.pop()
        prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "run_name": sample.path.name,
                        "checkpoints": [checkpoint],
                        "patch_sites": sites,
                        "folds": 3,
                    },
                    "records": records,
                }
            )
            + "\n"
        )
        if causal_output_valid(prefix, sample, 150_000):
            raise AssertionError("partial synthetic causal output was accepted")
        command = command_for(
            script=Path(__file__).with_name("causal_reuse.py"),
            run=sample,
            prefix=prefix,
            final_step=150_000,
            device="cuda",
        )
        if (
            command[0] != sys.executable
            or command[command.index("--steps") + 1] != "150000"
            or command[command.index("--checkpoint-glob") + 1]
            != "weights-150000.pt"
        ):
            raise AssertionError("endpoint command is not pinned and reproducible")
    print(
        "self-test passed: validated 64 runs, deterministic 4x16 shards, "
        "strict output skipping, and a pinned sys.executable command"
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.shard_index is None:
        raise ValueError("--shard-index is required")
    if args.final_step != 150_000:
        raise ValueError("the causal endpoint launcher is pinned to step 150000")
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("poll interval and timeout must be positive")
    if args.min_free_gb < 0 or args.max_log_mb <= 0:
        raise ValueError("disk and log guards must be positive")

    runs = wait_for_runs(args)
    selected = shard_runs(
        runs,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    shard_log_root = args.log_root / f"shard-{args.shard_index}"
    shard_log_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("causal_reuse.py")
    manifest = {
        "created_at": now(),
        "python": sys.executable,
        "final_step": args.final_step,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "run_count": len(selected),
        "runs": [run.slug for run in selected],
        "device": args.device,
        "min_free_gb": args.min_free_gb,
        "max_log_mb": args.max_log_mb,
        "dry_run": args.dry_run,
    }
    atomic_json(
        args.log_root / f"shard-{args.shard_index}-manifest.json",
        manifest,
    )
    print(
        f"{now()} shard {args.shard_index}/{args.shard_count}: "
        f"{len(selected)} validated runs",
        flush=True,
    )
    results: list[dict[str, object]] = []
    for index, run in enumerate(selected, start=1):
        result = run_one(
            run=run,
            script=script,
            log_root=shard_log_root,
            final_step=args.final_step,
            device=args.device,
            min_free_gb=args.min_free_gb,
            max_log_mb=args.max_log_mb,
            dry_run=args.dry_run,
        )
        results.append(result)
        atomic_json(
            args.log_root / f"shard-{args.shard_index}-results.json",
            results,
        )
        print(
            f"{now()} [{index}/{len(selected)}] {run.slug}: "
            f"{result['status']}",
            flush=True,
        )
    failures = [result for result in results if result["status"] == "failed"]
    if failures:
        raise SystemExit(f"{len(failures)} causal endpoint jobs failed")
    print(
        f"{now()} shard {args.shard_index} complete: "
        f"{len(results) - len(failures)} runs",
        flush=True,
    )


if __name__ == "__main__":
    main()
