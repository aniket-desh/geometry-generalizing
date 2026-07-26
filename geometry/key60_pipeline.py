from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from key60_common import (
    CAUSAL_CONTROLS,
    CAUSAL_FOLDS,
    FINAL_STEP,
    KEY_CONDITIONS,
    KEY_COUNT,
    OPERATOR_STEPS,
    PRESETS,
    SEEDS,
    CausalJob,
    KeyRun,
    atomic_json,
    causal_output_valid,
    causal_prefix,
    causal_schedule,
    exclusive_lock,
    load_json,
    now,
    operator_output_valid,
    operator_prefix,
    ready_runs,
    wait_for_marker,
    wait_for_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated, marker-gated 18-run key60 pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=("operator", "causal-shard", "causal-join", "finalize"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/workspace/geometry-reuse-results"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/workspace/geometry-key60-logs/key60/manifest.json"),
    )
    parser.add_argument(
        "--training-results",
        type=Path,
        default=Path("/workspace/geometry-key60-logs/key60/results.json"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/workspace/geometry-key60-logs"),
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("/workspace/geometry-key60-figures"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--max-log-mb", type=float, default=16.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def _run_with_capped_log(
    command: list[str],
    *,
    log_path: Path,
    max_bytes: int,
    cwd: Path,
) -> int:
    written = 0
    truncated = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
                marker = f"{now()} LOG LIMIT REACHED; child output discarded\n"
                log.write(marker)
                written += len(marker.encode())
                truncated = True
        returncode = process.wait()
        log.write(f"{now()} EXIT {returncode}\n")
    return returncode


def _operator_command(run: KeyRun, *, device: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("operator_reuse.py")),
        "--run-dir",
        str(run.path),
        "--output-prefix",
        str(operator_prefix(run)),
        "--checkpoint-glob",
        "weights-*.pt",
        "--steps",
        ",".join(str(step) for step in OPERATOR_STEPS),
        "--views",
        "output",
        "--folds",
        "5",
        "--max-dimension",
        "16",
        "--device",
        device,
    ]
    if run.condition == "random":
        command.extend(["--generator-relation", "1"])
    return command


def _causal_command(job: CausalJob, *, device: str) -> list[str]:
    config = load_json(job.run.path / "config.json")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"missing model config for {job.run.path}")
    depth = int(config["model"]["depth"])
    return [
        sys.executable,
        str(Path(__file__).with_name("causal_reuse.py")),
        "--run-dir",
        str(job.run.path),
        "--output-prefix",
        str(causal_prefix(job)),
        "--checkpoint-glob",
        f"weights-{job.step:06d}.pt",
        "--steps",
        str(job.step),
        "--positions",
        "output",
        "--layers",
        str(depth),
        "--folds",
        str(CAUSAL_FOLDS),
        "--max-dimension",
        "16",
        "--batch-size",
        "4096",
        "--device",
        device,
    ]


def _run_operator(
    run: KeyRun,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    prefix = operator_prefix(run)
    with exclusive_lock(run.path / ".key60-operator.lock"):
        if operator_output_valid(run):
            return {
                "run": run.slug,
                "status": "skipped_valid",
                "output": str(prefix),
            }
        available = min(free_gb(run.path), free_gb(args.log_root))
        if available < args.min_free_gb:
            return {
                "run": run.slug,
                "status": "failed",
                "returncode": 75,
                "error": f"disk guard: {available:.1f} GiB free",
            }
        command = _operator_command(run, device=args.device)
        if args.dry_run:
            return {
                "run": run.slug,
                "status": "dry_run",
                "command": command,
            }
        started = time.monotonic()
        returncode = _run_with_capped_log(
            command,
            log_path=args.log_root / "operator" / f"{run.slug}.log",
            max_bytes=max(1, math.ceil(args.max_log_mb * 1024**2)),
            cwd=Path(__file__).parent,
        )
        valid = operator_output_valid(run)
        return {
            "run": run.slug,
            "status": "complete" if returncode == 0 and valid else "failed",
            "returncode": returncode,
            "validated_output": valid,
            "elapsed_seconds": time.monotonic() - started,
            "output": str(prefix),
        }


def operator_stage(args: argparse.Namespace) -> None:
    runs = wait_for_runs(
        results_root=args.results_root,
        manifest_path=args.manifest,
        results_path=args.training_results,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    marker = args.log_root / "operator-complete.json"
    if isinstance(load_json(marker), dict) and all(
        operator_output_valid(run) for run in runs
    ):
        print(f"{now()} operator stage already complete", flush=True)
        return
    manifest = {
        "created_at": now(),
        "run_count": len(runs),
        "runs": [run.slug for run in runs],
        "steps": list(OPERATOR_STEPS),
        "views": ["output"],
        "layers": "all output-position residual layers; final layer rendered",
        "folds": 5,
        "workers": args.workers,
        "usable_mdl": (
            "min(lookup_bits, null_bits) - shared_successor_bits, "
            "measured on held-out aliases"
        ),
    }
    atomic_json(args.log_root / "operator-manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_operator, run, args=args) for run in runs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            atomic_json(args.log_root / "operator-results.json", results)
            print(
                f"{now()} operator {result['run']}: {result['status']}",
                flush=True,
            )
    failures = [
        result
        for result in results
        if result["status"] not in {"complete", "skipped_valid"}
    ]
    if failures or not all(operator_output_valid(run) for run in runs):
        raise RuntimeError(f"{len(failures)} key60 operator jobs failed")
    atomic_json(
        marker,
        {
            "status": "complete",
            "completed_at": now(),
            "run_count": len(runs),
            "steps": list(OPERATOR_STEPS),
            "behavior_gate": True,
            "geometry_gate": True,
            "usable_mdl_gate": True,
        },
    )


def _run_causal(
    job: CausalJob,
    *,
    args: argparse.Namespace,
    shard_root: Path,
) -> dict[str, object]:
    prefix = causal_prefix(job)
    with exclusive_lock(job.run.path / f".key60-causal-{job.step:06d}.lock"):
        if causal_output_valid(job):
            return {
                "job": job.slug,
                "status": "skipped_valid",
                "output": str(prefix),
            }
        available = min(free_gb(job.run.path), free_gb(shard_root))
        if available < args.min_free_gb:
            return {
                "job": job.slug,
                "status": "failed",
                "returncode": 75,
                "error": f"disk guard: {available:.1f} GiB free",
            }
        command = _causal_command(job, device=args.device)
        if args.dry_run:
            return {
                "job": job.slug,
                "status": "dry_run",
                "command": command,
            }
        started = time.monotonic()
        returncode = _run_with_capped_log(
            command,
            log_path=shard_root / f"{job.slug}.log",
            max_bytes=max(1, math.ceil(args.max_log_mb * 1024**2)),
            cwd=Path(__file__).parent,
        )
        valid = causal_output_valid(job)
        return {
            "job": job.slug,
            "status": "complete" if returncode == 0 and valid else "failed",
            "returncode": returncode,
            "validated_output": valid,
            "elapsed_seconds": time.monotonic() - started,
            "output": str(prefix),
        }


def _shard_jobs(
    jobs: list[CausalJob],
    *,
    shard_index: int,
    shard_count: int,
) -> list[CausalJob]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid causal shard index or count")
    return [job for index, job in enumerate(jobs) if index % shard_count == shard_index]


def causal_shard_stage(args: argparse.Namespace) -> None:
    if args.shard_index is None:
        raise ValueError("--shard-index is required for causal-shard")
    runs = wait_for_runs(
        results_root=args.results_root,
        manifest_path=args.manifest,
        results_path=args.training_results,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    selected = _shard_jobs(
        causal_schedule(runs),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    shard_root = args.log_root / "causal" / f"shard-{args.shard_index}"
    shard_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": now(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "job_count": len(selected),
        "jobs": [job.slug for job in selected],
        "patch_site": "output residual stream, final layer",
        "folds": CAUSAL_FOLDS,
        "controls": sorted(CAUSAL_CONTROLS),
    }
    atomic_json(
        args.log_root / "causal" / f"shard-{args.shard_index}-manifest.json",
        manifest,
    )
    if args.dry_run:
        manifest["commands"] = [
            _causal_command(job, device=args.device) for job in selected
        ]
        print(json.dumps(manifest, indent=2))
        return
    results: list[dict[str, object]] = []
    for index, job in enumerate(selected, start=1):
        result = _run_causal(job, args=args, shard_root=shard_root)
        results.append(result)
        atomic_json(
            args.log_root / "causal" / f"shard-{args.shard_index}-results.json",
            results,
        )
        print(
            f"{now()} causal shard {args.shard_index} "
            f"[{index}/{len(selected)}] {job.slug}: {result['status']}",
            flush=True,
        )
    failures = [
        result
        for result in results
        if result["status"] not in {"complete", "skipped_valid"}
    ]
    if failures or not all(causal_output_valid(job) for job in selected):
        raise RuntimeError(f"{len(failures)} key60 causal jobs failed")
    atomic_json(
        args.log_root / "causal" / f"shard-{args.shard_index}-complete.json",
        {
            "status": "complete",
            "completed_at": now(),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "jobs": [job.slug for job in selected],
        },
    )


def causal_join_stage(args: argparse.Namespace) -> None:
    runs = wait_for_runs(
        results_root=args.results_root,
        manifest_path=args.manifest,
        results_path=args.training_results,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    for shard in range(args.shard_count):
        wait_for_marker(
            args.log_root / "causal" / f"shard-{shard}-complete.json",
            poll_seconds=args.poll_seconds,
            timeout_hours=args.timeout_hours,
        )
    jobs = causal_schedule(runs)
    invalid = [job.slug for job in jobs if not causal_output_valid(job)]
    if invalid:
        raise RuntimeError(f"invalid key60 causal outputs: {invalid}")
    endpoint_jobs = [job for job in jobs if job.step == FINAL_STEP]
    transition_jobs = [job for job in jobs if job.step != FINAL_STEP]
    atomic_json(
        args.log_root / "causal-complete.json",
        {
            "status": "complete",
            "completed_at": now(),
            "job_count": len(jobs),
            "endpoint_jobs": len(endpoint_jobs),
            "transition_jobs": len(transition_jobs),
            "endpoint_scope": (
                "all 18 runs at step 60000, output position, final layer, "
                "3 folds and 5 controls"
            ),
            "transition_scope": (
                "grok seed 0 for clean, corrupt15 and random at steps "
                "10000 and 30000, with the same patch site, folds and controls"
            ),
            "jobs": [job.slug for job in jobs],
        },
    )


def finalize_stage(args: argparse.Namespace) -> None:
    runs = wait_for_runs(
        results_root=args.results_root,
        manifest_path=args.manifest,
        results_path=args.training_results,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    operator_marker = args.log_root / "operator-complete.json"
    causal_marker = args.log_root / "causal-complete.json"
    wait_for_marker(
        operator_marker,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    wait_for_marker(
        causal_marker,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    gates = {
        "behavior": all(
            isinstance(load_json(run.path / "done.json"), dict) for run in runs
        ),
        "geometry": all(operator_output_valid(run) for run in runs),
        "usable_mdl": all(operator_output_valid(run) for run in runs),
        "causal": all(causal_output_valid(job) for job in causal_schedule(runs)),
    }
    if not all(gates.values()):
        raise RuntimeError(f"key60 final gates failed: {gates}")
    if free_gb(args.figure_root.parent) < args.min_free_gb:
        raise RuntimeError("disk guard stopped key60 rendering")
    render_log = args.log_root / "render.log"
    render_command = [
        sys.executable,
        str(Path(__file__).with_name("render_key60.py")),
        "--output",
        str(args.figure_root),
    ]
    for run in runs:
        render_command.extend(["--run", str(run.path)])
    if args.dry_run:
        print(json.dumps({"gates": gates, "command": render_command}, indent=2))
        return
    returncode = _run_with_capped_log(
        render_command,
        log_path=render_log,
        max_bytes=max(1, math.ceil(args.max_log_mb * 1024**2)),
        cwd=Path(__file__).parent,
    )
    render_manifest = load_json(args.figure_root / "key60-render-manifest.json")
    if (
        returncode != 0
        or not isinstance(render_manifest, dict)
        or render_manifest.get("status") != "complete"
        or render_manifest.get("gates") != gates
    ):
        raise RuntimeError(f"key60 rendering failed; see {render_log}")
    atomic_json(
        args.log_root / "key60-complete.json",
        {
            "status": "complete",
            "completed_at": now(),
            "run_count": len(runs),
            "final_step": FINAL_STEP,
            "gates": gates,
            "operator_marker": str(operator_marker),
            "causal_marker": str(causal_marker),
            "figure_root": str(args.figure_root),
            "render_manifest": str(args.figure_root / "key60-render-manifest.json"),
        },
    )


def _write_synthetic_analysis(run: KeyRun) -> None:
    operator_records = [
        {
            "step": step,
            "checkpoint": f"weights-{step:06d}.pt",
            "view": "output",
            "layer": 1,
            "joint_cv_error": 0.5,
            "usable_reuse_gain_bits": 100.0,
        }
        for step in OPERATOR_STEPS
    ]
    prefix = operator_prefix(run)
    atomic_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "run_name": run.path.name,
                "folds": 5,
            },
            "records": operator_records,
        },
    )
    prefix.with_suffix(".jsonl").write_text("{}\n")
    prefix.with_suffix(".csv").write_text("step\n60000\n")
    for job in causal_schedule([run]):
        prefix = causal_prefix(job)
        checkpoint = f"weights-{job.step:06d}.pt"
        records = [
            {
                "step": job.step,
                "checkpoint": checkpoint,
                "fold": fold,
                "position": "output",
                "layer": 1,
                "control": control,
            }
            for fold in range(CAUSAL_FOLDS)
            for control in sorted(CAUSAL_CONTROLS)
        ]
        atomic_json(
            prefix.with_suffix(".json"),
            {
                "metadata": {
                    "run_name": run.path.name,
                    "folds": CAUSAL_FOLDS,
                    "checkpoints": [checkpoint],
                    "patch_sites": [{"position": "output", "layer": 1}],
                },
                "records": records,
            },
        )
        prefix.with_suffix(".jsonl").write_text("{}\n")
        prefix.with_suffix(".csv").write_text("step\n60000\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="key60-") as temporary:
        root = Path(temporary)
        results_root = root / "results"
        results_root.mkdir()
        log_root = root / "logs"
        specs: list[dict[str, object]] = []
        training_results: list[dict[str, object]] = []
        for task, corruption, condition in KEY_CONDITIONS:
            for preset in PRESETS:
                for seed in SEEDS:
                    spec = {
                        "task": task,
                        "corruption": corruption,
                        "condition": condition,
                        "preset": preset,
                        "seed": seed,
                        "steps": FINAL_STEP,
                    }
                    specs.append(spec)
                    training_results.append({**spec, "returncode": 0})
                    run_dir = results_root / f"{condition}-{preset}-s{seed}"
                    run_dir.mkdir()
                    atomic_json(
                        run_dir / "config.json",
                        {
                            "run_name": run_dir.name,
                            "task": task,
                            "task_corruption_fraction": corruption,
                            "preset": preset,
                            "seed": seed,
                            "model": {"depth": 1},
                        },
                    )
                    atomic_json(
                        run_dir / "done.json",
                        {"run_name": run_dir.name, "final_step": FINAL_STEP},
                    )
                    (run_dir / "metrics.jsonl").write_text(
                        json.dumps({"step": FINAL_STEP, "test_accuracy": 1.0}) + "\n"
                    )
                    for step in OPERATOR_STEPS:
                        (run_dir / f"weights-{step:06d}.pt").touch()
        manifest = log_root / "key60" / "manifest.json"
        training = log_root / "key60" / "results.json"
        atomic_json(manifest, {"profile": "key60", "runs": specs})
        atomic_json(training, training_results)
        runs = ready_runs(
            results_root=results_root,
            manifest_path=manifest,
            results_path=training,
        )
        if runs is None or len(runs) != KEY_COUNT:
            raise AssertionError("strict synthetic key60 matrix did not validate")
        for run in runs:
            _write_synthetic_analysis(run)
        if not all(operator_output_valid(run) for run in runs):
            raise AssertionError("synthetic operator outputs did not validate")
        jobs = causal_schedule(runs)
        if len(jobs) != 24 or not all(causal_output_valid(job) for job in jobs):
            raise AssertionError("synthetic causal schedule did not validate")
        shards = [
            _shard_jobs(jobs, shard_index=index, shard_count=6) for index in range(6)
        ]
        if [len(shard) for shard in shards] != [4] * 6:
            raise AssertionError("24 causal jobs did not shard into 6x4")
        if len({job.slug for shard in shards for job in shard}) != 24:
            raise AssertionError("causal shards overlap")
        runs[0].path.joinpath("weights-060000.pt").unlink()
        if (
            ready_runs(
                results_root=results_root,
                manifest_path=manifest,
                results_path=training,
            )
            is not None
        ):
            raise AssertionError("missing exact final checkpoint was accepted")
    print(
        "self-test passed: strict 18-run training gate, 18 operator outputs "
        "at 10k/30k/60k, 18 causal endpoints plus 6 pre/post jobs, and "
        "deterministic 6x4 causal shards"
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.stage is None:
        raise ValueError("--stage is required unless --self-test is used")
    if args.workers < 1 or args.shard_count < 1:
        raise ValueError("worker and shard counts must be positive")
    if (
        min(
            args.poll_seconds,
            args.timeout_hours,
            args.min_free_gb,
            args.max_log_mb,
        )
        <= 0
    ):
        raise ValueError("poll, timeout, disk, and log guards must be positive")
    args.log_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "operator":
        operator_stage(args)
    elif args.stage == "causal-shard":
        causal_shard_stage(args)
    elif args.stage == "causal-join":
        causal_join_stage(args)
    elif args.stage == "finalize":
        finalize_stage(args)


if __name__ == "__main__":
    main()
