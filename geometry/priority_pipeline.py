from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from key60_common import (
    CAUSAL_CONTROLS,
    CAUSAL_FOLDS,
    KEY_CONDITIONS,
    PRESETS,
    SEEDS,
    CausalJob,
    KeyRun,
    atomic_json,
    exclusive_lock,
    load_json,
    now,
    wait_for_marker,
)
from key60_pipeline import _run_with_capped_log
from priority_common import (
    HORIZONS,
    analysis_slot,
    causal_output_valid,
    causal_prefix,
    causal_schedule,
    discover_runs,
    expected_specs,
    horizon_for,
    operator_output_valid,
    operator_prefix,
    operator_steps_for,
    wait_for_run,
    wait_for_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the marker-gated mixed-horizon priority pipeline."
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
        "--log-root",
        type=Path,
        default=Path("/workspace/geometry-priority-logs"),
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("/workspace/geometry-priority-figures"),
    )
    parser.add_argument(
        "--presets",
        default="grok,micro",
        help="Comma-separated pair of matched model presets.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--analysis-slots", type=int, default=2)
    parser.add_argument("--analysis-slot-root", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=10.0)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--max-log-mb", type=float, default=16.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _parse_presets(value: str) -> tuple[str, ...]:
    presets = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if len(presets) != 2:
        raise ValueError("--presets must select exactly two distinct model presets")
    if any(preset not in {"grok", "micro", "small", "medium"} for preset in presets):
        raise ValueError(f"unsupported priority presets: {presets}")
    return presets


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def _operator_command(run: KeyRun, device: str) -> list[str]:
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
        ",".join(str(step) for step in operator_steps_for(run)),
        "--views",
        "output",
        "--folds",
        "5",
        "--max-dimension",
        "16",
        "--successor-mode",
        "latent-cycle",
        "--projection-fit",
        "inductive",
        "--device",
        device,
    ]
    return command


def _causal_command(job: CausalJob, device: str) -> list[str]:
    config = load_json(job.run.path / "config.json")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"missing model config for {job.run.path}")
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
        "node,output",
        "--layers",
        f"0,{int(config['model']['depth'])}",
        "--successor-mode",
        "latent-cycle",
        "--folds",
        str(CAUSAL_FOLDS),
        "--max-dimension",
        "16",
        "--batch-size",
        "4096",
        "--device",
        device,
    ]


def _wait_spec(args: argparse.Namespace, spec: tuple[str, float, str, str, int]) -> KeyRun:
    _, _, condition, preset, seed = spec
    return wait_for_run(
        args.results_root,
        condition=condition,
        preset=preset,
        seed=seed,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
        presets=args.presets,
    )


def _run_operator(
    spec: tuple[str, float, str, str, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    run = _wait_spec(args, spec)
    with exclusive_lock(run.path / ".priority-operator.lock"):
        if operator_output_valid(run):
            return {"run": run.slug, "status": "skipped_valid"}
        if min(_free_gb(run.path), _free_gb(args.log_root)) < args.min_free_gb:
            return {"run": run.slug, "status": "failed", "returncode": 75}
        command = _operator_command(run, args.device)
        started = time.monotonic()
        with analysis_slot(
            args.analysis_slot_root,
            count=args.analysis_slots,
            poll_seconds=args.poll_seconds,
            timeout_hours=args.timeout_hours,
        ):
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
        }


def operator_stage(args: argparse.Namespace) -> None:
    marker = args.log_root / "operator-complete.json"
    specs = sorted(
        expected_specs(args.presets),
        key=lambda spec: (HORIZONS[spec[2]], spec[3], spec[2], spec[4]),
    )
    results: list[dict[str, object]] = []
    atomic_json(
        args.log_root / "operator-manifest.json",
        {
            "created_at": now(),
            "run_count": len(specs),
            "horizons": HORIZONS,
            "operator_steps": {
                "clean": [10_000, 30_000, 60_000],
                "corrupt15": [10_000, 30_000],
                "random": [10_000, 30_000],
            },
            "workers": args.workers,
            "views": ["output"],
            "folds": 5,
            "projection_fit": "inductive_state_alias_fold",
            "probe": "preregistered canonical latent-label k -> k + 1 cycle",
            "successor_mode": "latent_label_plus_one",
        },
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_operator, spec, args) for spec in specs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            atomic_json(args.log_root / "operator-results.json", results)
            print(f"{now()} priority operator {result['run']}: {result['status']}", flush=True)
    failures = [result for result in results if result["status"] not in {"complete", "skipped_valid"}]
    runs = discover_runs(args.results_root, args.presets)
    if failures or runs is None or not all(operator_output_valid(run) for run in runs):
        raise RuntimeError(f"{len(failures)} priority operator jobs failed")
    atomic_json(
        marker,
        {
            "status": "complete",
            "completed_at": now(),
            "run_count": len(runs),
            "horizons": HORIZONS,
        },
    )


def _job_specs(
    presets: tuple[str, ...],
) -> list[tuple[str, float, str, str, int]]:
    return sorted(
        expected_specs(presets),
        key=lambda spec: (HORIZONS[spec[2]], spec[3], spec[2], spec[4]),
    )


def _shard_specs(
    specs: list[tuple[str, float, str, str, int]],
    index: int,
    count: int,
) -> list[tuple[str, float, str, str, int]]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid causal shard index or count")
    return [spec for offset, spec in enumerate(specs) if offset % count == index]


def _run_causal(
    spec: tuple[str, float, str, str, int],
    args: argparse.Namespace,
    shard_root: Path,
) -> dict[str, object]:
    run = _wait_spec(args, spec)
    job = CausalJob(run, horizon_for(run))
    with exclusive_lock(run.path / f".priority-causal-{job.step:06d}.lock"):
        if causal_output_valid(job):
            return {"job": job.slug, "status": "skipped_valid"}
        if min(_free_gb(run.path), _free_gb(shard_root)) < args.min_free_gb:
            return {"job": job.slug, "status": "failed", "returncode": 75}
        started = time.monotonic()
        with analysis_slot(
            args.analysis_slot_root,
            count=args.analysis_slots,
            poll_seconds=args.poll_seconds,
            timeout_hours=args.timeout_hours,
        ):
            returncode = _run_with_capped_log(
                _causal_command(job, args.device),
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
        }


def causal_shard_stage(args: argparse.Namespace) -> None:
    if args.shard_index is None:
        raise ValueError("--shard-index is required for causal-shard")
    specs = _shard_specs(
        _job_specs(args.presets),
        args.shard_index,
        args.shard_count,
    )
    shard_root = args.log_root / "causal" / f"shard-{args.shard_index}"
    shard_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.log_root / "causal" / f"shard-{args.shard_index}-manifest.json",
        {
            "created_at": now(),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "jobs": [f"{spec[2]}-{spec[3]}-s{spec[4]}-p{HORIZONS[spec[2]]:06d}" for spec in specs],
            "patch_sites": [
                "input node, layer 0",
                "output residual stream, final layer",
            ],
            "successor_mode": "latent_label_plus_one",
            "folds": CAUSAL_FOLDS,
            "controls": sorted(CAUSAL_CONTROLS),
        },
    )
    results = []
    for offset, spec in enumerate(specs, start=1):
        result = _run_causal(spec, args, shard_root)
        results.append(result)
        atomic_json(
            args.log_root / "causal" / f"shard-{args.shard_index}-results.json",
            results,
        )
        print(
            f"{now()} priority causal {args.shard_index} [{offset}/{len(specs)}] "
            f"{result['job']}: {result['status']}",
            flush=True,
        )
    failures = [result for result in results if result["status"] not in {"complete", "skipped_valid"}]
    if failures:
        raise RuntimeError(f"{len(failures)} priority causal jobs failed")
    atomic_json(
        args.log_root / "causal" / f"shard-{args.shard_index}-complete.json",
        {
            "status": "complete",
            "completed_at": now(),
            "shard_index": args.shard_index,
            "jobs": [result["job"] for result in results],
        },
    )


def causal_join_stage(args: argparse.Namespace) -> None:
    for shard in range(args.shard_count):
        wait_for_marker(
            args.log_root / "causal" / f"shard-{shard}-complete.json",
            poll_seconds=args.poll_seconds,
            timeout_hours=args.timeout_hours,
        )
    runs = wait_for_runs(
        results_root=args.results_root,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
        presets=args.presets,
    )
    jobs = causal_schedule(runs)
    invalid = [job.slug for job in jobs if not causal_output_valid(job)]
    if invalid:
        raise RuntimeError(f"invalid priority causal outputs: {invalid}")
    atomic_json(
        args.log_root / "causal-complete.json",
        {
            "status": "complete",
            "completed_at": now(),
            "job_count": len(jobs),
            "horizons": HORIZONS,
            "patch_sites": [
                "input node, layer 0",
                "output residual stream, final layer",
            ],
            "successor_mode": "latent_label_plus_one",
            "folds": CAUSAL_FOLDS,
            "controls": sorted(CAUSAL_CONTROLS),
            "jobs": [job.slug for job in jobs],
        },
    )


def finalize_stage(args: argparse.Namespace) -> None:
    runs = wait_for_runs(
        results_root=args.results_root,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
        presets=args.presets,
    )
    operator_marker = args.log_root / "operator-complete.json"
    causal_marker = args.log_root / "causal-complete.json"
    wait_for_marker(operator_marker, poll_seconds=args.poll_seconds, timeout_hours=args.timeout_hours)
    wait_for_marker(causal_marker, poll_seconds=args.poll_seconds, timeout_hours=args.timeout_hours)
    gates = {
        "behavior": True,
        "geometry": all(operator_output_valid(run) for run in runs),
        "usable_mdl": all(operator_output_valid(run) for run in runs),
        "causal": all(causal_output_valid(job) for job in causal_schedule(runs)),
    }
    if not all(gates.values()):
        raise RuntimeError(f"priority evidence gates failed: {gates}")
    command = [
        sys.executable,
        str(Path(__file__).with_name("render_priority.py")),
        "--output",
        str(args.figure_root),
    ]
    for preset in args.presets:
        command.extend(["--preset", preset])
    for run in runs:
        command.extend(["--run", str(run.path)])
    returncode = _run_with_capped_log(
        command,
        log_path=args.log_root / "render.log",
        max_bytes=max(1, math.ceil(args.max_log_mb * 1024**2)),
        cwd=Path(__file__).parent,
    )
    manifest = load_json(args.figure_root / "priority-render-manifest.json")
    if (
        returncode != 0
        or not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("gates") != gates
    ):
        raise RuntimeError(f"priority rendering failed; see {args.log_root / 'render.log'}")
    atomic_json(
        args.log_root / "priority-complete.json",
        {
            "status": "complete",
            "completed_at": now(),
            "run_count": len(runs),
            "horizons": HORIZONS,
            "gates": gates,
            "figure_root": str(args.figure_root),
        },
    )


def _write_synthetic_analysis(run: KeyRun) -> None:
    config = load_json(run.path / "config.json")
    if not isinstance(config, dict):
        raise ValueError(f"missing synthetic config: {run.path}")
    order = int(config["task_order"])
    successor = (np.arange(order, dtype=np.int64) + 1) % order
    successor_metadata = {
        "successor_mode": "latent_label_plus_one",
        "successor_preregistered": True,
        "successor_vector": successor.tolist(),
        "successor_sha256": hashlib.sha256(
            np.asarray(successor, dtype="<i8").tobytes()
        ).hexdigest(),
        "generator_relation": None,
    }
    prefix = operator_prefix(run)
    records = [
        {
            "step": step,
            "checkpoint": f"weights-{step:06d}.pt",
            "view": "output",
            "layer": 1,
            "joint_cv_error": 0.5,
            "usable_reuse_gain_bits": 100.0,
        }
        for step in operator_steps_for(run)
    ]
    atomic_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "run_name": run.path.name,
                "folds": 5,
                "projection_fit": "inductive_state_alias_fold",
                **successor_metadata,
            },
            "records": records,
        },
    )
    prefix.with_suffix(".jsonl").write_text("{}\n")
    prefix.with_suffix(".csv").write_text("step\n")
    job = CausalJob(run, horizon_for(run))
    prefix = causal_prefix(job)
    checkpoint = f"weights-{job.step:06d}.pt"
    depth = 1
    sites = (("node", 0), ("output", depth))
    causal_records = [
        {
            "step": job.step,
            "checkpoint": checkpoint,
            "fold": fold,
            "position": position,
            "layer": layer,
            "control": control,
        }
        for fold in range(CAUSAL_FOLDS)
        for position, layer in sites
        for control in sorted(CAUSAL_CONTROLS)
    ]
    atomic_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "run_name": run.path.name,
                "folds": CAUSAL_FOLDS,
                "checkpoints": [checkpoint],
                "patch_sites": [
                    {"position": position, "layer": layer}
                    for position, layer in sites
                ],
                "successor_mode": "latent_label_plus_one",
                **successor_metadata,
            },
            "records": causal_records,
        },
    )
    prefix.with_suffix(".jsonl").write_text("{}\n")
    prefix.with_suffix(".csv").write_text("step\n")


def self_test(presets: tuple[str, ...]) -> None:
    with tempfile.TemporaryDirectory(prefix="priority-pipeline-") as temporary:
        root = Path(temporary)
        for task, corruption, condition, preset, seed in expected_specs(presets):
            run_dir = root / f"{condition}-{preset}-s{seed}"
            run_dir.mkdir()
            atomic_json(
                run_dir / "config.json",
                {
                    "run_name": run_dir.name,
                    "task": task,
                    "task_corruption_fraction": corruption,
                    "task_order": 113,
                    "preset": preset,
                    "seed": seed,
                    "split_seed": seed,
                    "task_seed": seed,
                    "token_seed": 100_000 + seed,
                    "aliases": 4,
                    "contexts": 16,
                    "batch_size": 2048 if preset == "medium" else 4096,
                    "train_fraction": 0.3,
                    "weight_decay": 1.0,
                    "model": {"depth": 1},
                },
            )
            horizon = HORIZONS[condition]
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"step": horizon, "test_accuracy": 1.0}) + "\n"
            )
            steps = (10_000, 30_000, 60_000) if condition == "clean" else (10_000, 30_000)
            for step in steps:
                (run_dir / f"weights-{step:06d}.pt").touch()
        runs = discover_runs(root, presets)
        expected_count = len(expected_specs(presets))
        if runs is None or len(runs) != expected_count:
            raise AssertionError("mixed-horizon matrix did not validate")
        for run in runs:
            _write_synthetic_analysis(run)
        if not all(operator_output_valid(run) for run in runs):
            raise AssertionError("operator validation failed")
        jobs = causal_schedule(runs)
        if len(jobs) != expected_count or not all(causal_output_valid(job) for job in jobs):
            raise AssertionError("causal validation failed")
        shards = [
            _shard_specs(_job_specs(presets), index, 4) for index in range(4)
        ]
        if [len(shard) for shard in shards] != [5, 5, 4, 4]:
            raise AssertionError("18 causal jobs did not partition across four shards")
        flattened = [spec for shard in shards for spec in shard]
        if len(flattened) != len(set(flattened)):
            raise AssertionError("causal shards overlap")
        control = next(run for run in runs if run.condition == "random")
        if (control.path / "weights-060000.pt").exists():
            raise AssertionError("synthetic control unexpectedly has a 60k checkpoint")
    print(
        "self-test passed: 18 mixed-horizon runs, per-condition operator steps, "
        "18 endpoint causal jobs, and four disjoint causal shards"
    )


def main() -> None:
    args = parse_args()
    args.presets = _parse_presets(args.presets)
    if args.analysis_slot_root is None:
        args.analysis_slot_root = args.log_root / "analysis-slots"
    if args.self_test:
        self_test(args.presets)
        return
    if args.stage is None:
        raise ValueError("--stage is required unless --self-test is used")
    if min(
        args.workers,
        args.analysis_slots,
        args.shard_count,
        args.poll_seconds,
        args.timeout_hours,
        args.min_free_gb,
        args.max_log_mb,
    ) <= 0:
        raise ValueError("workers, slots, shards, poll, timeout, disk, and log guards must be positive")
    args.log_root.mkdir(parents=True, exist_ok=True)
    args.figure_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "operator":
        operator_stage(args)
    elif args.stage == "causal-shard":
        causal_shard_stage(args)
    elif args.stage == "causal-join":
        causal_join_stage(args)
    else:
        finalize_stage(args)


if __name__ == "__main__":
    main()
