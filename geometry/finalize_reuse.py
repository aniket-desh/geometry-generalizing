from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


CONDITIONS = {
    ("cycle113", 0.00): "clean",
    ("cycle113", 0.02): "corrupt02",
    ("cycle113", 0.05): "corrupt05",
    ("cycle113", 0.15): "corrupt15",
    ("cycle113", 0.30): "corrupt30",
    ("cycle113", 0.60): "corrupt60",
    ("cycle113", 1.00): "corrupt100",
    ("random113", 0.00): "random",
}
CAUSAL_CONDITIONS = {"clean", "corrupt15", "random"}


@dataclass(frozen=True)
class Run:
    path: Path
    task: str
    corruption: float
    condition: str
    preset: str
    seed: int
    final_step: int

    @property
    def slug(self) -> str:
        return f"{self.condition}-{self.preset}-s{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the p=113 confirmation sweep, run the reusable-operator "
            "and causal analyses, and render the complete figure suite."
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
        default=Path("/workspace/geometry-reuse-logs/finalize"),
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("/workspace/geometry-reuse-figures"),
    )
    parser.add_argument("--final-step", type=int, default=150_000)
    parser.add_argument(
        "--expected-matrix",
        choices=("full64", "manifest"),
        default="full64",
        help=(
            "Require the canonical 64-run matrix, or trust the exact nonempty "
            "subset recorded in the supplied confirmation manifest."
        ),
    )
    parser.add_argument(
        "--operator-presets",
        default="grok",
        help="Comma-separated presets to include in operator analysis.",
    )
    parser.add_argument(
        "--causal-presets",
        default="grok",
        help="Comma-separated presets to include in causal analysis.",
    )
    parser.add_argument(
        "--output-tag",
        default="zz_final",
        help="Suffix for step-specific analysis outputs inside each run.",
    )
    parser.add_argument("--step-every", type=int, default=10_000)
    parser.add_argument(
        "--causal-step-every",
        type=int,
        default=0,
        help=(
            "Causal checkpoint interval; zero analyzes only the final checkpoint. "
            "The dense seed-0 trajectory is run separately."
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def condition_for(task: str, corruption: float) -> str:
    for (candidate_task, candidate_corruption), condition in CONDITIONS.items():
        if task == candidate_task and abs(corruption - candidate_corruption) < 1e-8:
            return condition
    raise ValueError(f"unexpected condition: task={task}, corruption={corruption}")


def expected_specs(
    manifest_path: Path, *, final_step: int, expected_matrix: str
) -> list[dict[str, object]] | None:
    payload = load_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("profile") != "confirmation":
        return None
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    if any(int(run.get("steps", 0)) < final_step for run in runs):
        return None
    expected = {
        (
            str(run["task"]),
            float(run.get("corruption", 0.0)),
            str(run["preset"]),
            int(run["seed"]),
        )
        for run in runs
    }
    canonical = {
        (task, corruption, preset, seed)
        for task, corruption in CONDITIONS
        for preset in ("grok", "micro")
        for seed in range(4)
    }
    if len(expected) != len(runs):
        raise ValueError("confirmation manifest contains duplicate runs")
    if not expected.issubset(canonical):
        raise ValueError(
            f"confirmation manifest contains unexpected runs: "
            f"{sorted(expected - canonical)}"
        )
    if expected_matrix == "full64" and expected != canonical:
        missing = sorted(canonical - expected)
        extra = sorted(expected - canonical)
        raise ValueError(
            f"confirmation manifest is not the 64-run matrix; "
            f"missing={missing}, extra={extra}"
        )
    return runs


def training_results_complete(
    results_path: Path, *, expected_count: int, final_step: int
) -> bool:
    payload = load_json(results_path)
    if not isinstance(payload, list) or len(payload) != expected_count:
        return False
    return all(
        int(result.get("steps", 0)) >= final_step
        and int(result.get("returncode", 1)) == 0
        for result in payload
        if isinstance(result, dict)
    ) and all(isinstance(result, dict) for result in payload)


def discover_runs(
    results_root: Path,
    specs: list[dict[str, object]],
    *,
    final_step: int,
) -> list[Run] | None:
    indexed: dict[tuple[str, float, str, int], Path] = {}
    for config_path in results_root.glob("*/config.json"):
        payload = load_json(config_path)
        if not isinstance(payload, dict):
            continue
        key = (
            str(payload.get("task")),
            float(
                payload.get(
                    "task_corruption_fraction",
                    payload.get("corruption", 0.0),
                )
            ),
            str(payload.get("preset")),
            int(payload.get("seed", -1)),
        )
        indexed[key] = config_path.parent

    runs: list[Run] = []
    for spec in specs:
        key = (
            str(spec["task"]),
            float(spec.get("corruption", 0.0)),
            str(spec["preset"]),
            int(spec["seed"]),
        )
        run_dir = indexed.get(key)
        if run_dir is None:
            return None
        done = load_json(run_dir / "done.json")
        if (
            not isinstance(done, dict)
            or int(done.get("final_step", 0)) < final_step
            or not (run_dir / f"weights-{final_step:06d}.pt").exists()
        ):
            return None
        runs.append(
            Run(
                path=run_dir,
                task=key[0],
                corruption=key[1],
                condition=condition_for(key[0], key[1]),
                preset=key[2],
                seed=key[3],
                final_step=final_step,
            )
        )
    return sorted(runs, key=lambda run: (run.preset, run.seed, run.condition))


def wait_for_confirmation(args: argparse.Namespace) -> list[Run]:
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    last_message = 0.0
    while time.monotonic() < deadline:
        specs = expected_specs(
            args.confirmation_manifest,
            final_step=args.final_step,
            expected_matrix=args.expected_matrix,
        )
        runs = None
        if specs is not None and training_results_complete(
            args.confirmation_results,
            expected_count=len(specs),
            final_step=args.final_step,
        ):
            runs = discover_runs(
                args.results_root, specs, final_step=args.final_step
            )
        if runs is not None:
            return runs
        if time.monotonic() - last_message >= 300.0:
            print(
                f"{now()} waiting for the validated "
                f"{args.final_step}-step confirmation matrix",
                flush=True,
            )
            last_message = time.monotonic()
        time.sleep(args.poll_seconds)
    raise TimeoutError("confirmation sweep did not complete before the timeout")


def requested_steps(final_step: int, every: int) -> list[int]:
    if every < 1 or final_step < 1:
        raise ValueError("step interval and final step must be positive")
    return sorted(set(range(every, final_step + 1, every)) | {final_step})


def parse_presets(value: str) -> set[str]:
    presets = {item.strip() for item in value.split(",") if item.strip()}
    invalid = presets - {"grok", "micro", "small", "medium", "large"}
    if invalid or not presets:
        raise ValueError(f"invalid preset selection: {sorted(invalid)}")
    return presets


def output_has_steps(prefix: Path, steps: list[int]) -> bool:
    payload = load_json(prefix.with_suffix(".json"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        return False
    observed = {
        int(record["step"])
        for record in payload["records"]
        if isinstance(record, dict) and "step" in record
    }
    return set(steps).issubset(observed)


def command_for(
    *,
    script: Path,
    run: Run,
    prefix: Path,
    steps: list[int],
    device: str,
) -> list[str]:
    command = [
        sys.executable,
        str(script),
        "--run-dir",
        str(run.path),
        "--output-prefix",
        str(prefix),
        "--checkpoint-glob",
        "weights-*.pt",
        "--steps",
        ",".join(str(step) for step in steps),
        "--device",
        device,
    ]
    if script.name == "operator_reuse.py":
        command.extend(["--folds", "5", "--max-dimension", "16"])
        if run.condition == "random":
            command.extend(["--generator-relation", "1"])
    else:
        command.extend(
            [
                "--folds",
                "3",
                "--max-dimension",
                "16",
                "--batch-size",
                "4096",
            ]
        )
    return command


def run_job(
    *,
    phase: str,
    script: Path,
    run: Run,
    log_root: Path,
    steps: list[int],
    device: str,
    min_free_gb: float,
    output_tag: str,
) -> dict[str, object]:
    # Render discovery is lexical; keep the validated final analysis last so
    # older pilot and trajectory files cannot override the same checkpoint.
    prefix = run.path / f"{phase}_{output_tag}"
    if output_has_steps(prefix, steps):
        return {"run": run.slug, "status": "skipped", "output": str(prefix)}
    available = free_gb(run.path)
    if available < min_free_gb:
        return {
            "run": run.slug,
            "status": "failed",
            "returncode": 75,
            "error": f"disk guard: only {available:.1f} GiB free",
        }
    command = command_for(
        script=script,
        run=run,
        prefix=prefix,
        steps=steps,
        device=device,
    )
    log_path = log_root / f"{phase}-{run.slug}.log"
    started = time.monotonic()
    with log_path.open("a") as log:
        log.write(f"\n{now()} START {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=script.parent,
        )
        log.write(f"{now()} EXIT {result.returncode}\n")
    status = (
        "complete"
        if result.returncode == 0 and output_has_steps(prefix, steps)
        else "failed"
    )
    return {
        "run": run.slug,
        "status": status,
        "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log_path),
        "output": str(prefix),
    }


def run_phase(
    *,
    phase: str,
    script: Path,
    runs: list[Run],
    log_root: Path,
    steps: list[int],
    workers: int,
    device: str,
    min_free_gb: float,
    output_tag: str,
) -> list[dict[str, object]]:
    print(f"{now()} starting {phase}: {len(runs)} runs", flush=True)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_job,
                phase=phase,
                script=script,
                run=run,
                log_root=log_root,
                steps=steps,
                device=device,
                min_free_gb=min_free_gb,
                output_tag=output_tag,
            )
            for run in runs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{now()} {phase} {result['run']}: {result['status']}",
                flush=True,
            )
    failures = [result for result in results if result["status"] == "failed"]
    atomic_json(log_root / f"{phase}-results.json", results)
    if failures:
        raise RuntimeError(f"{len(failures)} {phase} jobs failed")
    return results


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    args.figure_root.mkdir(parents=True, exist_ok=True)
    operator_steps = requested_steps(args.final_step, args.step_every)
    causal_steps = (
        [args.final_step]
        if args.causal_step_every == 0
        else requested_steps(args.final_step, args.causal_step_every)
    )
    script_root = Path(__file__).parent

    runs = wait_for_confirmation(args)
    operator_presets = parse_presets(args.operator_presets)
    causal_presets = parse_presets(args.causal_presets)
    operator_runs = [run for run in runs if run.preset in operator_presets]
    causal_runs = [
        run
        for run in runs
        if run.preset in causal_presets
        and run.condition in CAUSAL_CONDITIONS
        and run.seed in range(4)
    ]
    manifest = {
        "created_at": now(),
        "confirmation_manifest": str(args.confirmation_manifest),
        "confirmation_results": str(args.confirmation_results),
        "expected_matrix": args.expected_matrix,
        "final_step": args.final_step,
        "operator_steps": operator_steps,
        "causal_steps": causal_steps,
        "workers": args.workers,
        "device": args.device,
        "min_free_gb": args.min_free_gb,
        "output_tag": args.output_tag,
        "operator_runs": [run.slug for run in operator_runs],
        "causal_runs": [run.slug for run in causal_runs],
        "figure_root": str(args.figure_root),
    }
    atomic_json(args.log_root / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    run_phase(
        phase="operator_reuse",
        script=script_root / "operator_reuse.py",
        runs=operator_runs,
        log_root=args.log_root,
        steps=operator_steps,
        workers=args.workers,
        device=args.device,
        min_free_gb=args.min_free_gb,
        output_tag=args.output_tag,
    )
    run_phase(
        phase="causal_reuse",
        script=script_root / "causal_reuse.py",
        runs=causal_runs,
        log_root=args.log_root,
        steps=causal_steps,
        workers=args.workers,
        device=args.device,
        min_free_gb=args.min_free_gb,
        output_tag=args.output_tag,
    )

    if free_gb(args.figure_root) < args.min_free_gb:
        raise RuntimeError("disk guard stopped rendering")
    render_log = args.log_root / "render.log"
    render_command = [
        sys.executable,
        str(script_root / "render_reuse.py"),
        "--output",
        str(args.figure_root),
        "--operator-view",
        "output",
        "--operator-layer",
        "last",
    ]
    for run in runs:
        render_command.extend(["--results", str(run.path)])
    with render_log.open("a") as log:
        log.write(f"\n{now()} START {' '.join(render_command)}\n")
        log.flush()
        result = subprocess.run(
            render_command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=script_root,
        )
        log.write(f"{now()} EXIT {result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"rendering failed; see {render_log}")
    atomic_json(
        args.log_root / "done.json",
        {
            "completed_at": now(),
            "figure_root": str(args.figure_root),
            "free_gb": free_gb(args.figure_root),
        },
    )
    print(f"{now()} finalization complete: {args.figure_root}", flush=True)


if __name__ == "__main__":
    main()
