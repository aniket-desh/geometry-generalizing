from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from key60_common import KEY_CONDITIONS, PRESETS, SEEDS, CausalJob, KeyRun, atomic_json, load_json, wait_for_marker
from priority_common import (
    HORIZONS,
    causal_prefix,
    horizon_for,
    operator_prefix,
    operator_steps_for,
    wait_for_runs,
)
from staged_packer import (
    GIB,
    MIB,
    Settings,
    run,
    self_test as staged_packer_self_test,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and pack the exact mixed-horizon evidence bundle locally, "
            "with optional explicit upload."
        )
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
        "--selection-root",
        type=Path,
        default=Path("/workspace/geometry-priority-selected"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/geometry-priority-stage-archives"),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=10.0)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--chunk-mib", type=int, default=42)
    parser.add_argument("--upload-endpoint", default="https://temp.sh/upload")
    parser.add_argument("--upload-retries", type=int, default=8)
    delivery = parser.add_mutually_exclusive_group()
    delivery.add_argument(
        "--local-only",
        dest="local_only",
        action="store_true",
        default=True,
        help="retain a checksummed local archive and chunks without uploading "
        "(default)",
    )
    delivery.add_argument(
        "--upload",
        dest="local_only",
        action="store_false",
        help="explicitly upload wrapped chunks to --upload-endpoint",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name.endswith(".lock") or path.name == "priority-pack.log":
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _freeze_metrics(source: Path, destination: Path, horizon: int) -> None:
    records = []
    for line in source.read_text().splitlines():
        try:
            record = json.loads(line)
            if int(record.get("step", -1)) <= horizon:
                records.append(record)
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            continue
    if not records or max(int(record["step"]) for record in records) != horizon:
        raise ValueError(f"metrics do not reach selected horizon {horizon}: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(record) + "\n" for record in records))


def build_selection(
    runs: list[KeyRun],
    *,
    log_root: Path,
    figure_root: Path,
    selection_root: Path,
) -> Path:
    parent = selection_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{selection_root.name}.tmp-", dir=parent))
    try:
        run_records = []
        for selected in runs:
            destination = temporary / "runs" / selected.slug
            horizon = horizon_for(selected)
            for name in ("config.json", "operation_table.npy", "train_mask.npy", "token_layout.npz"):
                source = selected.path / name
                if not source.is_file():
                    raise FileNotFoundError(source)
                _link_or_copy(source, destination / name)
            _freeze_metrics(
                selected.path / "metrics.jsonl",
                destination / "metrics.jsonl",
                horizon,
            )
            selected_steps = operator_steps_for(selected)
            for step in selected_steps:
                for template in ("weights-{step:06d}.pt", "activations-{step:06d}.npz"):
                    source = selected.path / template.format(step=step)
                    if source.is_file():
                        _link_or_copy(source, destination / source.name)
            for prefix in (
                operator_prefix(selected),
                causal_prefix(CausalJob(selected, horizon)),
            ):
                for suffix in (".json", ".jsonl", ".csv"):
                    source = prefix.with_suffix(suffix)
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    _link_or_copy(source, destination / source.name)
            atomic_json(
                destination / "selection.json",
                {
                    "source": str(selected.path),
                    "condition": selected.condition,
                    "preset": selected.preset,
                    "seed": selected.seed,
                    "endpoint_step": horizon,
                    "operator_steps": list(selected_steps),
                    "causal_step": horizon,
                },
            )
            run_records.append(
                {
                    "slug": selected.slug,
                    "source": str(selected.path),
                    "endpoint_step": horizon,
                    "operator_steps": list(selected_steps),
                }
            )
        _copy_tree(log_root, temporary / "logs")
        _copy_tree(figure_root, temporary / "figures")
        files = [
            {
                "path": str(path.relative_to(temporary)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        ]
        marker = temporary / "selection-complete.json"
        atomic_json(
            marker,
            {
                "status": "complete",
                "horizons": HORIZONS,
                "run_count": len(runs),
                "runs": run_records,
                "file_count": len(files),
                "files": files,
            },
        )
        if selection_root.exists():
            shutil.rmtree(selection_root)
        temporary.replace(selection_root)
        return selection_root / marker.name
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def selection_valid(path: Path) -> bool:
    payload = load_json(path)
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        return False
    root = path.parent
    files = payload.get("files")
    return (
        isinstance(files, list)
        and files
        and all(
            isinstance(record, dict)
            and (candidate := root / str(record.get("path", ""))).is_file()
            and candidate.stat().st_size == int(record.get("bytes", -1))
            and sha256(candidate) == record.get("sha256")
            for record in files
        )
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="pack-priority-") as temporary:
        root = Path(temporary)
        logs = root / "logs"
        figures = root / "figures"
        logs.mkdir()
        figures.mkdir()
        (logs / "run.log").write_text("complete\n")
        (logs / "slot.lock").touch()
        (figures / "plot.png").write_bytes(b"plot")
        runs = []
        for task, corruption, condition in KEY_CONDITIONS:
            for preset in PRESETS:
                for seed in SEEDS:
                    path = root / "source" / f"{condition}-{preset}-s{seed}"
                    path.mkdir(parents=True)
                    run = KeyRun(path, task, corruption, condition, preset, seed)
                    runs.append(run)
                    for name in ("config.json", "operation_table.npy", "train_mask.npy", "token_layout.npz"):
                        (path / name).write_bytes(b"{}" if name.endswith(".json") else b"x")
                    horizon = horizon_for(run)
                    (path / "metrics.jsonl").write_text(
                        json.dumps({"step": horizon, "test_accuracy": 1.0}) + "\n"
                        + json.dumps({"step": 60_000, "test_accuracy": 0.5}) + "\n"
                    )
                    for step in (*operator_steps_for(run), 60_000):
                        (path / f"weights-{step:06d}.pt").touch()
                    for prefix in (operator_prefix(run), causal_prefix(CausalJob(run, horizon))):
                        for suffix in (".json", ".jsonl", ".csv"):
                            prefix.with_suffix(suffix).write_text("{}\n")
        selected = root / "selected"
        marker = build_selection(
            runs,
            log_root=logs,
            figure_root=figures,
            selection_root=selected,
        )
        if not selection_valid(marker):
            raise AssertionError("selection manifest did not validate")
        random_run = selected / "runs" / "random-grok-s0"
        if (random_run / "weights-060000.pt").exists():
            raise AssertionError("control 60k checkpoint leaked into mixed-horizon bundle")
        if (selected / "logs" / "slot.lock").exists():
            raise AssertionError("lock file leaked into bundle")
    staged_packer_self_test()
    print(
        "self-test passed: mixed-horizon selection and local-only packing "
        "validated"
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not 1 <= args.chunk_mib <= 42:
        raise ValueError("--chunk-mib must be between 1 and 42")
    if min(
        args.poll_seconds,
        args.timeout_hours,
        args.min_free_gib,
        args.upload_retries,
    ) <= 0:
        raise ValueError("poll, timeout, disk, and retry guards must be positive")
    final_marker = args.log_root / "priority-complete.json"
    operator_marker = args.log_root / "operator-complete.json"
    causal_marker = args.log_root / "causal-complete.json"
    wait_for_marker(
        final_marker,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    runs = wait_for_runs(
        results_root=args.results_root,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    selection_marker = args.selection_root / "selection-complete.json"
    if not selection_valid(selection_marker):
        selection_marker = build_selection(
            runs,
            log_root=args.log_root,
            figure_root=args.figure_root,
            selection_root=args.selection_root,
        )
    if not selection_valid(selection_marker):
        raise RuntimeError("priority selection did not validate")
    run(
        Settings(
            stage="priority",
            marker=final_marker,
            required_markers=(operator_marker, causal_marker, selection_marker),
            roots=(args.selection_root,),
            output_root=args.output_root,
            archive_prefix="vi-priority-mixed-endpoints",
            chunk_bytes=args.chunk_mib * MIB,
            min_free_bytes=int(args.min_free_gib * GIB),
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_hours * 3600,
            local_only=args.local_only,
            upload_endpoint=args.upload_endpoint,
            upload_retries=args.upload_retries,
        )
    )


if __name__ == "__main__":
    main()
