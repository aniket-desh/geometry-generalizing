from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from summarize_priority_evidence import (
    EvidenceError,
    _synthetic_fixture,
    summarize,
)


SUITE_COUNTS = {"scale": 18, "large": 9}
CAPACITY_PRESETS = ("small", "medium", "large")
CAPACITY_COUNT = 27
FIGURE_NAMES = {
    "capacity-trajectories.png",
    "capacity-trajectories.pdf",
    "capacity-causal-sites.png",
    "capacity-causal-sites.pdf",
    "capacity-output-final-controls.png",
    "capacity-output-final-controls.pdf",
    "capacity-endpoint-summary.json",
    "capacity-figure-captions.json",
    "priority-render-manifest.json",
}


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read valid JSON from {path}") from error


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolved_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise EvidenceError(f"{label} does not resolve: {path}") from error
    if not resolved.is_dir():
        raise EvidenceError(f"{label} is not a directory: {resolved}")
    return resolved


def _validate_paths(scale_root: Path, large_root: Path, output_root: Path) -> None:
    if (
        scale_root == large_root
        or _is_within(scale_root, large_root)
        or _is_within(large_root, scale_root)
    ):
        raise EvidenceError("scale and large results roots must be disjoint")
    for source in (scale_root, large_root):
        if (
            source == output_root
            or _is_within(source, output_root)
            or _is_within(output_root, source)
        ):
            raise EvidenceError(
                f"output and results roots must be disjoint: {source}"
            )


def _suite_sources(
    root: Path,
    suite: str,
) -> tuple[dict[str, Path], dict[str, object]]:
    expected_count = SUITE_COUNTS[suite]
    summary = summarize(results_root=root, suite=suite)
    validation = summary.get("validation")
    if (
        summary.get("suite") != suite
        or not isinstance(validation, dict)
        or validation.get("status") != "complete"
        or validation.get("exact_run_count") != expected_count
    ):
        raise EvidenceError(f"{suite} evidence did not pass its existing validator")

    config_paths = sorted(root.glob("*/config.json"))
    expected_names = {
        str(run["run_name"])
        for run in summary["runs"]
        if isinstance(run, dict) and "run_name" in run
    }
    observed_names = {path.parent.name for path in config_paths}
    if (
        len(config_paths) != expected_count
        or len(observed_names) != expected_count
        or observed_names != expected_names
    ):
        raise EvidenceError(
            f"{suite} root must contain exactly {expected_count} run configs"
        )

    sources: dict[str, Path] = {}
    for config_path in config_paths:
        run_dir = config_path.parent
        if run_dir.is_symlink():
            raise EvidenceError(f"source run may not be a symlink: {run_dir}")
        resolved = run_dir.resolve(strict=True)
        if resolved.parent != root:
            raise EvidenceError(f"source run escaped its results root: {run_dir}")
        sources[run_dir.name] = resolved
    return sources, summary


@contextmanager
def _output_lock(output_root: Path) -> Iterator[None]:
    lock_path = output_root / ".capacity-finalize.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise EvidenceError(
                f"another finalizer holds {lock_path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _prepare_union(union_root: Path, sources: dict[str, Path]) -> list[Path]:
    if union_root.is_symlink():
        raise EvidenceError(f"run union may not be a symlink: {union_root}")
    if union_root.exists() and not union_root.is_dir():
        raise EvidenceError(f"run union is not a directory: {union_root}")
    union_root.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in union_root.iterdir() if path.name not in sources
    )
    if unexpected:
        raise EvidenceError(f"run union contains unexpected entries: {unexpected}")

    links = []
    for name, source in sorted(sources.items()):
        link = union_root / name
        if os.path.lexists(link):
            if not link.is_symlink():
                raise EvidenceError(f"run-union collision is not a symlink: {link}")
            try:
                target = link.resolve(strict=True)
            except OSError as error:
                raise EvidenceError(f"run-union link is broken: {link}") from error
            if target != source:
                raise EvidenceError(
                    f"run-union link points to {target}, expected {source}"
                )
        else:
            os.symlink(source, link, target_is_directory=True)
        links.append(link)
    if len(links) != CAPACITY_COUNT:
        raise EvidenceError(
            f"capacity union requires {CAPACITY_COUNT} links, found {len(links)}"
        )
    return links


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = log_path.with_suffix(log_path.suffix + ".tmp")
    with temporary.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    temporary.replace(log_path)
    if returncode != 0:
        raise EvidenceError(
            f"command failed with return code {returncode}; see {log_path}"
        )


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _validate_outputs(output_root: Path) -> None:
    figure_root = output_root / "figures"
    missing = sorted(
        name
        for name in FIGURE_NAMES
        if not (figure_root / name).is_file()
        or (figure_root / name).stat().st_size == 0
    )
    if missing:
        raise EvidenceError(f"capacity renderer missed artifacts: {missing}")
    render_manifest = _load_json(figure_root / "priority-render-manifest.json")
    summary = _load_json(output_root / "capacity-evidence.json")
    validation = summary.get("validation") if isinstance(summary, dict) else None
    if (
        not isinstance(render_manifest, dict)
        or render_manifest.get("status") != "complete"
        or render_manifest.get("suite") != "capacity"
        or render_manifest.get("run_count") != CAPACITY_COUNT
        or render_manifest.get("presets") != list(CAPACITY_PRESETS)
        or not isinstance(render_manifest.get("gates"), dict)
        or not all(render_manifest["gates"].values())
        or not isinstance(summary, dict)
        or summary.get("suite") != "capacity"
        or not isinstance(validation, dict)
        or validation.get("status") != "complete"
        or validation.get("exact_run_count") != CAPACITY_COUNT
        or validation.get("presets") != list(CAPACITY_PRESETS)
        or not (output_root / "capacity-evidence.md").is_file()
    ):
        raise EvidenceError("capacity render or summary failed final validation")


def finalize_capacity(
    *,
    scale_results_root: Path,
    large_results_root: Path,
    output_root: Path,
) -> dict[str, object]:
    scale_root = _resolved_directory(scale_results_root, "scale results root")
    large_root = _resolved_directory(large_results_root, "large results root")
    expanded_output = output_root.expanduser()
    if expanded_output.is_symlink():
        raise EvidenceError(f"output root may not be a symlink: {expanded_output}")
    resolved_output = expanded_output.resolve(strict=False)
    _validate_paths(scale_root, large_root, resolved_output)
    if resolved_output.exists() and not resolved_output.is_dir():
        raise EvidenceError(f"output root is not a directory: {resolved_output}")
    resolved_output.mkdir(parents=True, exist_ok=True)

    with _output_lock(resolved_output):
        scale_sources, scale_summary = _suite_sources(scale_root, "scale")
        large_sources, large_summary = _suite_sources(large_root, "large")
        collisions = sorted(set(scale_sources) & set(large_sources))
        if collisions:
            raise EvidenceError(f"run directory names collide: {collisions}")
        sources = {**scale_sources, **large_sources}
        identities = {
            (str(run["condition"]), str(run["preset"]), int(run["seed"]))
            for summary in (scale_summary, large_summary)
            for run in summary["runs"]
        }
        if len(sources) != CAPACITY_COUNT or len(identities) != CAPACITY_COUNT:
            raise EvidenceError("capacity inputs do not form 27 unique identities")

        union_root = resolved_output / "run-union"
        links = _prepare_union(union_root, sources)
        source_manifest = {
            "schema_version": 1,
            "scale_results_root": str(scale_root),
            "large_results_root": str(large_root),
            "run_union": str(union_root),
            "run_count": len(links),
            "links": [
                {"name": link.name, "target": str(link.resolve(strict=True))}
                for link in links
            ],
        }
        _atomic_json(
            resolved_output / "capacity-source-manifest.json",
            source_manifest,
        )

        figure_root = resolved_output / "figures"
        log_root = resolved_output / "logs"
        for directory in (figure_root, log_root):
            if directory.is_symlink():
                raise EvidenceError(f"managed directory may not be a symlink: {directory}")
            directory.mkdir(exist_ok=True)

        render_command = [
            sys.executable,
            str(Path(__file__).with_name("render_priority.py")),
            "--output",
            str(figure_root),
        ]
        for preset in CAPACITY_PRESETS:
            render_command.extend(["--preset", preset])
        for link in links:
            render_command.extend(["--run", str(link)])
        _run_logged(render_command, log_root / "render-capacity.log")

        _run_logged(
            [
                sys.executable,
                str(Path(__file__).with_name("summarize_priority_evidence.py")),
                "--results-root",
                str(union_root),
                "--suite",
                "capacity",
                "--output-json",
                str(resolved_output / "capacity-evidence.json"),
                "--output-markdown",
                str(resolved_output / "capacity-evidence.md"),
            ],
            log_root / "summarize-capacity.log",
        )
        _validate_outputs(resolved_output)

        manifest = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "run_count": CAPACITY_COUNT,
            "presets": list(CAPACITY_PRESETS),
            "scale_results_root": str(scale_root),
            "large_results_root": str(large_root),
            "run_union": str(union_root),
            "figures": str(figure_root),
            "summary_json": str(resolved_output / "capacity-evidence.json"),
            "summary_markdown": str(resolved_output / "capacity-evidence.md"),
        }
        _atomic_json(
            resolved_output / "capacity-finalize-manifest.json",
            manifest,
        )
        return manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="capacity-finalizer-") as temporary:
        root = Path(temporary)
        scale_root = root / "scale"
        large_root = root / "large"
        output_root = root / "capacity"
        _synthetic_fixture(scale_root, suite="scale")
        _synthetic_fixture(large_root, suite="large")

        first = finalize_capacity(
            scale_results_root=scale_root,
            large_results_root=large_root,
            output_root=output_root,
        )
        second = finalize_capacity(
            scale_results_root=scale_root,
            large_results_root=large_root,
            output_root=output_root,
        )
        links = list((output_root / "run-union").iterdir())
        if (
            first.get("status") != "complete"
            or second.get("status") != "complete"
            or len(links) != CAPACITY_COUNT
            or not all(path.is_symlink() for path in links)
        ):
            raise AssertionError("capacity finalizer is not complete and idempotent")

        stale_link, wrong_link = sorted(links)[:2]
        expected_target = stale_link.resolve(strict=True)
        stale_link.unlink()
        os.symlink(wrong_link.resolve(strict=True), stale_link)
        try:
            try:
                finalize_capacity(
                    scale_results_root=scale_root,
                    large_results_root=large_root,
                    output_root=output_root,
                )
            except EvidenceError:
                pass
            else:
                raise AssertionError("stale union link was accepted")
        finally:
            stale_link.unlink()
            os.symlink(expected_target, stale_link)

        operator = next(large_root.glob("*/operator_reuse_zz_priority.json"))
        hidden = operator.with_suffix(".json.hidden")
        operator.replace(hidden)
        try:
            try:
                finalize_capacity(
                    scale_results_root=scale_root,
                    large_results_root=large_root,
                    output_root=output_root,
                )
            except EvidenceError:
                pass
            else:
                raise AssertionError("incomplete analysis was accepted")
        finally:
            hidden.replace(operator)

        extra = scale_root / "unexpected-run"
        extra.mkdir()
        (extra / "config.json").write_text("{}\n")
        try:
            try:
                finalize_capacity(
                    scale_results_root=scale_root,
                    large_results_root=large_root,
                    output_root=output_root,
                )
            except EvidenceError:
                pass
            else:
                raise AssertionError("an extra run config was accepted")
        finally:
            (extra / "config.json").unlink()
            extra.rmdir()

        try:
            finalize_capacity(
                scale_results_root=scale_root,
                large_results_root=large_root,
                output_root=scale_root / "nested-output",
            )
        except EvidenceError:
            pass
        else:
            raise AssertionError("nested output was accepted")
    print(
        "self-test passed: exact scale18 + large9 union, render, summary, "
        "idempotent resume, and collision/incomplete/path rejection"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the exact 27-run small/medium/large capacity comparison "
            "without copying raw data."
        )
    )
    parser.add_argument("--scale-results-root", type=Path)
    parser.add_argument("--large-results-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    missing = [
        flag
        for flag, value in (
            ("--scale-results-root", args.scale_results_root),
            ("--large-results-root", args.large_results_root),
            ("--output-root", args.output_root),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"required arguments missing: {', '.join(missing)}")
    manifest = finalize_capacity(
        scale_results_root=args.scale_results_root,
        large_results_root=args.large_results_root,
        output_root=args.output_root,
    )
    print(
        f"capacity finalization {manifest['status']}: "
        f"{manifest['run_count']} runs in {manifest['figures']}"
    )


if __name__ == "__main__":
    main()
