from __future__ import annotations

import argparse
from pathlib import Path

from key60_common import wait_for_marker, wait_for_runs
from staged_packer import GIB, MIB, Settings, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack the gated key60 result bundle locally, with optional "
            "explicit upload."
        )
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/geometry-key60-stage-archives"),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--chunk-mib", type=int, default=42)
    parser.add_argument(
        "--upload-endpoint",
        default="https://temp.sh/upload",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.chunk_mib <= 42:
        raise ValueError("--chunk-mib must be between 1 and 42")
    if (
        min(
            args.poll_seconds,
            args.timeout_hours,
            args.min_free_gib,
            args.upload_retries,
        )
        <= 0
    ):
        raise ValueError("poll, timeout, disk, and retry guards must be positive")
    final_marker = args.log_root / "key60-complete.json"
    operator_marker = args.log_root / "operator-complete.json"
    causal_marker = args.log_root / "causal-complete.json"
    wait_for_marker(
        final_marker,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    runs = wait_for_runs(
        results_root=args.results_root,
        manifest_path=args.manifest,
        results_path=args.training_results,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    settings = Settings(
        stage="key60",
        marker=final_marker,
        required_markers=(operator_marker, causal_marker),
        roots=(
            *(run.path for run in runs),
            args.log_root,
            args.figure_root,
        ),
        output_root=args.output_root,
        archive_prefix="vi-key60",
        chunk_bytes=args.chunk_mib * MIB,
        min_free_bytes=int(args.min_free_gib * GIB),
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_hours * 3600,
        local_only=args.local_only,
        upload_endpoint=args.upload_endpoint,
        upload_retries=args.upload_retries,
    )
    run(settings)


if __name__ == "__main__":
    main()
