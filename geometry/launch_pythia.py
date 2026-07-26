from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


EARLY_AND_LATE = (
    "step0",
    "step1",
    "step4",
    "step16",
    "step64",
    "step256",
    "step1000",
    "step4000",
    "step16000",
    "step64000",
    "step143000",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    return parser.parse_args()


def matrix(profile: str) -> list[tuple[str, tuple[str, ...]]]:
    if profile == "quick":
        return [
            ("EleutherAI/pythia-70m", EARLY_AND_LATE),
            ("EleutherAI/pythia-160m", EARLY_AND_LATE),
        ]
    return [
        ("EleutherAI/pythia-70m", EARLY_AND_LATE),
        ("EleutherAI/pythia-160m", EARLY_AND_LATE),
        (
            "EleutherAI/pythia-410m",
            (
                "step0",
                "step4",
                "step64",
                "step1000",
                "step4000",
                "step16000",
                "step64000",
                "step143000",
            ),
        ),
        (
            "EleutherAI/pythia-1b",
            (
                "step0",
                "step16",
                "step256",
                "step4000",
                "step32000",
                "step143000",
            ),
        ),
    ]


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for model, revisions in matrix(args.profile):
        slug = model.replace("/", "--")
        command = [
            sys.executable,
            str(Path(__file__).with_name("pythia_geometry.py")),
            "--model",
            model,
            "--revisions",
            *revisions,
            "--cache-dir",
            str(args.cache_dir),
            "--output-root",
            str(args.output_root),
        ]
        log_path = args.log_root / f"{slug}.log"
        started = time.time()
        with log_path.open("a") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record = {
            "model": model,
            "revisions": revisions,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
            "log": str(log_path),
        }
        results.append(record)
        print(
            f"finished {model} rc={result.returncode} "
            f"in {record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if result.returncode != 0:
            break
    result_path = args.log_root / f"pythia-{args.profile}-results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    if any(result["returncode"] != 0 for result in results):
        raise SystemExit("a Pythia extraction failed")


if __name__ == "__main__":
    main()
