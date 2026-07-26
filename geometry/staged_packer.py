from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


MIB = 1024**2
GIB = 1024**3
TEMP_SH_URL = re.compile(r"https://temp\.sh/[^\s<>\"']+")


@dataclass(frozen=True)
class Profile:
    marker: Path
    required_markers: tuple[Path, ...]
    roots: tuple[Path, ...]
    archive_prefix: str


PROFILES = {
    "full60": Profile(
        marker=Path(
            "/workspace/geometry-reuse-logs/finalize-60k/done.json"
        ),
        required_markers=(
            Path(
                "/workspace/geometry-reuse-logs/"
                "causal-final-60k/complete.json"
            ),
        ),
        roots=(
            Path("/workspace/geometry-reuse-results"),
            Path("/workspace/geometry-reuse-logs"),
            Path("/workspace/geometry-reuse-figures"),
        ),
        archive_prefix="vi-full60",
    ),
    "final150": Profile(
        marker=Path(
            "/workspace/geometry-reuse-extend-logs/"
            "analysis-handoff-complete.json"
        ),
        required_markers=(
            Path(
                "/workspace/geometry-reuse-logs/"
                "finalize-60k/done.json"
            ),
            Path(
                "/workspace/geometry-reuse-extend-logs/"
                "finalize-150k/done.json"
            ),
            Path(
                "/workspace/geometry-reuse-extend-logs/"
                "causal-final-150k/complete.json"
            ),
        ),
        roots=(
            Path("/workspace/geometry-reuse-results"),
            Path("/workspace/geometry-reuse-logs"),
            Path("/workspace/geometry-reuse-extend-logs"),
            Path("/workspace/geometry-reuse-figures"),
        ),
        archive_prefix="vi-final150",
    ),
}


@dataclass(frozen=True)
class Settings:
    stage: str
    marker: Path
    required_markers: tuple[Path, ...]
    roots: tuple[Path, ...]
    output_root: Path
    archive_prefix: str
    chunk_bytes: int
    min_free_bytes: int
    poll_seconds: float
    timeout_seconds: float
    upload_endpoint: str
    upload_retries: int


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def marker_ready(path: Path) -> bool:
    if not path.exists():
        return False
    if path.suffix != ".json":
        return True
    return isinstance(load_json(path), dict)


def wait_for_markers(settings: Settings, log: Callable[[str], None]) -> None:
    pending = (settings.marker, *settings.required_markers)
    deadline = time.monotonic() + settings.timeout_seconds
    last_log = 0.0
    while time.monotonic() < deadline:
        missing = [path for path in pending if not marker_ready(path)]
        if not missing:
            for path in pending:
                log(f"marker ready: {path}")
            return
        if time.monotonic() - last_log >= 300:
            log("waiting for " + ", ".join(str(path) for path in missing))
            last_log = time.monotonic()
        time.sleep(settings.poll_seconds)
    raise TimeoutError("markers did not complete before timeout")


def iter_files(root: Path) -> Iterator[Path]:
    if root.is_file() or root.is_symlink():
        yield root
        return
    for directory, names, files in os.walk(root):
        names.sort()
        files.sort()
        base = Path(directory)
        for name in files:
            yield base / name


def inventory(roots: tuple[Path, ...]) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    total = 0
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"artifact root does not exist: {root}")
        for path in iter_files(root):
            stat = path.lstat()
            total += stat.st_size
            records.append(
                {
                    "path": str(path),
                    "bytes": stat.st_size,
                    "kind": "symlink" if path.is_symlink() else "file",
                }
            )
    return records, total


def disk_guard(
    settings: Settings,
    *,
    additional_bytes: int,
    phase: str,
) -> None:
    free = shutil.disk_usage(settings.output_root).free
    required = additional_bytes + settings.min_free_bytes
    if free < required:
        raise OSError(
            f"{phase} disk guard: {free / GIB:.2f} GiB free, "
            f"{required / GIB:.2f} GiB required before starting"
        )


def build_archive(
    archive: Path,
    roots: tuple[Path, ...],
) -> None:
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as bundle:
        for root in roots:
            bundle.add(root, arcname=str(root).lstrip("/"))
    temporary.replace(archive)


def split_archive(
    archive: Path,
    chunks_root: Path,
    chunk_bytes: int,
) -> list[dict[str, object]]:
    chunks_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with archive.open("rb") as source:
        index = 0
        while block := source.read(chunk_bytes):
            name = f"{archive.name}.part-{index:04d}"
            path = chunks_root / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(block)
            temporary.replace(path)
            records.append(
                {
                    "index": index,
                    "raw_chunk": str(path),
                    "raw_bytes": len(block),
                    "raw_sha256": hashlib.sha256(block).hexdigest(),
                    "wrapper": str(chunks_root / f"{name}.tar.gz"),
                    "url": None,
                }
            )
            index += 1
    if not records:
        raise RuntimeError("archive split produced no chunks")
    return records


def wrap_chunk(chunk: Path, wrapper: Path) -> None:
    temporary = wrapper.with_suffix(wrapper.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as bundle:
        bundle.add(chunk, arcname=chunk.name)
    temporary.replace(wrapper)


def upload_temp_sh(
    wrapper: Path,
    *,
    endpoint: str,
    retries: int,
    log: Callable[[str], None],
) -> str:
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                "1800",
                "-F",
                f"file=@{wrapper}",
                endpoint,
            ],
            text=True,
            capture_output=True,
        )
        match = TEMP_SH_URL.search(result.stdout)
        if result.returncode == 0 and match is not None:
            return match.group(0)
        message = result.stderr.strip() or result.stdout.strip()
        log(
            f"upload attempt {attempt}/{retries} failed for "
            f"{wrapper.name}: {message}"
        )
        if attempt < retries:
            time.sleep(min(10 * 2 ** (attempt - 1), 120))
    raise RuntimeError(f"upload failed after {retries} attempts: {wrapper}")


def valid_url(value: object) -> bool:
    return isinstance(value, str) and TEMP_SH_URL.fullmatch(value) is not None


def resumable_parts(
    manifest_path: Path,
    *,
    archive: Path,
    archive_sha: str,
    chunk_bytes: int,
) -> list[dict[str, object]] | None:
    payload = load_json(manifest_path)
    if (
        not isinstance(payload, dict)
        or payload.get("archive") != str(archive)
        or payload.get("archive_sha256") != archive_sha
        or payload.get("archive_bytes") != archive.stat().st_size
        or payload.get("chunk_bytes") != chunk_bytes
    ):
        return None
    parts = payload.get("parts")
    if not isinstance(parts, list) or not parts:
        return None
    validated: list[dict[str, object]] = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or part.get("index") != index:
            return None
        path = Path(str(part.get("raw_chunk", "")))
        if (
            not path.is_file()
            or path.stat().st_size != part.get("raw_bytes")
            or path.stat().st_size > chunk_bytes
            or sha256(path) != part.get("raw_sha256")
            or (part.get("url") is not None and not valid_url(part.get("url")))
        ):
            return None
        validated.append(dict(part))
    if sum(int(part["raw_bytes"]) for part in validated) != archive.stat().st_size:
        return None
    return validated


def completion_valid(path: Path) -> bool:
    payload = load_json(path)
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        return False
    archive = Path(str(payload.get("archive", "")))
    if (
        not archive.is_file()
        or archive.stat().st_size != payload.get("archive_bytes")
        or sha256(archive) != payload.get("archive_sha256")
    ):
        return False
    parts = payload.get("parts")
    manifest_parts = resumable_parts(
        Path(str(payload.get("parts_manifest", ""))),
        archive=archive,
        archive_sha=str(payload["archive_sha256"]),
        chunk_bytes=int(payload.get("chunk_bytes", 0)),
    )
    if (
        not isinstance(parts, list)
        or not parts
        or manifest_parts is None
        or len(parts) != len(manifest_parts)
    ):
        return False
    return all(
        isinstance(part, dict)
        and valid_url(part.get("url"))
        and part.get("index") == manifest.get("index")
        and part.get("raw_sha256") == manifest.get("raw_sha256")
        and part.get("url") == manifest.get("url")
        for part, manifest in zip(parts, manifest_parts, strict=True)
    )


def validate_paths(settings: Settings) -> None:
    output = settings.output_root.resolve()
    for root in settings.roots:
        source = root.resolve()
        if output == source or output.is_relative_to(source):
            raise ValueError(
                f"output root {output} must not be inside source root {source}"
            )


def run(
    settings: Settings,
    *,
    uploader: Callable[[Path], str] | None = None,
) -> Path:
    validate_paths(settings)
    stage_root = settings.output_root / settings.stage
    stage_root.mkdir(parents=True, exist_ok=True)
    completion_path = stage_root / "complete.json"
    log_path = stage_root / "pack.log"

    def log(message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        with log_path.open("a") as stream:
            stream.write(line + "\n")

    lock = (stage_root / "pack.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"stage already has a live packer: {settings.stage}") from error

    if completion_valid(completion_path):
        log(f"already complete: {completion_path}")
        return completion_path

    wait_for_markers(settings, log)
    inventory_path = stage_root / "inventory.json"
    archive = stage_root / f"{settings.archive_prefix}.tar.gz"
    if archive.exists():
        if not inventory_path.exists():
            raise RuntimeError(
                f"archive exists without its inventory: {archive}"
            )
        log(f"resuming retained archive: {archive}")
    else:
        records, source_bytes = inventory(settings.roots)
        atomic_json(
            inventory_path,
            {
                "created_at": now(),
                "marker": str(settings.marker),
                "required_markers": [
                    str(path) for path in settings.required_markers
                ],
                "roots": [str(path) for path in settings.roots],
                "file_count": len(records),
                "source_bytes": source_bytes,
                "files": records,
            },
        )
        tar_overhead = (len(records) + len(settings.roots) + 1024) * 4096
        disk_guard(
            settings,
            additional_bytes=source_bytes
            + source_bytes // 50
            + tar_overhead,
            phase="archive",
        )
        build_archive(archive, settings.roots)
    archive_sha = sha256(archive)
    archive_sha_path = archive.with_suffix(archive.suffix + ".sha256")
    archive_sha_path.write_text(f"{archive_sha}  {archive.name}\n")
    log(
        f"archive ready: {archive} "
        f"({archive.stat().st_size} bytes, sha256={archive_sha})"
    )

    disk_guard(
        settings,
        additional_bytes=archive.stat().st_size + 2 * settings.chunk_bytes,
        phase="chunk",
    )
    manifest_path = stage_root / "parts.json"
    chunks = resumable_parts(
        manifest_path,
        archive=archive,
        archive_sha=archive_sha,
        chunk_bytes=settings.chunk_bytes,
    )
    if chunks is None:
        chunks = split_archive(
            archive,
            stage_root / f"{archive.name}.parts",
            settings.chunk_bytes,
        )

    def write_manifest() -> None:
        atomic_json(
            manifest_path,
            {
                "updated_at": now(),
                "stage": settings.stage,
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": archive_sha,
                "chunk_bytes": settings.chunk_bytes,
                "parts": chunks,
            },
        )

    write_manifest()
    for part in chunks:
        if valid_url(part["url"]):
            continue
        raw_chunk = Path(str(part["raw_chunk"]))
        wrapper = Path(str(part["wrapper"]))
        wrap_chunk(raw_chunk, wrapper)
        if uploader is None:
            url = upload_temp_sh(
                wrapper,
                endpoint=settings.upload_endpoint,
                retries=settings.upload_retries,
                log=log,
            )
        else:
            url = uploader(wrapper)
        if not valid_url(url):
            raise RuntimeError(f"uploader returned an invalid URL: {url}")
        part["wrapper_bytes"] = wrapper.stat().st_size
        part["url"] = url
        write_manifest()
        wrapper.unlink()
        log(f"uploaded part {part['index'] + 1}/{len(chunks)}: {url}")

    if not all(valid_url(part.get("url")) for part in chunks):
        raise RuntimeError("refusing to complete with missing part URLs")
    atomic_json(
        completion_path,
        {
            "completed_at": now(),
            "status": "complete",
            "stage": settings.stage,
            "marker": str(settings.marker),
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
            "archive_sha256_file": str(archive_sha_path),
            "chunk_bytes": settings.chunk_bytes,
            "inventory": str(inventory_path),
            "parts_manifest": str(manifest_path),
            "sources_deleted": False,
            "archive_and_raw_chunks_retained": True,
            "parts": chunks,
        },
    )
    log(f"complete: {completion_path}")
    return completion_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--stage")
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--required-marker", action="append", type=Path)
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/geometry-stage-archives"),
    )
    parser.add_argument("--archive-prefix")
    parser.add_argument("--chunk-mib", type=int, default=42)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument(
        "--upload-endpoint",
        default="https://temp.sh/upload",
    )
    parser.add_argument("--upload-retries", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> Settings:
    profile = PROFILES.get(args.profile) if args.profile else None
    marker = args.marker or (profile.marker if profile else None)
    roots = tuple(args.root or (profile.roots if profile else ()))
    required_markers = tuple(
        args.required_marker
        or (profile.required_markers if profile else ())
    )
    stage = args.stage or args.profile
    prefix = args.archive_prefix or (
        profile.archive_prefix if profile else stage
    )
    if not marker or not roots or not stage or not prefix:
        raise ValueError(
            "provide --profile or explicit --stage, --marker, --root, "
            "and --archive-prefix"
        )
    if not 1 <= args.chunk_mib <= 42:
        raise ValueError("--chunk-mib must be between 1 and 42")
    if min(
        args.min_free_gib,
        args.poll_seconds,
        args.timeout_hours,
        args.upload_retries,
    ) <= 0:
        raise ValueError("sizes, polling, timeout, and retries must be positive")
    return Settings(
        stage=stage,
        marker=marker,
        required_markers=required_markers,
        roots=roots,
        output_root=args.output_root,
        archive_prefix=prefix,
        chunk_bytes=args.chunk_mib * MIB,
        min_free_bytes=int(args.min_free_gib * GIB),
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_hours * 3600,
        upload_endpoint=args.upload_endpoint,
        upload_retries=args.upload_retries,
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        (source / "small.json").write_text('{"ok": true}\n')
        (source / "random.bin").write_bytes(os.urandom(700_000))
        marker = root / "done.json"
        marker.write_text('{"status": "complete"}\n')
        settings = Settings(
            stage="tiny",
            marker=marker,
            required_markers=(),
            roots=(source,),
            output_root=root / "artifacts",
            archive_prefix="tiny",
            chunk_bytes=256 * 1024,
            min_free_bytes=1,
            poll_seconds=0.01,
            timeout_seconds=1,
            upload_endpoint="unused",
            upload_retries=1,
        )

        def fake_upload(path: Path) -> str:
            with tarfile.open(path, "r:gz") as bundle:
                members = bundle.getmembers()
                assert len(members) == 1 and members[0].isfile()
            return f"https://temp.sh/test/{path.name}"

        completion = run(
            settings,
            uploader=fake_upload,
        )
        payload = load_json(completion)
        assert completion_valid(completion)
        assert isinstance(payload, dict)
        parts = payload["parts"]
        assert isinstance(parts, list) and len(parts) >= 2
        assert all(
            int(part["raw_bytes"]) <= settings.chunk_bytes for part in parts
        )
        assert all(Path(str(part["raw_chunk"])).exists() for part in parts)
        assert not any(
            Path(str(part["wrapper"])).exists() for part in parts
        )
        reconstructed = b"".join(
            Path(str(part["raw_chunk"])).read_bytes() for part in parts
        )
        assert hashlib.sha256(reconstructed).hexdigest() == payload[
            "archive_sha256"
        ]
        with tarfile.open(Path(str(payload["archive"])), "r:gz") as bundle:
            assert not any(
                member.name.startswith("artifact-metadata/")
                for member in bundle.getmembers()
            )

        def fail_upload(_: Path) -> str:
            raise AssertionError("completed stage attempted another upload")

        assert run(settings, uploader=fail_upload) == completion
        assert (source / "small.json").exists()
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(settings_from_args(args))


if __name__ == "__main__":
    main()
