"""Dataset downloading, safe archive extraction, image verification, and manifest generation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import shutil
import tarfile
import time
from typing import Any, Callable, Iterable, Sequence
import urllib.error
import urllib.request
import zipfile

from PIL import Image

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
ARCHIVE_EXTENSIONS: tuple[str, ...] = (".tar.gz", ".tgz", ".tar", ".zip")
DEFAULT_CHUNK_SIZE: int = 1 << 20


class DownloadError(Exception):
    """Raised when a file download fails or fails size/checksum verification."""


class UnsafeArchiveError(Exception):
    """Raised when an archive contains members that escape the destination directory."""


@dataclass(frozen=True)
class Source:
    """Metadata and integrity constraints for a dataset file source."""

    name: str
    url: str
    filename: str
    size: int | None = None
    sha256: str | None = None
    licence: str = ""
    licence_url: str = ""


def sha256_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Compute the hexadecimal SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _matches_expected(path: Path, source: Source) -> bool:
    """Return True if an existing file matches the expected size and SHA-256."""
    if not path.is_file():
        return False
    if source.size is not None and path.stat().st_size != source.size:
        return False
    if source.sha256 is not None and sha256_file(path).lower() != source.sha256.lower():
        return False
    return True


def _stream_to_part(
    source: Source,
    part_path: Path,
    opener: Callable[..., Any],
    chunk_size: int,
    timeout: float,
) -> tuple[int, str]:
    """Stream URL content into part_path and return (total_bytes, sha256_hex)."""
    hasher = hashlib.sha256()
    total = 0
    with opener(source.url, timeout=timeout) as response, part_path.open("wb") as out:
        while chunk := response.read(chunk_size):
            out.write(chunk)
            hasher.update(chunk)
            total += len(chunk)
    return total, hasher.hexdigest()


def download(
    source: Source,
    dest_dir: str | Path,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: float = 60,
) -> Path:
    """Download a source file atomically with retries, size check, and SHA-256 check."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    final_path = dest / source.filename

    if _matches_expected(final_path, source):
        return final_path

    part_path = dest / f"{source.filename}.part"
    last_error: Exception | None = None

    for attempt in range(1, max(1, retries) + 1):
        try:
            total, digest = _stream_to_part(source, part_path, opener, chunk_size, timeout)
            break
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            part_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(backoff * attempt)
    else:
        raise DownloadError(f"Failed to download {source.url} after {retries} attempts") from last_error

    if source.size is not None and total != source.size:
        part_path.unlink(missing_ok=True)
        raise DownloadError(f"Size mismatch for {source.filename}: expected {source.size}, got {total}")

    if source.sha256 is not None and digest.lower() != source.sha256.lower():
        part_path.unlink(missing_ok=True)
        raise DownloadError(
            f"SHA-256 mismatch for {source.filename}: expected {source.sha256}, got {digest}"
        )

    os.replace(part_path, final_path)
    return final_path


def _is_archive_name(name: str) -> bool:
    """Return True if the filename has a supported archive extension."""
    lower = name.lower()
    return any(lower.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def _is_absolute_member(member_name: str) -> bool:
    """Return True if an archive member path is absolute on POSIX or Windows."""
    return (
        member_name.startswith(("/", "\\"))
        or PurePosixPath(member_name).is_absolute()
        or PureWindowsPath(member_name).is_absolute()
    )


def _validate_target_path(member_name: str, dest_resolved: Path) -> Path:
    """Validate that member_name stays strictly inside dest_resolved and return its resolved path."""
    if _is_absolute_member(member_name):
        raise UnsafeArchiveError(f"Absolute path in archive: {member_name!r}")
    target = (dest_resolved / member_name).resolve()
    if not target.is_relative_to(dest_resolved):
        raise UnsafeArchiveError(f"Archive member escapes destination: {member_name!r}")
    return target


def _validate_tar_member(member: tarfile.TarInfo, dest_resolved: Path) -> None:
    """Validate a tar member path and any symlink/hardlink target."""
    target = _validate_target_path(member.name, dest_resolved)
    if member.issym() or member.islnk():
        if _is_absolute_member(member.linkname):
            raise UnsafeArchiveError(f"Absolute link target in archive: {member.linkname!r}")
        base = target.parent if member.issym() else dest_resolved
        link_target = (base / member.linkname).resolve()
        if not link_target.is_relative_to(dest_resolved):
            raise UnsafeArchiveError(f"Archive link escapes destination: {member.linkname!r}")


def _extract_zip(archive_path: Path, dest_path: Path, dest_resolved: Path) -> list[Path]:
    """Validate and extract a ZIP archive into dest_path."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = zf.infolist()
        for info in infos:
            _validate_target_path(info.filename, dest_resolved)

        dest_path.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        for info in infos:
            out_path = dest_path / info.filename
            if info.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(out_path)
        return extracted


def _extract_tar(archive_path: Path, dest_path: Path, dest_resolved: Path) -> list[Path]:
    """Validate and extract a TAR/TAR.GZ/TGZ archive into dest_path."""
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        for member in members:
            _validate_tar_member(member, dest_resolved)

        dest_path.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        for member in members:
            out_path = dest_path / member.name
            if member.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is not None:
                    with src, out_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                extracted.append(out_path)
            # Links, devices and FIFOs are skipped: datasets never need them, and a Colab machine runs as root.
        return extracted


def safe_extract(archive: str | Path, dest: str | Path) -> list[Path]:
    """Safely extract a .zip, .tar, .tar.gz, or .tgz archive into dest."""
    archive_path = Path(archive)
    dest_path = Path(dest)

    if not _is_archive_name(archive_path.name):
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    marker = dest_path / f".extracted-{archive_path.name}"
    if marker.exists():
        return []

    dest_resolved = dest_path.resolve()
    if archive_path.name.lower().endswith(".zip"):
        extracted = _extract_zip(archive_path, dest_path, dest_resolved)
    else:
        extracted = _extract_tar(archive_path, dest_path, dest_resolved)

    dest_path.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    return extracted


def count_images(folder: str | Path) -> int:
    """Recursively count image files in folder (case-insensitive extension check)."""
    root = Path(folder)
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith("._")
    )


def verify_images(
    paths: Sequence[str | Path],
    *,
    sample: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Verify image integrity with Pillow, optionally checking a deterministic random sample."""
    items = list(paths)
    if sample is not None and sample < len(items):
        chosen = random.Random(seed).sample(items, sample)
    else:
        chosen = items

    bad: list[str] = []
    for item in chosen:
        try:
            with Image.open(item) as img:
                img.verify()
        except Exception:
            bad.append(str(item))

    return {"checked": len(chosen), "bad": bad}


def write_manifest(
    path: str | Path,
    sources: Iterable[Source],
    counts: dict[str, int],
) -> dict[str, Any]:
    """Write a dataset manifest JSON file and return the manifest dictionary."""
    manifest: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "sources": [dataclasses.asdict(source) for source in sources],
        "counts": dict(counts),
    }
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def fetch(
    sources: Iterable[Source],
    root: str | Path,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Download all sources into root, extract archives, count images, and write manifest.json."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    source_list = list(sources)
    counts: dict[str, int] = {}

    for source in source_list:
        downloaded = download(source, root_path, opener=opener)
        extract_dir = root_path / source.name
        if _is_archive_name(downloaded.name):
            safe_extract(downloaded, extract_dir)
        counts[source.name] = count_images(extract_dir)

    return write_manifest(root_path / "manifest.json", source_list, counts)
