"""Downloads datasets, unpacks archives safely, checks images and writes a manifest of what was fetched.

The module is written to run in a notebook on Google Colab (a hosted notebook service) as well as on a normal
computer. Every download is checked against the expected size and SHA-256 (a checksum that identifies the exact
contents of a file) when the source gives them, and archives are unpacked only when no member could write outside
the target folder.
"""

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
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, Iterable, Sequence
import urllib.error
import urllib.request
import zipfile

from PIL import Image

# File extensions that count as images.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# File extensions that the module can unpack.
ARCHIVE_EXTENSIONS: tuple[str, ...] = (".tar.gz", ".tgz", ".tar", ".zip")

# Number of bytes read from a file or a network connection at one time (1 MiB).
DEFAULT_CHUNK_SIZE: int = 1 << 20


class DownloadError(Exception):
    """Raised when a file download fails or fails the size or checksum verification."""


class UnsafeArchiveError(Exception):
    """Raised when an archive contains members that would be written outside the destination directory."""


class DownloadStopped(Exception):
    """Raised inside a download or an extraction when the caller has set its stop signal."""


def _check_stop(stop: threading.Event | None, what: str) -> None:
    """Raises DownloadStopped, carrying the name `what`, if the stop signal has been set."""
    if stop is not None and stop.is_set():
        raise DownloadStopped(what)


@dataclass(frozen=True)
class Source:
    """Describes one downloadable dataset file, together with the values used to check it and its licence."""

    # Short name of the dataset. It is also the name of the folder that the archive is unpacked into.
    name: str

    # Address that the file is downloaded from.
    url: str

    # Name that the downloaded file is saved under.
    filename: str

    # Expected size of the file in bytes. None skips the size check.
    size: int | None = None

    # Expected SHA-256 checksum of the file as hexadecimal text. None skips the checksum check.
    sha256: str | None = None

    # Name of the licence that the dataset is published under.
    licence: str = ""

    # Address of the full licence text.
    licence_url: str = ""


def sha256_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Returns the SHA-256 checksum of a file as hexadecimal text, reading the file `chunk_size` bytes at a time."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _matches_expected(path: Path, source: Source) -> bool:
    """Returns True if the file at `path` exists and has the size and SHA-256 that `source` expects.

    A check is skipped when the source does not give that value.
    """
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
    stop: threading.Event | None = None,
) -> tuple[int, str]:
    """Saves the content of the source's URL to `part_path` and returns the byte count and the SHA-256 checksum.

    The stop signal is checked after each chunk, so a stop request takes effect within one chunk.
    """
    hasher = hashlib.sha256()
    total = 0
    with opener(source.url, timeout=timeout) as response, part_path.open("wb") as out:
        while chunk := response.read(chunk_size):
            _check_stop(stop, source.filename)
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
    stop: threading.Event | None = None,
) -> Path:
    """Downloads a source file into `dest_dir` and returns its path.

    The data is first written to a temporary file with the ending ".part". The file is renamed to its final name only
    after the whole stream has been read and the size and the SHA-256 checksum, where the source gives them, have
    been checked. If a file that passes these checks is already in place, nothing is downloaded.

    `retries` is the total number of attempts. After failed attempt n, the function waits `backoff` times n seconds
    before the next attempt. When `stop` is set, the download ends within one chunk (or within one network timeout at
    worst), removes its partial file and raises DownloadStopped. A finished file is kept, so a second call skips it.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    final_path = dest / source.filename

    # A file that is already complete is not downloaded again.
    if _matches_expected(final_path, source):
        return final_path

    part_path = dest / f"{source.filename}.part"
    last_error: Exception | None = None

    # Download with retries. The loop ends with "break" on success. Its "else" branch runs only when every attempt
    # failed.
    for attempt in range(1, max(1, retries) + 1):
        try:
            total, digest = _stream_to_part(source, part_path, opener, chunk_size, timeout, stop)
            break
        except DownloadStopped:
            part_path.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            part_path.unlink(missing_ok=True)

            # Wait before the next attempt. When a stop signal exists, the wait ends early if the signal is set.
            if attempt < retries:
                if stop is not None:
                    if stop.wait(backoff * attempt):
                        raise DownloadStopped(source.filename) from exc
                else:
                    time.sleep(backoff * attempt)
    else:
        raise DownloadError(f"Failed to download {source.url} after {retries} attempts") from last_error

    # Check the size and the checksum of the finished download. A file that fails is deleted.
    if source.size is not None and total != source.size:
        part_path.unlink(missing_ok=True)
        raise DownloadError(f"Size mismatch for {source.filename}: expected {source.size}, got {total}")

    if source.sha256 is not None and digest.lower() != source.sha256.lower():
        part_path.unlink(missing_ok=True)
        raise DownloadError(
            f"SHA-256 mismatch for {source.filename}: expected {source.sha256}, got {digest}"
        )

    # Give the file its final name in one step.
    os.replace(part_path, final_path)
    return final_path


def _is_archive_name(name: str) -> bool:
    """Returns True if the file name ends with a supported archive extension, ignoring upper and lower case."""
    lower = name.lower()
    return any(lower.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def _is_absolute_member(member_name: str) -> bool:
    """Returns True if an archive member path is absolute, in the style of either Linux or Windows."""
    return (
        member_name.startswith(("/", "\\"))
        or PurePosixPath(member_name).is_absolute()
        or PureWindowsPath(member_name).is_absolute()
    )


def _validate_target_path(member_name: str, dest_resolved: Path) -> Path:
    """Returns the full path that an archive member would be written to, after checking that it is safe.

    Raises UnsafeArchiveError if the member name is absolute or if the path, once ".." parts and links are resolved,
    leaves `dest_resolved`.
    """
    if _is_absolute_member(member_name):
        raise UnsafeArchiveError(f"Absolute path in archive: {member_name!r}")
    target = (dest_resolved / member_name).resolve()
    if not target.is_relative_to(dest_resolved):
        raise UnsafeArchiveError(f"Archive member escapes destination: {member_name!r}")
    return target


def _validate_tar_member(member: tarfile.TarInfo, dest_resolved: Path) -> None:
    """Checks that a tar member and, for a link, the file that it points to stay inside `dest_resolved`.

    A symbolic link (a file that points to another path) is resolved from the folder that holds the link. A hard link
    (a second name for an existing file) is resolved from the extraction root. Raises UnsafeArchiveError if either
    leaves the destination.
    """
    target = _validate_target_path(member.name, dest_resolved)
    if member.issym() or member.islnk():
        if _is_absolute_member(member.linkname):
            raise UnsafeArchiveError(f"Absolute link target in archive: {member.linkname!r}")
        base = target.parent if member.issym() else dest_resolved
        link_target = (base / member.linkname).resolve()
        if not link_target.is_relative_to(dest_resolved):
            raise UnsafeArchiveError(f"Archive link escapes destination: {member.linkname!r}")


def _extract_zip(archive_path: Path, dest_path: Path, dest_resolved: Path,
                 stop: threading.Event | None = None) -> list[Path]:
    """Checks every member of a ZIP archive and then extracts it into `dest_path`.

    Returns the files that were written. All members are checked before the first one is written, so an unsafe
    archive leaves nothing behind. The stop signal is checked before each member.
    """
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = zf.infolist()

        # Check all member paths first.
        for info in infos:
            _validate_target_path(info.filename, dest_resolved)

        # Write the members one by one.
        dest_path.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        for info in infos:
            _check_stop(stop, archive_path.name)
            out_path = dest_path / info.filename
            if info.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(out_path)
        return extracted


def _extract_tar(archive_path: Path, dest_path: Path, dest_resolved: Path,
                 stop: threading.Event | None = None) -> list[Path]:
    """Checks every member of a TAR, TAR.GZ or TGZ archive, then extracts its folders and regular files to `dest_path`.

    Returns the files that were written. All members are checked before the first one is written, so an unsafe
    archive leaves nothing behind. The stop signal is checked before each member.
    """
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()

        # Check all member paths and link targets first.
        for member in members:
            _validate_tar_member(member, dest_resolved)

        # Write the members one by one.
        dest_path.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        for member in members:
            _check_stop(stop, archive_path.name)
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
            # Links, devices and FIFOs (named pipes) are skipped. Datasets never need them, and a Colab machine
            # runs as root, so extracting them could be unsafe.
        return extracted


def safe_extract(archive: str | Path, dest: str | Path, *, stop: threading.Event | None = None) -> list[Path]:
    """Extracts a .zip, .tar, .tar.gz or .tgz archive into `dest` and returns the files that were written.

    Raises ValueError for any other file type and UnsafeArchiveError if a member would be written outside `dest`.
    After a complete extraction the function writes a marker file named ".extracted-<archive name>" into `dest`. When
    the marker already exists, the archive is not unpacked again and the function returns an empty list.

    When `stop` is set, extraction ends before the next member and raises DownloadStopped without writing the
    marker, so the next call extracts the archive again from the start.
    """
    archive_path = Path(archive)
    dest_path = Path(dest)

    if not _is_archive_name(archive_path.name):
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    # An earlier complete extraction has left its marker, so there is nothing to do.
    marker = dest_path / f".extracted-{archive_path.name}"
    if marker.exists():
        return []

    dest_resolved = dest_path.resolve()
    if archive_path.name.lower().endswith(".zip"):
        extracted = _extract_zip(archive_path, dest_path, dest_resolved, stop)
    else:
        extracted = _extract_tar(archive_path, dest_path, dest_resolved, stop)

    # Record that the extraction finished.
    dest_path.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    return extracted


def count_images(folder: str | Path) -> int:
    """Returns the number of image files in a folder and all its subfolders, or 0 if the folder does not exist.

    The extension check ignores upper and lower case. Files whose names start with "._" are not counted. Archives
    made on macOS often contain these small metadata copies, and they are not images.
    """
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
    """Checks that image files can be opened with Pillow (an image library) and are not damaged.

    If `sample` is smaller than the number of files, only that many randomly chosen files are checked. The same `seed`
    always picks the same files. Returns a dictionary with "checked", the number of files examined, and "bad", the
    list of paths that failed the check.
    """
    items = list(paths)
    if sample is not None and sample < len(items):
        chosen = random.Random(seed).sample(items, sample)
    else:
        chosen = items

    # Any error while opening or verifying a file counts as a damaged image.
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
    """Writes a JSON manifest that records what was fetched, and returns the same content as a dictionary.

    The manifest holds the creation time in UTC, the description of every source (including its licence) and the
    number of images found for each dataset name in `counts`. Missing parent folders are created.
    """
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
    """Downloads the sources one after another into `root`, unpacks archives, counts images and writes a manifest.

    Each archive is unpacked into a subfolder of `root` that carries the source's name. The manifest is saved as
    manifest.json in `root` and is also returned.
    """
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


def _bytes_on_disk(sources: Sequence[Source], root: Path) -> int:
    """Returns the bytes written so far for these sources, counting finished files and ".part" files together."""
    total = 0
    for source in sources:
        for path in (root / source.filename, root / f"{source.filename}.part"):
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def fetch_all(
    sources: Iterable[Source],
    root: str | Path,
    *,
    extract_root: str | Path | None = None,
    local_names: Iterable[str] = (),
    workers: int = 6,
    retries: int = 4,
    report_every: float = 30.0,
    poll: float = 0.5,
    log: Callable[[str], Any] = print,
    download_fn: Callable[..., Path] | None = None,
) -> list[dict[str, Any]]:
    """Downloads and unpacks every source in parallel for the notebook's data cell, and reports as it goes.

    Downloaded files live in `root`. On Colab this is the data/raw folder of the artifacts folder in Google Drive, so
    a file downloaded once is found there by every later machine. Archives are unpacked into `extract_root`, which is
    the machine's own disk, or into `root` when `extract_root` is not given. Sources named in `local_names` are
    downloaded into `extract_root` instead, so their files never reach `root`. A complete file already in place is
    not downloaded again.

    The function logs one line for each source as soon as that source is done, in the order the sources finish. Every
    `report_every` seconds it also logs a progress line that names what is still running and how many gigabytes are
    on disk. It waits in steps of `poll` seconds, so an interrupt (Ctrl+C, or the notebook's stop button) reaches it
    at once. After an interrupt it sets the stop signal, waits for the downloads to end, keeps every finished file
    and raises KeyboardInterrupt. Running it again skips the finished downloads. A failed download stops the others
    in the same way and raises its error.

    Returns one record per source, in the order the sources were given. Each record holds the name, the path of the
    file, its size in bytes, the number of files extracted, the seconds spent and whether the file was "downloaded" or
    "already there".
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    unpack_path = Path(extract_root) if extract_root is not None else root_path
    unpack_path.mkdir(parents=True, exist_ok=True)
    local = set(local_names)
    source_list = list(sources)

    # The folder that each source's file is downloaded into.
    folder_of = {source.name: unpack_path if source.name in local else root_path for source in source_list}
    download_fn = download_fn or download

    # The stop signal is shared by all workers. The state of each source is shown in the progress lines.
    stop = threading.Event()
    state: dict[str, str] = {source.name: "queued" for source in source_list}
    results: dict[str, dict[str, Any]] = {}

    def one(source: Source) -> dict[str, Any]:
        """Downloads and, for an archive, unpacks one source, and returns its record for the final result."""
        started = time.monotonic()
        folder = folder_of[source.name]
        origin = "already there" if _matches_expected(folder / source.filename, source) else "downloaded"
        state[source.name] = "checking" if origin == "already there" else "downloading"
        path = download_fn(source, folder, retries=retries, stop=stop)
        extracted = 0
        if _is_archive_name(path.name):
            state[source.name] = "unpacking"
            extracted = len(safe_extract(path, unpack_path / source.name, stop=stop))
        state[source.name] = "done"
        return {"name": source.name, "path": str(path), "size": path.stat().st_size, "extracted": extracted,
                "seconds": round(time.monotonic() - started, 1), "origin": origin}

    # Start all sources in a pool of worker threads. The pool runs at most `workers` of them at the same time.
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    pending = {pool.submit(one, source) for source in source_list}
    started = last_report = time.monotonic()
    last_bytes = sum(_bytes_on_disk([x], folder_of[x.name]) for x in source_list)
    try:
        while pending:
            # Wait a short time for a source to finish, so that an interrupt is noticed quickly.
            done, pending = wait(pending, timeout=poll, return_when=FIRST_COMPLETED)

            # Log one line for each source that has just finished.
            for future in done:
                record = future.result()
                results[record["name"]] = record
                log(f"{record['name']:22s} {record['size'] / 1e9:6.2f} GB  extracted {record['extracted']:7d} files  "
                    f"{record['seconds']:6.1f} s  {record['origin']}")

            # Log a progress line when `report_every` seconds have passed since the last one.
            now = time.monotonic()
            if pending and now - last_report >= report_every:
                on_disk = sum(_bytes_on_disk([x], folder_of[x.name]) for x in source_list)
                rate = (on_disk - last_bytes) / max(now - last_report, 1e-9) / 1e6
                working = ", ".join(f"{name} ({how})" for name, how in state.items() if how not in ("queued", "done"))
                log(f"[{(now - started) / 60:4.1f} min] {len(results)} of {len(source_list)} done; still working on "
                    f"{working or 'queued downloads'}; {on_disk / 1e9:.2f} GB so far, {rate:.1f} MB/s")
                last_report, last_bytes = now, on_disk
    except KeyboardInterrupt:
        # The user interrupted. Tell the workers to stop, wait for them, and keep the files that are finished.
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        log(f"Stopped after {(time.monotonic() - started) / 60:.1f} min. {len(results)} of {len(source_list)} sources "
            "are complete and kept; run this cell again to continue with the rest.")
        raise
    except BaseException:
        # Any other error, for example a failed download or an unsafe archive, stops the other workers before it is
        # passed on.
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [results[source.name] for source in source_list]
