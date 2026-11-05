"""Tests facekit.datasets: download with size and checksum checks, safe extraction, image checks and the manifest.

Every test runs offline. The network is replaced by a fake opener that serves bytes from memory. A source is the
description of one downloadable file, with its address, expected size and expected SHA-256 checksum (a fingerprint of
the file's bytes). A manifest is the JSON file that records which sources were fetched and how many images each gave.
"""
import hashlib
import io
import json
import tarfile
import urllib.error
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from facekit import datasets
from facekit.datasets import DownloadError, Source, UnsafeArchiveError


class FakeResponse(io.BytesIO):
    """Imitates the response of urllib's urlopen.

    It can be read in chunks, it can be used as a context manager, and it carries a Content-Length header unless the
    length is switched off.
    """

    def __init__(self, payload, length=True):
        """Stores the payload and adds a Content-Length header unless `length` is false."""
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))} if length else {}


class FakeOpener:
    """Imitates urlopen by serving payloads from memory, with the address as the key.

    The argument `failures` makes the first N calls raise a transient network error before any call succeeds.
    """

    def __init__(self, payloads, failures=0):
        """Stores the payloads and the number of calls that must fail, and starts an empty list of called addresses."""
        self.payloads = payloads
        self.failures = failures
        self.calls = []

    def __call__(self, url, timeout=None):
        """Records the address, then either raises a transient error or returns the payload as a response."""
        self.calls.append(url)
        if self.failures > 0:
            self.failures -= 1
            raise urllib.error.URLError("transient")
        return FakeResponse(self.payloads[url])


def _sha(data):
    """Returns the SHA-256 checksum of the bytes as a hexadecimal text."""
    return hashlib.sha256(data).hexdigest()


def _zip_bytes(members):
    """Returns the bytes of a zip archive that holds the given members, a dictionary of file names and contents."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _png_bytes(color=(200, 10, 10)):
    """Returns the bytes of an 8 x 8 pixel PNG image in one flat colour."""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


# Tests of the download.


def test_download_writes_file_and_leaves_no_part_file(tmp_path):
    """A download writes the finished file and removes the temporary .part file in which it is first saved."""
    payload = b"x" * 5000
    source = Source(name="demo", url="https://example.org/demo.bin", filename="demo.bin", size=len(payload),
                    sha256=_sha(payload))
    opener = FakeOpener({source.url: payload})

    # The small chunk size makes the download read the payload in several pieces.
    path = datasets.download(source, tmp_path, opener=opener, chunk_size=1024)
    assert path == tmp_path / "demo.bin"
    assert path.read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))


def test_download_skips_existing_file_that_matches(tmp_path):
    """A file that is already on disk with the right size and checksum is not downloaded again."""
    payload = b"abc" * 100
    source = Source(name="demo", url="https://example.org/a.bin", filename="a.bin", size=len(payload),
                    sha256=_sha(payload))
    (tmp_path / "a.bin").write_bytes(payload)

    # The fake opener has no payloads, so any attempt to download would fail.
    opener = FakeOpener({})
    assert datasets.download(source, tmp_path, opener=opener) == tmp_path / "a.bin"
    assert opener.calls == []


def test_download_replaces_existing_file_with_wrong_size(tmp_path):
    """A file on disk whose size is wrong, such as a truncated earlier download, is replaced by a fresh download."""
    payload = b"abc" * 100
    source = Source(name="demo", url="https://example.org/a.bin", filename="a.bin", size=len(payload))
    (tmp_path / "a.bin").write_bytes(b"truncated")
    opener = FakeOpener({source.url: payload})
    datasets.download(source, tmp_path, opener=opener)
    assert (tmp_path / "a.bin").read_bytes() == payload
    assert len(opener.calls) == 1


def test_download_rejects_checksum_mismatch_and_keeps_nothing(tmp_path):
    payload = b"payload"

    # The expected checksum is all zeros, so it cannot match the payload.
    source = Source(name="demo", url="https://example.org/b.bin", filename="b.bin", sha256="0" * 64)
    with pytest.raises(DownloadError):
        datasets.download(source, tmp_path, opener=FakeOpener({source.url: payload}), retries=1)

    # Neither the finished file nor the temporary .part file may remain.
    assert not (tmp_path / "b.bin").exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_size_mismatch(tmp_path):
    payload = b"payload"

    # The expected size of 999 bytes differs from the real size of the payload.
    source = Source(name="demo", url="https://example.org/c.bin", filename="c.bin", size=999)
    with pytest.raises(DownloadError):
        datasets.download(source, tmp_path, opener=FakeOpener({source.url: payload}), retries=1)
    assert not (tmp_path / "c.bin").exists()


def test_download_retries_after_transient_error(tmp_path):
    """A download that fails twice with a network error succeeds on the third of three allowed attempts."""
    payload = b"ok"
    source = Source(name="demo", url="https://example.org/d.bin", filename="d.bin")
    opener = FakeOpener({source.url: payload}, failures=2)
    path = datasets.download(source, tmp_path, opener=opener, retries=3, backoff=0)
    assert path.read_bytes() == payload
    assert len(opener.calls) == 3


def test_download_gives_up_after_retries(tmp_path):
    """A download that keeps failing stops after the allowed number of attempts and raises an error."""
    source = Source(name="demo", url="https://example.org/e.bin", filename="e.bin")
    opener = FakeOpener({source.url: b"never"}, failures=10)
    with pytest.raises(DownloadError):
        datasets.download(source, tmp_path, opener=opener, retries=2, backoff=0)
    assert len(opener.calls) == 2


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    assert datasets.sha256_file(path) == _sha(b"hello world")


# Tests of the extraction of archives.


def test_safe_extract_zip_and_marker_makes_second_call_a_no_op(tmp_path):
    """Extracting a zip file lists the files it wrote, and extracting it again does nothing and lists no files."""
    archive = tmp_path / "a.zip"
    archive.write_bytes(_zip_bytes({"set/one.txt": b"1", "set/two/three.txt": b"3"}))
    out = tmp_path / "out"
    written = datasets.safe_extract(archive, out)
    assert sorted(p.relative_to(out).as_posix() for p in written) == ["set/one.txt", "set/two/three.txt"]
    assert (out / "set" / "two" / "three.txt").read_bytes() == b"3"

    # The first extraction leaves a marker file behind, which makes the second extraction skip the archive.
    assert datasets.safe_extract(archive, out) == []


def test_safe_extract_refuses_path_traversal(tmp_path):
    """A member whose name climbs out of the target folder with ".." is refused, and nothing is written outside."""
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip_bytes({"../evil.txt": b"x"}))
    out = tmp_path / "out"
    with pytest.raises(UnsafeArchiveError):
        datasets.safe_extract(archive, out)
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_refuses_absolute_member(tmp_path):
    """A member whose name is an absolute path, which would be written outside the target folder, is refused."""
    archive = tmp_path / "abs.zip"
    archive.write_bytes(_zip_bytes({"/etc/evil.txt": b"x"}))
    with pytest.raises(UnsafeArchiveError):
        datasets.safe_extract(archive, tmp_path / "out")


def test_safe_extract_tar_gz(tmp_path):
    """A gzip-compressed tar archive is extracted in the same way as a zip file."""
    archive = tmp_path / "a.tar.gz"
    data = b"tar content"

    # The archive is built in code with one file inside a folder.
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("folder/file.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    out = tmp_path / "out"
    datasets.safe_extract(archive, out)
    assert (out / "folder" / "file.txt").read_bytes() == data


def test_safe_extract_rejects_unknown_format(tmp_path):
    """A file that is neither a zip nor a tar archive, here named like a RAR file, raises a ValueError."""
    archive = tmp_path / "a.rar"
    archive.write_bytes(b"not an archive")
    with pytest.raises(ValueError):
        datasets.safe_extract(archive, tmp_path / "out")


# Tests of the image checks and of the manifest.


def test_count_images_is_recursive_and_case_insensitive(tmp_path):
    """Images are counted in sub-folders too, whatever the case of the extension, and other files are ignored."""
    (tmp_path / "a").mkdir()

    # Four image names, with the extensions jpg, PNG, jpeg and webp, and one text file that is not counted.
    for name in ("a/x.jpg", "a/y.PNG", "z.jpeg", "notes.txt", "a/w.webp"):
        (tmp_path / name).write_bytes(b"0")
    assert datasets.count_images(tmp_path) == 4


def test_verify_images_flags_corrupt_files(tmp_path):
    """Checking images lists the files that cannot be decoded and counts every file that was checked."""
    good = tmp_path / "good.png"
    good.write_bytes(_png_bytes())
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG broken")
    report = datasets.verify_images([good, bad])
    assert report["checked"] == 2
    assert [Path(p).name for p in report["bad"]] == ["bad.png"]


def test_verify_images_sample_limits_the_number_checked(tmp_path):
    """Asking for a sample of 3 checks only 3 of the 10 images, and none of them is reported as bad."""
    paths = []
    for i in range(10):
        path = tmp_path / f"{i}.png"
        path.write_bytes(_png_bytes())
        paths.append(path)
    report = datasets.verify_images(paths, sample=3, seed=0)
    assert report["checked"] == 3
    assert report["bad"] == []


def test_write_manifest_records_sources_and_counts(tmp_path):
    """The manifest file holds each source with its licence, the image counts and the time of creation."""
    source = Source(name="demo", url="https://example.org/demo.zip", filename="demo.zip", licence="CC BY 4.0",
                    licence_url="https://example.org/licence")
    path = tmp_path / "manifest.json"
    manifest = datasets.write_manifest(path, [source], {"demo": 12})

    # The manifest that the function returns is the same as the one that was saved to disk.
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["sources"][0]["name"] == "demo"
    assert on_disk["sources"][0]["licence"] == "CC BY 4.0"
    assert on_disk["counts"] == {"demo": 12}
    assert "created" in on_disk


# Tests of the whole path from download to manifest.


def test_fetch_downloads_extracts_and_counts(tmp_path):
    """Fetching downloads the archive, extracts it, counts the images and writes the manifest."""
    # The archive holds two small images.
    payload = _zip_bytes({"imgs/a.png": _png_bytes(), "imgs/b.png": _png_bytes((0, 0, 255))})
    source = Source(name="tiny", url="https://example.org/tiny.zip", filename="tiny.zip", size=len(payload),
                    sha256=_sha(payload))
    manifest = datasets.fetch([source], tmp_path, opener=FakeOpener({source.url: payload}))
    assert manifest["counts"] == {"tiny": 2}
    assert (tmp_path / "tiny" / "imgs" / "a.png").exists()
    assert (tmp_path / "manifest.json").exists()

    # A second fetch downloads nothing and extracts nothing, but it still reports the same counts.
    opener = FakeOpener({})
    assert datasets.fetch([source], tmp_path, opener=opener)["counts"] == {"tiny": 2}
    assert opener.calls == []


def test_count_images_skips_macos_resource_forks(tmp_path):
    """The hidden files that macOS adds to archives, whose names start with "._", are not counted as images."""
    (tmp_path / "001_03.jpg").write_bytes(b"x")
    (tmp_path / "._001_03.jpg").write_bytes(b"resource fork")
    assert datasets.count_images(tmp_path) == 1
