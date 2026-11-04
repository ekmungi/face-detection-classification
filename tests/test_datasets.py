"""Tests for facekit.datasets: download with size and checksum checks, safe extraction, image checks and the manifest.

Every test runs offline: the network is replaced by a fake opener that serves bytes from memory.
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
    """A urlopen-style response: readable in chunks, usable as a context manager, with a Content-Length header."""

    def __init__(self, payload, length=True):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))} if length else {}


class FakeOpener:
    """Serves payloads by URL; `failures` makes the first N calls raise a transient network error."""

    def __init__(self, payloads, failures=0):
        self.payloads = payloads
        self.failures = failures
        self.calls = []

    def __call__(self, url, timeout=None):
        self.calls.append(url)
        if self.failures > 0:
            self.failures -= 1
            raise urllib.error.URLError("transient")
        return FakeResponse(self.payloads[url])


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _zip_bytes(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _png_bytes(color=(200, 10, 10)):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


# --- download -----------------------------------------------------------------------------------------------


def test_download_writes_file_and_leaves_no_part_file(tmp_path):
    payload = b"x" * 5000
    source = Source(name="demo", url="https://example.org/demo.bin", filename="demo.bin", size=len(payload), sha256=_sha(payload))
    opener = FakeOpener({source.url: payload})
    path = datasets.download(source, tmp_path, opener=opener, chunk_size=1024)
    assert path == tmp_path / "demo.bin"
    assert path.read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))


def test_download_skips_existing_file_that_matches(tmp_path):
    payload = b"abc" * 100
    source = Source(name="demo", url="https://example.org/a.bin", filename="a.bin", size=len(payload), sha256=_sha(payload))
    (tmp_path / "a.bin").write_bytes(payload)
    opener = FakeOpener({})
    assert datasets.download(source, tmp_path, opener=opener) == tmp_path / "a.bin"
    assert opener.calls == []


def test_download_replaces_existing_file_with_wrong_size(tmp_path):
    payload = b"abc" * 100
    source = Source(name="demo", url="https://example.org/a.bin", filename="a.bin", size=len(payload))
    (tmp_path / "a.bin").write_bytes(b"truncated")
    opener = FakeOpener({source.url: payload})
    datasets.download(source, tmp_path, opener=opener)
    assert (tmp_path / "a.bin").read_bytes() == payload
    assert len(opener.calls) == 1


def test_download_rejects_checksum_mismatch_and_keeps_nothing(tmp_path):
    payload = b"payload"
    source = Source(name="demo", url="https://example.org/b.bin", filename="b.bin", sha256="0" * 64)
    with pytest.raises(DownloadError):
        datasets.download(source, tmp_path, opener=FakeOpener({source.url: payload}), retries=1)
    assert not (tmp_path / "b.bin").exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_size_mismatch(tmp_path):
    payload = b"payload"
    source = Source(name="demo", url="https://example.org/c.bin", filename="c.bin", size=999)
    with pytest.raises(DownloadError):
        datasets.download(source, tmp_path, opener=FakeOpener({source.url: payload}), retries=1)
    assert not (tmp_path / "c.bin").exists()


def test_download_retries_after_transient_error(tmp_path):
    payload = b"ok"
    source = Source(name="demo", url="https://example.org/d.bin", filename="d.bin")
    opener = FakeOpener({source.url: payload}, failures=2)
    path = datasets.download(source, tmp_path, opener=opener, retries=3, backoff=0)
    assert path.read_bytes() == payload
    assert len(opener.calls) == 3


def test_download_gives_up_after_retries(tmp_path):
    source = Source(name="demo", url="https://example.org/e.bin", filename="e.bin")
    opener = FakeOpener({source.url: b"never"}, failures=10)
    with pytest.raises(DownloadError):
        datasets.download(source, tmp_path, opener=opener, retries=2, backoff=0)
    assert len(opener.calls) == 2


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    assert datasets.sha256_file(path) == _sha(b"hello world")


# --- extraction ---------------------------------------------------------------------------------------------


def test_safe_extract_zip_and_marker_makes_second_call_a_no_op(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(_zip_bytes({"set/one.txt": b"1", "set/two/three.txt": b"3"}))
    out = tmp_path / "out"
    written = datasets.safe_extract(archive, out)
    assert sorted(p.relative_to(out).as_posix() for p in written) == ["set/one.txt", "set/two/three.txt"]
    assert (out / "set" / "two" / "three.txt").read_bytes() == b"3"
    assert datasets.safe_extract(archive, out) == []


def test_safe_extract_refuses_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip_bytes({"../evil.txt": b"x"}))
    out = tmp_path / "out"
    with pytest.raises(UnsafeArchiveError):
        datasets.safe_extract(archive, out)
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_refuses_absolute_member(tmp_path):
    archive = tmp_path / "abs.zip"
    archive.write_bytes(_zip_bytes({"/etc/evil.txt": b"x"}))
    with pytest.raises(UnsafeArchiveError):
        datasets.safe_extract(archive, tmp_path / "out")


def test_safe_extract_tar_gz(tmp_path):
    archive = tmp_path / "a.tar.gz"
    data = b"tar content"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("folder/file.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    out = tmp_path / "out"
    datasets.safe_extract(archive, out)
    assert (out / "folder" / "file.txt").read_bytes() == data


def test_safe_extract_rejects_unknown_format(tmp_path):
    archive = tmp_path / "a.rar"
    archive.write_bytes(b"not an archive")
    with pytest.raises(ValueError):
        datasets.safe_extract(archive, tmp_path / "out")


# --- image checks and manifest --------------------------------------------------------------------------


def test_count_images_is_recursive_and_case_insensitive(tmp_path):
    (tmp_path / "a").mkdir()
    for name in ("a/x.jpg", "a/y.PNG", "z.jpeg", "notes.txt", "a/w.webp"):
        (tmp_path / name).write_bytes(b"0")
    assert datasets.count_images(tmp_path) == 4


def test_verify_images_flags_corrupt_files(tmp_path):
    good = tmp_path / "good.png"
    good.write_bytes(_png_bytes())
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG broken")
    report = datasets.verify_images([good, bad])
    assert report["checked"] == 2
    assert [Path(p).name for p in report["bad"]] == ["bad.png"]


def test_verify_images_sample_limits_the_number_checked(tmp_path):
    paths = []
    for i in range(10):
        path = tmp_path / f"{i}.png"
        path.write_bytes(_png_bytes())
        paths.append(path)
    report = datasets.verify_images(paths, sample=3, seed=0)
    assert report["checked"] == 3
    assert report["bad"] == []


def test_write_manifest_records_sources_and_counts(tmp_path):
    source = Source(name="demo", url="https://example.org/demo.zip", filename="demo.zip", licence="CC BY 4.0",
                    licence_url="https://example.org/licence")
    path = tmp_path / "manifest.json"
    manifest = datasets.write_manifest(path, [source], {"demo": 12})
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["sources"][0]["name"] == "demo"
    assert on_disk["sources"][0]["licence"] == "CC BY 4.0"
    assert on_disk["counts"] == {"demo": 12}
    assert "created" in on_disk


# --- fetch: the whole path ------------------------------------------------------------------------------


def test_fetch_downloads_extracts_and_counts(tmp_path):
    payload = _zip_bytes({"imgs/a.png": _png_bytes(), "imgs/b.png": _png_bytes((0, 0, 255))})
    source = Source(name="tiny", url="https://example.org/tiny.zip", filename="tiny.zip", size=len(payload), sha256=_sha(payload))
    manifest = datasets.fetch([source], tmp_path, opener=FakeOpener({source.url: payload}))
    assert manifest["counts"] == {"tiny": 2}
    assert (tmp_path / "tiny" / "imgs" / "a.png").exists()
    assert (tmp_path / "manifest.json").exists()
    # A second fetch downloads nothing and extracts nothing.
    opener = FakeOpener({})
    assert datasets.fetch([source], tmp_path, opener=opener)["counts"] == {"tiny": 2}
    assert opener.calls == []


def test_count_images_skips_macos_resource_forks(tmp_path):
    (tmp_path / "001_03.jpg").write_bytes(b"x")
    (tmp_path / "._001_03.jpg").write_bytes(b"resource fork")
    assert datasets.count_images(tmp_path) == 1
