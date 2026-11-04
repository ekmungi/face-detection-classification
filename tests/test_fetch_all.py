"""Tests for facekit.datasets.fetch_all and the stop signal: the notebook's download cell must show progress while it
works and must stop cleanly, without hanging the kernel, when it is interrupted.

Every test runs offline: downloads are stand-ins that sleep, write a partial file, or wait for the stop signal.
"""
import _thread
import io
import threading
import time
import zipfile
from pathlib import Path

import pytest

from facekit import datasets
from facekit.datasets import DownloadError, DownloadStopped, Source


def _source(name, filename=None):
    return Source(name=name, url=f"https://example.test/{name}", filename=filename or f"{name}.bin")


def _fake_download(delays, part_bytes=0):
    """A download stand-in: sleeps for its source's delay, writes the file, and honours the stop signal."""
    def download(source, dest, *, retries, stop):
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if part_bytes:
            (dest / f"{source.filename}.part").write_bytes(b"x" * part_bytes)
        end = time.monotonic() + delays[source.name]
        while time.monotonic() < end:
            if stop.is_set():
                raise DownloadStopped(source.name)
            time.sleep(0.01)
        (dest / f"{source.filename}.part").unlink(missing_ok=True)
        path = dest / source.filename
        path.write_bytes(b"done")
        return path
    return download


def test_fetch_all_reports_each_source_as_it_finishes_and_returns_them_in_list_order(tmp_path):
    lines = []
    sources = [_source("slow"), _source("fast")]
    results = datasets.fetch_all(sources, tmp_path, workers=2, report_every=60, poll=0.01, log=lines.append,
                                 download_fn=_fake_download({"slow": 0.4, "fast": 0.0}))
    finished = [line.split()[0] for line in lines if " GB " in line]
    assert finished == ["fast", "slow"]
    assert [r["name"] for r in results] == ["slow", "fast"]
    assert all(r["size"] == 4 and r["extracted"] == 0 for r in results)


def test_fetch_all_prints_progress_while_downloads_are_still_running(tmp_path):
    lines = []
    datasets.fetch_all([_source("big")], tmp_path, workers=1, report_every=0.05, poll=0.01, log=lines.append,
                       download_fn=_fake_download({"big": 0.4}, part_bytes=2_000_000))
    progress = [line for line in lines if "still working on" in line]
    assert progress, lines
    assert "big" in progress[0] and "GB so far" in progress[0]


def test_fetch_all_unpacks_zip_archives(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a/one.png", b"1")
        archive.writestr("a/two.png", b"2")

    def download(source, dest, *, retries, stop):
        path = Path(dest) / source.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.getvalue())
        return path

    results = datasets.fetch_all([_source("pack", "pack.zip")], tmp_path, workers=1, poll=0.01, log=lambda _: None,
                                 download_fn=download)
    assert results[0]["extracted"] == 2
    assert (tmp_path / "pack" / "a" / "two.png").is_file()


def test_fetch_all_stops_cleanly_when_interrupted(tmp_path):
    lines = []
    saw_stop = []

    def download(source, dest, *, retries, stop):
        while not stop.wait(0.01):
            pass
        saw_stop.append(source.name)
        raise DownloadStopped(source.name)

    threading.Timer(0.2, _thread.interrupt_main).start()
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        datasets.fetch_all([_source("a"), _source("b")], tmp_path, workers=2, poll=0.01, log=lines.append,
                           download_fn=download)
    assert time.monotonic() - started < 5
    assert sorted(saw_stop) == ["a", "b"]
    assert any("Stopped" in line for line in lines)


def test_fetch_all_stops_the_other_downloads_when_one_fails(tmp_path):
    saw_stop = []

    def download(source, dest, *, retries, stop):
        if source.name == "bad":
            raise DownloadError("server said no")
        while not stop.wait(0.01):
            pass
        saw_stop.append(source.name)
        raise DownloadStopped(source.name)

    with pytest.raises(DownloadError, match="server said no"):
        datasets.fetch_all([_source("good"), _source("bad")], tmp_path, workers=2, poll=0.01, log=lambda _: None,
                           download_fn=download)
    assert saw_stop == ["good"]


class EndlessResponse:
    """A urlopen-style response that never ends, so only the stop signal can end the download."""

    def __init__(self, stop, set_after):
        self.stop, self.set_after, self.reads = stop, set_after, 0

    def read(self, size):
        self.reads += 1
        if self.reads == self.set_after:
            self.stop.set()
        return b"y" * 16

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_stops_between_chunks_and_leaves_no_partial_file(tmp_path):
    stop = threading.Event()
    with pytest.raises(DownloadStopped):
        datasets.download(_source("endless"), tmp_path, opener=lambda url, timeout: EndlessResponse(stop, 3), stop=stop)
    assert list(tmp_path.iterdir()) == []


def test_safe_extract_stops_on_the_signal_and_writes_no_marker(tmp_path):
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(5):
            zf.writestr(f"img{i}.png", b"z")
    stop = threading.Event()
    stop.set()
    with pytest.raises(DownloadStopped):
        datasets.safe_extract(archive, tmp_path / "out", stop=stop)
    assert not (tmp_path / "out" / ".extracted-pack.zip").exists()


# --- Downloaded files in one folder (Google Drive), unpacked folders in another (the machine's disk) ----------

def _recording_download(calls, payload=b"done"):
    def download(source, dest, *, retries, stop):
        calls.append((source.name, Path(dest)))
        path = Path(dest) / source.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return path
    return download


def _zip_payload():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a/one.png", b"1")
    return buffer.getvalue()


def test_fetch_all_keeps_archives_in_root_and_unpacks_into_extract_root(tmp_path):
    drive, disk, calls = tmp_path / "drive", tmp_path / "disk", []
    records = datasets.fetch_all([_source("pack", "pack.zip")], drive, extract_root=disk, workers=1, poll=0.01,
                                 log=lambda _: None, download_fn=_recording_download(calls, _zip_payload()))
    assert calls == [("pack", drive)]
    assert (drive / "pack.zip").is_file() and not (drive / "pack").exists()
    assert (disk / "pack" / "a" / "one.png").is_file()
    assert records[0]["path"] == str(drive / "pack.zip") and records[0]["extracted"] == 1


def test_fetch_all_keeps_local_sources_off_the_archive_folder(tmp_path):
    drive, disk, calls = tmp_path / "drive", tmp_path / "disk", []
    records = datasets.fetch_all([_source("open"), _source("private")], drive, extract_root=disk,
                                 local_names={"private"}, workers=2, poll=0.01, log=lambda _: None,
                                 download_fn=_recording_download(calls))
    assert sorted(calls) == [("open", drive), ("private", disk)]
    assert not (drive / "private.bin").exists()
    assert [Path(r["path"]).parent for r in records] == [drive, disk]


def test_fetch_all_says_which_files_were_already_there(tmp_path):
    drive, lines = tmp_path / "drive", []
    drive.mkdir()
    (drive / "big.bin").write_bytes(b"done")
    records = datasets.fetch_all([_source("big"), _source("new")], drive, workers=2, poll=0.01, log=lines.append,
                                 download_fn=_recording_download([]))
    assert [r["origin"] for r in records] == ["already there", "downloaded"]
    assert any("already there" in line for line in lines)
