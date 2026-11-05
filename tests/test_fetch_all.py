"""Tests that facekit.datasets.fetch_all shows progress while it works and stops cleanly when it is interrupted.

The notebook's download cell uses fetch_all to download several sources at the same time. The cell must show progress
while it works. When it is interrupted, it must stop cleanly and must not hang the kernel (the process that runs the
notebook code). The stop signal is a threading event that every download checks while it works.

Every test runs offline. The downloads are stand-ins that sleep, write a partial file or wait for the stop signal.
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
    """Returns a Source with a made-up address, and with the file name "<name>.bin" unless a file name is given."""
    return Source(name=name, url=f"https://example.test/{name}", filename=filename or f"{name}.bin")


def _fake_download(delays, part_bytes=0):
    """Returns a download stand-in that waits for the delay of its source, writes the file and obeys the stop signal.

    The argument `delays` gives the waiting time in seconds for each source name. When `part_bytes` is not zero, the
    stand-in first writes a temporary .part file of that many bytes, which imitates a download in progress.
    """
    def download(source, dest, *, retries, stop):
        """Imitates facekit.datasets.download for one source."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if part_bytes:
            (dest / f"{source.filename}.part").write_bytes(b"x" * part_bytes)

        # Wait for the delay of this source in small steps, and give up at once if the stop signal is set.
        end = time.monotonic() + delays[source.name]
        while time.monotonic() < end:
            if stop.is_set():
                raise DownloadStopped(source.name)
            time.sleep(0.01)

        # The download is finished, so the temporary file is replaced by the real file.
        (dest / f"{source.filename}.part").unlink(missing_ok=True)
        path = dest / source.filename
        path.write_bytes(b"done")
        return path
    return download


def test_fetch_all_reports_each_source_as_it_finishes_and_returns_them_in_list_order(tmp_path):
    lines = []

    # The first source is slow and the second is fast, so the fast one finishes first.
    sources = [_source("slow"), _source("fast")]
    results = datasets.fetch_all(sources, tmp_path, workers=2, report_every=60, poll=0.01, log=lines.append,
                                 download_fn=_fake_download({"slow": 0.4, "fast": 0.0}))

    # Each finished source is reported in a log line that starts with its name and gives its size in GB.
    finished = [line.split()[0] for line in lines if " GB " in line]
    assert finished == ["fast", "slow"]

    # The returned records follow the order of the list that was passed in, not the order of finishing.
    assert [r["name"] for r in results] == ["slow", "fast"]
    assert all(r["size"] == 4 and r["extracted"] == 0 for r in results)


def test_fetch_all_prints_progress_while_downloads_are_still_running(tmp_path):
    lines = []

    # The report interval of 0.05 seconds is shorter than the 0.4 seconds that the download takes, and the stand-in
    # leaves a partial file of 2 million bytes, so a progress line with the size so far must appear.
    datasets.fetch_all([_source("big")], tmp_path, workers=1, report_every=0.05, poll=0.01, log=lines.append,
                       download_fn=_fake_download({"big": 0.4}, part_bytes=2_000_000))
    progress = [line for line in lines if "still working on" in line]
    assert progress, lines
    assert "big" in progress[0] and "GB so far" in progress[0]


def test_fetch_all_unpacks_zip_archives(tmp_path):
    """A downloaded zip file is unpacked into a folder named after its source, and the records count the files."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a/one.png", b"1")
        archive.writestr("a/two.png", b"2")

    def download(source, dest, *, retries, stop):
        """Writes the prepared zip file instead of downloading anything."""
        path = Path(dest) / source.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.getvalue())
        return path

    results = datasets.fetch_all([_source("pack", "pack.zip")], tmp_path, workers=1, poll=0.01, log=lambda _: None,
                                 download_fn=download)
    assert results[0]["extracted"] == 2
    assert (tmp_path / "pack" / "a" / "two.png").is_file()


def test_fetch_all_stops_cleanly_when_interrupted(tmp_path):
    """An interrupt reaches every running download through the stop signal, and fetch_all then re-raises it quickly."""
    lines = []
    saw_stop = []

    def download(source, dest, *, retries, stop):
        """Waits until the stop signal is set, notes that it saw the signal and stops."""
        while not stop.wait(0.01):
            pass
        saw_stop.append(source.name)
        raise DownloadStopped(source.name)

    # After 0.2 seconds, a timer sends the main thread the same interrupt as the stop button of a notebook.
    threading.Timer(0.2, _thread.interrupt_main).start()
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        datasets.fetch_all([_source("a"), _source("b")], tmp_path, workers=2, poll=0.01, log=lines.append,
                           download_fn=download)

    # The call must return within seconds, which shows that it did not hang, and both downloads must have stopped.
    assert time.monotonic() - started < 5
    assert sorted(saw_stop) == ["a", "b"]
    assert any("Stopped" in line for line in lines)


def test_fetch_all_stops_the_other_downloads_when_one_fails(tmp_path):
    """When one download fails, the others are told to stop and the error of the failed one is raised."""
    saw_stop = []

    def download(source, dest, *, retries, stop):
        """Fails at once for the source "bad" and waits for the stop signal for every other source."""
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
    """Imitates the response of urlopen, but never ends, so that only the stop signal can end the download."""

    def __init__(self, stop, set_after):
        """Stores the stop signal and the number of the read call at which the signal is set."""
        self.stop, self.set_after, self.reads = stop, set_after, 0

    def read(self, size):
        """Returns 16 bytes, and sets the stop signal when this is the read call that was chosen for it."""
        self.reads += 1
        if self.reads == self.set_after:
            self.stop.set()
        return b"y" * 16

    def __enter__(self):
        """Lets the response be used in a with statement."""
        return self

    def __exit__(self, *exc):
        """Leaves the with statement without hiding any error."""
        return False


def test_download_stops_between_chunks_and_leaves_no_partial_file(tmp_path):
    """A download stops as soon as the signal is set, and neither a finished file nor a .part file remains."""
    stop = threading.Event()

    # The signal is set during the third read, so the download must stop before it reads another chunk.
    with pytest.raises(DownloadStopped):
        datasets.download(_source("endless"), tmp_path, opener=lambda url, timeout: EndlessResponse(stop, 3), stop=stop)
    assert list(tmp_path.iterdir()) == []


def test_safe_extract_stops_on_the_signal_and_writes_no_marker(tmp_path):
    """Extraction with the signal already set raises at once and does not write the marker of a finished extraction."""
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(5):
            zf.writestr(f"img{i}.png", b"z")

    # The signal is set before the extraction starts.
    stop = threading.Event()
    stop.set()
    with pytest.raises(DownloadStopped):
        datasets.safe_extract(archive, tmp_path / "out", stop=stop)
    assert not (tmp_path / "out" / ".extracted-pack.zip").exists()


# Tests of the case where the downloaded files are kept in one folder (Google Drive) and the unpacked folders are
# kept in another (the machine's disk).


def _recording_download(calls, payload=b"done"):
    """Returns a download stand-in that writes the payload and adds the source name and folder to the list `calls`."""
    def download(source, dest, *, retries, stop):
        """Imitates facekit.datasets.download for one source and records where it was asked to save the file."""
        calls.append((source.name, Path(dest)))
        path = Path(dest) / source.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return path
    return download


def _zip_payload():
    """Returns the bytes of a zip archive that holds one small file."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a/one.png", b"1")
    return buffer.getvalue()


def test_fetch_all_keeps_archives_in_root_and_unpacks_into_extract_root(tmp_path):
    """The archive stays in the download folder, and its contents are unpacked into the separate extraction folder."""
    drive, disk, calls = tmp_path / "drive", tmp_path / "disk", []
    records = datasets.fetch_all([_source("pack", "pack.zip")], drive, extract_root=disk, workers=1, poll=0.01,
                                 log=lambda _: None, download_fn=_recording_download(calls, _zip_payload()))

    # The download goes to the drive folder, which holds the archive but no unpacked folder.
    assert calls == [("pack", drive)]
    assert (drive / "pack.zip").is_file() and not (drive / "pack").exists()

    # The unpacked files are in the disk folder, and the record points to the archive on the drive.
    assert (disk / "pack" / "a" / "one.png").is_file()
    assert records[0]["path"] == str(drive / "pack.zip") and records[0]["extracted"] == 1


def test_fetch_all_keeps_local_sources_off_the_archive_folder(tmp_path):
    """A source marked as local is downloaded to the local disk folder only, never to the synchronised drive folder."""
    drive, disk, calls = tmp_path / "drive", tmp_path / "disk", []
    records = datasets.fetch_all([_source("open"), _source("private")], drive, extract_root=disk,
                                 local_names={"private"}, workers=2, poll=0.01, log=lambda _: None,
                                 download_fn=_recording_download(calls))

    # The open source goes to the drive folder and the private source goes to the disk folder.
    assert sorted(calls) == [("open", drive), ("private", disk)]
    assert not (drive / "private.bin").exists()
    assert [Path(r["path"]).parent for r in records] == [drive, disk]


def test_fetch_all_says_which_files_were_already_there(tmp_path):
    """Each record says whether its file was already on disk or was downloaded, and the log mentions the former."""
    drive, lines = tmp_path / "drive", []
    drive.mkdir()

    # The file of the source "big" exists before the call, and the file of the source "new" does not.
    (drive / "big.bin").write_bytes(b"done")
    records = datasets.fetch_all([_source("big"), _source("new")], drive, workers=2, poll=0.01, log=lines.append,
                                 download_fn=_recording_download([]))
    assert [r["origin"] for r in records] == ["already there", "downloaded"]
    assert any("already there" in line for line in lines)
