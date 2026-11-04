"""Tests for the two data modes: fetch (download in the session) and store (a kept copy in the artifacts data folder)."""
import zipfile

import pytest

import datastore

# Several tests build a dataset without a source note on purpose; the warning that loading one raises is tested in
# test_datastore_verify.py.
pytestmark = pytest.mark.filterwarnings("ignore:.*has no source note:UserWarning")


def fake_fetch(calls):
    """Returns a fetch function that records its calls and writes one file into the folder it is given."""
    def fetch(folder):
        calls.append(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "one.txt").write_text("downloaded")
    return fetch


def test_fetch_mode_downloads_into_the_work_folder_and_leaves_the_data_folder_alone(tmp_path):
    calls = []
    folder = datastore.resolve("toy", "fetch", tmp_path / "data", tmp_path / "work", fetch=fake_fetch(calls))
    assert folder == tmp_path / "work" / "toy" and (folder / "one.txt").read_text() == "downloaded"
    assert not (tmp_path / "data").exists()


def test_fetch_mode_downloads_only_once_per_work_folder(tmp_path):
    calls = []
    for _ in range(3):
        datastore.resolve("toy", "fetch", tmp_path / "data", tmp_path / "work", fetch=fake_fetch(calls))
    assert len(calls) == 1


def test_fetch_mode_without_a_fetch_function_is_refused(tmp_path):
    with pytest.raises(ValueError):
        datastore.resolve("toy", "fetch", tmp_path / "data", tmp_path / "work")


def test_a_failed_fetch_leaves_no_marker_so_the_next_call_tries_again(tmp_path):
    def broken(folder):
        raise ConnectionError("no network")

    with pytest.raises(ConnectionError):
        datastore.resolve("toy", "fetch", tmp_path / "data", tmp_path / "work", fetch=broken)
    calls = []
    datastore.resolve("toy", "fetch", tmp_path / "data", tmp_path / "work", fetch=fake_fetch(calls))
    assert len(calls) == 1


def test_store_mode_reads_the_kept_copy_and_never_calls_fetch(tmp_path):
    (tmp_path / "data" / "toy").mkdir(parents=True)
    (tmp_path / "data" / "toy" / "one.txt").write_text("kept")
    calls = []
    folder = datastore.resolve("toy", "store", tmp_path / "data", tmp_path / "work", fetch=fake_fetch(calls))
    assert (folder / "one.txt").read_text() == "kept" and calls == []


def test_store_mode_with_nothing_kept_raises_instead_of_downloading(tmp_path):
    calls = []
    with pytest.raises(datastore.DatasetMissing):
        datastore.resolve("toy", "store", tmp_path / "data", tmp_path / "work", fetch=fake_fetch(calls))
    assert calls == []


def test_an_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError):
        datastore.resolve("toy", "sometimes", tmp_path / "data", tmp_path / "work")


def test_keep_copy_copies_a_fetched_dataset_into_the_data_folder_with_a_manifest(tmp_path):
    datastore.resolve("toy", "fetch", tmp_path / "data", tmp_path / "work", fetch=fake_fetch([]))
    kept = datastore.keep_copy("toy", tmp_path / "work", tmp_path / "data", source="https://example.org/toy",
                               licence="CC0", acquired_by="Claude")
    assert kept == tmp_path / "data" / "toy" and (kept / "one.txt").read_text() == "downloaded"
    assert datastore.check_manifest(kept) == []
    assert not (kept / datastore.READY_MARKER).exists() and not (kept / datastore.FETCHED_MARKER).exists()


def test_keep_copy_can_store_a_zip_archive_with_its_own_manifest(tmp_path):
    datastore.resolve("toy", "fetch", tmp_path / "data", tmp_path / "work", fetch=fake_fetch([]))
    kept = datastore.keep_copy("toy", tmp_path / "work", tmp_path / "data", source="s", licence="l", acquired_by="me",
                               archive=True)
    assert kept == tmp_path / "data" / "toy.zip" and zipfile.is_zipfile(kept)
    assert (tmp_path / "data" / "toy.source.md").is_file() and datastore.check_manifest(kept) == []
    unpacked = datastore.resolve("toy", "store", tmp_path / "data", tmp_path / "other")
    assert (unpacked / "one.txt").read_text() == "downloaded"


def test_keep_copy_never_overwrites_a_kept_dataset(tmp_path):
    datastore.resolve("toy", "fetch", tmp_path / "data", tmp_path / "work", fetch=fake_fetch([]))
    datastore.keep_copy("toy", tmp_path / "work", tmp_path / "data", source="s", licence="l", acquired_by="me")
    with pytest.raises(FileExistsError):
        datastore.keep_copy("toy", tmp_path / "work", tmp_path / "data", source="s", licence="l", acquired_by="me")


def kept_thing(data_root, kind):
    """Creates something already kept under the name `toy` in data_root: a folder, a zip, a tar.gz or a plain file."""
    data_root.mkdir(parents=True, exist_ok=True)
    if kind == "folder":
        (data_root / "toy").mkdir()
        (data_root / "toy" / "old.txt").write_text("old")
    elif kind == "file":
        (data_root / "toy").write_text("old")
    else:
        (data_root / f"toy.{kind}").write_bytes(b"old")


@pytest.mark.parametrize("archive", [False, True])
@pytest.mark.parametrize("kind", ["folder", "zip", "tar.gz", "tgz", "tar", "file"])
def test_keep_copy_refuses_whatever_is_already_kept_under_the_name_and_changes_nothing(tmp_path, kind, archive):
    datastore.resolve("toy", "fetch", tmp_path / "data", tmp_path / "work", fetch=fake_fetch([]))
    kept_thing(tmp_path / "data", kind)
    before = sorted(p.name for p in (tmp_path / "data").rglob("*"))
    with pytest.raises(FileExistsError, match="kept copy of 'toy' already exists"):
        datastore.keep_copy("toy", tmp_path / "work", tmp_path / "data", source="s", licence="l", acquired_by="me",
                            archive=archive)
    assert sorted(p.name for p in (tmp_path / "data").rglob("*")) == before


def test_keep_copy_of_a_dataset_that_was_never_fetched_is_refused(tmp_path):
    with pytest.raises(datastore.DatasetMissing):
        datastore.keep_copy("toy", tmp_path / "work", tmp_path / "data", source="s", licence="l", acquired_by="me")


def test_an_archive_manifest_detects_a_changed_archive(tmp_path):
    (tmp_path / "toy.zip").write_bytes(b"PK-not-really")
    datastore.write_manifest(tmp_path / "toy.zip", source="s", licence="l", acquired_by="me")
    (tmp_path / "toy.zip").write_bytes(b"different")
    assert any("toy.zip" in p and "changed" in p for p in datastore.check_manifest(tmp_path / "toy.zip"))


def test_a_login_is_recorded_as_needed_but_never_stored(tmp_path):
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "one.txt").write_text("1")
    datastore.write_manifest(tmp_path / "toy", source="https://example.org/gated", licence="research only",
                             acquired_by="Tester", credentials_needed=True)
    text = (tmp_path / "toy" / "source.md").read_text()
    assert "credentials needed: yes" in text and "never stored" in text
