"""Tests for run-time verification in store mode: a kept dataset is checked against its source note, and a replaced archive is unpacked again."""
import hashlib
import warnings
import zipfile

import pytest

import datastore

# Several tests below build a dataset without a note on purpose; the warning that loading it raises is tested separately.
pytestmark = pytest.mark.filterwarnings("ignore:.*has no source note:UserWarning")


def make_zip(path, files):
    """Writes a zip archive at `path` holding {name: text} and returns its sha256."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keep_folder(data_root, files=None, note=True):
    """Creates a kept dataset folder <data_root>/toy with the given {name: text} files and, by default, its source note."""
    folder = data_root / "toy"
    for name, text in (files or {"one.txt": "1", "sub/two.txt": "2"}).items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)
    if note:
        datastore.write_manifest(folder, source="s", licence="l", acquired_by="me")
    return folder


def keep_archive(data_root, files=None, note=True):
    """Creates a kept dataset archive <data_root>/toy.zip and, by default, its source note; returns the archive path."""
    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / "toy.zip"
    make_zip(archive, files or {"one.txt": "1"})
    if note:
        datastore.write_manifest(archive, source="s", licence="l", acquired_by="me")
    return archive


def load_with_resolve(data_root, work):
    """Loads dataset `toy` through resolve() in store mode."""
    return datastore.resolve("toy", "store", data_root, work)


def load_with_ensure(data_root, work):
    """Loads dataset `toy` through ensure_local_copy()."""
    return datastore.ensure_local_copy("toy", data_root, work)


LOADERS = [load_with_resolve, load_with_ensure]


@pytest.mark.parametrize("load", LOADERS)
def test_a_kept_folder_whose_file_changed_is_refused(tmp_path, load):
    folder = keep_folder(tmp_path / "data")
    (folder / "one.txt").write_text("tampered")
    with pytest.raises(datastore.ChecksumMismatch, match="one.txt"):
        load(tmp_path / "data", tmp_path / "work")


@pytest.mark.parametrize("load", LOADERS)
def test_a_kept_folder_with_a_missing_file_is_refused(tmp_path, load):
    folder = keep_folder(tmp_path / "data")
    (folder / "sub" / "two.txt").unlink()
    with pytest.raises(datastore.ChecksumMismatch, match="two.txt"):
        load(tmp_path / "data", tmp_path / "work")


@pytest.mark.parametrize("load", LOADERS)
def test_a_kept_folder_with_a_file_the_note_does_not_list_is_refused(tmp_path, load):
    folder = keep_folder(tmp_path / "data")
    (folder / "extra.txt").write_text("added later")
    with pytest.raises(datastore.ChecksumMismatch, match="extra.txt"):
        load(tmp_path / "data", tmp_path / "work")


@pytest.mark.parametrize("load", LOADERS)
def test_a_kept_folder_that_matches_its_note_is_used_in_place(tmp_path, load):
    folder = keep_folder(tmp_path / "data")
    assert load(tmp_path / "data", tmp_path / "work") == folder


@pytest.mark.parametrize("load", LOADERS)
def test_a_kept_folder_without_a_note_still_loads_but_warns_that_it_was_not_verified(tmp_path, load):
    folder = keep_folder(tmp_path / "data", note=False)
    with pytest.warns(UserWarning, match=r"toy has no source note \(source.md\).*without a checksum check"):
        assert load(tmp_path / "data", tmp_path / "work") == folder


@pytest.mark.parametrize("load", LOADERS)
def test_a_kept_dataset_with_a_note_loads_without_any_warning(tmp_path, load):
    keep_folder(tmp_path / "data")
    keep_archive(tmp_path / "data2")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load(tmp_path / "data", tmp_path / "work")
        datastore.ensure_local_copy("toy", tmp_path / "data2", tmp_path / "work2")


@pytest.mark.parametrize("load", LOADERS)
def test_a_note_that_lists_no_files_is_refused(tmp_path, load):
    folder = keep_folder(tmp_path / "data", note=False)
    (folder / "source.md").write_text("# toy\n\n- source: s\n")
    with pytest.raises(datastore.ChecksumMismatch, match="no files"):
        load(tmp_path / "data", tmp_path / "work")


@pytest.mark.parametrize("load", LOADERS)
def test_a_kept_archive_that_changed_after_its_note_was_written_is_refused(tmp_path, load):
    archive = keep_archive(tmp_path / "data")
    make_zip(archive, {"one.txt": "replaced"})
    with pytest.raises(datastore.ChecksumMismatch, match="toy.zip"):
        load(tmp_path / "data", tmp_path / "work")
    assert not (tmp_path / "work" / "toy").exists()


@pytest.mark.parametrize("load", LOADERS)
def test_a_kept_archive_that_matches_its_note_is_unpacked(tmp_path, load):
    keep_archive(tmp_path / "data")
    assert (load(tmp_path / "data", tmp_path / "work") / "one.txt").read_text() == "1"


@pytest.mark.parametrize("load", LOADERS)
def test_a_kept_archive_without_a_note_still_loads_but_warns_that_it_was_not_verified(tmp_path, load):
    keep_archive(tmp_path / "data", note=False)
    with pytest.warns(UserWarning, match=r"toy.zip has no source note \(toy.source.md\)"):
        assert (load(tmp_path / "data", tmp_path / "work") / "one.txt").read_text() == "1"


def test_the_expected_checksum_is_still_honoured_when_there_is_no_note(tmp_path):
    archive = keep_archive(tmp_path / "data", note=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # the archive was verified by the expected checksum, so no warning
        assert datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work", expected_sha256=digest).is_dir()
    with pytest.raises(datastore.ChecksumMismatch):
        datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work2", expected_sha256="0" * 64)


def test_the_unpacked_marker_records_the_archive_checksum(tmp_path):
    archive = keep_archive(tmp_path / "data", note=False)
    folder = datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work")
    assert hashlib.sha256(archive.read_bytes()).hexdigest() in (folder / datastore.READY_MARKER).read_text()


def test_a_replaced_archive_is_unpacked_again_into_a_clean_folder(tmp_path):
    archive = keep_archive(tmp_path / "data", {"old.txt": "old", "one.txt": "1"}, note=False)
    first = datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work")
    assert (first / "old.txt").exists()
    make_zip(archive, {"one.txt": "new"})
    again = datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work")
    assert (again / "one.txt").read_text() == "new"
    assert not (again / "old.txt").exists()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() in (again / datastore.READY_MARKER).read_text()


def test_an_unpacked_marker_without_a_checksum_is_treated_as_stale(tmp_path):
    keep_archive(tmp_path / "data", note=False)
    stale = tmp_path / "work" / "toy"
    stale.mkdir(parents=True)
    (stale / "leftover.txt").write_text("from an older unpack")
    (stale / datastore.READY_MARKER).write_text("unpacked\n")
    folder = datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work")
    assert (folder / "one.txt").read_text() == "1" and not (folder / "leftover.txt").exists()


def test_an_unchanged_archive_is_not_unpacked_again(tmp_path):
    keep_archive(tmp_path / "data", note=False)
    first = datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work")
    (first / "one.txt").write_text("changed locally")
    again = datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work")
    assert (again / "one.txt").read_text() == "changed locally"


def test_a_failed_unpack_leaves_no_marker_so_the_next_call_tries_again(tmp_path):
    archive = keep_archive(tmp_path / "data", note=False)
    archive.write_bytes(b"not an archive at all")
    with pytest.raises(Exception):
        datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work")
    assert not (tmp_path / "work" / "toy" / datastore.READY_MARKER).exists()
