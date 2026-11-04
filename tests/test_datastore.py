"""Tests for datastore.py: datasets live in the artifacts data folder, archives are unpacked on the machine's disk, nothing is downloaded."""
import hashlib
import zipfile

import pytest

import datastore

# Several tests build a dataset without a source note on purpose; the warning that loading one raises is tested in
# test_datastore_verify.py.
pytestmark = pytest.mark.filterwarnings("ignore:.*has no source note:UserWarning")


def make_zip(path, files):
    """Writes a zip archive at `path` holding {name: text} and returns its sha256."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_zip_archive_is_unpacked_into_the_work_folder(tmp_path):
    data_root, work = tmp_path / "data", tmp_path / "work"
    data_root.mkdir()
    make_zip(data_root / "toy.zip", {"a/one.txt": "1", "two.txt": "2"})
    folder = datastore.ensure_local_copy("toy", data_root, work)
    assert folder == work / "toy"
    assert (folder / "a" / "one.txt").read_text() == "1"
    assert (folder / "two.txt").read_text() == "2"


def test_a_second_call_reuses_the_unpacked_copy(tmp_path):
    data_root, work = tmp_path / "data", tmp_path / "work"
    data_root.mkdir()
    make_zip(data_root / "toy.zip", {"one.txt": "1"})
    first = datastore.ensure_local_copy("toy", data_root, work)
    (first / "one.txt").write_text("changed locally")
    again = datastore.ensure_local_copy("toy", data_root, work)
    assert (again / "one.txt").read_text() == "changed locally"


def test_an_unpacked_folder_in_the_data_root_is_used_in_place(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "toy").mkdir(parents=True)
    (data_root / "toy" / "one.txt").write_text("1")
    assert datastore.ensure_local_copy("toy", data_root, tmp_path / "work") == data_root / "toy"


def test_a_missing_dataset_says_how_to_acquire_it_and_never_downloads(tmp_path):
    with pytest.raises(datastore.DatasetMissing) as error:
        datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work")
    assert "store mode" in str(error.value) and "never downloaded" in str(error.value)


def test_an_archive_with_the_wrong_checksum_is_refused(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    make_zip(data_root / "toy.zip", {"one.txt": "1"})
    with pytest.raises(datastore.ChecksumMismatch):
        datastore.ensure_local_copy("toy", data_root, tmp_path / "work", expected_sha256="0" * 64)


def test_an_archive_with_the_right_checksum_is_accepted(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    digest = make_zip(data_root / "toy.zip", {"one.txt": "1"})
    assert datastore.ensure_local_copy("toy", data_root, tmp_path / "work", expected_sha256=digest).is_dir()


def test_an_archive_member_that_escapes_the_folder_is_refused(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    make_zip(data_root / "toy.zip", {"../evil.txt": "x"})
    with pytest.raises(ValueError):
        datastore.ensure_local_copy("toy", data_root, tmp_path / "work")
    assert not (tmp_path / "evil.txt").exists()


def test_the_manifest_records_every_file_with_its_checksum(tmp_path):
    folder = tmp_path / "toy"
    (folder / "sub").mkdir(parents=True)
    (folder / "one.txt").write_text("1")
    (folder / "sub" / "two.txt").write_text("2")
    datastore.write_manifest(folder, source="https://example.org/toy", licence="CC0", acquired_by="Claude")
    text = (folder / "source.md").read_text()
    assert "https://example.org/toy" in text and "CC0" in text and "Claude" in text
    assert "one.txt" in text and "sub/two.txt" in text
    assert datastore.check_manifest(folder) == []


def test_a_changed_or_missing_file_is_reported_by_the_manifest_check(tmp_path):
    folder = tmp_path / "toy"
    folder.mkdir()
    (folder / "one.txt").write_text("1")
    (folder / "two.txt").write_text("2")
    datastore.write_manifest(folder, source="s", licence="l", acquired_by="me")
    (folder / "one.txt").write_text("tampered")
    (folder / "two.txt").unlink()
    problems = datastore.check_manifest(folder)
    assert any("one.txt" in p and "changed" in p for p in problems)
    assert any("two.txt" in p and "missing" in p for p in problems)


def test_the_manifest_check_without_a_manifest_says_so(tmp_path):
    (tmp_path / "toy").mkdir()
    assert datastore.check_manifest(tmp_path / "toy") == ["source.md is missing"]
