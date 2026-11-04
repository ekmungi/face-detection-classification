"""Tests for the archive unpacker in datastore.py: what it refuses (escaping names, links, special files) and what it accepts."""
import io
import stat
import tarfile
import zipfile

import pytest

import datastore

# Several tests build a dataset without a source note on purpose; the warning that loading one raises is tested in
# test_datastore_verify.py.
pytestmark = pytest.mark.filterwarnings("ignore:.*has no source note:UserWarning")


def make_tar(path, members, mode="w"):
    """Writes a tar archive from (name, type, payload, linkname) tuples; nothing real is created on disk for links or devices."""
    with tarfile.open(path, mode) as bundle:
        for name, kind, payload, linkname in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = linkname
            if kind == tarfile.REGTYPE:
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
            else:
                bundle.addfile(info)


def unpack(tmp_path):
    """Unpacks the archive for dataset `toy` found in <tmp_path>/data into <tmp_path>/work and returns the folder."""
    return datastore.ensure_local_copy("toy", tmp_path / "data", tmp_path / "work")


def tar_in(tmp_path, members, name="toy.tar", mode="w"):
    """Creates <tmp_path>/data/<name> from members."""
    (tmp_path / "data").mkdir(exist_ok=True)
    make_tar(tmp_path / "data" / name, members, mode)


GOOD = ("ok.txt", tarfile.REGTYPE, b"fine", "")


@pytest.mark.parametrize("name, mode", [("toy.tar", "w"), ("toy.tar.gz", "w:gz"), ("toy.tgz", "w:gz")])
def test_a_tar_archive_of_files_and_folders_is_unpacked(tmp_path, name, mode):
    tar_in(tmp_path, [("sub", tarfile.DIRTYPE, b"", ""), ("sub/one.txt", tarfile.REGTYPE, b"1", ""), GOOD], name, mode)
    folder = unpack(tmp_path)
    assert (folder / "sub" / "one.txt").read_bytes() == b"1" and (folder / "ok.txt").read_bytes() == b"fine"


def test_a_tar_member_that_escapes_the_folder_is_refused_and_nothing_is_unpacked(tmp_path):
    tar_in(tmp_path, [GOOD, ("../evil.txt", tarfile.REGTYPE, b"x", "")])
    with pytest.raises(ValueError, match="outside"):
        unpack(tmp_path)
    assert not (tmp_path / "work" / "evil.txt").exists()
    assert not (tmp_path / "work" / "toy" / "ok.txt").exists()


def test_a_tar_member_with_an_absolute_name_is_refused(tmp_path):
    target = (tmp_path / "elsewhere" / "evil.txt").resolve().as_posix()
    tar_in(tmp_path, [(target, tarfile.REGTYPE, b"x", "")])
    with pytest.raises(ValueError, match="outside"):
        unpack(tmp_path)
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_a_tar_link_to_a_file_inside_the_archive_is_refused(tmp_path, kind):
    tar_in(tmp_path, [GOOD, ("alias.txt", kind, b"", "ok.txt")])
    with pytest.raises(ValueError, match="link"):
        unpack(tmp_path)
    assert not (tmp_path / "work" / "toy" / "alias.txt").exists()


@pytest.mark.parametrize("kind", [tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE])
def test_a_tar_fifo_or_device_is_refused(tmp_path, kind):
    tar_in(tmp_path, [GOOD, ("special", kind, b"", "")])
    with pytest.raises(ValueError, match="regular file"):
        unpack(tmp_path)
    assert not (tmp_path / "work" / "toy" / "ok.txt").exists()


def test_a_zip_entry_that_escapes_the_folder_is_refused(tmp_path):
    (tmp_path / "data").mkdir()
    with zipfile.ZipFile(tmp_path / "data" / "toy.zip", "w") as bundle:
        bundle.writestr("ok.txt", "fine")
        bundle.writestr("a/../../evil.txt", "x")
    with pytest.raises(ValueError, match="outside"):
        unpack(tmp_path)
    assert not (tmp_path / "work" / "evil.txt").exists() and not (tmp_path / "work" / "toy" / "ok.txt").exists()


def test_a_zip_symbolic_link_entry_is_refused(tmp_path):
    (tmp_path / "data").mkdir()
    link = zipfile.ZipInfo("alias.txt")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(tmp_path / "data" / "toy.zip", "w") as bundle:
        bundle.writestr("ok.txt", "fine")
        bundle.writestr(link, "ok.txt")
    with pytest.raises(ValueError, match="link"):
        unpack(tmp_path)
    assert not (tmp_path / "work" / "toy" / "alias.txt").exists()


@pytest.mark.parametrize("file_type", [stat.S_IFIFO, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFSOCK])
def test_a_zip_entry_recorded_as_a_fifo_or_device_is_refused(tmp_path, file_type):
    (tmp_path / "data").mkdir()
    special = zipfile.ZipInfo("special")
    special.external_attr = (file_type | 0o644) << 16
    with zipfile.ZipFile(tmp_path / "data" / "toy.zip", "w") as bundle:
        bundle.writestr("ok.txt", "fine")
        bundle.writestr(special, "")
    with pytest.raises(ValueError, match="regular file"):
        unpack(tmp_path)
    assert not (tmp_path / "work" / "toy").exists()


def test_a_zip_entry_that_is_a_regular_file_or_folder_with_unix_modes_is_accepted(tmp_path):
    (tmp_path / "data").mkdir()
    folder, regular = zipfile.ZipInfo("sub/"), zipfile.ZipInfo("sub/one.txt")
    for info, mode in ((folder, stat.S_IFDIR | 0o755), (regular, stat.S_IFREG | 0o644)):
        info.create_system = 3
        info.external_attr = mode << 16
    with zipfile.ZipFile(tmp_path / "data" / "toy.zip", "w") as bundle:
        bundle.writestr(folder, "")
        bundle.writestr(regular, "1")
    assert (unpack(tmp_path) / "sub" / "one.txt").read_text() == "1"


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "toy ", " toy", "toy\t", "a:b", "C:", "a\x00b"])
def test_a_dataset_name_that_is_not_one_plain_folder_name_is_refused_before_anything_is_deleted(tmp_path, name):
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "keep.txt").write_text("precious")
    with pytest.raises(ValueError, match="plain folder name"):
        datastore.ensure_local_copy(name, tmp_path / "data", tmp_path / "work")
    assert (tmp_path / "work" / "keep.txt").read_text() == "precious"
