"""Tests for how data.load_mnist chooses between downloading in the session (fetch) and reading a kept copy (store)."""
import struct

import numpy as np
import pytest

import data
import datastore

pytest.importorskip("torchvision")

# Several tests build a dataset without a source note on purpose; the warning that loading one raises is tested in
# test_datastore_verify.py.
pytestmark = pytest.mark.filterwarnings("ignore:.*has no source note:UserWarning")


def write_idx(path, array):
    """Writes a uint8 array as an uncompressed IDX file, the form torchvision reads once MNIST is unpacked."""
    header = struct.pack(">HBB", 0, 0x08, array.ndim) + b"".join(struct.pack(">I", d) for d in array.shape)
    path.write_bytes(header + array.astype(np.uint8).tobytes())


def fake_mnist_fetch(folder):
    """Stands in for the torchvision download: writes four tiny MNIST files into <folder>/raw."""
    raw = folder / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    write_idx(raw / "train-images-idx3-ubyte", rng.integers(0, 255, (6, 28, 28)))
    write_idx(raw / "train-labels-idx1-ubyte", np.arange(6) % 10)
    write_idx(raw / "t10k-images-idx3-ubyte", rng.integers(0, 255, (4, 28, 28)))
    write_idx(raw / "t10k-labels-idx1-ubyte", np.arange(4) % 10)


def test_fetch_mode_loads_from_a_fresh_download_and_keeps_nothing_in_the_data_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "fetch_mnist", fake_mnist_fetch)
    train_full, val_source, test = data.load_mnist(tmp_path / "data", work_dir=tmp_path / "work", mode="fetch")
    assert (len(train_full), len(val_source), len(test)) == (6, 6, 4)
    image, label = train_full[3]
    assert tuple(image.shape) == (1, 28, 28) and label == 3
    assert not (tmp_path / "data").exists()


def test_the_default_mode_is_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "fetch_mnist", fake_mnist_fetch)
    assert len(data.load_mnist(tmp_path / "data", work_dir=tmp_path / "work")[0]) == 6


def test_store_mode_with_nothing_kept_raises_and_does_not_download(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(data, "fetch_mnist", lambda folder: called.append(folder))
    with pytest.raises(datastore.DatasetMissing):
        data.load_mnist(tmp_path / "data", work_dir=tmp_path / "work", mode="store")
    assert called == []


def test_store_mode_reads_a_kept_copy(tmp_path):
    fake_mnist_fetch(tmp_path / "data" / "MNIST")
    train_full, _, test = data.load_mnist(tmp_path / "data", work_dir=tmp_path / "work", mode="store")
    assert len(train_full) == 6 and len(test) == 4


def test_raw_images_come_back_as_pictures_for_display(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "fetch_mnist", fake_mnist_fetch)
    raw = data.load_mnist_raw(tmp_path / "data", work_dir=tmp_path / "work")
    image, label = raw[2]
    assert image.size == (28, 28) and label == 2 and len(raw.targets) == 6


def test_store_mode_checks_a_kept_copy_against_its_source_note(tmp_path):
    fake_mnist_fetch(tmp_path / "data" / "MNIST")
    datastore.write_manifest(tmp_path / "data" / "MNIST", source="s", licence="l", acquired_by="Tester")
    assert len(data.load_mnist(tmp_path / "data", work_dir=tmp_path / "work", mode="store")[0]) == 6
    (tmp_path / "data" / "MNIST" / "raw" / "train-labels-idx1-ubyte").write_bytes(b"tampered")
    with pytest.raises(datastore.ChecksumMismatch, match="train-labels-idx1-ubyte"):
        data.load_mnist(tmp_path / "data", work_dir=tmp_path / "work", mode="store")
    with pytest.raises(datastore.ChecksumMismatch, match="train-labels-idx1-ubyte"):
        data.load_mnist_raw(tmp_path / "data", work_dir=tmp_path / "work", mode="store")

