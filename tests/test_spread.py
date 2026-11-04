"""Tests for the evenly spread subsets that the smoke and medium modes take from WIDER FACE.

Both WIDER lists are sorted by the kind of event: the first 3,000 training photos cover 13 of the 61 kinds and the
first 500 validation photos cover 6. A capped subset is therefore taken at even steps through the whole list.
"""
import cv2
import numpy as np

from facekit import prepare


def test_evenly_spaced_indices_reach_from_the_start_to_the_end_of_the_list():
    indices = prepare.evenly_spaced_indices(12880, 3000)
    assert len(indices) == 3000 and len(set(indices)) == 3000
    assert indices[0] == 0 and indices[-1] >= 12880 - 12880 / 3000 - 1   # the last one is within a step of the end
    assert indices == sorted(indices)


def test_without_a_cap_or_with_a_large_cap_every_item_is_kept():
    assert prepare.evenly_spaced_indices(5, None) == [0, 1, 2, 3, 4]
    assert prepare.evenly_spaced_indices(5, 10) == [0, 1, 2, 3, 4]


def test_evenly_spaced_keeps_the_order_of_the_items():
    assert prepare.evenly_spaced(list("abcdefghij"), 5) == ["a", "c", "e", "g", "i"]
    assert prepare.evenly_spaced(list("abc"), None) == ["a", "b", "c"]


def fake_validation_file(monkeypatch, rows=10):
    """Stands in for the WIDER validation parquet: ten one-face photos named 0.jpg to 9.jpg."""
    png = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
    table = [{"image": {"bytes": png}, "image_path": f"{n}.jpg", "bboxes": [[1, 1, 4, 4]],
              "easy_keep_indices": [0], "medium_keep_indices": [0], "hard_keep_indices": [0]} for n in range(rows)]
    monkeypatch.setattr(prepare, "iter_parquet", lambda paths, columns, batch_size=64: iter(table))
    monkeypatch.setattr(prepare, "parquet_row_count", lambda path: rows)


def test_spread_validation_records_come_from_the_whole_file(monkeypatch):
    fake_validation_file(monkeypatch)
    picked = [r["path"] for r in prepare.wider_val_records("val.parquet", max_items=3, spread=True)]
    assert picked == ["0.jpg", "3.jpg", "6.jpg"]


def test_validation_records_without_spread_are_the_first_ones(monkeypatch):
    fake_validation_file(monkeypatch)
    picked = [r["path"] for r in prepare.wider_val_records("val.parquet", max_items=3)]
    assert picked == ["0.jpg", "1.jpg", "2.jpg"]
