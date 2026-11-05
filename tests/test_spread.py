"""Tests for the evenly spread subsets that the smoke and medium modes take from WIDER FACE.

WIDER FACE is a public dataset of photos with labelled faces. Both of its lists are sorted by the kind of event: the
first 3,000 training photos cover 13 of the 61 kinds and the first 500 validation photos cover 6. A capped subset is
therefore taken at even steps through the whole list, so that it covers every kind of event.
"""
import cv2
import numpy as np

from facekit import prepare


def test_evenly_spaced_indices_reach_from_the_start_to_the_end_of_the_list():
    indices = prepare.evenly_spaced_indices(12880, 3000)
    assert len(indices) == 3000 and len(set(indices)) == 3000

    # The first index is the start of the list and the last one is within one step of the end.
    assert indices[0] == 0 and indices[-1] >= 12880 - 12880 / 3000 - 1
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

    # The two replaced functions are the only ones that read the file, so the real parquet file is not needed.
    monkeypatch.setattr(prepare, "iter_parquet", lambda paths, columns, batch_size=64: iter(table))
    monkeypatch.setattr(prepare, "parquet_row_count", lambda path: rows)


def test_spread_validation_records_come_from_the_whole_file(monkeypatch):
    fake_validation_file(monkeypatch)

    # Three of ten photos at even steps are the photos at positions 0, 3 and 6.
    picked = [r["path"] for r in prepare.wider_val_records("val.parquet", max_items=3, spread=True)]
    assert picked == ["0.jpg", "3.jpg", "6.jpg"]


def test_validation_records_without_spread_are_the_first_ones(monkeypatch):
    fake_validation_file(monkeypatch)
    picked = [r["path"] for r in prepare.wider_val_records("val.parquet", max_items=3)]
    assert picked == ["0.jpg", "1.jpg", "2.jpg"]
