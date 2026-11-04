"""Offline tests for facekit.det_data: annotation parsers, augmentation and the training dataset."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from facekit.det_data import (
    WiderFaceTrain,
    detection_collate,
    hflip,
    parse_retinaface_labels,
    parse_wider_official,
    photometric_distort,
    random_square_crop,
)

RETINAFACE_TRAIN_SAMPLE = """\
# 0--Parade/0_Parade_marchingband_1_849.jpg
449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.830 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82
# 0--Parade/0_Parade_Parade_0_904.jpg
361 98 263 339 424.143 251.656 0.0 547.134 232.571 0.0 494.121 325.875 0.0 453.83 339.86 0.0 561.978 320.857 0.0 0.89
78 221 7 8 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
50 60 0 10 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
"""

RETINAFACE_VAL_SAMPLE = """\
# 0--Parade/0_Parade_marchingband_1_465.jpg
345 211 4 4
331 126 3 3
"""

WIDER_OFFICIAL_SAMPLE = """\
0--Parade/0_Parade_marchingband_1_465.jpg
2
345 211 4 4 2 0 0 0 0 0
331 126 3 3 2 0 0 1 1 0
0--Parade/0_Parade_Parade_0_452.jpg
0
0 0 0 0 0 0 0 0 0 0
0--Parade/0_Parade_marchingband_1_1031.jpg
1
100 200 30 40 0 0 0 0 2 1
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_retinaface_train_labels(tmp_path: Path) -> None:
    records = parse_retinaface_labels(write(tmp_path, "label.txt", RETINAFACE_TRAIN_SAMPLE))
    assert [r["path"] for r in records] == [
        "0--Parade/0_Parade_marchingband_1_849.jpg",
        "0--Parade/0_Parade_Parade_0_904.jpg",
    ]
    first, second = records
    assert first["boxes"].dtype == np.float32 and first["landmarks"].dtype == np.float32
    assert np.allclose(first["boxes"], [[449, 330, 571, 479]])
    assert np.allclose(first["landmarks"][0, 0], [488.906, 373.643])
    assert np.allclose(first["landmarks"][0, 4], [538.357, 431.491])
    # The zero-width face is dropped; the -1 landmarks become NaN.
    assert second["boxes"].shape == (2, 4) and second["landmarks"].shape == (2, 5, 2)
    assert np.allclose(second["boxes"][1], [78, 221, 85, 229])
    assert np.isfinite(second["landmarks"][0]).all()
    assert np.isnan(second["landmarks"][1]).all()


def test_parse_retinaface_val_labels_have_no_landmarks(tmp_path: Path) -> None:
    records = parse_retinaface_labels(write(tmp_path, "label.txt", RETINAFACE_VAL_SAMPLE))
    assert len(records) == 1
    assert np.allclose(records[0]["boxes"], [[345, 211, 349, 215], [331, 126, 334, 129]])
    assert np.isnan(records[0]["landmarks"]).all() and records[0]["landmarks"].shape == (2, 5, 2)


def test_parse_wider_official_including_zero_faces(tmp_path: Path) -> None:
    records = parse_wider_official(write(tmp_path, "gt.txt", WIDER_OFFICIAL_SAMPLE))
    assert [r["path"] for r in records] == [
        "0--Parade/0_Parade_marchingband_1_465.jpg",
        "0--Parade/0_Parade_Parade_0_452.jpg",
        "0--Parade/0_Parade_marchingband_1_1031.jpg",
    ]
    first, empty, last = records
    assert np.allclose(first["boxes"], [[345, 211, 349, 215], [331, 126, 334, 129]])
    assert first["invalid"].tolist() == [False, True]
    assert first["blur"].tolist() == [2, 2] and first["occlusion"].tolist() == [0, 1]
    assert empty["boxes"].shape == (0, 4) and empty["invalid"].shape == (0,)
    assert np.allclose(last["boxes"], [[100, 200, 130, 240]])
    assert last["occlusion"].tolist() == [2] and last["pose"].tolist() == [1] and last["invalid"].tolist() == [False]


def test_hflip_mirrors_coordinates_and_swaps_sides() -> None:
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[:, 0] = 255
    boxes = np.array([[10, 20, 30, 50]], dtype=np.float32)
    landmarks = np.array([[[15, 30], [25, 31], [20, 38], [16, 45], [24, 46]]], dtype=np.float32)

    flipped, new_boxes, new_landmarks = hflip(image, boxes, landmarks)

    assert flipped[:, -1].min() == 255 and flipped[:, 0].max() == 0
    assert np.allclose(new_boxes, [[70, 20, 90, 50]])
    expected = [[75, 31], [85, 30], [80, 38], [76, 46], [84, 45]]  # the image-left eye is the mirrored right eye
    assert np.allclose(new_landmarks[0], expected)
    assert new_landmarks[0, 0, 0] < new_landmarks[0, 1, 0] and new_landmarks[0, 3, 0] < new_landmarks[0, 4, 0]


def test_random_square_crop_keeps_faces_inside() -> None:
    rng = np.random.default_rng(1)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    boxes = np.array([[100, 80, 140, 130], [260, 10, 290, 40]], dtype=np.float32)
    landmarks = np.full((2, 5, 2), np.nan, dtype=np.float32)
    for _ in range(20):
        crop, new_boxes, new_landmarks = random_square_crop(image, boxes, landmarks, rng)
        side = crop.shape[0]
        assert crop.shape[0] == crop.shape[1] and len(new_boxes) >= 1 and len(new_boxes) == len(new_landmarks)
        assert (new_boxes >= 0).all() and (new_boxes <= side).all()


def test_random_square_crop_without_faces_pads_the_whole_image() -> None:
    image = np.full((50, 80, 3), 7, dtype=np.uint8)
    crop, boxes, _ = random_square_crop(image, np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32),
                                        np.random.default_rng(0))
    assert crop.shape == (80, 80, 3) and crop[49, 79, 0] == 7 and crop[79, 0, 0] == 0 and boxes.shape == (0, 4)


def test_photometric_distort_keeps_shape_and_type() -> None:
    image = np.random.default_rng(0).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    for seed in range(5):
        out = photometric_distort(image, np.random.default_rng(seed))
        assert out.shape == image.shape and out.dtype == np.uint8


def make_dataset_files(tmp_path: Path) -> list[dict]:
    """Writes a 120 x 200 image with one face and returns its record."""
    image = np.full((120, 200, 3), 40, dtype=np.uint8)
    image[30:90, 50:110] = (30, 60, 200)  # BGR on disk
    cv2.imwrite(str(tmp_path / "face.jpg"), image)
    landmarks = np.array([[[65, 50], [95, 50], [80, 62], [68, 75], [92, 75]]], dtype=np.float32)
    return [{"path": "face.jpg", "boxes": np.array([[50, 30, 110, 90]], np.float32), "landmarks": landmarks}]


def test_wider_face_train_without_augmentation(tmp_path: Path) -> None:
    dataset = WiderFaceTrain(make_dataset_files(tmp_path), tmp_path, size=64, augment=False)
    image, target = dataset[0]
    assert image.dtype == torch.uint8 and image.shape == (3, 64, 64)
    # Read as RGB: the red channel of the face (200 in BGR's third channel) comes first.
    assert image[0, 15, 25] > 150 and image[2, 15, 25] < 80
    # Padded to 200 x 200 and resized by 0.32: centre (80, 60) -> (25.6, 19.2) -> cell (6, 4).
    assert target["heatmap"].shape == (1, 16, 16) and target["heatmap"][0, 4, 6] == 1.0
    assert target["mask"].sum() == 1 and target["landmark_mask"][0].sum() == 10


def test_wider_face_train_with_augmentation_and_loader(tmp_path: Path) -> None:
    dataset = WiderFaceTrain(make_dataset_files(tmp_path), tmp_path, size=64, augment=True, seed=3)
    for _ in range(5):
        image, target = dataset[0]
        assert image.shape == (3, 64, 64) and target["heatmap"].shape == (1, 16, 16)
        assert target["mask"].sum() <= 1
    loader = DataLoader(torch.utils.data.ConcatDataset([dataset, dataset]), batch_size=2,
                        collate_fn=detection_collate)
    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 64, 64) and images.dtype == torch.uint8
    assert targets["heatmap"].shape == (2, 1, 16, 16) and targets["index"].shape == (2, 1000)
