"""Tests facekit.det_data: the annotation parsers, the image augmentation and the training dataset.

The tests run offline on small made-up annotation files and images. A box is written as xyxy, which means the left,
top, right and bottom pixel positions. A face has five landmark points, which are the two eyes, the nose and the two
mouth corners. WIDER FACE is a public data set of photos with labelled faces. The authors of the RetinaFace
detector published a version of its labels that adds the landmarks.
"""
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

# A training label file in the RetinaFace format. A line that starts with "#" gives an image path. Each line after it
# describes one face with the box as x, y, width and height, then five landmark points of three numbers each, and a
# final score. The second image has one face with landmarks, one face with the landmark values -1, which means that
# they are missing, and a last face with a width of 0, which is not a usable face.
RETINAFACE_TRAIN_SAMPLE = """\
# 0--Parade/0_Parade_marchingband_1_849.jpg
449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.830 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82
# 0--Parade/0_Parade_Parade_0_904.jpg
361 98 263 339 424.143 251.656 0.0 547.134 232.571 0.0 494.121 325.875 0.0 453.83 339.86 0.0 561.978 320.857 0.0 0.89
78 221 7 8 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
50 60 0 10 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
"""

# A validation label file in the RetinaFace format. It has the box as x, y, width and height and has no landmarks.
RETINAFACE_VAL_SAMPLE = """\
# 0--Parade/0_Parade_marchingband_1_465.jpg
345 211 4 4
331 126 3 3
"""

# An annotation file in the official WIDER FACE format. Each image has its path, the number of faces and then one line
# per face. A face line has the box as x, y, width and height, followed by six attributes: blur, expression,
# illumination, invalid, occlusion and pose. The second image has no faces, but the format still gives it one line of
# zeros.
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
    """Writes the text to a file of the given name in the folder and returns the path of the file."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_retinaface_train_labels(tmp_path: Path) -> None:
    """The parser reads the paths, boxes and landmarks, drops faces without width and marks missing landmarks as NaN."""
    records = parse_retinaface_labels(write(tmp_path, "label.txt", RETINAFACE_TRAIN_SAMPLE))
    assert [r["path"] for r in records] == [
        "0--Parade/0_Parade_marchingband_1_849.jpg",
        "0--Parade/0_Parade_Parade_0_904.jpg",
    ]
    first, second = records

    # The first image has one face. Its box is converted from x, y, width and height to the xyxy form.
    assert first["boxes"].dtype == np.float32 and first["landmarks"].dtype == np.float32
    assert np.allclose(first["boxes"], [[449, 330, 571, 479]])
    assert np.allclose(first["landmarks"][0, 0], [488.906, 373.643])
    assert np.allclose(first["landmarks"][0, 4], [538.357, 431.491])

    # The face with a width of 0 is dropped, so two faces remain. The landmark values of -1 become NaN.
    assert second["boxes"].shape == (2, 4) and second["landmarks"].shape == (2, 5, 2)
    assert np.allclose(second["boxes"][1], [78, 221, 85, 229])
    assert np.isfinite(second["landmarks"][0]).all()
    assert np.isnan(second["landmarks"][1]).all()


def test_parse_retinaface_val_labels_have_no_landmarks(tmp_path: Path) -> None:
    """Validation labels have no landmarks, so every landmark of every face is NaN."""
    records = parse_retinaface_labels(write(tmp_path, "label.txt", RETINAFACE_VAL_SAMPLE))
    assert len(records) == 1
    assert np.allclose(records[0]["boxes"], [[345, 211, 349, 215], [331, 126, 334, 129]])
    assert np.isnan(records[0]["landmarks"]).all() and records[0]["landmarks"].shape == (2, 5, 2)


def test_parse_wider_official_including_zero_faces(tmp_path: Path) -> None:
    """The parser reads boxes and attributes in the official format, including an image that has no faces."""
    records = parse_wider_official(write(tmp_path, "gt.txt", WIDER_OFFICIAL_SAMPLE))
    assert [r["path"] for r in records] == [
        "0--Parade/0_Parade_marchingband_1_465.jpg",
        "0--Parade/0_Parade_Parade_0_452.jpg",
        "0--Parade/0_Parade_marchingband_1_1031.jpg",
    ]
    first, empty, last = records

    # The first image has two faces, and the second face is marked invalid in the file.
    assert np.allclose(first["boxes"], [[345, 211, 349, 215], [331, 126, 334, 129]])
    assert first["invalid"].tolist() == [False, True]
    assert first["blur"].tolist() == [2, 2] and first["occlusion"].tolist() == [0, 1]

    # The image without faces gives empty arrays, and its line of zeros is not read as a face.
    assert empty["boxes"].shape == (0, 4) and empty["invalid"].shape == (0,)

    # The last image has one face with an occlusion of 2 and a pose of 1.
    assert np.allclose(last["boxes"], [[100, 200, 130, 240]])
    assert last["occlusion"].tolist() == [2] and last["pose"].tolist() == [1] and last["invalid"].tolist() == [False]


def test_hflip_mirrors_coordinates_and_swaps_sides() -> None:
    """A horizontal flip mirrors the image, the boxes and the landmarks, and swaps the left and right landmarks."""
    # The image is black except for a white column on its left edge.
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[:, 0] = 255
    boxes = np.array([[10, 20, 30, 50]], dtype=np.float32)
    landmarks = np.array([[[15, 30], [25, 31], [20, 38], [16, 45], [24, 46]]], dtype=np.float32)

    flipped, new_boxes, new_landmarks = hflip(image, boxes, landmarks)

    # The white column is now on the right edge. The box that spanned x from 10 to 30 now spans x from 70 to 90, because
    # the image is 100 pixels wide.
    assert flipped[:, -1].min() == 255 and flipped[:, 0].max() == 0
    assert np.allclose(new_boxes, [[70, 20, 90, 50]])

    # After the flip, the first landmark is the eye on the left of the image. It is the mirrored copy of the old
    # second landmark, and the same holds for the mouth corners.
    expected = [[75, 31], [85, 30], [80, 38], [76, 46], [84, 45]]
    assert np.allclose(new_landmarks[0], expected)
    assert new_landmarks[0, 0, 0] < new_landmarks[0, 1, 0] and new_landmarks[0, 3, 0] < new_landmarks[0, 4, 0]


def test_random_square_crop_keeps_faces_inside() -> None:
    """Every random square crop is square, keeps at least one face and has all its boxes inside the crop."""
    rng = np.random.default_rng(1)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    boxes = np.array([[100, 80, 140, 130], [260, 10, 290, 40]], dtype=np.float32)
    landmarks = np.full((2, 5, 2), np.nan, dtype=np.float32)

    # The crop is random, so the check is repeated 20 times.
    for _ in range(20):
        crop, new_boxes, new_landmarks = random_square_crop(image, boxes, landmarks, rng)
        side = crop.shape[0]
        assert crop.shape[0] == crop.shape[1] and len(new_boxes) >= 1 and len(new_boxes) == len(new_landmarks)
        assert (new_boxes >= 0).all() and (new_boxes <= side).all()


def test_random_square_crop_without_faces_pads_the_whole_image() -> None:
    """With no faces, the crop is the whole image padded to a square, with the padding below the picture."""
    image = np.full((50, 80, 3), 7, dtype=np.uint8)
    crop, boxes, _ = random_square_crop(image, np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32),
                                        np.random.default_rng(0))

    # The 50 x 80 image becomes 80 x 80. The last pixel of the picture keeps its value of 7, and the padding is 0.
    assert crop.shape == (80, 80, 3) and crop[49, 79, 0] == 7 and crop[79, 0, 0] == 0 and boxes.shape == (0, 4)


def test_photometric_distort_keeps_shape_and_type() -> None:
    """The random colour changes leave the image with the same shape and the uint8 type, for several seeds."""
    image = np.random.default_rng(0).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    for seed in range(5):
        out = photometric_distort(image, np.random.default_rng(seed))
        assert out.shape == image.shape and out.dtype == np.uint8


def make_dataset_files(tmp_path: Path) -> list[dict]:
    """Writes a 120 x 200 image with one face to the folder and returns the list with the record of that image."""
    image = np.full((120, 200, 3), 40, dtype=np.uint8)

    # The face is a coloured rectangle. The colour is written in BGR order, which is the order that OpenCV saves.
    image[30:90, 50:110] = (30, 60, 200)
    cv2.imwrite(str(tmp_path / "face.jpg"), image)
    landmarks = np.array([[[65, 50], [95, 50], [80, 62], [68, 75], [92, 75]]], dtype=np.float32)
    return [{"path": "face.jpg", "boxes": np.array([[50, 30, 110, 90]], np.float32), "landmarks": landmarks}]


def test_wider_face_train_without_augmentation(tmp_path: Path) -> None:
    """Without augmentation the dataset returns the padded, resized RGB image and the encoded targets of the face."""
    dataset = WiderFaceTrain(make_dataset_files(tmp_path), tmp_path, size=64, augment=False)
    image, target = dataset[0]
    assert image.dtype == torch.uint8 and image.shape == (3, 64, 64)

    # The image is returned in RGB order. The red value of the face, 200, is the third channel in BGR, so in RGB the
    # red channel of the face comes first and is high, and the blue channel is low.
    assert image[0, 15, 25] > 150 and image[2, 15, 25] < 80

    # The image is padded to 200 x 200 and resized by a factor of 0.32. The face centre moves from (80, 60) to
    # (25.6, 19.2) pixels, which is column 6 and row 4 of the grid, because one grid cell is 4 pixels wide.
    assert target["heatmap"].shape == (1, 16, 16) and target["heatmap"][0, 4, 6] == 1.0
    assert target["mask"].sum() == 1 and target["landmark_mask"][0].sum() == 10


def test_wider_face_train_with_augmentation_and_loader(tmp_path: Path) -> None:
    """With augmentation the outputs keep their shapes, and a data loader can combine the items into a batch."""
    dataset = WiderFaceTrain(make_dataset_files(tmp_path), tmp_path, size=64, augment=True, seed=3)

    # The augmentation is random, so the item is read several times. Every version must keep the shapes and hold at
    # most the one face.
    for _ in range(5):
        image, target = dataset[0]
        assert image.shape == (3, 64, 64) and target["heatmap"].shape == (1, 16, 16)
        assert target["mask"].sum() <= 1

    # A batch of two items is built from two copies of the dataset, with the collate function of the module.
    loader = DataLoader(torch.utils.data.ConcatDataset([dataset, dataset]), batch_size=2,
                        collate_fn=detection_collate)
    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 64, 64) and images.dtype == torch.uint8
    assert targets["heatmap"].shape == (2, 1, 16, 16) and targets["index"].shape == (2, 1000)
