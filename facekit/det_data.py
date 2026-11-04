"""WIDER FACE annotation parsers and the training dataset for the face detector in :mod:`facekit.det_model`.

Two annotation formats are read:

- the RetinaFace file ``retinaface_gt_v1.1/train/label.txt`` (boxes plus five landmarks; the validation file of the
  same release has boxes only);
- the official ``wider_face_split/wider_face_val_bbx_gt.txt`` (boxes plus blur, occlusion, pose and invalid flags).

Boxes are returned as float32 xyxy pixels and landmarks as float32 [N,5,2] in the order image-left eye, image-right eye,
nose tip, image-left mouth corner, image-right mouth corner, with NaN where a face has no landmark annotation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from facekit.det_model import collate_targets, encode_targets

CROP_SCALES = (0.3, 0.45, 0.6, 0.8, 1.0)
CROP_ATTEMPTS = 25
FLIP_ORDER = (1, 0, 2, 4, 3)
PHOTOMETRIC_PROBABILITY = 0.5
BRIGHTNESS_DELTA = 32.0
CONTRAST_RANGE = (0.5, 1.5)
SATURATION_RANGE = (0.5, 1.5)
HUE_DELTA = 9  # OpenCV hue units of 2 degrees, so +-18 degrees.
MISSING_LANDMARK = -1.0


# --------------------------------------------------------------------------------------------------------------------
# Annotation parsers
# --------------------------------------------------------------------------------------------------------------------


def _xywh_to_xyxy(values: np.ndarray) -> np.ndarray:
    """Converts [N,4] x, y, w, h rows to float32 x1, y1, x2, y2."""
    boxes = np.asarray(values, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _parse_landmarks(values: list[float]) -> np.ndarray:
    """Reads five (x, y, flag) triples into a [5,2] array; a point with both coordinates -1 becomes NaN."""
    points = np.full((5, 2), np.nan, dtype=np.float32)
    if len(values) < 15:
        return points
    triples = np.asarray(values[:15], dtype=np.float32).reshape(5, 3)
    annotated = ~((triples[:, 0] == MISSING_LANDMARK) & (triples[:, 1] == MISSING_LANDMARK))
    points[annotated] = triples[annotated, :2]
    return points


def _retinaface_record(path: str, rows: list[list[float]]) -> dict:
    """Builds one image record from its face rows, dropping boxes with zero or negative width or height."""
    if rows:
        boxes = _xywh_to_xyxy(np.array([row[:4] for row in rows]))
        landmarks = np.stack([_parse_landmarks(row[4:]) for row in rows])
    else:
        boxes, landmarks = np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return {"path": path, "boxes": boxes[keep], "landmarks": landmarks[keep]}


def parse_retinaface_labels(path: str | Path) -> list[dict]:
    """Parses a RetinaFace ``label.txt`` into ``{"path", "boxes" [N,4] xyxy, "landmarks" [N,5,2]}`` records.

    Each image starts with ``# relative/path.jpg``; each face line holds ``x y w h`` and, in the training file, five
    ``lx ly flag`` triples and a score. Landmarks given as -1 are NaN; boxes with width or height <= 0 are dropped.
    """
    records: list[dict] = []
    current_path: str | None = None
    rows: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current_path is not None:
                records.append(_retinaface_record(current_path, rows))
            current_path, rows = line[1:].strip(), []
        elif current_path is not None:
            rows.append([float(value) for value in line.split()])
    if current_path is not None:
        records.append(_retinaface_record(current_path, rows))
    return records


def _wider_record(path: str, rows: np.ndarray) -> dict:
    """Builds one official-format record from its [N,10] attribute rows."""
    rows = rows.reshape(-1, 10)
    boxes = _xywh_to_xyxy(rows[:, :4])
    degenerate = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
    return {
        "path": path,
        "boxes": boxes,
        "invalid": (rows[:, 7] != 0) | degenerate,
        "blur": rows[:, 4].astype(np.int64),
        "occlusion": rows[:, 8].astype(np.int64),
        "pose": rows[:, 9].astype(np.int64),
    }


def parse_wider_official(path: str | Path) -> list[dict]:
    """Parses the official WIDER FACE ``*_bbx_gt.txt`` file.

    Format per image: the image path, the face count n, then max(n, 1) lines of
    ``x y w h blur expression illumination invalid occlusion pose`` (an image with n = 0 still has one line of zeros,
    which is skipped). Returns ``{"path", "boxes" xyxy float32 [N,4], "invalid" bool [N], "blur", "occlusion",
    "pose" int [N]}``. Boxes with zero width or height are kept but marked invalid, so evaluation can ignore them.
    """
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    records: list[dict] = []
    position = 0
    while position < len(lines):
        image_path, count = lines[position], int(lines[position + 1])
        face_lines = lines[position + 2 : position + 2 + max(count, 1)]
        rows = np.array([[float(v) for v in face.split()[:10]] for face in face_lines[:count]], dtype=np.float32)
        records.append(_wider_record(image_path, rows))
        position += 2 + max(count, 1)
    return records


# --------------------------------------------------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------------------------------------------------


def hflip(image: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirrors an image [H,W,3] with its boxes and landmarks; swaps left/right eyes and left/right mouth corners."""
    width = image.shape[1]
    flipped = np.ascontiguousarray(image[:, ::-1])
    new_boxes = boxes.copy()
    new_boxes[:, 0], new_boxes[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    new_landmarks = landmarks.copy()
    new_landmarks[:, :, 0] = width - landmarks[:, :, 0]
    return flipped, new_boxes, new_landmarks[:, list(FLIP_ORDER)]


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Pads an image bottom/right with black so it is square; coordinates are unchanged."""
    height, width = image.shape[:2]
    side = max(height, width)
    return cv2.copyMakeBorder(image, 0, side - height, 0, side - width, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def random_square_crop(
    image: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RetinaFace crop: a random square of 0.3-1.0 of the short side that contains at least one face centre.

    Faces whose centres lie in the crop are kept, with boxes clipped to it. After 25 failed attempts the whole image is
    padded to a square instead.
    """
    height, width = image.shape[:2]
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    for _ in range(CROP_ATTEMPTS):
        side = max(1, int(rng.choice(CROP_SCALES) * min(height, width)))
        left = int(rng.integers(0, width - side + 1))
        top = int(rng.integers(0, height - side + 1))
        inside = (
            (centres[:, 0] >= left)
            & (centres[:, 0] < left + side)
            & (centres[:, 1] >= top)
            & (centres[:, 1] < top + side)
        )
        if not inside.any():
            continue
        shift = np.array([left, top], dtype=np.float32)
        new_boxes = (boxes[inside].reshape(-1, 2, 2) - shift).clip(0, side).reshape(-1, 4)
        return image[top : top + side, left : left + side], new_boxes, landmarks[inside] - shift
    return pad_to_square(image), boxes, landmarks


def photometric_distort(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random brightness, contrast, saturation and hue changes on an RGB uint8 image, each with probability 0.5."""
    result = image.astype(np.float32)
    if rng.random() < 0.5:
        result += rng.uniform(-BRIGHTNESS_DELTA, BRIGHTNESS_DELTA)
    if rng.random() < 0.5:
        result *= rng.uniform(*CONTRAST_RANGE)
    result = np.clip(result, 0, 255).astype(np.uint8)
    change_saturation, change_hue = rng.random() < 0.5, rng.random() < 0.5
    if not (change_saturation or change_hue):
        return result
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.int32)
    if change_saturation:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(*SATURATION_RANGE), 0, 255)
    if change_hue:
        hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-HUE_DELTA, HUE_DELTA + 1))) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def resize_square(
    image: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resizes a square image to size x size and scales its boxes and landmarks to match."""
    side = image.shape[0]
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    resized = cv2.resize(image, (size, size), interpolation=interpolation)
    factor = size / side
    return resized, boxes * factor, landmarks * factor


# --------------------------------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------------------------------


def read_rgb(path: str | Path) -> np.ndarray:
    """Reads an image file with OpenCV and returns it as RGB uint8 [H,W,3]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class WiderFaceTrain(Dataset):
    """WIDER FACE training images with RetinaFace-style augmentation and :func:`encode_targets` targets.

    ``records`` come from :func:`parse_retinaface_labels`; their paths are relative to ``image_root``. Each item is
    ``(image uint8 [3,size,size] RGB, target dict)``. The random generator is created per process from the seed and
    ``torch.initial_seed()``, so DataLoader workers draw different augmentations and each epoch differs.
    """

    def __init__(
        self, records: Sequence[dict], image_root: str | Path, size: int = 640, augment: bool = True, seed: int = 0
    ) -> None:
        if size % 32:
            raise ValueError(f"size must be a multiple of 32, got {size}")
        self.records = list(records)
        self.image_root = Path(image_root)
        self.size = size
        self.augment = augment
        self.seed = seed
        self._rng: np.random.Generator | None = None
        self._rng_key: int | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _generator(self) -> np.random.Generator:
        """Returns this process's generator, creating it again when the process (or worker seed) changes."""
        key = torch.initial_seed()
        if self._rng is None or self._rng_key != key:
            self._rng = np.random.default_rng([self.seed, key & 0xFFFFFFFF, key >> 32])
            self._rng_key = key
        return self._rng

    def _augment(
        self, image: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Applies crop, resize, horizontal flip and photometric jitter."""
        image, boxes, landmarks = random_square_crop(image, boxes, landmarks, rng)
        image, boxes, landmarks = resize_square(image, boxes, landmarks, self.size)
        if rng.random() < 0.5:
            image, boxes, landmarks = hflip(image, boxes, landmarks)
        if rng.random() < PHOTOMETRIC_PROBABILITY:
            image = photometric_distort(image, rng)
        return image, boxes, landmarks

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        record = self.records[index]
        image = read_rgb(self.image_root / record["path"])
        boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
        landmarks = np.asarray(record["landmarks"], dtype=np.float32).reshape(-1, 5, 2)
        if self.augment:
            image, boxes, landmarks = self._augment(image, boxes, landmarks, self._generator())
        else:
            image, boxes, landmarks = resize_square(pad_to_square(image), boxes, landmarks, self.size)
        target = encode_targets(boxes, landmarks, self.size, self.size)
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous(), target


def detection_collate(batch: Sequence[tuple[Tensor, dict[str, Tensor]]]) -> tuple[Tensor, dict[str, Tensor]]:
    """Stacks dataset items into ``(images uint8 [B,3,S,S], targets dict of batched tensors)``."""
    images = torch.stack([image for image, _ in batch])
    return images, collate_targets([target for _, target in batch])
