"""Reads WIDER FACE annotation files and provides the training dataset for the face detector in `facekit.det_model`.

WIDER FACE is a public collection of photos with many faces. RetinaFace is a published face detector whose authors
released extra annotations for WIDER FACE. Two annotation formats are read:

- The RetinaFace file `retinaface_gt_v1.1/train/label.txt` holds boxes plus five landmarks. The validation file of the
  same release has boxes only.
- The official file `wider_face_split/wider_face_val_bbx_gt.txt` holds boxes plus flags for blur, occlusion, pose and
  invalid faces.

Boxes are returned as float32 xyxy pixels, which means (left, top, right, bottom). Landmarks are returned as float32
[N, 5, 2] arrays in the order image-left eye, image-right eye, nose tip, image-left mouth corner and image-right mouth
corner. A landmark is NaN where a face has no landmark annotation.
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

# The side of a random crop is one of these fractions of the image's shorter side.
CROP_SCALES = (0.3, 0.45, 0.6, 0.8, 1.0)

# The number of random crops tried before the whole image is padded to a square and used instead.
CROP_ATTEMPTS = 25

# The landmark order after a horizontal flip, which swaps the two eyes and the two mouth corners.
FLIP_ORDER = (1, 0, 2, 4, 3)

# The probability that the colour changes are applied to a training image.
PHOTOMETRIC_PROBABILITY = 0.5

# The largest brightness change, in pixel values on the 0 to 255 scale, in either direction.
BRIGHTNESS_DELTA = 32.0

# The range of the random factor that multiplies the pixel values to change the contrast.
CONTRAST_RANGE = (0.5, 1.5)

# The range of the random factor that multiplies the saturation.
SATURATION_RANGE = (0.5, 1.5)

# The largest hue change, in OpenCV hue units of 2 degrees, so 9 units are 18 degrees in either direction.
HUE_DELTA = 9

# The coordinate value that the RetinaFace file uses for a landmark that is not annotated.
MISSING_LANDMARK = -1.0


# --------------------------------------------------------------------------------------------------------------------
# Annotation parsers
# --------------------------------------------------------------------------------------------------------------------


def _xywh_to_xyxy(values: np.ndarray) -> np.ndarray:
    """Converts [N, 4] rows of (x, y, width, height) to float32 rows of (x1, y1, x2, y2) corner coordinates."""
    boxes = np.asarray(values, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _parse_landmarks(values: list[float]) -> np.ndarray:
    """Reads five (x, y, flag) triples into a [5, 2] array. A point with both coordinates -1 becomes NaN.

    The values are the part of a face row after the box. When fewer than 15 values are given, the face has no landmark
    annotation and all five points stay NaN. The flag of each triple is not used.
    """
    points = np.full((5, 2), np.nan, dtype=np.float32)
    if len(values) < 15:
        return points

    triples = np.asarray(values[:15], dtype=np.float32).reshape(5, 3)
    annotated = ~((triples[:, 0] == MISSING_LANDMARK) & (triples[:, 1] == MISSING_LANDMARK))
    points[annotated] = triples[annotated, :2]
    return points


def _retinaface_record(path: str, rows: list[list[float]]) -> dict:
    """Builds one image record from its face rows, dropping boxes with zero or negative width or height."""
    # An image without faces gets empty arrays of the right shape.
    if rows:
        boxes = _xywh_to_xyxy(np.array([row[:4] for row in rows]))
        landmarks = np.stack([_parse_landmarks(row[4:]) for row in rows])
    else:
        boxes, landmarks = np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32)

    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return {"path": path, "boxes": boxes[keep], "landmarks": landmarks[keep]}


def parse_retinaface_labels(path: str | Path) -> list[dict]:
    """Parses a RetinaFace `label.txt` into `{"path", "boxes" [N, 4] xyxy, "landmarks" [N, 5, 2]}` records.

    Each image starts with a line `# relative/path.jpg`. Each face line holds `x y w h` and, in the training file,
    five `lx ly flag` triples and a score. Landmarks given as -1 are NaN, and boxes with a width or height of 0 or
    less are dropped.
    """
    records: list[dict] = []
    current_path: str | None = None
    rows: list[list[float]] = []

    # A line that starts with "#" begins a new image, so the image before it is finished. Any other line is a face
    # of the current image.
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

    # The last image has no following "#" line, so it is finished here.
    if current_path is not None:
        records.append(_retinaface_record(current_path, rows))
    return records


def _wider_record(path: str, rows: np.ndarray) -> dict:
    """Builds one official-format record from its [N, 10] attribute rows.

    A box counts as invalid when the file flags it as invalid or when its width or height is zero or less.
    """
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
    """Parses the official WIDER FACE `*_bbx_gt.txt` file.

    The file holds, for each image, the image path, the face count n, and then max(n, 1) lines of
    `x y w h blur expression illumination invalid occlusion pose`. An image with n = 0 still has one line of zeros,
    which is skipped. The function returns records of the form `{"path", "boxes" xyxy float32 [N, 4], "invalid" bool
    [N], "blur", "occlusion", "pose" int [N]}`. Boxes with zero width or height are kept but marked invalid, so that
    the evaluation can ignore them.
    """
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    records: list[dict] = []
    position = 0

    # Each pass reads one image: its path, its face count and its face lines. The position then moves past them.
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
    """Mirrors an image [H, W, 3] left to right together with its boxes and landmarks.

    After the mirror, the landmark order is changed so that the image-left eye is again first and the image-left mouth
    corner is again fourth. Landmarks that are NaN stay NaN.
    """
    width = image.shape[1]
    flipped = np.ascontiguousarray(image[:, ::-1])
    new_boxes = boxes.copy()
    new_boxes[:, 0], new_boxes[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    new_landmarks = landmarks.copy()
    new_landmarks[:, :, 0] = width - landmarks[:, :, 0]
    return flipped, new_boxes, new_landmarks[:, list(FLIP_ORDER)]


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Pads an image with black on the bottom and right edges until it is square.

    The padding is added after the existing pixels, so the box and landmark coordinates do not change.
    """
    height, width = image.shape[:2]
    side = max(height, width)
    return cv2.copyMakeBorder(image, 0, side - height, 0, side - width, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def random_square_crop(
    image: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cuts the random square crop that RetinaFace uses, which contains at least one face centre.

    The side of the square is one of the fractions in `CROP_SCALES` (0.3 to 1.0) of the image's shorter side. Faces
    whose centres lie in the crop are kept, with their boxes clipped to the crop. If no crop that contains a face centre
    is found within `CROP_ATTEMPTS` (25) tries, the whole image is padded to a square instead.
    """
    height, width = image.shape[:2]
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2

    for _ in range(CROP_ATTEMPTS):
        # Pick a random square that lies inside the image.
        side = max(1, int(rng.choice(CROP_SCALES) * min(height, width)))
        left = int(rng.integers(0, width - side + 1))
        top = int(rng.integers(0, height - side + 1))

        # Keep the faces whose centres lie inside the square. A square without any such face is tried again.
        inside = (
            (centres[:, 0] >= left)
            & (centres[:, 0] < left + side)
            & (centres[:, 1] >= top)
            & (centres[:, 1] < top + side)
        )
        if not inside.any():
            continue

        # Move the boxes and landmarks into the coordinates of the crop, and clip the boxes to the crop.
        shift = np.array([left, top], dtype=np.float32)
        new_boxes = (boxes[inside].reshape(-1, 2, 2) - shift).clip(0, side).reshape(-1, 4)
        return image[top : top + side, left : left + side], new_boxes, landmarks[inside] - shift

    return pad_to_square(image), boxes, landmarks


def photometric_distort(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Returns a copy of an RGB uint8 image with random brightness, contrast, saturation and hue changes.

    Each of the four changes is applied with probability 0.5, independently of the others.
    """
    result = image.astype(np.float32)

    # Brightness and contrast. The brightness change adds one number to every pixel and the contrast change multiplies
    # every pixel by one factor. The values are then clipped back to the range 0 to 255.
    if rng.random() < 0.5:
        result += rng.uniform(-BRIGHTNESS_DELTA, BRIGHTNESS_DELTA)
    if rng.random() < 0.5:
        result *= rng.uniform(*CONTRAST_RANGE)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Saturation and hue. These are changed in the HSV colour space (hue, saturation, value), where colour and
    # brightness are separate. The step is skipped when neither change was drawn.
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
    """Resizes a square image to `size` x `size` and scales its boxes and landmarks to match.

    An image that is shrunk is resized by area averaging, and any other image by bilinear interpolation.
    """
    side = image.shape[0]
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    resized = cv2.resize(image, (size, size), interpolation=interpolation)
    factor = size / side
    return resized, boxes * factor, landmarks * factor


# --------------------------------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------------------------------


def read_rgb(path: str | Path) -> np.ndarray:
    """Reads an image file with OpenCV and returns it as RGB uint8 [H, W, 3].

    Raises FileNotFoundError when the file is missing or cannot be decoded. OpenCV reads the colours as BGR, so the
    channels are reordered.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class WiderFaceTrain(Dataset):
    """Holds the WIDER FACE training images, with RetinaFace-style augmentation and `encode_targets` targets.

    `records` come from `parse_retinaface_labels`, and their paths are relative to `image_root`. Each item is
    `(image uint8 [3, size, size] RGB, target dict)`. The random generator is created per process from the seed and
    `torch.initial_seed()`, so DataLoader workers draw different augmentations and each epoch differs.
    """

    def __init__(
        self, records: Sequence[dict], image_root: str | Path, size: int = 640, augment: bool = True, seed: int = 0
    ) -> None:
        """Stores the records and settings. The `size` must be a multiple of 32.

        With `augment` switched off, an image is only padded to a square and resized, without any random change.
        """
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
        """Returns the number of images in the dataset."""
        return len(self.records)

    def _generator(self) -> np.random.Generator:
        """Returns this process's random generator, creating it again when the process or the worker seed changes."""
        key = torch.initial_seed()
        if self._rng is None or self._rng_key != key:
            self._rng = np.random.default_rng([self.seed, key & 0xFFFFFFFF, key >> 32])
            self._rng_key = key
        return self._rng

    def _augment(
        self, image: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Applies a random crop, a resize, a horizontal flip and colour changes to an image and its annotations."""
        image, boxes, landmarks = random_square_crop(image, boxes, landmarks, rng)
        image, boxes, landmarks = resize_square(image, boxes, landmarks, self.size)

        # The flip happens with probability 0.5. The colour changes follow with the probability set for them.
        if rng.random() < 0.5:
            image, boxes, landmarks = hflip(image, boxes, landmarks)
        if rng.random() < PHOTOMETRIC_PROBABILITY:
            image = photometric_distort(image, rng)
        return image, boxes, landmarks

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        """Returns the image as uint8 [3, size, size] and its encoded detector targets."""
        record = self.records[index]
        image = read_rgb(self.image_root / record["path"])
        boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
        landmarks = np.asarray(record["landmarks"], dtype=np.float32).reshape(-1, 5, 2)

        # With augmentation switched on, the image gets the random changes. Otherwise it is only padded to a square and
        # resized.
        if self.augment:
            image, boxes, landmarks = self._augment(image, boxes, landmarks, self._generator())
        else:
            image, boxes, landmarks = resize_square(pad_to_square(image), boxes, landmarks, self.size)
        target = encode_targets(boxes, landmarks, self.size, self.size)
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous(), target


def detection_collate(batch: Sequence[tuple[Tensor, dict[str, Tensor]]]) -> tuple[Tensor, dict[str, Tensor]]:
    """Stacks dataset items into `(images uint8 [B, 3, S, S], targets dict of batched tensors)`.

    The images all have the same size S, so they stack directly. The targets are combined by `collate_targets`.
    """
    images = torch.stack([image for image, _ in batch])
    return images, collate_targets([target for _, target in batch])
