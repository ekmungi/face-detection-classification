"""Face alignment: map five facial landmarks onto a fixed template with a similarity transform.

The same function prepares crops for training and for inference, so both models always see faces in the same pose and
scale. The template is the widely used ArcFace 112 x 112 layout: image-left eye, image-right eye, nose tip, image-left
mouth corner, image-right mouth corner.
"""
from __future__ import annotations

import cv2
import numpy as np

# Landmark positions in a 112 x 112 crop, in pixels, ordered image-left eye, image-right eye, nose, mouth left, mouth right.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def template_for(size: int) -> np.ndarray:
    """Returns the ArcFace template scaled to a square crop of `size` pixels."""
    return ARCFACE_TEMPLATE_112 * (size / 112.0)


def estimate_transform(landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Returns the 2 x 3 similarity transform (rotation, uniform scale, translation) that maps `landmarks` onto the template.

    Args:
        landmarks: array of shape (5, 2) in image pixels, in template order.
        size: side of the output crop in pixels.

    Raises:
        ValueError: when the landmarks do not have shape (5, 2) or the transform cannot be estimated.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2):
        raise ValueError(f"expected landmarks of shape (5, 2), got {points.shape}")
    matrix, _ = cv2.estimateAffinePartial2D(points, template_for(size), method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("could not estimate an alignment transform from these landmarks")
    return matrix.astype(np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Returns the aligned square face crop of side `size` (same channel order and dtype as `image`)."""
    matrix = estimate_transform(landmarks, size)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def center_crop_resize(image: np.ndarray, size: int = 112) -> np.ndarray:
    """Returns the largest centred square of `image` resized to `size`: the fallback for crops where no face is detected."""
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = image[top:top + side, left:left + side]
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(square, (size, size), interpolation=interpolation)
