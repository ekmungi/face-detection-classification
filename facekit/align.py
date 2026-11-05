"""Aligns a face by mapping its five landmarks onto a fixed template with a similarity transform.

A similarity transform is a rotation, a uniform scale and a shift. The same function prepares crops for training and
for inference, so both models always see faces in the same pose and scale.

The template is the widely used ArcFace 112 x 112 layout. Its five points are the image-left eye, the image-right
eye, the nose tip, the image-left mouth corner and the image-right mouth corner. Image-left means the left side of the
picture, not the left side of the person.
"""
from __future__ import annotations

import cv2
import numpy as np

# The landmark positions in a 112 x 112 crop, in pixels. The rows are the image-left eye, the image-right eye, the nose,
# the image-left mouth corner and the image-right mouth corner.
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
    """Returns the 2 x 3 similarity transform that maps `landmarks` onto the template for a crop of `size` pixels.

    Args:
        landmarks: An array of shape (5, 2) with the landmark positions in image pixels, in template order.
        size: The side of the output crop in pixels.

    Raises:
        ValueError: When the landmarks do not have shape (5, 2), are not finite, or do not allow the transform to be
            estimated.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        raise ValueError(f"expected finite landmarks of shape (5, 2), got {points.shape}")
    return umeyama(points, template_for(size).astype(np.float64)).astype(np.float32)


def umeyama(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Returns the 2 x 3 least-squares similarity transform from the source points to the target points.

    The transform is the rotation, uniform scale and translation that bring the source points as close as possible
    to the target points, which Umeyama's method computes directly. Every point counts equally, so one noisy landmark
    shifts the crop a little instead of being dropped.

    Raises ValueError for degenerate points, which are points that all lie in one place.
    """
    # Centre both point sets on their means. The variance of the source points is needed for the scale, and a variance
    # near zero means that the points are degenerate.
    mu_s, mu_t = source.mean(0), target.mean(0)
    s_c, t_c = source - mu_s, target - mu_t
    var_s = (s_c ** 2).sum() / len(source)
    if var_s < 1e-9:
        raise ValueError("could not estimate an alignment transform from these landmarks")

    # The singular value decomposition (SVD) of the covariance between the two sets gives the best rotation. The sign
    # matrix turns a mirroring into a proper rotation when the determinants indicate one.
    cov = t_c.T @ s_c / len(source)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt

    # Derive the scale and the translation from the rotation, and join the three parts into one 2 x 3 matrix.
    scale = np.trace(np.diag(d) @ sign) / var_s
    translation = mu_t - scale * rotation @ mu_s
    return np.hstack([scale * rotation, translation[:, None]])


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Returns the aligned square face crop of side `size`, with the same channel order and dtype as `image`.

    Pixels that fall outside the original image repeat its nearest border pixel.
    """
    matrix = estimate_transform(landmarks, size)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def center_crop_resize(image: np.ndarray, size: int = 112) -> np.ndarray:
    """Returns the largest centred square of `image`, resized to `size` pixels.

    This is the fallback for crops in which no face is detected.
    """
    # Cut the largest centred square out of the image.
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = image[top:top + side, left:left + side]

    # Area interpolation is the usual choice for shrinking, and cubic interpolation is the usual choice for enlarging.
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(square, (size, size), interpolation=interpolation)
