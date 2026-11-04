"""The trained face detector behind one interface, running either the PyTorch model or its ONNX export.

ONNX Runtime is the production path: it runs the network on a laptop CPU without PyTorch's training machinery, and
the decoding step (peaks, boxes, landmarks, non-maximum suppression) is shared with training through det_model.decode.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from facekit.det_model import _resize_and_pad, decode

OUTPUT_NAMES = ("heatmap", "size", "offset", "landmarks")


@dataclasses.dataclass(frozen=True)
class Detection:
    """One detected face: box (x1, y1, x2, y2), five landmarks (5 x 2) in template order, and a score in [0, 1]."""

    box: tuple[float, float, float, float]
    landmarks: np.ndarray
    score: float

    def as_dict(self) -> dict:
        """Returns a JSON-friendly copy."""
        return {
            "box": [round(float(v), 1) for v in self.box],
            "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in self.landmarks],
            "score": round(float(self.score), 4),
        }


class FaceDetector:
    """Detects faces in RGB uint8 images.

    Args:
        run: a function from a float32 array [1, 3, H, W] in [0, 1] to the four output maps (numpy or torch).
        score_threshold: faces below this score are dropped (chosen on validation data; see the model card).
        nms_iou: overlap above which the weaker of two boxes is dropped.
        max_side: images are shrunk so the longest side is at most this before detection.
        min_face: faces whose shorter box side is below this many pixels are dropped.
    """

    def __init__(self, run: Callable[[np.ndarray], dict], score_threshold: float = 0.5, nms_iou: float = 0.4,
                 max_side: int = 1280, min_face: int = 16):
        self._run = run
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_side = max_side
        self.min_face = min_face

    @classmethod
    def from_onnx(cls, path: str | Path, providers: list[str] | None = None, **kwargs) -> "FaceDetector":
        """Loads the ONNX export with ONNX Runtime (CPU unless other providers are given)."""
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        names = [o.name for o in session.get_outputs()]

        def run(batch: np.ndarray) -> dict:
            return dict(zip(names, session.run(None, {input_name: batch})))
        return cls(run, **kwargs)

    @classmethod
    def from_torch(cls, model: torch.nn.Module, device: str = "cpu", **kwargs) -> "FaceDetector":
        """Wraps a trained FaceDetNet."""
        model = model.to(device).eval()

        @torch.no_grad()
        def run(batch: np.ndarray) -> dict:
            return model(torch.from_numpy(batch).to(device))
        return cls(run, **kwargs)

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """Returns the faces in an RGB uint8 image [H, W, 3], highest score first, in original pixels."""
        if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("expected an RGB image of shape (height, width, 3)")
        height, width = image_rgb.shape[:2]
        padded, scale_x, scale_y = _resize_and_pad(image_rgb, self.max_side)
        batch = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        outputs = self._run(batch)
        outputs = {k: torch.as_tensor(np.asarray(outputs[k].cpu() if torch.is_tensor(outputs[k]) else outputs[k]))
                   for k in OUTPUT_NAMES}
        found = decode(outputs, score_threshold=self.score_threshold, nms_iou=self.nms_iou)[0]
        scale = torch.tensor([scale_x, scale_y])
        boxes = (found["boxes"].reshape(-1, 2, 2) / scale).reshape(-1, 4)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
        landmarks = found["landmarks"] / scale
        faces = [Detection(tuple(float(v) for v in box), points.numpy().astype(np.float32), float(score))
                 for box, score, points in zip(boxes, found["scores"], landmarks)]
        faces = [f for f in faces if min(f.box[2] - f.box[0], f.box[3] - f.box[1]) >= self.min_face]
        return sorted(faces, key=lambda f: f.score, reverse=True)

    def as_detect_fn(self) -> Callable[[np.ndarray], list[dict]]:
        """Returns a function with the shape facekit.prepare expects (faces as dicts with "landmarks")."""
        def detect(image_rgb: np.ndarray) -> list[dict]:
            return [{"landmarks": f.landmarks, "score": f.score, "box": f.box} for f in self.detect(image_rgb)]
        return detect
