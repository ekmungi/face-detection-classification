"""The face analysis pipeline: detect, align, classify the expression, embed and identify.

`FaceAnalyzer.from_model_dir(folder)` loads the three ONNX models and `model_card.json` from one folder, so the
pipeline runs on a laptop CPU with no network. A gallery of enrolled people is optional.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable

import numpy as np

from facekit import align
from facekit.detect import Detection, FaceDetector
from facekit.gallery import Gallery

Net = Callable[[np.ndarray], np.ndarray]   # float32 [B, 3, S, S] in [0, 1] -> [B, K]


@dataclasses.dataclass
class FaceResult:
    """Everything the pipeline says about one face."""

    detection: Detection
    expression: str
    expression_probs: dict[str, float]
    identity: str | None
    identity_score: float
    embedding: np.ndarray

    def as_dict(self, with_embedding: bool = False) -> dict:
        """Returns a JSON-friendly copy; the embedding is left out unless asked for (it is biometric data)."""
        out = {**self.detection.as_dict(), "expression": self.expression,
               "expression_probs": {k: round(v, 4) for k, v in self.expression_probs.items()},
               "identity": self.identity, "identity_score": round(float(self.identity_score), 4)}
        if with_embedding:
            out["embedding"] = [round(float(v), 6) for v in self.embedding]
        return out


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _onnx_net(path: Path) -> Net:
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return lambda batch: session.run(None, {name: batch})[0]


def _to_batch(crops: list[np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(np.stack(crops).transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


class FaceAnalyzer:
    """Runs the whole pipeline on RGB uint8 images.

    Args:
        detector: a FaceDetector.
        expression_net, embedding_net: functions from float batches to logits / embeddings.
        classes: expression class names in the model's output order.
        expression_size: side of the aligned crop for the expression model (224); the identity crop is 112.
        temperature: the expression model's calibration temperature (from the model card).
        gallery: enrolled people, or None to report every face as unknown.
    """

    def __init__(self, detector: FaceDetector, expression_net: Net, embedding_net: Net, classes: list[str],
                 expression_size: int = 224, temperature: float = 1.0, gallery: Gallery | None = None):
        self.detector = detector
        self.expression_net = expression_net
        self.embedding_net = embedding_net
        self.classes = list(classes)
        self.expression_size = expression_size
        self.temperature = temperature
        self.gallery = gallery

    @classmethod
    def from_model_dir(cls, folder: str | Path, gallery: str | Path | None = None, max_side: int = 1280
                       ) -> "FaceAnalyzer":
        """Loads detector.onnx, expression.onnx, embedder.onnx and model_card.json from `folder`."""
        folder = Path(folder)
        card = json.loads((folder / "model_card.json").read_text(encoding="utf-8"))
        detector = FaceDetector.from_onnx(folder / "detector.onnx", score_threshold=card["detector"]["score_threshold"],
                                          max_side=max_side)
        if gallery and not Path(gallery).exists():
            raise FileNotFoundError(f"gallery not found: {gallery} (enrol people first with `facekit enroll`)")
        loaded = Gallery.load(gallery) if gallery else None
        return cls(detector, _onnx_net(folder / "expression.onnx"), _onnx_net(folder / "embedder.onnx"),
                   card["expression"]["classes"], card["expression"]["input_size"],
                   card["expression"].get("temperature", 1.0), loaded)

    def embed_crops(self, crops112: list[np.ndarray]) -> np.ndarray:
        """Returns L2-normalised embeddings, adding the mirrored face's embedding as in evaluation."""
        batch = _to_batch(crops112)
        both = self.embedding_net(np.concatenate([batch, np.ascontiguousarray(batch[:, :, :, ::-1])]))  # one call
        embeddings = both[: len(batch)] + both[len(batch):]
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)

    def embed_largest_face(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Returns the embedding of the largest face in the photo, or None; enrolment needs nothing else."""
        detections = self.detector.detect(image_rgb)
        if not detections:
            return None
        largest = max(detections, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        return self.embed_crops([align.align_face(image_rgb, largest.landmarks, 112)])[0]

    def analyze(self, image_rgb: np.ndarray) -> list[FaceResult]:
        """Returns one FaceResult per detected face, highest detection score first."""
        detections = self.detector.detect(image_rgb)
        if not detections:
            return []
        crops_expr = [align.align_face(image_rgb, d.landmarks, self.expression_size) for d in detections]
        crops_id = [align.align_face(image_rgb, d.landmarks, 112) for d in detections]
        probs = softmax(self.expression_net(_to_batch(crops_expr)), self.temperature)
        embeddings = self.embed_crops(crops_id)
        results = []
        for d, p, e in zip(detections, probs, embeddings):
            name, score = self.gallery.identify(e) if self.gallery is not None else (None, 0.0)
            results.append(FaceResult(d, self.classes[int(np.argmax(p))],
                                      {c: float(v) for c, v in zip(self.classes, p)}, name, score, e))
        return results
