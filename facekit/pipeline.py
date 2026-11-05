"""The face analysis pipeline, which detects faces, aligns them, classifies the expression, embeds and identifies them.

`FaceAnalyzer.from_model_dir(folder)` loads the three ONNX models and `model_card.json` from one folder, so the
pipeline runs on a laptop CPU with no network. ONNX is a file format for sharing trained models. A gallery of enrolled
people is optional.
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

# A neural network seen as a function. It takes a float32 batch [B, 3, S, S] with values between 0 and 1, and returns
# an array [B, K], where K is the number of outputs per image.
Net = Callable[[np.ndarray], np.ndarray]


@dataclasses.dataclass
class FaceResult:
    """Everything the pipeline says about one face."""

    # The box of the face in the image, its landmarks and its detection score.
    detection: Detection

    # The name of the most likely expression, and the probability of every expression class.
    expression: str
    expression_probs: dict[str, float]

    # The name of the matching enrolled person, or None when nobody matches. The score is the cosine similarity of the
    # closest enrolled person, even when it is below the threshold, and 0 when the gallery is missing or empty.
    identity: str | None
    identity_score: float

    # The face embedding that the identity decision is based on, L2-normalised (scaled to a length of 1).
    embedding: np.ndarray

    def as_dict(self, with_embedding: bool = False) -> dict:
        """Returns a JSON-friendly copy of the result.

        The embedding is left out unless `with_embedding` is set, because it is biometric data.
        """
        out = {**self.detection.as_dict(), "expression": self.expression,
               "expression_probs": {k: round(v, 4) for k, v in self.expression_probs.items()},
               "identity": self.identity, "identity_score": round(float(self.identity_score), 4)}
        if with_embedding:
            out["embedding"] = [round(float(v), 6) for v in self.embedding]
        return out


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turns logits [B, K], the raw class scores, into probabilities that add up to 1 in each row.

    A temperature above 1 makes the probabilities less extreme and a temperature below 1 makes them more extreme.
    """
    z = logits / temperature

    # Subtracting the row maximum does not change the result, but it prevents numerical overflow in the exponential.
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _onnx_net(path: Path) -> Net:
    """Loads an ONNX model on the CPU and returns it as a function from an input batch to the first output."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return lambda batch: session.run(None, {name: batch})[0]


def _to_batch(crops: list[np.ndarray]) -> np.ndarray:
    """Stacks RGB uint8 crops [H, W, 3] into one float32 batch [B, 3, H, W] with values between 0 and 1."""
    return np.ascontiguousarray(np.stack(crops).transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


class FaceAnalyzer:
    """Runs the whole pipeline on RGB uint8 images.

    Args:
        detector: A FaceDetector.
        expression_net: A function from a float batch to expression logits.
        embedding_net: A function from a float batch to face embeddings.
        classes: The expression class names, in the order of the model's output.
        expression_size: The side of the aligned crop for the expression model, which is normally 224. The identity
            crop is always 112.
        temperature: The calibration temperature of the expression model, taken from the model card.
        gallery: The enrolled people, or None to report every face as unknown.
    """

    def __init__(self, detector: FaceDetector, expression_net: Net, embedding_net: Net, classes: list[str],
                 expression_size: int = 224, temperature: float = 1.0, gallery: Gallery | None = None):
        """Stores the models and settings. See the class description for the meaning of each argument."""
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
        """Loads detector.onnx, expression.onnx, embedder.onnx and model_card.json from `folder`.

        `gallery` is the path of an optional gallery file, and `max_side` is passed on to the detector. Raises
        FileNotFoundError when a gallery path is given but the file does not exist.
        """
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
        """Returns L2-normalised embeddings of aligned 112 x 112 crops.

        Like the evaluation does, it adds the embedding of the mirrored face to the embedding of each face.
        """
        batch = _to_batch(crops112)

        # The original and the mirrored crops go through the network in one call, which saves a second call.
        both = self.embedding_net(np.concatenate([batch, np.ascontiguousarray(batch[:, :, :, ::-1])]))
        embeddings = both[: len(batch)] + both[len(batch):]
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)

    def embed_largest_face(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Returns the embedding of the largest face in the photo, or None when there is no face.

        Enrolment needs nothing else, so this skips the expression model.
        """
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

        # Cut out each face twice, since the expression model and the identity model need different crop sizes.
        crops_expr = [align.align_face(image_rgb, d.landmarks, self.expression_size) for d in detections]
        crops_id = [align.align_face(image_rgb, d.landmarks, 112) for d in detections]
        probs = softmax(self.expression_net(_to_batch(crops_expr)), self.temperature)
        embeddings = self.embed_crops(crops_id)

        # Combine the expression probabilities and the identity match of each face into one result.
        results = []
        for d, p, e in zip(detections, probs, embeddings):
            name, score = self.gallery.identify(e) if self.gallery is not None else (None, 0.0)
            results.append(FaceResult(d, self.classes[int(np.argmax(p))],
                                      {c: float(v) for c, v in zip(self.classes, p)}, name, score, e))
        return results
