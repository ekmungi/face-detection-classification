"""ONNX export parity and the end-to-end pipeline with small untrained models (offline, CPU)."""
import json

import numpy as np
import pytest
import torch

from facekit import export
from facekit.det_model import FaceDetNet
from facekit.detect import Detection, FaceDetector
from facekit.expression import CLASSES, ExpressionNet
from facekit.gallery import Gallery
from facekit.identity import IResNet
from facekit.pipeline import FaceAnalyzer, softmax

onnxruntime = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(0)
    folder = tmp_path_factory.mktemp("models")
    detector = FaceDetNet("resnet18", pretrained_backbone=False)
    expression = ExpressionNet(backbone="resnet18", pretrained=False, input_size=64)
    embedder = IResNet("iresnet18")
    export.export_detector(detector, folder / "detector.onnx", size=128)
    export.export_classifier(expression, folder / "expression.onnx", 64, "logits")
    export.export_classifier(embedder.eval(), folder / "embedder.onnx", 112, "embedding")
    card = {"detector": {"score_threshold": 0.5}, "identity": {"threshold": 0.3},
            "expression": {"classes": list(CLASSES), "input_size": 64, "temperature": 1.5}}
    (folder / "model_card.json").write_text(json.dumps(card))
    return folder, detector, expression, embedder


def test_onnx_parity_for_all_three_models(model_dir):
    folder, detector, expression, embedder = model_dir
    assert export.parity(detector, folder / "detector.onnx", torch.rand(1, 3, 96, 160)) < 1e-3
    assert export.parity(expression, folder / "expression.onnx", torch.rand(3, 3, 64, 64)) < 1e-3
    assert export.parity(embedder, folder / "embedder.onnx", torch.rand(3, 3, 112, 112)) < 1e-3


def test_detector_onnx_and_torch_agree_on_an_image(model_dir):
    folder, detector, _, _ = model_dir
    image = np.random.default_rng(0).integers(0, 255, (100, 150, 3), dtype=np.uint8)
    a = FaceDetector.from_onnx(folder / "detector.onnx", score_threshold=0.05).detect(image)
    b = FaceDetector.from_torch(detector, score_threshold=0.05).detect(image)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.allclose(x.box, y.box, atol=0.5)


class FixedDetector:
    """A detector stand-in that always finds one face at the template position of a 224-pixel image."""

    def detect(self, image):
        from facekit.align import template_for
        return [Detection((40.0, 40.0, 180.0, 200.0), template_for(224).copy(), 0.99)]


def test_analyzer_returns_expression_and_identity(model_dir):
    folder, _, _, _ = model_dir
    analyzer = FaceAnalyzer.from_model_dir(folder)
    analyzer.detector = FixedDetector()
    image = np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    first = analyzer.analyze(image)
    assert len(first) == 1
    face = first[0]
    assert face.expression in CLASSES
    assert abs(sum(face.expression_probs.values()) - 1.0) < 1e-5
    assert face.identity is None
    assert abs(np.linalg.norm(face.embedding) - 1.0) < 1e-4
    gallery = Gallery(threshold=0.9)
    gallery.enroll("same-person", face.embedding[None])
    analyzer.gallery = gallery
    again = analyzer.analyze(image)[0]
    assert again.identity == "same-person" and again.identity_score > 0.99
    record = again.as_dict()
    assert "embedding" not in record and record["identity"] == "same-person"


def test_softmax_temperature_flattens():
    logits = np.array([[2.0, 0.0]])
    assert softmax(logits, 2.0)[0, 0] < softmax(logits, 1.0)[0, 0]
