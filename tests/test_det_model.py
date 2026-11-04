"""Offline tests for facekit.det_model: network shapes, target encoding, loss, decoding and image-level detection."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from facekit.det_model import (
    STRIDE,
    FaceDetNet,
    collate_targets,
    decode,
    detect_image,
    detection_loss,
    encode_targets,
)

NAN_POINTS = np.full((5, 2), np.nan, dtype=np.float32)


def face_landmarks(box: list[float]) -> np.ndarray:
    """Returns plausible five-point landmarks inside an xyxy box, in template order."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    relative = np.array([[0.3, 0.35], [0.7, 0.35], [0.5, 0.55], [0.35, 0.75], [0.65, 0.75]], dtype=np.float32)
    return relative * np.array([w, h], dtype=np.float32) + np.array([x1, y1], dtype=np.float32)


def perfect_outputs(targets: dict[str, torch.Tensor], height: int, width: int) -> dict[str, torch.Tensor]:
    """Builds head outputs that decode to the encoded faces: logit 20 at the centres, -20 elsewhere."""
    out_h, out_w = height // STRIDE, width // STRIDE
    batch = targets["heatmap"].shape[0]
    heat = torch.full((batch, 1, out_h * out_w), -20.0)
    maps = {name: torch.zeros(batch, channels, out_h * out_w) for name, channels in (("size", 2), ("offset", 2))}
    maps["landmarks"] = torch.zeros(batch, 10, out_h * out_w)
    for image in range(batch):
        valid = targets["mask"][image] > 0
        cells = targets["index"][image][valid]
        heat[image, 0, cells] = 20.0
        for name in ("size", "offset", "landmarks"):
            maps[name][image][:, cells] = targets[name][image][valid].t()
    outputs = {name: value.reshape(batch, -1, out_h, out_w) for name, value in maps.items()}
    outputs["heatmap"] = heat.reshape(batch, 1, out_h, out_w)
    return outputs


def synthetic_face_image(size: int = 128) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """A grey image with one bright rectangle standing in for a face."""
    image = torch.full((1, 3, size, size), 0.2)
    box = [40.0, 36.0, 88.0, 100.0]
    image[:, :, 36:100, 40:88] = 0.9
    return image, np.array([box], dtype=np.float32), face_landmarks(box)[None]


@pytest.mark.parametrize("backbone", ["resnet18", "resnet34", "mobilenet_v3_large"])
def test_forward_shapes_for_each_backbone(backbone: str) -> None:
    model = FaceDetNet(backbone=backbone, pretrained_backbone=False, head_channels=32).eval()
    with torch.no_grad():
        outputs = model(torch.rand(2, 3, 128, 128))
    expected = {"heatmap": 1, "size": 2, "offset": 2, "landmarks": 10}
    assert set(outputs) == set(expected)
    for name, channels in expected.items():
        assert outputs[name].shape == (2, channels, 32, 32)
    # At the start of training (batch-norm in train mode), the heatmap bias of -2.19 gives a face probability of ~0.1.
    with torch.no_grad():
        heatmap = model.train()(torch.rand(2, 3, 128, 128))["heatmap"]
    assert abs(torch.sigmoid(heatmap).mean().item() - 0.1) < 0.05


def test_forward_accepts_uint8_like_float() -> None:
    model = FaceDetNet(pretrained_backbone=False, head_channels=16).eval()
    image = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    with torch.no_grad():
        from_uint8 = model(image)["heatmap"]
        from_float = model(image.float() / 255.0)["heatmap"]
    assert torch.allclose(from_uint8, from_float)


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ValueError):
        FaceDetNet(backbone="vgg16", pretrained_backbone=False)


def test_encode_targets_peak_and_masks() -> None:
    boxes = np.array(
        [[40, 24, 80, 72], [90, 90, 120, 120], [10, 10, 11, 30], [200, 10, 240, 50]], dtype=np.float32
    )  # normal, no landmarks, 1 px wide (skipped), centre outside the map (skipped)
    landmarks = np.stack([face_landmarks(list(boxes[0])), NAN_POINTS, NAN_POINTS, NAN_POINTS])
    target = encode_targets(boxes, landmarks, height=128, width=128)

    assert target["heatmap"].shape == (1, 32, 32)
    assert target["heatmap"][0, 12, 15] == 1.0  # centre (60, 48) / 4
    assert target["heatmap"][0, 26, 26] == 1.0  # centre (105, 105) / 4
    assert int((target["heatmap"] == 1.0).sum()) == 2
    assert target["mask"].tolist()[:3] == [1.0, 1.0, 0.0]
    assert target["mask"].sum() == 2 and target["index"].shape == (1000,)
    assert int(target["index"][0]) == 12 * 32 + 15
    assert torch.allclose(target["size"][0], torch.tensor([math.log(10.0), math.log(12.0)]))
    assert torch.allclose(target["offset"][0], torch.zeros(2))
    assert target["landmark_mask"][0].sum() == 10 and target["landmark_mask"][1].sum() == 0
    assert torch.isfinite(target["landmarks"]).all()


def test_collate_targets_stacks() -> None:
    first = encode_targets(np.array([[0, 0, 20, 20]]), None, 64, 64)
    second = encode_targets(np.zeros((0, 4)), None, 64, 64)
    batch = collate_targets([first, second])
    assert batch["heatmap"].shape == (2, 1, 16, 16) and batch["landmarks"].shape == (2, 1000, 10)


def test_loss_is_finite_for_zero_faces_and_for_a_face() -> None:
    model = FaceDetNet(pretrained_backbone=False, head_channels=16)
    image, boxes, landmarks = synthetic_face_image()
    outputs = model(image)
    empty = collate_targets([encode_targets(np.zeros((0, 4)), None, 128, 128)])
    loss_empty, stats_empty = detection_loss(outputs, empty)
    full = collate_targets([encode_targets(boxes, landmarks, 128, 128)])
    loss_full, stats_full = detection_loss(outputs, full)
    assert torch.isfinite(loss_empty) and torch.isfinite(loss_full)
    assert stats_empty["size"] == 0.0 and stats_empty["landmarks"] == 0.0
    assert set(stats_full) == {"heatmap", "size", "offset", "landmarks", "total"}
    loss_full.backward()


def test_loss_is_finite_in_half_precision_outputs() -> None:
    outputs = {
        "heatmap": torch.full((1, 1, 16, 16), 30.0, dtype=torch.float16),
        "size": torch.zeros(1, 2, 16, 16, dtype=torch.float16),
        "offset": torch.zeros(1, 2, 16, 16, dtype=torch.float16),
        "landmarks": torch.zeros(1, 10, 16, 16, dtype=torch.float16),
    }
    targets = collate_targets([encode_targets(np.array([[10, 10, 40, 40]]), None, 64, 64)])
    loss, _ = detection_loss(outputs, targets)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_loss_decreases_when_training_on_one_image() -> None:
    torch.manual_seed(0)
    model = FaceDetNet(pretrained_backbone=False, head_channels=32).train()
    image, boxes, landmarks = synthetic_face_image()
    images = image.repeat(2, 1, 1, 1)
    targets = collate_targets([encode_targets(boxes, landmarks, 128, 128)] * 2)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        loss, _ = detection_loss(model(images), targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert all(math.isfinite(value) for value in losses)
    assert np.mean(losses[-3:]) < 0.5 * np.mean(losses[:3])


def test_encode_decode_round_trip() -> None:
    height, width = 160, 224
    boxes = np.array([[20.5, 30.25, 70.5, 95.75], [120, 40, 180, 110], [150, 120, 190, 158]], dtype=np.float32)
    landmarks = np.stack([face_landmarks(list(box)) for box in boxes])
    targets = collate_targets([encode_targets(boxes, landmarks, height, width)])
    detections = decode(perfect_outputs(targets, height, width), score_threshold=0.5)[0]

    order = detections["boxes"][:, 0].argsort()
    assert len(order) == 3
    assert torch.allclose(detections["boxes"][order], torch.from_numpy(boxes), atol=1.0)
    assert torch.allclose(detections["landmarks"][order], torch.from_numpy(landmarks), atol=1.0)
    assert (detections["scores"] > 0.99).all()


def test_decode_nms_removes_duplicates() -> None:
    outputs = {
        "heatmap": torch.full((1, 1, 32, 32), -20.0),
        "size": torch.full((1, 2, 32, 32), math.log(40 / STRIDE)),
        "offset": torch.zeros(1, 2, 32, 32),
        "landmarks": torch.zeros(1, 10, 32, 32),
    }
    outputs["heatmap"][0, 0, 10, 10] = 3.0
    outputs["heatmap"][0, 0, 10, 12] = 2.0  # 8 px to the right: same face, IoU about 0.67
    without_nms = decode(outputs, score_threshold=0.3, nms_iou=1.0)[0]
    with_nms = decode(outputs, score_threshold=0.3, nms_iou=0.4)[0]
    assert len(without_nms["boxes"]) == 2
    assert len(with_nms["boxes"]) == 1
    assert torch.allclose(with_nms["boxes"][0], torch.tensor([20.0, 20.0, 60.0, 60.0]))


class PerfectModel(nn.Module):
    """A stand-in detector whose output decodes to fixed faces given in resized-input pixels."""

    def __init__(self, boxes: np.ndarray, landmarks: np.ndarray) -> None:
        super().__init__()
        self.boxes, self.landmarks = boxes, landmarks
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = x.shape[-2:]
        self.input_shapes.append(tuple(x.shape))
        targets = collate_targets([encode_targets(self.boxes, self.landmarks, height, width)])
        return perfect_outputs(targets, height, width)


def test_detect_image_maps_back_to_original_pixels() -> None:
    original_box = [100.0, 60.0, 200.0, 180.0]
    original_landmarks = face_landmarks(original_box)
    image = np.zeros((300, 500, 3), dtype=np.uint8)  # max_side 250 halves it to 150 x 250, padded to 160 x 256
    model = PerfectModel(np.array([original_box], np.float32) / 2, original_landmarks[None] / 2).train()

    faces = detect_image(model, image, max_side=250, score_threshold=0.5)

    assert model.input_shapes == [(1, 3, 160, 256)]
    assert model.training  # the previous mode is restored
    assert len(faces) == 1
    assert np.allclose(faces[0]["box"], original_box, atol=2.0)
    assert np.allclose(faces[0]["landmarks"], original_landmarks, atol=2.0)
    assert 0.99 < faces[0]["score"] <= 1.0


def test_detect_image_runs_a_real_model_on_a_small_image() -> None:
    model = FaceDetNet(pretrained_backbone=False, head_channels=16)
    image = np.random.default_rng(0).integers(0, 256, (90, 130, 3), dtype=np.uint8)
    faces = detect_image(model, image, score_threshold=0.0)
    assert faces == sorted(faces, key=lambda face: face["score"], reverse=True)
    for face in faces:
        assert len(face["box"]) == 4 and np.array(face["landmarks"]).shape == (5, 2)
        assert 0 <= face["box"][0] <= 130 and 0 <= face["box"][3] <= 90
