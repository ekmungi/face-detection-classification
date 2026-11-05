"""Tests facekit.det_model: network shapes, target encoding, loss, decoding and detection on whole images.

The tests run offline. The detector predicts, for each cell of a coarse grid over the image, the chance that a face
centre lies in that cell (the heatmap), the face size, a small offset inside the cell and five landmark points. The
STRIDE is the number of image pixels covered by one grid cell along each side. A box is written as xyxy, which means
the left, top, right and bottom pixel positions. IoU (intersection over union) measures how much two boxes overlap,
and NMS (non-maximum suppression) removes the lower-scoring one of two boxes that overlap too much.
"""
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

# Landmarks for a face that has none. The values are NaN (not a number), which the target encoding treats as missing.
NAN_POINTS = np.full((5, 2), np.nan, dtype=np.float32)


def face_landmarks(box: list[float]) -> np.ndarray:
    """Returns plausible five-point landmarks inside an xyxy box, in template order.

    The five points are the two eyes, the nose and the two mouth corners. Each point sits at a fixed fraction of the
    width and height of the box.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    relative = np.array([[0.3, 0.35], [0.7, 0.35], [0.5, 0.55], [0.35, 0.75], [0.65, 0.75]], dtype=np.float32)
    return relative * np.array([w, h], dtype=np.float32) + np.array([x1, y1], dtype=np.float32)


def perfect_outputs(targets: dict[str, torch.Tensor], height: int, width: int) -> dict[str, torch.Tensor]:
    """Builds head outputs that decode to the faces in the encoded targets.

    The heatmap logit (the score before it is turned into a probability) is 20 at each face centre and -20 everywhere
    else. The size, offset and landmark maps hold the encoded values at the face centres and zero elsewhere.
    """
    out_h, out_w = height // STRIDE, width // STRIDE
    batch = targets["heatmap"].shape[0]

    # The maps are first built with the grid flattened into one dimension, so that cells can be set by their index.
    heat = torch.full((batch, 1, out_h * out_w), -20.0)
    maps = {name: torch.zeros(batch, channels, out_h * out_w) for name, channels in (("size", 2), ("offset", 2))}
    maps["landmarks"] = torch.zeros(batch, 10, out_h * out_w)
    for image in range(batch):
        # Only the used face slots count, and each one gives the index of the grid cell that holds its centre.
        valid = targets["mask"][image] > 0
        cells = targets["index"][image][valid]
        heat[image, 0, cells] = 20.0
        for name in ("size", "offset", "landmarks"):
            maps[name][image][:, cells] = targets[name][image][valid].t()

    # The flattened grid is turned back into the two-dimensional grid that the model outputs have.
    outputs = {name: value.reshape(batch, -1, out_h, out_w) for name, value in maps.items()}
    outputs["heatmap"] = heat.reshape(batch, 1, out_h, out_w)
    return outputs


def synthetic_face_image(size: int = 128) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Returns (image, boxes, landmarks) for a grey image with one bright rectangle that stands in for a face.

    The image is a float tensor of shape [1, 3, size, size]. The boxes have shape [1, 4] in xyxy pixels, and the
    landmarks have shape [1, 5, 2].
    """
    image = torch.full((1, 3, size, size), 0.2)
    box = [40.0, 36.0, 88.0, 100.0]
    image[:, :, 36:100, 40:88] = 0.9
    return image, np.array([box], dtype=np.float32), face_landmarks(box)[None]


@pytest.mark.parametrize("backbone", ["resnet18", "resnet34", "mobilenet_v3_large"])
def test_forward_shapes_for_each_backbone(backbone: str) -> None:
    """Every backbone gives four output maps with the expected channels.

    Each map is a grid with a quarter of the image's height and width.
    """
    model = FaceDetNet(backbone=backbone, pretrained_backbone=False, head_channels=32).eval()
    with torch.no_grad():
        outputs = model(torch.rand(2, 3, 128, 128))
    expected = {"heatmap": 1, "size": 2, "offset": 2, "landmarks": 10}
    assert set(outputs) == set(expected)
    for name, channels in expected.items():
        assert outputs[name].shape == (2, channels, 32, 32)

    # At the start of training the heatmap bias of -2.19 gives a face probability of about 0.1. The model is put in
    # training mode first, because batch normalisation then uses the statistics of the batch.
    with torch.no_grad():
        heatmap = model.train()(torch.rand(2, 3, 128, 128))["heatmap"]
    assert abs(torch.sigmoid(heatmap).mean().item() - 0.1) < 0.05


def test_forward_accepts_uint8_like_float() -> None:
    """An integer image with values from 0 to 255 gives the same output as the same image scaled to the range 0 to 1."""
    model = FaceDetNet(pretrained_backbone=False, head_channels=16).eval()
    image = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    with torch.no_grad():
        from_uint8 = model(image)["heatmap"]
        from_float = model(image.float() / 255.0)["heatmap"]
    assert torch.allclose(from_uint8, from_float)


def test_unknown_backbone_is_rejected() -> None:
    """A backbone name that the model does not offer raises a ValueError."""
    with pytest.raises(ValueError):
        FaceDetNet(backbone="vgg16", pretrained_backbone=False)


def test_encode_targets_peak_and_masks() -> None:
    """Encoding puts a peak of 1 at each usable face centre and marks which face slots and landmarks are used."""
    # Four boxes on a 128 x 128 image. The first is a normal face. The second has no landmarks. The third is only 1
    # pixel wide, so it is skipped. The fourth has its centre outside the image, so it is skipped too.
    boxes = np.array(
        [[40, 24, 80, 72], [90, 90, 120, 120], [10, 10, 11, 30], [200, 10, 240, 50]], dtype=np.float32
    )
    landmarks = np.stack([face_landmarks(list(boxes[0])), NAN_POINTS, NAN_POINTS, NAN_POINTS])
    target = encode_targets(boxes, landmarks, height=128, width=128)

    # The stride is 4 pixels, so the image side of 128 pixels gives a grid of 32 x 32 cells.
    assert target["heatmap"].shape == (1, 32, 32)

    # The first face has its centre at (60, 48) pixels, which is column 15 and row 12. The second face has its centre
    # at (105, 105) pixels, which is column 26 and row 26. Only these two peaks exist.
    assert target["heatmap"][0, 12, 15] == 1.0
    assert target["heatmap"][0, 26, 26] == 1.0
    assert int((target["heatmap"] == 1.0).sum()) == 2

    # Two of the 1000 face slots are used, the first two in the order of the boxes. The index of the first slot is
    # the position of its grid cell when the grid is read row by row.
    assert target["mask"].tolist()[:3] == [1.0, 1.0, 0.0]
    assert target["mask"].sum() == 2 and target["index"].shape == (1000,)
    assert int(target["index"][0]) == 12 * 32 + 15

    # The first face is 40 x 48 pixels, which is 10 x 12 grid cells. The size is stored as the logarithm of that. The
    # face centre lies exactly on a grid position, so the offset is zero.
    assert torch.allclose(target["size"][0], torch.tensor([math.log(10.0), math.log(12.0)]))
    assert torch.allclose(target["offset"][0], torch.zeros(2))

    # Only the first face has landmarks, which are 10 numbers, and the second face has none.
    assert target["landmark_mask"][0].sum() == 10 and target["landmark_mask"][1].sum() == 0
    assert torch.isfinite(target["landmarks"]).all()


def test_collate_targets_stacks() -> None:
    """Collating the targets of two images, one with a face and one with none, stacks them into one batch."""
    first = encode_targets(np.array([[0, 0, 20, 20]]), None, 64, 64)
    second = encode_targets(np.zeros((0, 4)), None, 64, 64)
    batch = collate_targets([first, second])
    assert batch["heatmap"].shape == (2, 1, 16, 16) and batch["landmarks"].shape == (2, 1000, 10)


def test_loss_is_finite_for_zero_faces_and_for_a_face() -> None:
    """The loss is a finite number for an image without faces and for an image with one face, and it can be trained."""
    model = FaceDetNet(pretrained_backbone=False, head_channels=16)
    image, boxes, landmarks = synthetic_face_image()
    outputs = model(image)

    # With no face in the target, the size and landmark losses are zero, because there is nothing to measure.
    empty = collate_targets([encode_targets(np.zeros((0, 4)), None, 128, 128)])
    loss_empty, stats_empty = detection_loss(outputs, empty)

    # With one face in the target, the statistics hold one entry for each loss part and for the total.
    full = collate_targets([encode_targets(boxes, landmarks, 128, 128)])
    loss_full, stats_full = detection_loss(outputs, full)
    assert torch.isfinite(loss_empty) and torch.isfinite(loss_full)
    assert stats_empty["size"] == 0.0 and stats_empty["landmarks"] == 0.0
    assert set(stats_full) == {"heatmap", "size", "offset", "landmarks", "total"}

    # Gradients can be computed from the loss.
    loss_full.backward()


def test_loss_is_finite_in_half_precision_outputs() -> None:
    """Half-precision outputs, even with a very large heatmap logit, still give a finite float32 loss."""
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
    """Thirty training steps on a single repeated image lower the loss to less than half of its starting value."""
    torch.manual_seed(0)
    model = FaceDetNet(pretrained_backbone=False, head_channels=32).train()
    image, boxes, landmarks = synthetic_face_image()
    images = image.repeat(2, 1, 1, 1)
    targets = collate_targets([encode_targets(boxes, landmarks, 128, 128)] * 2)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    # Each step computes the loss, clears the old gradients, computes new ones and updates the weights.
    for _ in range(30):
        loss, _ = detection_loss(model(images), targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    # The average of the last three losses is compared with the average of the first three, to reduce the noise.
    assert all(math.isfinite(value) for value in losses)
    assert np.mean(losses[-3:]) < 0.5 * np.mean(losses[:3])


def test_encode_decode_round_trip() -> None:
    """Encoding faces into targets and decoding perfect outputs for them gives back the same boxes and landmarks."""
    height, width = 160, 224
    boxes = np.array([[20.5, 30.25, 70.5, 95.75], [120, 40, 180, 110], [150, 120, 190, 158]], dtype=np.float32)
    landmarks = np.stack([face_landmarks(list(box)) for box in boxes])
    targets = collate_targets([encode_targets(boxes, landmarks, height, width)])
    detections = decode(perfect_outputs(targets, height, width), score_threshold=0.5)[0]

    # The detections are sorted by the left edge of their box, so that they can be compared with the input boxes.
    # The boxes and landmarks must agree to within one pixel.
    order = detections["boxes"][:, 0].argsort()
    assert len(order) == 3
    assert torch.allclose(detections["boxes"][order], torch.from_numpy(boxes), atol=1.0)
    assert torch.allclose(detections["landmarks"][order], torch.from_numpy(landmarks), atol=1.0)
    assert (detections["scores"] > 0.99).all()


def test_decode_nms_removes_duplicates() -> None:
    """Decoding keeps two overlapping detections of one face without NMS and only the stronger one with NMS."""
    # Every cell predicts a 40 pixel box with zero offset and zero landmark outputs. The heatmap is low except at two
    # nearby cells.
    outputs = {
        "heatmap": torch.full((1, 1, 32, 32), -20.0),
        "size": torch.full((1, 2, 32, 32), math.log(40 / STRIDE)),
        "offset": torch.zeros(1, 2, 32, 32),
        "landmarks": torch.zeros(1, 10, 32, 32),
    }
    outputs["heatmap"][0, 0, 10, 10] = 3.0

    # The second peak is 8 pixels to the right of the first one. It is the same face, with an IoU of about 0.67.
    outputs["heatmap"][0, 0, 10, 12] = 2.0

    # An NMS limit of 1.0 removes nothing, and a limit of 0.4 removes the weaker box.
    without_nms = decode(outputs, score_threshold=0.3, nms_iou=1.0)[0]
    with_nms = decode(outputs, score_threshold=0.3, nms_iou=0.4)[0]
    assert len(without_nms["boxes"]) == 2
    assert len(with_nms["boxes"]) == 1

    # The box that remains is the one around the stronger peak.
    assert torch.allclose(with_nms["boxes"][0], torch.tensor([20.0, 20.0, 60.0, 60.0]))


class PerfectModel(nn.Module):
    """A stand-in detector whose output decodes to fixed faces given in resized-input pixels."""

    def __init__(self, boxes: np.ndarray, landmarks: np.ndarray) -> None:
        """Stores the faces to report and starts an empty list of the input shapes that the model receives."""
        super().__init__()
        self.boxes, self.landmarks = boxes, landmarks
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Records the input shape and returns outputs that decode to the stored faces at that input size."""
        height, width = x.shape[-2:]
        self.input_shapes.append(tuple(x.shape))
        targets = collate_targets([encode_targets(self.boxes, self.landmarks, height, width)])
        return perfect_outputs(targets, height, width)


def test_detect_image_maps_back_to_original_pixels() -> None:
    """Detection resizes the image, runs the model and maps the faces back to the pixels of the original image."""
    original_box = [100.0, 60.0, 200.0, 180.0]
    original_landmarks = face_landmarks(original_box)

    # The image is 300 x 500 pixels. A longest side of 250 halves it to 150 x 250, and it is then padded to 160 x 256.
    # The stand-in model reports the faces at half size, as it would find them in the resized image.
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    model = PerfectModel(np.array([original_box], np.float32) / 2, original_landmarks[None] / 2).train()

    faces = detect_image(model, image, max_side=250, score_threshold=0.5)

    assert model.input_shapes == [(1, 3, 160, 256)]

    # The model was in training mode before the call, and detection puts it back in that mode afterwards.
    assert model.training
    assert len(faces) == 1
    assert np.allclose(faces[0]["box"], original_box, atol=2.0)
    assert np.allclose(faces[0]["landmarks"], original_landmarks, atol=2.0)
    assert 0.99 < faces[0]["score"] <= 1.0


def test_detect_image_runs_a_real_model_on_a_small_image() -> None:
    """A real untrained model on a noise image gives faces sorted by score, with box edges inside the image.

    Only the left and bottom edge of each box are checked.
    """
    model = FaceDetNet(pretrained_backbone=False, head_channels=16)
    image = np.random.default_rng(0).integers(0, 256, (90, 130, 3), dtype=np.uint8)

    # A score threshold of 0 drops no face for a low score, so that there are faces whose order and shapes can be
    # checked.
    faces = detect_image(model, image, score_threshold=0.0)
    assert faces == sorted(faces, key=lambda face: face["score"], reverse=True)
    for face in faces:
        assert len(face["box"]) == 4 and np.array(face["landmarks"]).shape == (5, 2)
        assert 0 <= face["box"][0] <= 130 and 0 <= face["box"][3] <= 90
