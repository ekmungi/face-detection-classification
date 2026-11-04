"""A CenterFace-style anchor-free face detector: one stride-4 heatmap, box sizes, centre offsets and five landmarks.

The network is trained by us on WIDER FACE. Only the backbone may start from ImageNet weights, which are not trained
for faces. The model takes RGB images in [0, 1] and normalises them itself, so an exported ONNX graph takes the same
input. Every face is represented by the cell of the stride-4 map that holds its box centre:

- ``heatmap`` (1 channel, logits): a Gaussian bump with value 1 at each face centre;
- ``size`` (2): log(box width / stride) and log(box height / stride);
- ``offset`` (2): the sub-pixel position of the centre inside its cell, in [0, 1);
- ``landmarks`` (10): for each of the five points, (point - box centre) divided by the box width (x) and height (y).

Landmark order: image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

STRIDE = 4
MAX_OBJECTS = 1000
HEATMAP_BIAS = -2.19
MIN_FACE_SIDE = 2.0
GAUSSIAN_MIN_OVERLAP = 0.7
MAX_LOG_SIZE = 10.0
DEFAULT_LOSS_WEIGHTS = {"heatmap": 1.0, "size": 0.1, "offset": 1.0, "landmarks": 0.1}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Channels of the stride 4, 8, 16 and 32 feature maps of each supported backbone.
BACKBONE_CHANNELS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "mobilenet_v3_large": (24, 40, 112, 960),
}
# Slices of torchvision's mobilenet_v3_large().features that end at strides 4, 8, 16 and 32.
MOBILENET_STAGE_BOUNDS = ((0, 4), (4, 7), (7, 13), (13, 17))


# --------------------------------------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------------------------------------


def _conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    """Returns a convolution, batch norm and ReLU block that keeps the spatial size."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _build_backbone_stages(backbone: str, pretrained_backbone: bool) -> nn.ModuleList:
    """Builds the four backbone stages that end at strides 4, 8, 16 and 32."""
    if backbone in ("resnet18", "resnet34"):
        weights_enum = (
            torchvision.models.ResNet18_Weights if backbone == "resnet18" else torchvision.models.ResNet34_Weights
        )
        net = getattr(torchvision.models, backbone)(weights=weights_enum.DEFAULT if pretrained_backbone else None)
        stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        return nn.ModuleList([nn.Sequential(stem, net.layer1), net.layer2, net.layer3, net.layer4])
    if backbone == "mobilenet_v3_large":
        weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        features = torchvision.models.mobilenet_v3_large(weights=weights).features
        return nn.ModuleList([features[start:end] for start, end in MOBILENET_STAGE_BOUNDS])
    raise ValueError(f"Unsupported backbone {backbone!r}; choose one of {sorted(BACKBONE_CHANNELS)}")


class _Head(nn.Module):
    """A 3x3 convolution, ReLU and a 1x1 output convolution."""

    def __init__(self, channels: int, out_channels: int, bias_fill: float = 0.0, weight_std: float = 0.001) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(channels, out_channels, 1)
        nn.init.normal_(self.out.weight, std=weight_std)
        nn.init.constant_(self.out.bias, bias_fill)

    def forward(self, x: Tensor) -> Tensor:
        """Returns the head's raw output at the input's resolution."""
        return self.out(self.body(x))


class FaceDetNet(nn.Module):
    """Backbone, FPN-style neck that merges strides 4/8/16/32 into one stride-4 map, and four prediction heads.

    Input: float tensor [B, 3, H, W], RGB in [0, 1] (a uint8 tensor in [0, 255] is also accepted and scaled), with H
    and W multiples of 32. Output: a dict with ``heatmap`` [B,1,H/4,W/4] (logits), ``size`` [B,2,...],
    ``offset`` [B,2,...] and ``landmarks`` [B,10,...].
    """

    def __init__(self, backbone: str = "resnet18", pretrained_backbone: bool = True, head_channels: int = 64) -> None:
        super().__init__()
        if backbone not in BACKBONE_CHANNELS:
            raise ValueError(f"Unsupported backbone {backbone!r}; choose one of {sorted(BACKBONE_CHANNELS)}")
        self.backbone_name = backbone
        self.stages = _build_backbone_stages(backbone, pretrained_backbone)
        channels = BACKBONE_CHANNELS[backbone]
        self.laterals = nn.ModuleList([_conv_bn_relu(c, head_channels, 1) for c in channels])
        self.smooths = nn.ModuleList([_conv_bn_relu(head_channels, head_channels, 3) for _ in channels[:-1]])
        self.heads = nn.ModuleDict(
            {
                "heatmap": _Head(head_channels, 1, bias_fill=HEATMAP_BIAS, weight_std=0.01),
                "size": _Head(head_channels, 2),
                "offset": _Head(head_channels, 2),
                "landmarks": _Head(head_channels, 10),
            }
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self._init_neck()

    def _init_neck(self) -> None:
        """Initialises the neck and the hidden head convolutions; the backbone keeps its own initialisation."""
        for module in list(self.laterals.modules()) + list(self.smooths.modules()):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        for head in self.heads.values():
            nn.init.kaiming_normal_(head.body[0].weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(head.body[0].bias)

    def _features(self, x: Tensor) -> list[Tensor]:
        """Returns the backbone feature maps at strides 4, 8, 16 and 32."""
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def _merge(self, features: list[Tensor]) -> Tensor:
        """Merges the feature maps top-down into one stride-4 map."""
        merged = self.laterals[-1](features[-1])
        for level in range(len(features) - 2, -1, -1):
            skip = self.laterals[level](features[level])
            merged = skip + F.interpolate(merged, size=skip.shape[-2:], mode="nearest")
            merged = self.smooths[level](merged)
        return merged

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Runs the detector on RGB images in [0, 1] and returns the raw stride-4 head outputs."""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = (x - self.mean) / self.std
        merged = self._merge(self._features(x))
        return {name: head(merged) for name, head in self.heads.items()}


# --------------------------------------------------------------------------------------------------------------------
# Training targets
# --------------------------------------------------------------------------------------------------------------------


def gaussian_radius(height: float, width: float, min_overlap: float = GAUSSIAN_MIN_OVERLAP) -> float:
    """CenterNet's radius rule: the largest centre shift that keeps a box of this size above ``min_overlap`` IoU.

    This is the formula of the CenterNet reference code, kept as published (its three roots are not divided by 2a),
    because the published detectors were trained with it.
    """
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1**2 - 4 * c1)) / 2
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2**2 - 16 * c2)) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Splats a Gaussian with peak 1.0 at cell (cx, cy) into ``heatmap`` [H, W] in place, keeping the maximum."""
    sigma = (2 * radius + 1) / 6.0
    ys, xs = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma)).astype(np.float32)
    height, width = heatmap.shape
    left, right = min(cx, radius), min(width - cx, radius + 1)
    top, bottom = min(cy, radius), min(height - cy, radius + 1)
    region = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    patch = kernel[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(region, patch, out=region)


def _as_numpy(array: np.ndarray | Tensor | None, shape_tail: tuple[int, ...]) -> np.ndarray:
    """Converts boxes or landmarks to a float32 numpy array; None becomes an empty array."""
    if array is None:
        return np.zeros((0, *shape_tail), dtype=np.float32)
    if isinstance(array, Tensor):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float32).reshape(-1, *shape_tail)


def encode_targets(
    boxes: np.ndarray | Tensor,
    landmarks: np.ndarray | Tensor | None,
    height: int,
    width: int,
    stride: int = STRIDE,
    max_objects: int = MAX_OBJECTS,
) -> dict[str, Tensor]:
    """Encodes the faces of ONE image (boxes [N,4] xyxy, landmarks [N,5,2] or None, input pixels) as training targets.

    Returns ``heatmap`` [1,H/s,W/s], and per object slot (``max_objects`` of them): ``index`` [M] (flat cell index),
    ``mask`` [M], ``size`` [M,2], ``offset`` [M,2], ``landmarks`` [M,10] and ``landmark_mask`` [M,10] (0 where the
    landmark is NaN). Faces smaller than 2 px on either side, or whose centre lies outside the map, are skipped. Every
    valid face is drawn on the heatmap; only the first ``max_objects`` get regression slots.
    """
    out_h, out_w = height // stride, width // stride
    boxes_np = _as_numpy(boxes, (4,))
    landmarks_np = _as_numpy(landmarks, (5, 2)) if landmarks is not None else np.full((len(boxes_np), 5, 2), np.nan)
    heatmap = np.zeros((out_h, out_w), dtype=np.float32)
    index = np.zeros(max_objects, dtype=np.int64)
    mask = np.zeros(max_objects, dtype=np.float32)
    size = np.zeros((max_objects, 2), dtype=np.float32)
    offset = np.zeros((max_objects, 2), dtype=np.float32)
    lm_target = np.zeros((max_objects, 10), dtype=np.float32)
    lm_mask = np.zeros((max_objects, 10), dtype=np.float32)
    slot = 0
    for box, points in zip(boxes_np, landmarks_np):
        box_w, box_h = float(box[2] - box[0]), float(box[3] - box[1])
        if box_w < MIN_FACE_SIDE or box_h < MIN_FACE_SIDE:
            continue
        cx, cy = float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2
        cell_x, cell_y = math.floor(cx / stride), math.floor(cy / stride)
        if not (0 <= cell_x < out_w and 0 <= cell_y < out_h):
            continue
        radius = max(0, int(gaussian_radius(box_h / stride, box_w / stride)))
        draw_gaussian(heatmap, cell_x, cell_y, radius)
        if slot >= max_objects:
            continue
        index[slot] = cell_y * out_w + cell_x
        mask[slot] = 1.0
        size[slot] = (math.log(box_w / stride), math.log(box_h / stride))
        offset[slot] = (cx / stride - cell_x, cy / stride - cell_y)
        relative = (points - np.array([cx, cy], dtype=np.float32)) / np.array([box_w, box_h], dtype=np.float32)
        valid = np.isfinite(relative).reshape(10)
        lm_target[slot] = np.where(valid, relative.reshape(10), 0.0)
        lm_mask[slot] = valid.astype(np.float32)
        slot += 1
    return {
        "heatmap": torch.from_numpy(heatmap[None]),
        "index": torch.from_numpy(index),
        "mask": torch.from_numpy(mask),
        "size": torch.from_numpy(size),
        "offset": torch.from_numpy(offset),
        "landmarks": torch.from_numpy(lm_target),
        "landmark_mask": torch.from_numpy(lm_mask),
    }


def collate_targets(targets: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Stacks per-image target dicts from :func:`encode_targets` into batch tensors."""
    return {key: torch.stack([target[key] for target in targets]) for key in targets[0]}


# --------------------------------------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------------------------------------


def focal_loss(logits: Tensor, target: Tensor) -> Tensor:
    """CornerNet's penalty-reduced focal loss in float32, normalised by the number of face centres (at least 1)."""
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    target = target.float()
    positive = target.eq(1).float()
    negative = 1.0 - positive
    pos_loss = torch.log(pred) * (1 - pred).pow(2) * positive
    neg_loss = torch.log(1 - pred) * pred.pow(2) * (1 - target).pow(4) * negative
    num_pos = positive.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def gather_at(feature_map: Tensor, index: Tensor) -> Tensor:
    """Gathers [B,C,H,W] values at flat cell indices [B,M] and returns [B,M,C]."""
    batch, channels = feature_map.shape[:2]
    flat = feature_map.reshape(batch, channels, -1)
    gathered = flat.gather(2, index.unsqueeze(1).expand(batch, channels, index.shape[1]))
    return gathered.permute(0, 2, 1)


def masked_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean absolute error over the elements where ``mask`` is 1 (zero when the mask is empty)."""
    mask = mask.float().expand_as(pred)
    total = (torch.abs(pred.float() - target.float()) * mask).sum()
    return total / (mask.sum() + 1e-4)


def detection_loss(
    outputs: dict[str, Tensor], targets: dict[str, Tensor], weights: dict[str, float] | None = None
) -> tuple[Tensor, dict[str, float]]:
    """Returns the weighted total loss and a dict of its float components (``heatmap``, ``size``, ``offset``,
    ``landmarks``, ``total``). Targets are batched (see :func:`collate_targets`) and moved to the outputs' device.
    """
    weights = {**DEFAULT_LOSS_WEIGHTS, **(weights or {})}
    device = outputs["heatmap"].device
    targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
    index = targets["index"]
    object_mask = targets["mask"].unsqueeze(-1)
    parts = {
        "heatmap": focal_loss(outputs["heatmap"], targets["heatmap"]),
        "size": masked_l1(gather_at(outputs["size"], index), targets["size"], object_mask),
        "offset": masked_l1(gather_at(outputs["offset"], index), targets["offset"], object_mask),
        "landmarks": masked_l1(
            gather_at(outputs["landmarks"], index), targets["landmarks"], targets["landmark_mask"] * object_mask
        ),
    }
    total = sum(weights[name] * value for name, value in parts.items())
    stats = {name: float(value.detach()) for name, value in parts.items()}
    stats["total"] = float(total.detach())
    return total, stats


# --------------------------------------------------------------------------------------------------------------------
# Decoding and inference
# --------------------------------------------------------------------------------------------------------------------


def _decode_one(
    heat: Tensor, outputs: dict[str, Tensor], image: int, score_threshold: float, top_k: int, stride: int
) -> dict[str, Tensor]:
    """Decodes the peaks of one image's heatmap (already peak-filtered) into boxes, scores and landmarks."""
    width = heat.shape[-1]
    scores, cells = heat[image].reshape(-1).topk(min(top_k, heat[image].numel()))
    keep = scores > score_threshold
    scores, cells = scores[keep], cells[keep]
    ys, xs = (cells // width).float(), (cells % width).float()
    size = outputs["size"][image].reshape(2, -1)[:, cells].float().clamp(max=MAX_LOG_SIZE).exp() * stride
    offset = outputs["offset"][image].reshape(2, -1)[:, cells].float()
    cx, cy = (xs + offset[0]) * stride, (ys + offset[1]) * stride
    box_w, box_h = size[0], size[1]
    boxes = torch.stack([cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2], dim=1)
    relative = outputs["landmarks"][image].reshape(10, -1)[:, cells].float().t().reshape(-1, 5, 2)
    landmarks = torch.stack([cx, cy], dim=1)[:, None, :] + relative * torch.stack([box_w, box_h], dim=1)[:, None, :]
    return {"boxes": boxes, "scores": scores, "landmarks": landmarks}


@torch.no_grad()
def decode(
    outputs: dict[str, Tensor],
    score_threshold: float = 0.3,
    nms_iou: float = 0.4,
    top_k: int = 1000,
    stride: int = STRIDE,
) -> list[dict[str, Tensor]]:
    """Turns raw head outputs into detections per image: ``boxes`` [K,4] xyxy, ``scores`` [K], ``landmarks`` [K,5,2],
    all in input pixels and sorted by score. Peaks are local maxima of a 3x3 window, followed by NMS.
    """
    heat = torch.sigmoid(outputs["heatmap"].float())
    peaks = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
    heat = heat * (peaks == heat).float()
    results = []
    for image in range(heat.shape[0]):
        detections = _decode_one(heat, outputs, image, score_threshold, top_k, stride)
        keep = torchvision.ops.nms(detections["boxes"], detections["scores"], nms_iou)
        results.append({key: value[keep] for key, value in detections.items()})
    return results


def _resize_and_pad(image: np.ndarray, max_side: int, multiple: int = 32) -> tuple[np.ndarray, float, float]:
    """Shrinks an image so its longest side is <= max_side and pads it bottom/right to a multiple of ``multiple``.

    Returns the padded image and the x and y scale factors from original to resized pixels.
    """
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    if (new_w, new_h) != (width, height):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_h, pad_w = (-new_h) % multiple, (-new_w) % multiple
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image, new_w / width, new_h / height


@torch.inference_mode()
def detect_image(
    model: nn.Module,
    image_rgb_uint8: np.ndarray,
    device: str | torch.device = "cpu",
    max_side: int = 1280,
    score_threshold: float = 0.5,
    nms_iou: float = 0.4,
) -> list[dict]:
    """Detects faces in one RGB uint8 image [H,W,3] and returns them sorted by score, in original pixels.

    Each face is ``{"box": [x1,y1,x2,y2], "score": float, "landmarks": [[x,y]]*5}``. The model runs in eval mode and
    its previous train/eval mode is restored afterwards. The model must already be on ``device``.
    """
    if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected an RGB image [H,W,3], got shape {image_rgb_uint8.shape}")
    height, width = image_rgb_uint8.shape[:2]
    padded, scale_x, scale_y = _resize_and_pad(image_rgb_uint8, max_side)
    tensor = torch.from_numpy(np.ascontiguousarray(padded)).permute(2, 0, 1)[None].to(device).float() / 255.0
    was_training = model.training
    model.eval()
    try:
        detections = decode(model(tensor), score_threshold=score_threshold, nms_iou=nms_iou)[0]
    finally:
        model.train(was_training)
    scale = torch.tensor([scale_x, scale_y], device=detections["boxes"].device)
    boxes = (detections["boxes"].reshape(-1, 2, 2) / scale).reshape(-1, 4)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    landmarks = detections["landmarks"] / scale
    faces = [
        {"box": box.tolist(), "score": float(score), "landmarks": points.tolist()}
        for box, score, points in zip(boxes.cpu(), detections["scores"].cpu(), landmarks.cpu())
    ]
    faces.sort(key=lambda face: face["score"], reverse=True)
    return faces
