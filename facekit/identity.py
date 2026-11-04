"""Face embedding network (IResNet, the ArcFace family's standard backbone) and the ArcFace margin head.

The embedding network is trained from scratch; it maps a 112 x 112 aligned face (RGB in [0, 1]) to a 512-number vector,
and two faces of the same person have vectors pointing the same way (high cosine similarity). The architecture follows
the public InsightFace IResNet design (MIT licence); no weights are taken from anywhere.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

DEPTHS: dict[str, list[int]] = {
    "iresnet18": [2, 2, 2, 2],
    "iresnet34": [3, 4, 6, 3],
    "iresnet50": [3, 4, 14, 3],
    "iresnet100": [3, 13, 30, 3],
}


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class IBasicBlock(nn.Module):
    """Pre-activation residual block: BN, conv, BN, PReLU, conv (strided), BN, plus the shortcut."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn3(self.conv2(self.prelu(self.bn2(self.conv1(self.bn1(x))))))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return out + shortcut


class IResNet(nn.Module):
    """IResNet embedding network for 112 x 112 input.

    Args:
        depth: one of DEPTHS ("iresnet18", "iresnet34", "iresnet50", "iresnet100").
        embedding_size: length of the output vector.
        dropout: dropout before the final fully connected layer.
    """

    def __init__(self, depth: str = "iresnet50", embedding_size: int = 512, dropout: float = 0.0):
        super().__init__()
        if depth not in DEPTHS:
            raise ValueError(f"unknown depth {depth!r}; choose from {sorted(DEPTHS)}")
        self.depth = depth
        self.embedding_size = embedding_size
        self.inplanes = 64
        self.conv1 = _conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        blocks = DEPTHS[depth]
        self.layer1 = self._make_layer(64, blocks[0], stride=2)
        self.layer2 = self._make_layer(128, blocks[1], stride=2)
        self.layer3 = self._make_layer(256, blocks[2], stride=2)
        self.layer4 = self._make_layer(512, blocks[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        for module in self.modules():
            if isinstance(module, IBasicBlock):
                nn.init.constant_(module.bn3.weight, 0)   # each residual block starts as the identity

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-5),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [IBasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns embeddings [B, embedding_size] (not normalised) for RGB input in [0, 1] of shape [B, 3, 112, 112]."""
        if x.shape[-1] != 112 or x.shape[-2] != 112:
            x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.bn2(x), 1)
        return self.features(self.fc(self.dropout(x)))


class ArcFaceHead(nn.Module):
    """Additive angular margin head (ArcFace): logits = s * cos(theta + m) for the true class, s * cos(theta) otherwise.

    The margin makes the network pull a person's faces into a tight cone, which is what makes cosine similarity a good
    identity score. Computed in float32 for stability under mixed precision.
    """

    def __init__(self, embedding_size: int, num_classes: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.normal_(self.weight, std=0.01)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.fallback = math.sin(math.pi - margin) * margin

    def cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns the cosine similarity [B, num_classes] between each embedding and each class centre."""
        return F.linear(F.normalize(embeddings.float()), F.normalize(self.weight.float())).clamp(-1 + 1e-7, 1 - 1e-7)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Returns scaled logits [B, num_classes] with the margin applied to each row's true class."""
        cosine = self.cosine(embeddings)
        rows = torch.arange(cosine.shape[0], device=cosine.device)
        target = cosine[rows, labels]
        sine = torch.sqrt((1.0 - target * target).clamp_min(0.0))
        with_margin = target * self.cos_m - sine * self.sin_m
        # Past theta = pi - m the margin would make the function non-monotonic; use a linear penalty there instead.
        with_margin = torch.where(target > self.threshold, with_margin, target - self.fallback)
        logits = cosine.clone()
        logits[rows, labels] = with_margin
        return logits * self.scale


class CosFaceHead(ArcFaceHead):
    """Additive cosine margin head (CosFace): logits = s * (cos(theta) - m) for the true class, s * cos(theta) otherwise.

    It converges more reliably than ArcFace in a short schedule from scratch, which is why SPEC-001 chose it.
    """

    def __init__(self, embedding_size: int, num_classes: int, scale: float = 64.0, margin: float = 0.35):
        super().__init__(embedding_size, num_classes, scale, margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = self.cosine(embeddings)
        rows = torch.arange(cosine.shape[0], device=cosine.device)
        logits = cosine.clone()
        logits[rows, labels] = cosine[rows, labels] - self.margin
        return logits * self.scale


class IdentityTrainer(nn.Module):
    """The embedding network and the margin head together, so one optimizer and one checkpoint hold both."""

    def __init__(self, num_identities: int, depth: str = "iresnet50", embedding_size: int = 512,
                 dropout: float = 0.0, scale: float = 64.0, margin: float = 0.35, loss: str = "cosface"):
        super().__init__()
        self.embedder = IResNet(depth, embedding_size, dropout)
        head = {"cosface": CosFaceHead, "arcface": ArcFaceHead}[loss]
        self.head = head(embedding_size, num_identities, scale, margin)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, accuracy of the nearest class centre) for a batch."""
        embeddings = self.embedder(images)
        logits = self.head(embeddings, labels)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            accuracy = (self.head.cosine(embeddings).argmax(1) == labels).float().mean()
        return loss, accuracy
