"""Face embedding network (IResNet) and the margin heads (ArcFace and CosFace) used to train it.

IResNet is the standard backbone of the ArcFace family of face recognition models. The embedding network is trained
from scratch. It maps a 112 x 112 aligned face (RGB values between 0 and 1) to a vector of 512 numbers, and two faces
of the same person get vectors that point in the same direction, which means a high cosine similarity.

The architecture follows the public InsightFace IResNet design (MIT licence). No pretrained weights are used.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# The number of residual blocks in each of the four stages, for every supported network depth.
DEPTHS: dict[str, list[int]] = {
    "iresnet18": [2, 2, 2, 2],
    "iresnet34": [3, 4, 6, 3],
    "iresnet50": [3, 4, 14, 3],
    "iresnet100": [3, 13, 30, 3],
}


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Returns a 3 x 3 convolution that keeps the image size when the stride is 1 and has no bias term."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class IBasicBlock(nn.Module):
    """A pre-activation residual block.

    The main path applies batch normalisation, a convolution, batch normalisation, a PReLU activation (a rectifier
    with a learned slope for negative values), a strided convolution and batch normalisation again. The block then
    adds the shortcut, which is either the input itself or the input passed through the `downsample` module when the
    shape changes.
    """

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        """Builds the layers of the block.

        `inplanes` and `planes` are the numbers of input and output channels. `stride` is the step of the second
        convolution, and a stride of 2 halves the image size. `downsample` reshapes the shortcut to the output shape.
        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the main path added to the shortcut."""
        out = self.bn3(self.conv2(self.prelu(self.bn2(self.conv1(self.bn1(x))))))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return out + shortcut


class IResNet(nn.Module):
    """IResNet embedding network for 112 x 112 input.

    Args:
        depth: One of the names in DEPTHS ("iresnet18", "iresnet34", "iresnet50" or "iresnet100").
        embedding_size: The length of the output vector.
        dropout: The dropout probability applied before the final fully connected layer.
    """

    def __init__(self, depth: str = "iresnet50", embedding_size: int = 512, dropout: float = 0.0):
        """Builds the network and initialises its weights. Raises ValueError when `depth` is not in DEPTHS."""
        super().__init__()
        if depth not in DEPTHS:
            raise ValueError(f"unknown depth {depth!r}; choose from {sorted(DEPTHS)}")
        self.depth = depth
        self.embedding_size = embedding_size
        self.inplanes = 64

        # The first convolution, followed by the four stages of residual blocks. Each stage halves the image size.
        self.conv1 = _conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        blocks = DEPTHS[depth]
        self.layer1 = self._make_layer(64, blocks[0], stride=2)
        self.layer2 = self._make_layer(128, blocks[1], stride=2)
        self.layer3 = self._make_layer(256, blocks[2], stride=2)
        self.layer4 = self._make_layer(512, blocks[3], stride=2)

        # The output stage. It turns the final 7 x 7 feature map into one embedding vector. The scale of the final
        # batch normalisation is fixed at 1 and is not trained.
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Weight initialisation. Convolutions use Kaiming initialisation (random weights scaled to the number of
        # connections of each layer), and the normalisation layers start at the identity transformation.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # The last normalisation layer of every residual block starts at zero, so that each block begins as the
        # identity function.
        for module in self.modules():
            if isinstance(module, IBasicBlock):
                nn.init.constant_(module.bn3.weight, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        """Builds one stage of `blocks` residual blocks with `planes` output channels.

        Only the first block uses the stride, so it is the one that changes the image size. It also gets a 1 x 1
        convolution on its shortcut whenever the shape of the input and the output differ.
        """
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
        """Returns embeddings [B, embedding_size], which are not normalised, for RGB input [B, 3, 112, 112].

        The input values lie between 0 and 1. An input of any other size is first resized to 112 x 112.
        """
        if x.shape[-1] != 112 or x.shape[-2] != 112:
            x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)

        # Rescale the values from the range 0 to 1 to the range -1 to 1, then run the stages in order.
        x = (x - 0.5) / 0.5
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.bn2(x), 1)
        return self.features(self.fc(self.dropout(x)))


class ArcFaceHead(nn.Module):
    """Additive angular margin head (ArcFace).

    The logits (the scores that the loss function compares with the labels) are s * cos(theta + m) for the true class
    and s * cos(theta) for every other class. Here theta is the angle between the embedding and the class centre, s is
    the scale and m is the margin.

    The margin makes the network pull a person's faces into a tight cone, which is what makes cosine similarity a good
    identity score. The head computes in float32 so that it stays stable under mixed precision training.
    """

    def __init__(self, embedding_size: int, num_classes: int, scale: float = 64.0, margin: float = 0.5):
        """Creates one learnable class centre of length `embedding_size` for each of the `num_classes` identities."""
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.normal_(self.weight, std=0.01)

        # Constants for the margin formula. The threshold marks where theta + m would pass 180 degrees, and the
        # fallback value is the matching linear penalty that replaces the formula beyond it.
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

        # Compute cos(theta + m) for the true class from the angle addition formula.
        target = cosine[rows, labels]
        sine = torch.sqrt((1.0 - target * target).clamp_min(0.0))
        with_margin = target * self.cos_m - sine * self.sin_m

        # Past theta = pi - m the margin would make the function non-monotonic, so a linear penalty replaces it there.
        with_margin = torch.where(target > self.threshold, with_margin, target - self.fallback)
        logits = cosine.clone()
        logits[rows, labels] = with_margin
        return logits * self.scale


class CosFaceHead(ArcFaceHead):
    """Additive cosine margin head (CosFace).

    The logits are s * (cos(theta) - m) for the true class and s * cos(theta) for every other class. This head
    converges more reliably than ArcFace in a short schedule from scratch, which is why this project uses it.
    """

    def __init__(self, embedding_size: int, num_classes: int, scale: float = 64.0, margin: float = 0.35):
        """Creates the class centres like ArcFaceHead does, with a smaller default margin."""
        super().__init__(embedding_size, num_classes, scale, margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Returns scaled logits [B, num_classes] with the margin subtracted from each row's true class."""
        cosine = self.cosine(embeddings)
        rows = torch.arange(cosine.shape[0], device=cosine.device)
        logits = cosine.clone()
        logits[rows, labels] = cosine[rows, labels] - self.margin
        return logits * self.scale


class IdentityTrainer(nn.Module):
    """The embedding network and the margin head together, so that one optimizer and one checkpoint hold both."""

    def __init__(self, num_identities: int, depth: str = "iresnet50", embedding_size: int = 512,
                 dropout: float = 0.0, scale: float = 64.0, margin: float = 0.35, loss: str = "cosface"):
        """Builds the network and the head. `loss` is "cosface" or "arcface" and selects the margin head."""
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
