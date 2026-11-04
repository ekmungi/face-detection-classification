"""Expression classifier: an ImageNet-pretrained backbone fine-tuned on aligned face crops, plus the EmoNet-Face label mapping.

The network takes RGB values in [0, 1] and normalises them itself, so the ONNX export needs no preprocessing beyond
alignment and scaling to [0, 1].
"""
from __future__ import annotations

import ast
from collections import Counter

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

CLASSES: tuple[str, ...] = ("neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt")

# EmoNet-Face category names (the part after "Group|" in the HQ set) mapped to the eight classes. SPEC-001 records why.
EMONET_TO_CLASS: dict[str, str] = {
    "Amusement": "happy",
    "Elation": "happy",
    "Pleasure/Ecstasy": "happy",
    "Contentment": "happy",
    "Sadness": "sad",
    "Disappointment": "sad",
    "Astonishment/Surprise": "surprise",
    "Fear": "fear",
    "Disgust": "disgust",
    "Anger": "anger",
    "Contempt": "contempt",
    "Emotional Numbness": "neutral",
    "Concentration": "neutral",
    "Contemplation": "neutral",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _short_name(category: str) -> str:
    """Strips the group prefix that the HQ set uses ("Negative High-Energy Emotions|Anger" -> "Anger")."""
    return category.split("|")[-1].strip()


def emonet_binary_target(label_text: str) -> tuple[str | None, bool]:
    """Reads one EmoNet-Face Binary label and returns (target category, confirmed).

    The label is a list of {annotator: {category: bool}} entries. The target is the category rated most often (the
    others are control questions); it is confirmed when a strict majority of its ratings are True. A tie between
    categories gives (None, False).
    """
    entries = ast.literal_eval(label_text) if isinstance(label_text, str) else label_text
    votes: dict[str, list[bool]] = {}
    for entry in entries:
        for ratings in entry.values():
            for category, present in ratings.items():
                votes.setdefault(_short_name(category), []).append(bool(present))
    if not votes:
        return None, False
    counts = Counter({category: len(v) for category, v in votes.items()}).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None, False
    target = counts[0][0]
    ratings = votes[target]
    return target, sum(ratings) * 2 > len(ratings)


def emonet_binary_class(label_text: str) -> int | None:
    """Returns the class index of a confirmed, mapped EmoNet-Face Binary image, or None when it is not used."""
    target, confirmed = emonet_binary_target(label_text)
    if not confirmed or target not in EMONET_TO_CLASS:
        return None
    return CLASSES.index(EMONET_TO_CLASS[target])


def emonet_hq_class(label_text: str, min_intensity: float = 3.0, neutral_max: float = 1.0) -> int | None:
    """Turns EmoNet-Face HQ expert intensity ratings (0 to 7) into one class index, or None when ambiguous.

    For each class the score is the highest mean rating among its member categories, averaged over annotators. The
    image gets the top non-neutral class when that score is at least `min_intensity` and is clearly above the
    second (by at least 1.0); it gets "neutral" when every non-neutral class scores at most `neutral_max`.
    """
    entries = ast.literal_eval(label_text) if isinstance(label_text, str) else label_text
    sums: dict[str, float] = {}
    count = 0
    for entry in entries:
        for ratings in entry.values():
            count += 1
            for category, value in ratings.items():
                name = _short_name(category)
                sums[name] = sums.get(name, 0.0) + float(value)
    if count == 0:
        return None
    means = {name: total / count for name, total in sums.items()}
    scores: dict[str, float] = {}
    for category, cls in EMONET_TO_CLASS.items():
        if category in means:
            scores[cls] = max(scores.get(cls, 0.0), means[category])
    emotional = sorted(((v, k) for k, v in scores.items() if k != "neutral"), reverse=True)
    if not emotional:
        return None
    if emotional[0][0] <= neutral_max:
        return CLASSES.index("neutral")
    second = emotional[1][0] if len(emotional) > 1 else 0.0
    if emotional[0][0] >= min_intensity and emotional[0][0] - second >= 1.0:
        return CLASSES.index(emotional[0][1])
    return None


class ExpressionNet(nn.Module):
    """An ImageNet backbone with a new classification head.

    Args:
        num_classes: number of expression classes.
        backbone: "convnext_tiny" (default), "resnet50" or "resnet18".
        pretrained: load torchvision ImageNet-1k weights (not trained on faces).
        input_size: side of the square input the backbone sees; other sizes are resized inside forward.
        dropout: dropout before the final layer.
    """

    def __init__(self, num_classes: int = len(CLASSES), backbone: str = "convnext_tiny", pretrained: bool = True,
                 input_size: int = 224, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.backbone_name = backbone
        if backbone == "convnext_tiny":
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            net = torchvision.models.convnext_tiny(weights=weights)
            features = net.classifier[2].in_features
            net.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
        elif backbone in ("resnet18", "resnet50"):
            weights = {"resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                       "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2}[backbone] if pretrained else None
            net = getattr(torchvision.models, backbone)(weights=weights)
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
        else:
            raise ValueError(f"unknown backbone {backbone!r}")
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, num_classes] for RGB input in [0, 1] of shape [B, 3, H, W]."""
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)
