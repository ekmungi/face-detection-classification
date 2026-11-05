"""Classifies facial expressions with a fine-tuned ImageNet backbone and maps EmoNet-Face labels to eight classes.

The network is an ImageNet-pretrained backbone fine-tuned on aligned face crops. It takes RGB values between 0 and 1
and normalises them itself, so the ONNX export (a file format for sharing trained models) needs no preprocessing
beyond alignment and scaling to the range 0 to 1.

The module also turns the two kinds of EmoNet-Face annotation, Binary and HQ, into one of the eight expression classes.
"""
from __future__ import annotations

import ast
from collections import Counter

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

# The eight expression classes. The position of a name in this tuple is the class index used everywhere else.
CLASSES: tuple[str, ...] = ("neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt")

# Maps each EmoNet-Face category name to one of the eight classes. The names are the part after "Group|" in the HQ
# set. The reasons for the mapping are recorded in the project notes.
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

# The per-channel mean and standard deviation of the ImageNet training images, in RGB order. The pretrained backbones
# expect their input to be normalised with these values.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _short_name(category: str) -> str:
    """Strips the group prefix that the HQ set uses, so "Negative High-Energy Emotions|Anger" becomes "Anger"."""
    return category.split("|")[-1].strip()


def emonet_binary_target(label_text: str) -> tuple[str | None, bool]:
    """Reads one EmoNet-Face Binary label and returns (target category, confirmed).

    The label is a list of {annotator: {category: bool}} entries. The target is the category that was rated most
    often, and the other categories are control questions. It is confirmed when a strict majority of its ratings are
    True. A tie between categories gives (None, False).
    """
    # Collect every true-or-false rating for each category.
    entries = ast.literal_eval(label_text) if isinstance(label_text, str) else label_text
    votes: dict[str, list[bool]] = {}
    for entry in entries:
        for ratings in entry.values():
            for category, present in ratings.items():
                votes.setdefault(_short_name(category), []).append(bool(present))
    if not votes:
        return None, False

    # The category with the most ratings is the target. Two categories with the same count make the target unclear.
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

    For each class, the score is the highest mean rating among the categories that belong to it. The mean is taken
    over the annotators. The image gets the strongest non-neutral class when that score is at least `min_intensity`,
    is clearly above the second strongest, by at least 1.0, and is not beaten by a category outside the eight classes.
    It gets "neutral" when every non-neutral category scores at most `neutral_max`, and None in all other cases.
    """
    # Add up each annotator's rating for every category, then divide by the number of annotators to get the means.
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

    # Give each class the highest mean among its categories, and rank the non-neutral classes by that score.
    scores: dict[str, float] = {}
    for category, cls in EMONET_TO_CLASS.items():
        if category in means:
            scores[cls] = max(scores.get(cls, 0.0), means[category])
    emotional = sorted(((v, k) for k, v in scores.items() if k != "neutral"), reverse=True)
    if not emotional:
        return None

    # The highest mean rating of any category that does not stand for neutral, including categories that the eight
    # classes do not cover. When it is low, no emotion of any kind is clearly present, so the image is neutral.
    neutral_proxies = {c for c, cls in EMONET_TO_CLASS.items() if cls == "neutral"}
    strongest_any = max((v for name, v in means.items() if name not in neutral_proxies), default=0.0)
    if strongest_any <= neutral_max:
        return CLASSES.index("neutral")

    # The dominant emotion is one that the eight classes do not cover, so the image is not used.
    if strongest_any > emotional[0][0]:
        return None

    # Accept the top class only when it is strong enough and clearly ahead of the second class.
    second = emotional[1][0] if len(emotional) > 1 else 0.0
    if emotional[0][0] >= min_intensity and emotional[0][0] - second >= 1.0:
        return CLASSES.index(emotional[0][1])
    return None


class ExpressionNet(nn.Module):
    """An ImageNet backbone with a new classification head.

    Args:
        num_classes: The number of expression classes.
        backbone: "convnext_tiny" (the default), "resnet50" or "resnet18".
        pretrained: Whether to load the torchvision ImageNet-1k weights, which were not trained on faces.
        input_size: The side of the square input that the backbone sees. Other sizes are resized inside `forward`.
        dropout: The dropout probability applied before the final layer.
    """

    def __init__(self, num_classes: int = len(CLASSES), backbone: str = "convnext_tiny", pretrained: bool = True,
                 input_size: int = 224, dropout: float = 0.2):
        """Builds the chosen backbone and replaces its last layer. Raises ValueError for an unknown backbone name."""
        super().__init__()
        self.input_size = input_size
        self.backbone_name = backbone

        # Load the backbone and swap its last layer for dropout followed by a new linear layer with one output per
        # expression class.
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

        # The normalisation constants are stored on the model, so that they move with it between devices. They are
        # not saved in checkpoints because they never change.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, num_classes], the raw class scores, for RGB input [B, 3, H, W] with values from 0 to 1."""
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)
