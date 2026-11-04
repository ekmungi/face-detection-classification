"""Where each dataset comes from and under which licence. SPEC-001 records the same table and the reasons for each choice.

Nothing here downloads anything; `facekit.datasets.fetch` does that on the training machine.
"""
from __future__ import annotations

from facekit.datasets import Source

HF = "https://huggingface.co/datasets"
DIGIFACE = "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023"

WIDER_LICENCE = "CC BY-NC-ND 4.0 (non-commercial research)"
WIDER_LICENCE_URL = f"{HF}/CUHK-CSE/wider_face"
DIGIFACE_LICENCE = "Microsoft Research Use of Data Agreement 1.0 (non-commercial research)"
DIGIFACE_LICENCE_URL = "https://github.com/microsoft/DigiFace1M/blob/main/LICENSE"
EMONET_LICENCE = "CC BY 4.0"

# Detection: WIDER FACE training images, the RetinaFace five-landmark annotations, and the validation set with the
# official easy, medium and hard face lists.
WIDER_TRAIN = Source(
    name="wider_train", url=f"{HF}/CUHK-CSE/wider_face/resolve/main/data/WIDER_train.zip",
    filename="WIDER_train.zip", size=1465602149, licence=WIDER_LICENCE, licence_url=WIDER_LICENCE_URL)
RETINAFACE_LABELS = Source(
    name="retinaface_labels",
    url=f"{HF}/yuuMQ/Retinaface_WiderFaceDataset/resolve/main/widerface/train/label.txt",
    filename="retinaface_train_label.txt", size=17147541,
    licence="Research annotations by the RetinaFace authors (no licence stated on the mirror)",
    licence_url="https://github.com/deepinsight/insightface/tree/master/detection/retinaface")
WIDER_VAL = Source(
    name="wider_val", url=f"{HF}/zhouxzh/retinaface_widerface/resolve/main/val/val-00000-of-00001.parquet",
    filename="wider_val.parquet", size=366027354, licence=WIDER_LICENCE, licence_url=WIDER_LICENCE_URL)

# Expression: EmoNet-Face Binary (training, validation and test) and EmoNet-Face HQ (a second test set).
EMONET_BINARY = [
    Source(name=f"emonet_binary_{i}",
           url=f"{HF}/laion/emonet-face-binary/resolve/main/data/train-0000{i}-of-00005.parquet",
           filename=f"emonet_binary_{i}.parquet", licence=EMONET_LICENCE,
           licence_url=f"{HF}/laion/emonet-face-binary")
    for i in range(5)
]
EMONET_HQ = [
    Source(name=f"emonet_hq_{i}", url=f"{HF}/laion/emonet-face-hq/resolve/main/data/train-0000{i}-of-00007.parquet",
           filename=f"emonet_hq_{i}.parquet", licence=EMONET_LICENCE, licence_url=f"{HF}/laion/emonet-face-hq")
    for i in range(7)
]

# Identity: DigiFace-1M parts of 2,000 synthetic identities with 72 images each.
DIGIFACE_PART_SIZES = {
    "0-1999": 2938553163,
    "2000-3999": 2936098225,
    "4000-5999": 2936895527,
    "6000-7999": 2940940004,
    "8000-9999": 2927632784,
}


def digiface_part(part: str) -> Source:
    """Returns the Source for one DigiFace-1M part, for example "0-1999"."""
    return Source(
        name=f"digiface_{part}", url=f"{DIGIFACE}/subjects_{part}_72_imgs.zip",
        filename=f"digiface_subjects_{part}_72_imgs.zip", size=DIGIFACE_PART_SIZES[part],
        licence=DIGIFACE_LICENCE, licence_url=DIGIFACE_LICENCE_URL)


# Identity test on real people who signed consent: Face Research Lab London Set, neutral and smiling front views.
LONDON_LICENCE_URL = "https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666"
LONDON_NEUTRAL = Source(
    name="london_neutral", url="https://ndownloader.figshare.com/files/8541961", filename="london_neutral_front.zip",
    size=33860607, licence="CC BY 4.0", licence_url=LONDON_LICENCE_URL)
LONDON_SMILING = Source(
    name="london_smiling", url="https://ndownloader.figshare.com/files/8541964", filename="london_smiling_front.zip",
    size=53522950, licence="CC BY 4.0", licence_url=LONDON_LICENCE_URL)

# ImageNet weights for the detector and expression backbones (torchvision, BSD-3 code; ImageNet terms for weights).
IMAGENET_NOTE = "torchvision ImageNet-1k weights: BSD-3-Clause code, weights under the ImageNet research terms"


def all_sources(digiface_train_parts: list[str], digiface_test_part: str, emonet_hq_files: int) -> list[Source]:
    """Returns every Source a full run downloads, for the manifest and the model card."""
    return (
        [WIDER_TRAIN, RETINAFACE_LABELS, WIDER_VAL]
        + EMONET_BINARY
        + EMONET_HQ[:emonet_hq_files]
        + [digiface_part(p) for p in digiface_train_parts + [digiface_test_part]]
        + [LONDON_NEUTRAL, LONDON_SMILING]
    )
