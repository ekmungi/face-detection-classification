"""Turns the downloaded datasets into the arrays and records that training and evaluation read.

Four datasets are prepared:

- DigiFace-1M is a set of synthetic faces that are already aligned. Its images are stored at 112 pixels as a uint8
  array, and its identities are split into train, validation and test sets.
- EmoNet-Face is a set of portraits stored in parquet files (a compressed column format). The portraits are decoded
  and aligned with the project's own trained detector, and their labels are mapped to eight expression classes.
- The WIDER FACE validation set provides images and the official easy, medium and hard face lists, read from parquet.
- The London set holds real faces whose owners consented to their use. The faces are aligned with the project's
  detector and used for the identity test.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from facekit import align
from facekit.arrays import parallel_map, write_array
from facekit.expression import emonet_binary_class, emonet_hq_class

# A face detector. It takes an RGB image and returns the faces found, best first. Each face is a dict with the keys
# "landmarks" and "score".
DetectFn = Callable[[np.ndarray], list[dict]]


def decode_rgb(data: bytes) -> np.ndarray | None:
    """Decodes encoded image bytes to an RGB uint8 array, or None when the bytes are not an image."""
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if array is None else cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def read_rgb(path: str | Path) -> np.ndarray | None:
    """Reads an image file as RGB uint8, or None when it cannot be read.

    The file is read as raw bytes and decoded afterwards, because OpenCV cannot open paths with non-ASCII characters
    on Windows. This function works for such paths.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    array = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return None if array is None else cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def iter_parquet(paths: list[Path], columns: list[str], batch_size: int = 64) -> Iterator[dict]:
    """Yields rows (dicts) of the given columns from parquet files.

    The rows are read a batch at a time, so the memory use stays small even for large files.
    """
    import pyarrow.parquet as pq

    for path in paths:
        handle = pq.ParquetFile(path)
        for batch in handle.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def parquet_row_count(path: Path) -> int:
    """Returns the number of rows in a parquet file, read from its footer without reading the rows."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def evenly_spaced_indices(total: int, n: int | None) -> list[int]:
    """Returns n positions at even steps through a list of `total` items, or all positions when n is None, 0 or not
    smaller than `total`.

    The WIDER lists are sorted by the kind of event, so the first n items would cover only a few kinds.
    """
    if not n or n >= total:
        return list(range(total))
    return [i * total // n for i in range(n)]


def evenly_spaced(items: list, n: int | None) -> list:
    """Returns n items taken at even steps through the list, or all of them when n is None or not smaller."""
    return [items[i] for i in evenly_spaced_indices(len(items), n)]


def image_bytes(cell) -> bytes:
    """Returns the encoded bytes of an image cell, which is either a dict {"bytes": ..., "path": ...} or raw bytes.

    The dict form is what a Hugging Face image column holds.
    """
    return cell["bytes"] if isinstance(cell, dict) else cell


def detect_and_align(detect: DetectFn, image_rgb: np.ndarray, size: int) -> np.ndarray | None:
    """Aligns the highest-scoring face to the alignment template, or returns None when no face is found.

    The detector lists the best face first, so the first face is the one that is aligned. The aligned crop is
    `size` x `size` pixels.
    """
    faces = detect(image_rgb)
    if not faces:
        return None
    return align.align_face(image_rgb, np.asarray(faces[0]["landmarks"], dtype=np.float32), size)


# --- DigiFace-1M ----------------------------------------------------------------------------------------------


def digiface_records(root: str | Path) -> tuple[list[str], list[str]]:
    """Returns (image paths, identity names) for every PNG under root.

    The name of the folder that holds an image is the identity of the person in it.
    """
    paths = sorted(Path(root).rglob("*.png"))
    return [str(p) for p in paths], [p.parent.name for p in paths]


def _load_112(path: str) -> np.ndarray:
    """Reads a DigiFace image as RGB and resizes it to 112 x 112 pixels when it has another size."""
    image = read_rgb(path)
    if image is None:
        raise ValueError(f"unreadable image {path}")
    if image.shape[:2] != (112, 112):
        image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    return image


def split_identities(identities: list[str], n_val: int, seed: int = 0, max_ids: int | None = None
                     ) -> tuple[list[str], list[str]]:
    """Splits identity names into (train, validation) by identity, so no person appears in both.

    The identities are shuffled with a fixed seed. With `max_ids`, only the first `max_ids` shuffled identities are
    used, which gives a random smaller set. The first `n_val` of those identities form the validation set, and the rest
    form the training set.
    """
    names = sorted(set(identities))
    rng = np.random.default_rng(seed)
    names = [names[i] for i in rng.permutation(len(names))]
    if max_ids:
        names = names[:max_ids]
    return sorted(names[n_val:]), sorted(names[:n_val])


def build_digiface_array(paths: list[str], identities: list[str], keep: list[str], out_dir: Path, name: str,
                         label_of: dict[str, int], max_per_id: int | None = None, workers: int = 8,
                         extra_meta: dict | None = None) -> dict:
    """Decodes the images of the kept identities into an array set and returns its description.

    The label of each image is `label_of[identity]`. With `max_per_id`, at most that many images are taken from each
    identity. The entries of `extra_meta` are added to the description.
    """
    kept: list[tuple[str, str]] = []
    per_id: dict[str, int] = {}
    keep_set = set(keep)

    # Choose the images to use, which are those of the kept identities, up to the limit for each identity.
    for path, identity in zip(paths, identities):
        if identity in keep_set and (max_per_id is None or per_id.get(identity, 0) < max_per_id):
            kept.append((path, identity))
            per_id[identity] = per_id.get(identity, 0) + 1

    # Decode the images in worker processes and write them to the array together with their labels and source ids.
    images = parallel_map(_load_112, [p for p, _ in kept], workers=workers, chunksize=256)
    items = ((img, label_of[identity], f"digiface:{identity}") for img, (_, identity) in zip(images, kept))
    return write_array(out_dir, name, items, count=len(kept), size=112,
                       meta={"identities": sorted(keep_set), "num_classes": len(label_of), **(extra_meta or {})})


# --- EmoNet-Face ----------------------------------------------------------------------------------------------


def emonet_selection(paths: list[Path], hq: bool = False, max_items: int | None = None
                     ) -> list[tuple[int, int, str]]:
    """Returns (row number, class index, prompt) for every usable image, reading only the label and prompt columns.

    An image is usable when its label maps to an expression class. With `hq`, the labels are read as EmoNet-Face HQ
    expert ratings, and otherwise as EmoNet-Face Binary labels. With `max_items`, the search stops after that many
    usable images.
    """
    chosen = []
    to_class = emonet_hq_class if hq else emonet_binary_class
    for row_number, row in enumerate(iter_parquet(paths, ["label", "prompt"], batch_size=1024)):
        cls = to_class(row["label"])
        if cls is not None:
            chosen.append((row_number, cls, str(row.get("prompt") or "")))
            if max_items and len(chosen) >= max_items:
                break
    return chosen


def build_emonet_array(paths: list[Path], selection: list[tuple[int, int, str]], detect: DetectFn, out_dir: Path,
                       name: str, size: int = 224, exclude_hashes: set[str] | None = None,
                       fallback_center_crop: bool = False, exclude_prompts: set[str] | None = None) -> dict:
    """Decodes the selected portraits, aligns each with the detector, writes an array set and returns its description.

    The id of each item is the SHA-1 hash of its image bytes, so another array can be checked for duplicates. Images
    whose hash is in `exclude_hashes` are skipped and counted as "duplicates". Images whose prompt is in
    `exclude_prompts` are skipped and counted as "shared_prompt". Portraits that cannot be decoded, or in which the
    detector finds no face, are left out and counted as "no_face". A decoded portrait without a face is kept only when
    `fallback_center_crop` is set. In that case the centre of the image is cropped instead and counted as "fallback".
    The smoke and medium runs use this option because their detectors are trained only briefly.

    The description's "groups" lists the prompt of each kept item, for the grouped split. The counts are stored in the
    description and in its .json file.
    """
    import hashlib

    wanted = {row: (cls, prompt) for row, cls, prompt in selection}
    counts = {"no_face": 0, "duplicates": 0, "fallback": 0, "shared_prompt": 0}
    if exclude_prompts:
        before = len(wanted)
        wanted = {row: v for row, v in wanted.items() if v[1] not in exclude_prompts}
        counts["shared_prompt"] = before - len(wanted)
    groups: list[str] = []

    def items():
        """Yields (aligned crop, class index, image hash) for every portrait that is kept, and counts the others."""
        for row_number, row in enumerate(iter_parquet(paths, ["path"], batch_size=32)):
            if row_number not in wanted:
                continue
            data = image_bytes(row["path"])
            digest = hashlib.sha1(data).hexdigest()
            if exclude_hashes and digest in exclude_hashes:
                counts["duplicates"] += 1
                continue
            image = decode_rgb(data)
            crop = None if image is None else detect_and_align(detect, image, size)
            if crop is None and fallback_center_crop and image is not None:
                crop = align.center_crop_resize(image, size)
                counts["fallback"] += 1
            if crop is None:
                counts["no_face"] += 1
                continue
            cls, prompt = wanted[row_number]
            groups.append(prompt)
            yield crop, cls, digest

    description = write_array(out_dir, name, items(), count=len(wanted), size=size, meta={"groups": groups})

    # The write above reads the generator to the end, so the counts are final here. They are stored with the
    # description, both in the returned dict and in its .json file.
    description.update(counts)
    meta_path = Path(out_dir) / f"{name}.json"
    import json as _json
    meta_path.write_text(_json.dumps(description), encoding="utf-8")
    return description


# --- WIDER FACE validation -------------------------------------------------------------------------------------


def wider_boxes_xyxy(cell) -> np.ndarray:
    """Converts the parquet's box cell to float32 xyxy rows [N, 4], meaning (left, top, right, bottom).

    The cell is either a dict {"x": [...], "y": [...], "w": [...], "h": [...]} or a list of rows of x, y, width and
    height.
    """
    if isinstance(cell, dict):
        columns = [np.asarray(cell.get(k) or [], dtype=np.float32) for k in ("x", "y", "w", "h")]
        boxes = np.stack(columns, axis=1) if len(columns[0]) else np.zeros((0, 4), dtype=np.float32)
    else:
        boxes = np.asarray(cell or [], dtype=np.float32).reshape(-1, 4)
    boxes = boxes.copy()
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def wider_val_records(parquet: Path, max_items: int | None = None, spread: bool = False) -> Iterator[dict]:
    """Yields validation images with ground truth: {"path", "image" RGB, "boxes" xyxy, "keep": {easy, medium, hard}}.

    The "keep" entry holds, for each of the easy, medium and hard lists, the indices of the boxes that belong to it.
    With `spread`, the `max_items` images are taken at even steps through the file instead of from its start, because
    the file is sorted by the kind of event.
    """
    columns = ["image", "image_path", "bboxes", "easy_keep_indices", "medium_keep_indices", "hard_keep_indices"]
    keep = set(evenly_spaced_indices(parquet_row_count(parquet), max_items)) if spread and max_items else None

    # Walk through the rows in order. With `spread`, only the chosen rows are kept and the walk ends after the last one.
    # Otherwise the walk ends after `max_items` rows.
    for n, row in enumerate(iter_parquet([parquet], columns, batch_size=16)):
        if keep is not None:
            if n > max(keep):
                return
            if n not in keep:
                continue
        elif max_items and n >= max_items:
            return
        boxes = wider_boxes_xyxy(row["bboxes"])
        yield {
            "path": row["image_path"],
            "image": decode_rgb(image_bytes(row["image"])),
            "boxes": boxes,
            "keep": {k: np.asarray(row[f"{k}_keep_indices"] or [], dtype=np.int64) for k in ("easy", "medium", "hard")},
        }


# --- London set ------------------------------------------------------------------------------------------------


def london_records(neutral_root: Path, smiling_root: Path) -> list[dict]:
    """Returns {"path", "person", "view"} for every front photo.

    The person id is the part of the file name before the first "_", which is a number. The view is "neutral" or
    "smiling", depending on which of the two folders holds the photo.
    """
    records = []
    for view, root in (("neutral", neutral_root), ("smiling", smiling_root)):
        for path in sorted(Path(root).rglob("*.jpg")):
            if path.name.startswith("._"):
                continue
            records.append({"path": str(path), "person": path.stem.split("_")[0], "view": view})
    return records


def encode_png(image_rgb: np.ndarray) -> bytes:
    """Encodes an RGB array as PNG bytes, which the notebook uses to show crops."""
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("could not encode image")
    return io.BytesIO(buffer.tobytes()).getvalue()
