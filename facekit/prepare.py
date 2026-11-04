"""Turns the downloaded datasets into what training and evaluation read.

- DigiFace-1M: aligned 112-pixel synthetic faces into a uint8 array, identities split into train, validation and test.
- EmoNet-Face: portraits decoded from parquet, aligned with OUR trained detector, mapped to eight expression classes.
- WIDER FACE validation: images and the official easy, medium and hard face lists from parquet.
- London set: real consented faces, aligned with our detector, for the identity test.
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

DetectFn = Callable[[np.ndarray], list[dict]]   # RGB image -> faces with "landmarks" and "score", best first


def decode_rgb(data: bytes) -> np.ndarray | None:
    """Decodes encoded image bytes to an RGB uint8 array, or None when the bytes are not an image."""
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if array is None else cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def read_rgb(path: str | Path) -> np.ndarray | None:
    """Reads an image file as RGB uint8, or None when it cannot be read (works for non-ASCII Windows paths)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    array = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return None if array is None else cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def iter_parquet(paths: list[Path], columns: list[str], batch_size: int = 64) -> Iterator[dict]:
    """Yields rows (dicts) of the given columns from parquet files, a batch at a time to bound memory."""
    import pyarrow.parquet as pq
    for path in paths:
        handle = pq.ParquetFile(path)
        for batch in handle.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def image_bytes(cell) -> bytes:
    """Returns the encoded bytes of a Hugging Face image cell ({"bytes": ..., "path": ...}) or raw bytes."""
    return cell["bytes"] if isinstance(cell, dict) else cell


def detect_and_align(detect: DetectFn, image_rgb: np.ndarray, size: int) -> np.ndarray | None:
    """Aligns the highest-scoring face to the template, or returns None when no face is found."""
    faces = detect(image_rgb)
    if not faces:
        return None
    return align.align_face(image_rgb, np.asarray(faces[0]["landmarks"], dtype=np.float32), size)


# --- DigiFace-1M ----------------------------------------------------------------------------------------------


def digiface_records(root: str | Path) -> tuple[list[str], list[str]]:
    """Returns (image paths, identity names) for every PNG under root, where the parent folder names the identity."""
    paths = sorted(Path(root).rglob("*.png"))
    return [str(p) for p in paths], [p.parent.name for p in paths]


def _load_112(path: str) -> np.ndarray:
    image = read_rgb(path)
    if image is None:
        raise ValueError(f"unreadable image {path}")
    if image.shape[:2] != (112, 112):
        image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    return image


def split_identities(identities: list[str], n_val: int, seed: int = 0, max_ids: int | None = None
                     ) -> tuple[list[str], list[str]]:
    """Splits identity names into (train, validation) by identity, so no person appears in both."""
    names = sorted(set(identities))
    rng = np.random.default_rng(seed)
    names = [names[i] for i in rng.permutation(len(names))]
    if max_ids:
        names = names[:max_ids]
    return sorted(names[n_val:]), sorted(names[:n_val])


def build_digiface_array(paths: list[str], identities: list[str], keep: list[str], out_dir: Path, name: str,
                         label_of: dict[str, int], max_per_id: int | None = None, workers: int = 8,
                         extra_meta: dict | None = None) -> dict:
    """Decodes the images of the kept identities into an array set; labels come from `label_of`."""
    kept: list[tuple[str, str]] = []
    per_id: dict[str, int] = {}
    keep_set = set(keep)
    for path, identity in zip(paths, identities):
        if identity in keep_set and (max_per_id is None or per_id.get(identity, 0) < max_per_id):
            kept.append((path, identity))
            per_id[identity] = per_id.get(identity, 0) + 1
    images = parallel_map(_load_112, [p for p, _ in kept], workers=workers, chunksize=256)
    items = ((img, label_of[identity], f"digiface:{identity}") for img, (_, identity) in zip(images, kept))
    return write_array(out_dir, name, items, count=len(kept), size=112,
                       meta={"identities": sorted(keep_set), "num_classes": len(label_of), **(extra_meta or {})})


# --- EmoNet-Face ----------------------------------------------------------------------------------------------


def emonet_selection(paths: list[Path], hq: bool = False, max_items: int | None = None
                     ) -> list[tuple[int, int, str]]:
    """Reads only the label and prompt columns and returns (row number, class index, prompt) for every usable image."""
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
    """Decodes the selected portraits, aligns each with the detector and writes an array set.

    Each item's id is the SHA-1 of its image bytes, so another array can be checked for duplicates; images whose hash
    is in `exclude_hashes` are skipped ("duplicates"). Portraits where the detector finds no face are left out
    ("no_face") unless `fallback_center_crop` is set, which only the smoke run uses with its barely trained detector.
    The description's "groups" lists each kept item's prompt, for the grouped split.
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
    # write_array reads the generator to the end, so the counts are final here; store them with the description.
    description.update(counts)
    meta_path = Path(out_dir) / f"{name}.json"
    import json as _json
    meta_path.write_text(_json.dumps(description), encoding="utf-8")
    return description


# --- WIDER FACE validation -------------------------------------------------------------------------------------


def wider_boxes_xyxy(cell) -> np.ndarray:
    """Converts the parquet's box cell to xyxy: either {"x": [...], "y": [...], "w": [...], "h": [...]} or rows of x y w h."""
    if isinstance(cell, dict):
        columns = [np.asarray(cell.get(k) or [], dtype=np.float32) for k in ("x", "y", "w", "h")]
        boxes = np.stack(columns, axis=1) if len(columns[0]) else np.zeros((0, 4), dtype=np.float32)
    else:
        boxes = np.asarray(cell or [], dtype=np.float32).reshape(-1, 4)
    boxes = boxes.copy()
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def wider_val_records(parquet: Path, max_items: int | None = None) -> Iterator[dict]:
    """Yields validation images with ground truth: {"path", "image" RGB, "boxes" xyxy, "keep": {easy, medium, hard}}."""
    columns = ["image", "image_path", "bboxes", "easy_keep_indices", "medium_keep_indices", "hard_keep_indices"]
    for n, row in enumerate(iter_parquet([parquet], columns, batch_size=16)):
        if max_items and n >= max_items:
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
    """Returns {"path", "person", "view"} for every front photo; the person id is the number before "_" in the name."""
    records = []
    for view, root in (("neutral", neutral_root), ("smiling", smiling_root)):
        for path in sorted(Path(root).rglob("*.jpg")):
            if path.name.startswith("._"):
                continue
            records.append({"path": str(path), "person": path.stem.split("_")[0], "view": view})
    return records


def encode_png(image_rgb: np.ndarray) -> bytes:
    """Encodes an RGB array as PNG bytes (used to show crops in the notebook)."""
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("could not encode image")
    return io.BytesIO(buffer.tobytes()).getvalue()
