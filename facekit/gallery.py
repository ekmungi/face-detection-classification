"""A gallery of enrolled people, where each person is the normalised mean of their face embeddings.

Identification compares a new face's embedding with every person by cosine similarity. It returns the best name when
the similarity reaches the threshold chosen on validation data, and otherwise the face is "unknown". The gallery file
holds embeddings of real people, so it stays on the owner's computer and never goes into the repository.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Returns the vectors as a float32 matrix [N, D] in which every row is scaled to a length of 1."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)


class Gallery:
    """Enrolled people and the decision threshold.

    The gallery keeps, for each person, a name, a template (the normalised mean embedding), the number of faces behind
    the template, and the sum of the normalised embeddings, which allows more faces to be added later.
    """

    def __init__(self, threshold: float):
        """Creates an empty gallery. `threshold` is the lowest cosine similarity that counts as a match."""
        self.threshold = float(threshold)
        self.names: list[str] = []
        self.templates = np.zeros((0, 0), dtype=np.float32)
        self.sums: dict[str, np.ndarray] = {}
        self.counts: list[int] = []

    def __len__(self) -> int:
        """Returns the number of enrolled people."""
        return len(self.names)

    def enroll(self, name: str, embeddings: np.ndarray) -> None:
        """Adds a person, or adds more faces to a person already enrolled (the template is re-averaged)."""
        vectors = _normalise(embeddings)

        # A person who is already enrolled gets the new faces added to the running sum, and the template is rebuilt.
        # A gallery loaded from an older file may have no sum, and then the template times the count replaces it.
        if name in self.names:
            i = self.names.index(name)
            previous = self.sums.get(name, self.templates[i] * self.counts[i])
            self.sums[name] = previous + vectors.sum(0)
            self.counts[i] += len(vectors)
            self.templates[i] = _normalise(self.sums[name])[0]
            return

        # A new person gets a new row in the template matrix.
        self.sums[name] = vectors.sum(0)
        template = _normalise(vectors.mean(0))
        self.templates = template if len(self.names) == 0 else np.vstack([self.templates, template])
        self.names.append(name)
        self.counts.append(len(vectors))

    def scores(self, embedding: np.ndarray) -> np.ndarray:
        """Returns the cosine similarity of one embedding with every enrolled person."""
        if not self.names:
            return np.zeros(0, dtype=np.float32)
        return self.templates @ _normalise(embedding)[0]

    def identify(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Returns (name, similarity) for the best match, or (None, similarity) when it is below the threshold.

        An empty gallery returns (None, 0.0).
        """
        sims = self.scores(embedding)
        if sims.size == 0:
            return None, 0.0
        best = int(np.argmax(sims))
        score = float(sims[best])
        return (self.names[best] if score >= self.threshold else None), score

    def save(self, path: str | Path) -> None:
        """Saves the gallery as a .npz file."""
        # Every person needs a stored sum. When one is missing, the template times the count stands in for it.
        sums = np.stack([self.sums.get(n, t * c) for n, t, c in zip(self.names, self.templates, self.counts)]) \
            if self.names else np.zeros((0, 0), dtype=np.float32)
        np.savez(path, names=np.array(self.names), templates=self.templates, counts=np.array(self.counts),
                 sums=sums, threshold=np.array(self.threshold))

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        """Loads a gallery saved with save()."""
        data = np.load(path, allow_pickle=False)
        gallery = cls(float(data["threshold"]))
        gallery.names = [str(n) for n in data["names"]]
        gallery.templates = data["templates"].astype(np.float32)
        gallery.counts = [int(c) for c in data["counts"]]

        # Files written by older versions have no sums, in which case the sums are rebuilt later when needed.
        if "sums" in data.files and len(data["sums"]):
            gallery.sums = {n: s.astype(np.float32) for n, s in zip(gallery.names, data["sums"])}
        return gallery
