"""Read the verified base-model features (stage 2, step 4) for training and evaluation.

The features live in data/features/<name>/ as <set>.part<i>of<n>.npy / .ids (see extract_job.py).
They are used only if scripts/check_features.py has written verified.json there. Values come
back as float32 with the storage scaling undone (features.yaml), so they are the base model's own
pooled values: per layer, the mean and then the std over time.

Test sets open only with allow_test=True. Training and tuning code never passes it, so the
test speakers stay unseen until the final evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.feature_items import read_split
from training.features import scale_vector

TEST_SETS = ("test", "test_pairs")


class NotVerifiedError(Exception):
    pass


@dataclass
class Part:
    ids: list[str]
    rows: np.ndarray                  # float16 memmap (items, layers, 2 * width)


class FeatureSet:
    """One set (e.g. "train"), all its parts, read block by block."""

    def __init__(self, name: str, parts: list[Part], scale: np.ndarray):
        self.name, self.parts = name, parts
        self.scale = scale.astype(np.float32)[None, :, None]
        self.ids = [i for p in parts for i in p.ids]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.parts[0].rows.shape[1:])

    def spans(self, size: int) -> list[tuple[int, int, int]]:
        """(part, start, stop) blocks of at most `size` rows, each inside one part (read in order)."""
        return [(k, s, min(s + size, len(p.ids))) for k, p in enumerate(self.parts)
                for s in range(0, len(p.ids), size)]

    def read(self, part: int, start: int, stop: int) -> tuple[list[str], np.ndarray]:
        p = self.parts[part]
        return p.ids[start:stop], np.asarray(p.rows[start:stop], dtype=np.float32) * self.scale

    def blocks(self, size: int, rng: np.random.Generator | None = None) -> Iterator[tuple[list[str], np.ndarray]]:
        """Every row once, in blocks; in random block order if `rng` is given."""
        spans = self.spans(size)
        order = rng.permutation(len(spans)) if rng is not None else range(len(spans))
        for k in order:
            yield self.read(*spans[k])


def open_set(folder: Path, name: str, storage_scale: dict[int, float] | None = None,
             allow_test: bool = False) -> FeatureSet:
    """`storage_scale` as in features.yaml: layers stored divided by a factor."""
    folder = Path(folder)
    if name in TEST_SETS and not allow_test:
        raise PermissionError(f"{name} is for the final evaluation only")
    verified = folder / "verified.json"
    if not verified.exists():
        raise NotVerifiedError(f"{folder} has no verified.json: run scripts/check_features.py first")
    n = json.loads(verified.read_text())["parts"][name]
    parts = []
    for i in range(n):
        stem = f"{name}.part{i}of{n}"
        parts.append(Part((folder / f"{stem}.ids").read_text(encoding="utf-8").split(),
                          np.load(folder / f"{stem}.npy", mmap_mode="r")))
    return FeatureSet(name, parts, scale_vector(parts[0].rows.shape[1], storage_scale))


@dataclass(frozen=True)
class ItemInfo:
    label: int                         # index into the class list
    speaker: str
    gender: str


def read_labels(local_table: Path, class_ids: list[str]) -> dict[str, ItemInfo]:
    """Item id -> class index, speaker and gender, from data/kaggle/<name>/local/items_local.tsv."""
    index = {c: k for k, c in enumerate(class_ids)}
    out = {}
    for row in read_split(local_table):
        if row["label"] not in index:
            raise ValueError(f"item {row['id']} has label {row['label']!r}, not in the class list")
        out[row["id"]] = ItemInfo(index[row["label"]], row["speaker"], row["gender"])
    return out
