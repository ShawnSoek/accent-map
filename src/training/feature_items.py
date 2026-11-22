"""The pieces of audio to run through the base model, made from the split lists.

Sets: "train" (each training clip), "train_aug" (each training clip again, as its augmented
copy), "dev" and "test" (each clip), "dev_pairs" and "test_pairs" (two clips of the same
speaker joined, about 10 s like the app's window; for evaluation only).

Two tables come out. The upload table names only item ids, sets and clip files, so the GPU
machine gets no speaker ids, accents or genders. The local table adds those, for training and
evaluation, and stays in data/.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from training.cv_split import rank

SETS = ("train", "train_aug", "dev", "test", "dev_pairs", "test_pairs")


@dataclass(frozen=True)
class Item:
    id: str                    # "<set>/<number>", e.g. "test_pairs/000012"
    set: str
    clips: tuple[str, ...]
    augment: bool
    label: str
    speaker: str
    gender: str
    seconds: float


def read_split(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def make_items(splits: dict[str, list[dict[str, str]]], seed: str) -> list[Item]:
    items: list[Item] = []
    numbers: dict[str, int] = defaultdict(int)

    def add(set_name: str, rows: list[dict[str, str]], augment: bool = False) -> None:
        n = numbers[set_name]
        numbers[set_name] += 1
        items.append(Item(f"{set_name}/{n:06d}", set_name, tuple(r["path"] for r in rows), augment,
                          rows[0]["label"], rows[0]["speaker"], rows[0]["gender"],
                          round(sum(float(r["seconds"]) for r in rows), 3)))

    for row in splits["train"]:
        add("train", [row])
    for row in splits["train"]:
        add("train_aug", [row], augment=True)
    for name in ("dev", "test"):
        for row in splits[name]:
            add(name, [row])
        by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in splits[name]:
            by_speaker[row["speaker"]].append(row)
        for speaker in sorted(by_speaker, key=lambda s: rank(seed, s)):
            rows = sorted(by_speaker[speaker], key=lambda r: rank(seed, r["path"]))
            for i in range(0, len(rows) - 1, 2):
                add(f"{name}_pairs", rows[i:i + 2])
    return items


UPLOAD_HEADER = "id\tset\taugment\tclips\n"
LOCAL_HEADER = "id\tset\taugment\tclips\tlabel\tspeaker\tgender\tseconds\n"


def write_upload_table(items: list[Item], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(UPLOAD_HEADER)
        for it in items:
            f.write(f"{it.id}\t{it.set}\t{int(it.augment)}\t{'+'.join(it.clips)}\n")


def write_local_table(items: list[Item], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(LOCAL_HEADER)
        for it in items:
            f.write(f"{it.id}\t{it.set}\t{int(it.augment)}\t{'+'.join(it.clips)}\t{it.label}\t{it.speaker}"
                    f"\t{it.gender}\t{it.seconds}\n")


@dataclass(frozen=True)
class Job:
    """One row of the upload table: what the GPU machine needs to know."""
    id: str
    set: str
    clips: tuple[str, ...]
    augment: bool


def read_upload_table(path: Path) -> list[Job]:
    jobs = []
    for row in read_split(path):
        jobs.append(Job(row["id"], row["set"], tuple(row["clips"].split("+")), row["augment"] == "1"))
    return jobs
