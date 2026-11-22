"""Speaker-disjoint train/dev/test clip lists for a dataset spec (training/<name>/dataset.yaml).

Per accent, dev and test speakers are drawn at random among speakers with enough clips; all
other speakers train. "Random" is a SHA-256 of the seed and the speaker id (or clip name), so the
split is the same on every run and does not depend on row order. Each speaker contributes at
most `max_clips_per_speaker` clips. Training clips whose sentence is also read in dev or test are
dropped first, so a model cannot score by recognizing sentences.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from training.cv_metadata import collect_speakers

SPLITS = ("train", "dev", "test")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Share(_Strict):
    fraction: float = Field(gt=0, lt=1)
    min_speakers: int = Field(ge=1)


class SplitSpec(_Strict):
    seed: str
    eligible_min_clips: int = Field(ge=1)
    test: Share
    dev: Share
    drop_train_clips_with_dev_test_sentences: bool = True


class ClassSpec(_Strict):
    id: str
    accent: str


class DatasetSpec(_Strict):
    release: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str
    max_clips_per_speaker: int = Field(ge=1)
    split: SplitSpec
    classes: list[ClassSpec] = Field(min_length=2)

    @model_validator(mode="after")
    def _unique(self):
        for what in ("id", "accent"):
            values = [getattr(c, what) for c in self.classes]
            if len(set(values)) != len(values):
                raise ValueError(f"class {what}s must be unique")
        return self


def load_spec(path: Path) -> DatasetSpec:
    return DatasetSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Clip:
    path: str
    speaker: str
    label: str          # class id
    sentence_id: str
    seconds: float
    gender: str         # the speaker's: f, m or other


def rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest()


@dataclass
class SplitResult:
    clips: dict[str, list[Clip]]
    dropped_for_sentences: int      # training clips left out because dev or test read the sentence


def make_splits(rows: Iterable[dict[str, str]], spec: DatasetSpec, durations: dict[str, float]) -> SplitResult:
    sentence_of: dict[str, str] = {}

    def remember(rows):
        for row in rows:
            sentence_of[row["path"]] = row["sentence_id"]
            yield row

    speakers = collect_speakers(remember(rows))
    class_of = {c.accent: c.id for c in spec.classes}
    by_class: dict[str, list[str]] = defaultdict(list)
    for speaker_id, s in speakers.items():
        if s.kind == "single" and s.accent in class_of:
            by_class[class_of[s.accent]].append(speaker_id)

    seed, cap = spec.split.seed, spec.max_clips_per_speaker
    split_of: dict[str, str] = {}
    for c in spec.classes:
        ids = by_class.get(c.id, [])
        eligible = sorted((i for i in ids if len(speakers[i].clips) >= spec.split.eligible_min_clips),
                          key=lambda i: rank(seed, i))
        n_test = max(spec.split.test.min_speakers, round(spec.split.test.fraction * len(eligible)))
        n_dev = max(spec.split.dev.min_speakers, round(spec.split.dev.fraction * len(eligible)))
        if n_test + n_dev > len(eligible):
            raise ValueError(f"{c.id}: {len(eligible)} speakers with at least {spec.split.eligible_min_clips} "
                             f"clips, but dev and test need {n_dev} + {n_test}")
        for i in eligible[:n_test]:
            split_of[i] = "test"
        for i in eligible[n_test:n_test + n_dev]:
            split_of[i] = "dev"
        for i in ids:
            split_of.setdefault(i, "train")
    label_of = {i: class_of[speakers[i].accent] for i in split_of}

    def clip(path: str, speaker_id: str) -> Clip:
        if path not in durations:
            raise KeyError(f"no duration for {path}")
        return Clip(path, speaker_id, label_of[speaker_id], sentence_of[path], durations[path],
                    speakers[speaker_id].gender)

    def ordered(speaker_id: str) -> list[str]:
        return sorted(speakers[speaker_id].clips, key=lambda p: rank(seed, p))

    out: dict[str, list[Clip]] = {name: [] for name in SPLITS}
    for i in sorted(split_of):
        if split_of[i] != "train":
            out[split_of[i]].extend(clip(p, i) for p in ordered(i)[:cap])
    held = {c.sentence_id for name in ("dev", "test") for c in out[name]}
    drop = spec.split.drop_train_clips_with_dev_test_sentences
    dropped = 0
    for i in sorted(split_of):
        if split_of[i] == "train":
            paths = ordered(i)
            kept = [p for p in paths if not (drop and sentence_of[p] in held)]
            dropped += len(paths) - len(kept)
            out["train"].extend(clip(p, i) for p in kept[:cap])
    return SplitResult(out, dropped)


def check_disjoint(result: SplitResult, sentences_too: bool) -> None:
    """Raise if a speaker (or, with sentences_too, a sentence) is in train and in dev or test."""
    speakers = {name: {c.speaker for c in result.clips[name]} for name in SPLITS}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if speakers[a] & speakers[b]:
            raise AssertionError(f"speakers in both {a} and {b}")
    if sentences_too:
        train = {c.sentence_id for c in result.clips["train"]}
        held = {c.sentence_id for name in ("dev", "test") for c in result.clips[name]}
        if train & held:
            raise AssertionError("sentences in both train and dev/test")


def write_clips(clips: list[Clip], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("path\tspeaker\tlabel\tsentence_id\tseconds\tgender\n")
        for c in clips:
            f.write(f"{c.path}\t{c.speaker}\t{c.label}\t{c.sentence_id}\t{c.seconds:.3f}\t{c.gender}\n")


def counts(result: SplitResult, class_ids: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """class id -> split -> {speakers, clips, hours, female, male}."""
    out = {cid: {name: {"speakers": 0, "clips": 0, "hours": 0.0, "female": 0, "male": 0} for name in SPLITS}
           for cid in class_ids}
    for name in SPLITS:
        seen = set()
        for c in result.clips[name]:
            row = out[c.label][name]
            row["clips"] += 1
            row["hours"] += c.seconds / 3600
            if c.speaker not in seen:
                seen.add(c.speaker)
                row["speakers"] += 1
                row["female"] += c.gender == "f"
                row["male"] += c.gender == "m"
    return out
