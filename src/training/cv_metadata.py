"""Common Voice metadata: clips, speakers and their self-reported accents.

A speaker is a `client_id`. Contributors choose their accent in their profile, from a list or
as free text; a clip row carries it in the `accents` column, several accents separated by "|".
For training, a speaker counts for an accent only if all their labelled clips name exactly that
one accent ("single"). Speakers who name several, or different ones over time, are "mixed".
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


def read_tsv(path: Path) -> Iterator[dict[str, str]]:
    """Rows of a Common Voice TSV as dicts. Fails on a row with the wrong number of fields."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{line_no}: {len(row)} fields, expected {len(header)}")
            yield dict(zip(header, row))


def split_accents(field: str) -> tuple[str, ...]:
    """The accents named in one `accents` field, without blanks or repeats."""
    return tuple(sorted({part.strip() for part in field.split("|") if part.strip()}))


def gender_group(value: str) -> str:
    """f, m or other (including not given). Common Voice has used "female" and "female_feminine"."""
    value = value.strip().lower()
    return "f" if value.startswith("female") else "m" if value.startswith("male") else "other"


@dataclass
class Speaker:
    labels: Counter            # accent tuple -> clips (the empty tuple: clips without an accent)
    clips: list[str]
    genders: Counter

    @property
    def named(self) -> set[tuple[str, ...]]:
        return {label for label in self.labels if label}

    @property
    def kind(self) -> str:
        named = self.named
        if not named:
            return "none"
        if len(named) == 1 and len(next(iter(named))) == 1:
            return "single"
        return "mixed"

    @property
    def accent(self) -> str | None:
        return next(iter(self.named))[0] if self.kind == "single" else None

    @property
    def gender(self) -> str:
        given = Counter({g: n for g, n in self.genders.items() if g != "other"})
        return given.most_common(1)[0][0] if given else "other"


def collect_speakers(rows: Iterable[dict[str, str]]) -> dict[str, Speaker]:
    speakers: dict[str, Speaker] = defaultdict(lambda: Speaker(Counter(), [], Counter()))
    for row in rows:
        s = speakers[row["client_id"]]
        s.labels[split_accents(row.get("accents", ""))] += 1
        s.clips.append(row["path"])
        s.genders[gender_group(row.get("gender", ""))] += 1
    return dict(speakers)


def read_durations(path: Path) -> dict[str, float]:
    """clip_durations.tsv: clip file name -> seconds."""
    out = {}
    for row in read_tsv(path):
        ms = row.get("duration[ms]") or row.get("duration")
        out[row["clip"]] = float(ms) / 1000
    return out


@dataclass
class AccentStats:
    accent: str
    speakers: int
    clips: int
    hours: float | None
    median_clips_per_speaker: float
    top_speaker_share: float        # share of the accent's clips that its most active speaker made
    top10_share: float              # ... that its 10 most active speakers made
    female: int
    male: int
    other_gender: int


def accent_stats(speakers: dict[str, Speaker], durations: dict[str, float] | None) -> list[AccentStats]:
    """Per accent, over "single" speakers only, most speakers first."""
    by_accent: dict[str, list[Speaker]] = defaultdict(list)
    for s in speakers.values():
        if s.kind == "single":
            by_accent[s.accent].append(s)
    out = []
    for accent, group in by_accent.items():
        counts = sorted((len(s.clips) for s in group), reverse=True)
        clips = sum(counts)
        hours = None
        if durations is not None:
            hours = sum(durations.get(c, 0.0) for s in group for c in s.clips) / 3600
        genders = Counter(s.gender for s in group)
        out.append(AccentStats(
            accent=accent, speakers=len(group), clips=clips, hours=hours,
            median_clips_per_speaker=statistics.median(counts),
            top_speaker_share=counts[0] / clips, top10_share=sum(counts[:10]) / clips,
            female=genders["f"], male=genders["m"], other_gender=genders["other"]))
    out.sort(key=lambda a: (-a.speakers, a.accent))
    return out


def summary(speakers: dict[str, Speaker]) -> dict:
    kinds = Counter(s.kind for s in speakers.values())
    clips = Counter()
    for s in speakers.values():
        clips[s.kind] += len(s.clips)
    combos = Counter()
    for s in speakers.values():
        if s.kind == "mixed":
            combos[" + ".join(sorted({a for label in s.named for a in label}))] += 1
    return {
        "speakers": {k: kinds[k] for k in ("single", "mixed", "none")},
        "clips": {k: clips[k] for k in ("single", "mixed", "none")},
        "top_mixed": combos.most_common(10),
    }


def as_dicts(stats: list[AccentStats]) -> list[dict]:
    return [asdict(a) for a in stats]
