"""Common Voice metadata: accents per speaker, and the per-accent counts used to choose classes."""

import pytest

from training.cv_metadata import (accent_stats, collect_speakers, gender_group, read_durations, read_tsv,
                                  split_accents, summary)

HEADER = ["client_id", "path", "sentence", "gender", "accents"]
MX = "México"
AR = "Rioplatense: Argentina, Uruguay, este de Bolivia, Paraguay"


def write_tsv(path, header, rows):
    path.write_text("\n".join("\t".join(r) for r in [header, *rows]) + "\n", encoding="utf-8")


def rows(*specs):
    # (speaker, clip, gender, accents)
    return [{"client_id": s, "path": c, "sentence": "Hola.", "gender": g, "accents": a} for s, c, g, a in specs]


def test_read_tsv_keeps_quotes_and_rejects_broken_rows(tmp_path):
    path = tmp_path / "v.tsv"
    write_tsv(path, HEADER, [["a", "1.mp3", 'Dijo "hola" y se fue.', "", MX]])
    assert list(read_tsv(path))[0]["sentence"] == 'Dijo "hola" y se fue.'
    write_tsv(path, HEADER, [["a", "1.mp3", "una\tfrase", "", MX]])
    with pytest.raises(ValueError, match="v.tsv:2: 6 fields"):
        list(read_tsv(path))


def test_split_accents():
    assert split_accents("") == ()
    assert split_accents(f" {MX} ") == (MX,)
    assert split_accents(f"{MX}|{AR}| |{MX}") == tuple(sorted([MX, AR]))


def test_gender_group():
    assert [gender_group(g) for g in ["female_feminine", "male_masculine", "female", "male", "", "intersex"]] == \
        ["f", "m", "f", "m", "other", "other"]


def test_speaker_kinds():
    speakers = collect_speakers(rows(
        ("a", "1", "male", MX), ("a", "2", "male", MX), ("a", "3", "", ""),    # one accent, one clip unlabelled
        ("b", "4", "", f"{MX}|{AR}"),                                         # names two
        ("c", "5", "", MX), ("c", "6", "", AR),                               # changed over time
        ("d", "7", "", ""),                                                   # none
    ))
    assert {k: s.kind for k, s in speakers.items()} == {"a": "single", "b": "mixed", "c": "mixed", "d": "none"}
    assert speakers["a"].accent == MX and speakers["a"].gender == "m"
    s = summary(speakers)
    assert s["speakers"] == {"single": 1, "mixed": 2, "none": 1} and s["clips"] == {"single": 3, "mixed": 3, "none": 1}
    assert s["top_mixed"] == [(" + ".join(sorted([MX, AR])), 2)]


def test_accent_stats(tmp_path):
    specs = [("big", f"b{i}.mp3", "female_feminine", MX) for i in range(8)]
    specs += [("s1", "s1.mp3", "male", MX), ("s2", "s2.mp3", "", MX), ("ar", "ar.mp3", "", AR)]
    durations_file = tmp_path / "clip_durations.tsv"
    write_tsv(durations_file, ["clip", "duration[ms]"], [[c, "4500"] for _, c, _, _ in specs])
    stats = accent_stats(collect_speakers(rows(*specs)), read_durations(durations_file))

    mx, ar = stats
    assert (mx.accent, mx.speakers, mx.clips, ar.accent, ar.speakers) == (MX, 3, 10, AR, 1)
    assert mx.hours == pytest.approx(10 * 4.5 / 3600)
    assert mx.median_clips_per_speaker == 1 and mx.top_speaker_share == 0.8 and mx.top10_share == 1.0
    assert (mx.female, mx.male, mx.other_gender) == (1, 1, 1)
    assert accent_stats(collect_speakers(rows(*specs)), None)[0].hours is None
