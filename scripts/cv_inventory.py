"""Count speakers, clips and hours per self-reported accent in a Common Voice release.

Usage:
  python -I scripts/cv_inventory.py data/commonvoice/cv-corpus-27.0-2026-09-11-es [--min-speakers 5]

Reads validated.tsv and clip_durations.tsv (run scripts/cv_extract.py first). Prints one row per
accent named by at least --min-speakers "single" speakers (see src/training/cv_metadata.py) and
writes everything to <release>.inventory.json next to the folder. Prints no speaker ids.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.cv_metadata import accent_stats, as_dicts, collect_speakers, read_durations, read_tsv, summary  # noqa: E402


def find_one(root: Path, name: str) -> Path | None:
    found = sorted(root.rglob(name))
    if len(found) > 1:
        sys.exit(f"more than one {name} under {root}: {found}")
    return found[0] if found else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release", type=Path, help="folder made by scripts/cv_extract.py")
    ap.add_argument("--min-speakers", type=int, default=5)
    args = ap.parse_args()
    validated = find_one(args.release, "validated.tsv")
    if validated is None:
        sys.exit(f"no validated.tsv under {args.release}")
    durations_file = find_one(args.release, "clip_durations.tsv")
    durations = read_durations(durations_file) if durations_file else None

    speakers = collect_speakers(read_tsv(validated))
    stats = accent_stats(speakers, durations)
    totals = summary(speakers)
    shown = [a for a in stats if a.speakers >= args.min_speakers]
    rest = [a for a in stats if a.speakers < args.min_speakers]

    print(f"{validated}: {sum(totals['clips'].values())} validated clips, {len(speakers)} speakers")
    for kind, label in [("single", "one accent"), ("mixed", "several accents"), ("none", "no accent")]:
        print(f"  {label:16s} {totals['speakers'][kind]:7d} speakers {totals['clips'][kind]:9d} clips")
    print()
    print(f"{'accent':62s} {'spk':>6s} {'clips':>8s} {'hours':>7s} {'med':>4s} {'top1':>5s} {'top10':>5s}"
          f" {'f/m/other':>12s}")
    for a in shown:
        hours = f"{a.hours:7.1f}" if a.hours is not None else "      ?"
        name = a.accent if len(a.accent) <= 62 else a.accent[:59] + "..."
        print(f"{name:62s} {a.speakers:6d} {a.clips:8d} {hours} {a.median_clips_per_speaker:4.0f}"
              f" {a.top_speaker_share:5.0%} {a.top10_share:5.0%} {f'{a.female}/{a.male}/{a.other_gender}':>12s}")
    if rest:
        print(f"... and {len(rest)} more accents named by fewer than {args.min_speakers} speakers "
              f"({sum(a.speakers for a in rest)} speakers, {sum(a.clips for a in rest)} clips)")
    print("\nmost common mixed labels:")
    for combo, n in totals["top_mixed"]:
        print(f"  {n:5d}  {combo[:150]}")

    out = args.release.parent / f"{args.release.name}.inventory.json"
    out.write_text(json.dumps({"validated": str(validated), "summary": totals, "accents": as_dicts(stats)},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
