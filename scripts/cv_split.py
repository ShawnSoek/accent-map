"""Make speaker-disjoint train/dev/test clip lists from a dataset spec.

Usage:
  python -I scripts/cv_split.py training/cv27_es10/dataset.yaml

Needs data/commonvoice/<release>/ from scripts/cv_extract.py, and checks that it came from the
archive whose SHA-256 the spec names. Writes data/commonvoice/splits/<spec folder>/{train,dev,
test}.tsv and counts.json (not committed: they contain speaker ids) and prints the counts.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from training.cv_metadata import read_durations, read_tsv  # noqa: E402
from training.cv_split import SPLITS, check_disjoint, counts, load_spec, make_splits, write_clips  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    args = ap.parse_args()
    spec = load_spec(args.spec)
    data = REPO / "data" / "commonvoice"
    checksum_file = data / f"{spec.release}.tar.gz.sha256"
    if not checksum_file.exists() or checksum_file.read_text().split()[0] != spec.archive_sha256:
        sys.exit(f"{checksum_file} is missing or does not match the spec's archive_sha256; run scripts/cv_extract.py")
    release = data / spec.release
    validated = next(release.rglob("validated.tsv"))
    durations = read_durations(next(release.rglob("clip_durations.tsv")))

    result = make_splits(read_tsv(validated), spec, durations)
    check_disjoint(result, sentences_too=spec.split.drop_train_clips_with_dev_test_sentences)
    out_dir = data / "splits" / args.spec.resolve().parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        write_clips(result.clips[name], out_dir / f"{name}.tsv")
    ids = [c.id for c in spec.classes]
    table = counts(result, ids)
    (out_dir / "counts.json").write_text(json.dumps(
        {"spec": str(args.spec), "dropped_for_sentences": result.dropped_for_sentences, "classes": table},
        indent=1), encoding="utf-8")

    print(f"{'class':16s} {'train spk':>9s} {'clips':>7s} {'h':>6s} | {'dev spk':>7s} {'h':>5s} |"
          f" {'test spk':>8s} {'h':>5s} {'f/m':>7s}")
    total = {name: [0, 0, 0.0] for name in SPLITS}
    for cid in ids:
        tr, dv, te = (table[cid][name] for name in SPLITS)
        print(f"{cid:16s} {tr['speakers']:9d} {tr['clips']:7d} {tr['hours']:6.1f} | {dv['speakers']:7d}"
              f" {dv['hours']:5.1f} | {te['speakers']:8d} {te['hours']:5.1f} {te['female']:>3d}/{te['male']:<3d}")
        for name in SPLITS:
            row = table[cid][name]
            total[name][0] += row["speakers"]
            total[name][1] += row["clips"]
            total[name][2] += row["hours"]
    print(f"{'total':16s} {total['train'][0]:9d} {total['train'][1]:7d} {total['train'][2]:6.1f} |"
          f" {total['dev'][0]:7d} {total['dev'][2]:5.1f} | {total['test'][0]:8d} {total['test'][2]:5.1f}")
    print(f"training clips left out because dev or test read the same sentence: {result.dropped_for_sentences}")
    print(f"speakers and held-out sentences checked disjoint. Written to {out_dir}")


if __name__ == "__main__":
    main()
