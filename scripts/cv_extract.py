"""Read a Common Voice archive once: checksum, list of members, and the metadata tables.

Usage:
  python -I scripts/cv_extract.py data/commonvoice/cv-corpus-27.0-2026-09-11-es.tar.gz
  python -I scripts/cv_extract.py <archive> --clips data/commonvoice/splits/cv27_es10

Writes next to the archive (all under data/, which is not committed):
  <name>/                  the archive's non-audio files (the .tsv tables and release notes)
  <name>.listing.tsv       every member: path, size in bytes, type
  <name>.tar.gz.sha256     the archive's SHA-256, in `sha256sum -c` format
Audio clips are not extracted then. With --clips, a second pass extracts only the clips named in
the train/dev/test.tsv lists of that folder (from scripts/cv_split.py), into <name>/.../clips/,
and checks that the archive is still the one first read.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.archives import scan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--clips", type=Path, help="folder with train/dev/test.tsv lists of clips to extract")
    args = ap.parse_args()
    archive = args.archive.resolve()
    name = archive.name.removesuffix(".tar.gz")
    out_dir = archive.parent / name
    checksum_file = archive.parent / f"{archive.name}.sha256"
    if args.clips:
        return extract_clips(archive, out_dir, checksum_file, args.clips)
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} is not empty; remove it first")
    total = archive.stat().st_size
    t0 = time.perf_counter()
    shown = [0]

    def progress(done: int) -> None:
        if done - shown[0] >= 5e9:
            shown[0] = done
            rate = done / (time.perf_counter() - t0)
            print(f"  {done / 1e9:5.1f} of {total / 1e9:.1f} GB, {rate / 1e6:.0f} MB/s", flush=True)

    result = scan(archive, out_dir, lambda n: not n.lower().endswith(".mp3"),
                  archive.parent / f"{name}.listing.tsv", progress)
    checksum_file.write_text(f"{result.sha256}  {archive.name}\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: {result.members} members, "
          f"{len(result.extracted)} extracted to {out_dir}")
    print(f"sha256 {result.sha256}")
    for name in result.extracted:
        print(f"  {name}  {(out_dir / name).stat().st_size / 1e6:.1f} MB")
    if result.skipped:
        print(f"skipped (unsafe or not a regular file): {result.skipped[:20]}")


def extract_clips(archive: Path, out_dir: Path, checksum_file: Path, lists: Path) -> None:
    if not checksum_file.exists():
        sys.exit(f"{checksum_file} is missing: run the first pass (without --clips) first")
    needed = set()
    for split in ("train", "dev", "test"):
        with open(lists / f"{split}.tsv", encoding="utf-8") as f:
            next(f)
            needed.update(line.split("\t", 1)[0] for line in f)
    print(f"extracting {len(needed)} clips named in {lists}", flush=True)
    t0 = time.perf_counter()

    def want(member: str) -> bool:
        folder, _, base = member.rpartition("/")
        return folder.endswith("/clips") and base in needed

    result = scan(archive, out_dir, want)
    found = {n.rsplit("/", 1)[-1] for n in result.extracted}
    print(f"done in {time.perf_counter() - t0:.0f} s: {len(found)} clips extracted")
    if result.sha256 != checksum_file.read_text().split()[0]:
        sys.exit(f"the archive's checksum changed since the first pass: {result.sha256}")
    missing = needed - found
    if missing or result.skipped:
        sys.exit(f"missing: {len(missing)} (e.g. {sorted(missing)[:3]}), skipped: {result.skipped[:5]}")
    print("all clips found; archive checksum unchanged")


if __name__ == "__main__":
    main()
