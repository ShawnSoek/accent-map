"""Read a Common Voice archive once: checksum, list of members, and the metadata tables.

Usage:
  python -I scripts/cv_extract.py data/commonvoice/cv-corpus-27.0-2026-09-11-es.tar.gz

Writes next to the archive (all under data/, which is not committed):
  <name>/                  the archive's non-audio files (the .tsv tables and release notes)
  <name>.listing.tsv       every member: path, size in bytes, type
  <name>.tar.gz.sha256     the archive's SHA-256, in `sha256sum -c` format
Audio clips are not extracted; a later step extracts only the clips it needs.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.cv_archive import scan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    args = ap.parse_args()
    archive = args.archive.resolve()
    name = archive.name.removesuffix(".tar.gz")
    out_dir = archive.parent / name
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
    (archive.parent / f"{archive.name}.sha256").write_text(f"{result.sha256}  {archive.name}\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: {result.members} members, "
          f"{len(result.extracted)} extracted to {out_dir}")
    print(f"sha256 {result.sha256}")
    for name in result.extracted:
        print(f"  {name}  {(out_dir / name).stat().st_size / 1e6:.1f} MB")
    if result.skipped:
        print(f"skipped (unsafe or not a regular file): {result.skipped[:20]}")


if __name__ == "__main__":
    main()
