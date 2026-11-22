"""Unpack a downloaded noise / room-echo collection (.zip) safely and record its checksum.

Usage:
  python -I scripts/noise_extract.py data/noise/rirs_noises.zip

Writes data/noise/<name>/ (the archive's files) and data/noise/<name>.zip.sha256. Refuses to
write into a folder that is not empty.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.archives import extract_zip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    archive = ap.parse_args().archive.resolve()
    out_dir = archive.parent / archive.name.removesuffix(".zip")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} is not empty; remove it first")
    t0 = time.perf_counter()
    result = extract_zip(archive, out_dir)
    (archive.parent / f"{archive.name}.sha256").write_text(f"{result.sha256}  {archive.name}\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: {len(result.extracted)} of {result.members} members "
          f"extracted to {out_dir}")
    print(f"sha256 {result.sha256}")
    if result.skipped:
        print(f"skipped (unsafe or a link): {result.skipped[:20]}")


if __name__ == "__main__":
    main()
