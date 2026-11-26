"""Check that every clip in the split lists decodes, and that its length matches clip_durations.tsv.

Usage:
  python -I scripts/cv_check_clips.py training/cv27_es10/dataset.yaml [--workers 8]

Decodes each MP3 at its own sample rate (no resampling) in parallel and prints the sample rates
seen, the largest length difference, and any clip that fails. Exits with an error if one fails.
"""

import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import soundfile as sf  # noqa: E402

from training.cv_split import SPLITS, load_spec  # noqa: E402


def check(job: tuple[str, float]) -> tuple[str, int | None, float | None, str | None]:
    path, listed = job
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        return path, rate, len(audio) / rate - listed, None
    except Exception as err:                       # report every failure, keep going
        return path, None, None, f"{type(err).__name__}: {err}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    spec = load_spec(args.spec)
    data = REPO / "data" / "commonvoice"
    clips_dir = next((data / spec.release).rglob("validated.tsv")).parent / "clips"
    jobs = []
    for split in SPLITS:
        with open(data / "splits" / args.spec.resolve().parent.name / f"{split}.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                fields = line.rstrip("\n").split("\t")
                jobs.append((str(clips_dir / fields[0]), float(fields[4])))
    t0 = time.perf_counter()
    rates, diffs, failed = Counter(), [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for path, rate, diff, error in pool.map(check, jobs, chunksize=256):
            if error:
                failed.append((path, error))
            else:
                rates[rate] += 1
                diffs.append(abs(diff))
    diffs.sort()
    print(f"{len(jobs)} clips in {time.perf_counter() - t0:.0f} s; sample rates {dict(rates)}")
    if diffs:
        print(f"length vs clip_durations.tsv: median {diffs[len(diffs) // 2] * 1000:.0f} ms, "
              f"99th percentile {diffs[int(len(diffs) * 0.99)] * 1000:.0f} ms, largest {diffs[-1] * 1000:.0f} ms")
    for path, error in failed[:20]:
        print(f"FAILED {Path(path).name}: {error}")
    if failed:
        sys.exit(f"{len(failed)} clips failed to decode")
    print("all clips decode")


if __name__ == "__main__":
    main()
