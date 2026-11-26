"""Write a few clean and augmented training clips as WAV files, to listen to.

Usage:
  python -I scripts/augment_preview.py training/cv27_es10 [--per-class 1]

Takes training clips (one per class by default, the same ones on every run), applies their
augmentation plan from augment.yaml, and writes <class>_<n>_clean.wav and _augmented.wav to
data/noise/preview/ (not committed). Prints what was done to each.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import soundfile as sf  # noqa: E402

from server.audio import decode_file  # noqa: E402
from training.augment import SAMPLE_RATE, NoiseBank, apply, load_augment_spec, plan_for  # noqa: E402
from training.cv_split import load_spec, rank  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="training/<name> with dataset.yaml and augment.yaml")
    ap.add_argument("--per-class", type=int, default=1)
    args = ap.parse_args()
    spec = load_spec(args.folder / "dataset.yaml")
    aug = load_augment_spec(args.folder / "augment.yaml")
    noise_dir = REPO / "data" / "noise"
    checksum = noise_dir / f"{aug.noise_archive}.sha256"
    if not checksum.exists() or checksum.read_text().split()[0] != aug.noise_sha256:
        sys.exit(f"{checksum} is missing or does not match augment.yaml; run scripts/noise_extract.py")
    bank = NoiseBank(noise_dir / aug.noise_archive.removesuffix(".zip"), aug.reverb.simulated_rooms)

    data = REPO / "data" / "commonvoice"
    clips_dir = next((data / spec.release).rglob("validated.tsv")).parent / "clips"
    by_class: dict[str, list[str]] = {}
    with open(data / "splits" / args.folder.resolve().name / "train.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            path, _, label = line.split("\t")[:3]
            by_class.setdefault(label, []).append(path)
    out = noise_dir / "preview"
    out.mkdir(exist_ok=True)
    for c in spec.classes:
        chosen = sorted(by_class[c.id], key=lambda p: rank("preview", p))[:args.per_class]
        for n, clip in enumerate(chosen, 1):
            audio = decode_file(clips_dir / clip, SAMPLE_RATE)
            plan = plan_for(clip, aug, bank)
            sf.write(out / f"{c.id}_{n}_clean.wav", audio, SAMPLE_RATE)
            sf.write(out / f"{c.id}_{n}_augmented.wav", apply(audio, plan, bank), SAMPLE_RATE)
            print(f"{c.id}_{n}: {len(audio) / SAMPLE_RATE:.1f} s, {plan.describe()}")
    try:
        windows = subprocess.run(["wslpath", "-w", str(out)], capture_output=True, text=True, check=True).stdout.strip()
        print(f"\nwritten to {out}\nin Windows: {windows}")
    except (OSError, subprocess.CalledProcessError):
        print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
