"""Load a model through the app's interface and classify whole audio files.

Usage:
  python scripts/check_model.py models/commonaccent_es6            # the model's sample clips
  python scripts/check_model.py models/commonaccent_es6 a.wav b.flac

Prints load time, peak memory, the model's class order and per-file probabilities.
Each file is classified as one piece (no windowing), resampled with soxr.
"""

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from server.audio import decode_file  # noqa: E402
from server.model_api import load_model  # noqa: E402
from server.paths import hf_cache_dir  # noqa: E402


def sample_clips(model) -> list[Path]:
    src = model.config.source
    snapshots = hf_cache_dir() / f"models--{src.repo.replace('/', '--')}" / "snapshots" / src.revision
    return sorted((snapshots / "data").glob("*.wav"))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    t0 = time.time()
    model = load_model(sys.argv[1])
    print(f"Loaded {model.config.display_name} in {time.time() - t0:.1f} s, "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.2f} GB")
    if hasattr(model, "model_labels"):
        by_label = {c.model_label: c.id for c in model.config.classes}
        print("Model output order (from model files) -> config id:")
        for i, label in enumerate(model.model_labels):
            print(f"  {i}: {by_label[label]:15s} <- {label}")

    files = [Path(p) for p in sys.argv[2:]] or sample_clips(model)
    ids = model.class_ids
    print(f"\n{'file':28s} {'sec':>5s}  " + " ".join(f"{i[:9]:>9s}" for i in ids) + "   top")
    for f in files:
        audio = decode_file(f, model.sample_rate)
        t0 = time.time()
        probs = model.predict(audio)
        ms = (time.time() - t0) * 1000
        top = max(probs, key=probs.get)
        print(f"{f.name:28s} {len(audio) / model.sample_rate:5.2f}  "
              + " ".join(f"{probs[i]:9.4f}" for i in ids) + f"   {top} ({ms:.0f} ms)")


if __name__ == "__main__":
    main()
