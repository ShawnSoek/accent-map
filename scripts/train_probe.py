"""Stage 2, step 5a: which of the base model's layers carry the accent.

Usage:
  python -I scripts/train_probe.py training/cv27_es10 [--epochs 4]

Reads the verified features in data/features/<name>/ and the labels in
data/kaggle/<name>/local/items_local.tsv. Trains one linear classifier per layer on the clean
training clips and scores each on dev (clips, and the ~10 s pairs) with the speaker-balanced
accuracy of src/training/metrics.py. Never opens the test sets.

Writes, in data/runs/<name>/ (not committed):
  train_stats.npz   mean and std per (layer, dimension) of the clean training clips
  probe.json        per layer: dev and dev-pair accuracy with 95 % intervals
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from training.cv_split import load_spec  # noqa: E402
from training.feature_store import open_set, read_labels  # noqa: E402
from training.features import load_features_spec  # noqa: E402
from training.metrics import balanced_accuracy, bootstrap_interval  # noqa: E402
from training.probe import layer_stats, probe_probabilities, train_probes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="training/<name>")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    name = args.folder.resolve().name
    class_ids = [c.id for c in load_spec(args.folder / "dataset.yaml").classes]
    storage_scale = load_features_spec(args.folder / "features.yaml").storage_scale
    features = REPO / "data" / "features" / name
    runs = REPO / "data" / "runs" / name
    runs.mkdir(parents=True, exist_ok=True)
    info = read_labels(REPO / "data" / "kaggle" / name / "local" / "items_local.tsv", class_ids)
    train = open_set(features, "train", storage_scale)
    dev_sets = [open_set(features, s, storage_scale) for s in ("dev", "dev_pairs")]
    print(f"train {len(train)} clips; dev {len(dev_sets[0])} clips, {len(dev_sets[1])} pairs; "
          f"{train.shape[0]} layers", flush=True)

    t0 = time.perf_counter()
    mean, std = layer_stats(train)
    np.savez(runs / "train_stats.npz", mean=mean, std=std)
    print(f"statistics of the clean training clips in {time.perf_counter() - t0:.0f} s", flush=True)

    t0 = time.perf_counter()
    model = train_probes(train, info, mean, std, len(class_ids), epochs=args.epochs,
                         log=lambda m: print(m, flush=True))
    print(f"probes trained in {time.perf_counter() - t0:.0f} s", flush=True)

    result = {"epochs": args.epochs, "classes": class_ids, "layers": []}
    scores = {}
    for fset in dev_sets:
        ids, probs = probe_probabilities(model, fset)
        y = np.array([info[i].label for i in ids])
        speakers = [info[i].speaker for i in ids]
        scores[fset.name] = [(balanced_accuracy(y, probs[:, k].argmax(1), speakers, len(class_ids)),
                              bootstrap_interval(y, probs[:, k].argmax(1), speakers, len(class_ids), rounds=1000))
                             for k in range(probs.shape[1])]
    print(f"\n{'layer':>5}  {'dev clips':>22}  {'dev pairs (~10 s)':>22}")
    for k in range(train.shape[0]):
        (a, (al, ah)), (b, (bl, bh)) = scores["dev"][k], scores["dev_pairs"][k]
        result["layers"].append({"layer": k, "dev": a, "dev_interval": [al, ah], "dev_pairs": b,
                                 "dev_pairs_interval": [bl, bh]})
        print(f"{k:>5}  {a:6.1%} ({al:5.1%}-{ah:5.1%})  {b:6.1%} ({bl:5.1%}-{bh:5.1%})")
    (runs / "probe.json").write_text(json.dumps(result, indent=1))
    print(f"\nchance is {1 / len(class_ids):.0%}; wrote {runs / 'probe.json'}")


if __name__ == "__main__":
    main()
