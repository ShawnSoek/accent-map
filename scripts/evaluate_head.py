"""Stage 2, step 5c: calibrate the chosen classifier on dev, then evaluate it once on test.

Usage:
  python -I scripts/evaluate_head.py training/cv27_es10 [--again]

Needs data/runs/<name>/head.pt from scripts/train_head.py. Then:
  1. Fits one temperature on the dev pairs (~10 s, the app's window) and saves it into head.pt.
     It changes no answer, only how sure the model claims to be.
  2. Scores dev and the dev pairs. Dev chose the layers and settings, so these are optimistic.
  3. Scores test and the test pairs: speakers no step has seen. This is meant to happen once;
     the script refuses if data/runs/<name>/evaluation.json exists, unless --again.

Writes data/runs/<name>/evaluation.json and prints the main numbers.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from training.cv_split import load_spec  # noqa: E402
from training.feature_store import open_set, read_labels  # noqa: E402
from training.features import load_features_spec  # noqa: E402
from training.head import TrainedHead, fit_temperature  # noqa: E402
from training.metrics import report  # noqa: E402


def score(head, fset, info, temperature=None):
    """Probabilities (at the given temperature) and labels of every item, in the set's order."""
    ids, probs = [], []
    for block_ids, x in fset.blocks(2000):
        ids += block_ids
        probs.append(head.probabilities(x, temperature))
    return (np.concatenate(probs), np.array([info[i].label for i in ids]), [info[i].speaker for i in ids],
            [info[i].gender for i in ids])


def with_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, None)) / t
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def show(name, r):
    lo, hi = r["interval_95"]
    b = r["before_temperature"]
    print(f"  {name:<11} {r['balanced_accuracy']:6.1%} ({lo:5.1%}-{hi:5.1%})   clips {r['clip_accuracy']:6.1%}   "
          f"log loss {b['log_loss']:.3f} -> {r['log_loss']:.3f}   calibration error {b['calibration_error']:.3f} -> "
          f"{r['calibration_error']:.3f}   {r['speakers']} speakers, {r['items']} items", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="training/<name>")
    ap.add_argument("--again", action="store_true", help="evaluate on test again (say why in DECISIONS.md)")
    args = ap.parse_args()
    name = args.folder.resolve().name
    class_ids = [c.id for c in load_spec(args.folder / "dataset.yaml").classes]
    storage_scale = load_features_spec(args.folder / "features.yaml").storage_scale
    features = REPO / "data" / "features" / name
    runs = REPO / "data" / "runs" / name
    out = runs / "evaluation.json"
    if out.exists() and not args.again:
        sys.exit(f"{out} exists: test was already evaluated. Use --again only with a reason in DECISIONS.md.")
    info = read_labels(REPO / "data" / "kaggle" / name / "local" / "items_local.tsv", class_ids)
    head = TrainedHead(runs / "head.pt")
    if head.class_ids != class_ids:
        sys.exit("head.pt was trained with another class list")

    t0 = time.perf_counter()
    probs, y, _, _ = score(head, open_set(features, "dev_pairs", storage_scale), info, temperature=1.0)
    temperature = fit_temperature(probs, y)
    saved = dict(head.saved, temperature=temperature)
    torch.save(saved, runs / "head.pt")
    head = TrainedHead(runs / "head.pt")
    print(f"temperature {temperature:.2f}, fitted on the dev pairs", flush=True)

    result = {"head": {k: head.saved[k] for k in ("layers", "hidden", "dropout", "temperature")},
              "classes": class_ids, "sets": {}}
    for set_name in ("dev", "dev_pairs", "test", "test_pairs"):
        fset = open_set(features, set_name, storage_scale, allow_test=set_name.startswith("test"))
        plain, y, speakers, genders = score(head, fset, info, temperature=1.0)
        probs = with_temperature(plain, temperature)
        r = report(probs, y, speakers, genders, class_ids)
        before = report(plain, y, speakers, genders, class_ids, with_interval=False)
        r["before_temperature"] = {k: before[k] for k in ("log_loss", "calibration_error")}
        result["sets"][set_name] = r
        if set_name == "dev":
            print("dev (chose the layers and settings, so optimistic):", flush=True)
        if set_name == "test":
            print("test (speakers no step has seen):", flush=True)
        show(set_name, r)
    result["seconds"] = round(time.perf_counter() - t0)
    out.write_text(json.dumps(result, indent=1))

    print("\nper class (balanced accuracy by speaker; test clips / test pairs):")
    for c in class_ids:
        a, b = result["sets"]["test"]["per_class"][c], result["sets"]["test_pairs"]["per_class"][c]
        print(f"  {c:<16} {a['accuracy']:6.1%} / {b['accuracy']:6.1%}   {a['speakers']:>3} speakers")
    print("by gender (test pairs): " + ", ".join(f"{g} {v['balanced_accuracy']:.1%} ({v['speakers']} speakers)"
                                                for g, v in result["sets"]["test_pairs"]["by_gender"].items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
