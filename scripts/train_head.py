"""Stage 2, step 5b: choose the layers and the classifier on dev, and train it.

Usage:
  python -I scripts/train_head.py training/cv27_es10 [--rebuild]

Needs scripts/train_probe.py's output (data/runs/<name>/train_stats.npz, probe.json). Then:
  1. Inputs: for each choice of layers (from the probe's dev-pair scores: the best layer,
     windows of 3, 5 and 9 around it, every layer within 5 points of the best, all layers), one
     vector per item, written once to data/runs/<name>/inputs/ from train, train_aug, dev and
     dev_pairs.
  2. A linear classifier on each choice; the best dev-pair score wins.
  3. A few classifier settings on that choice; the best dev-pair score wins.
Training always uses the clean and the noisy copies: dev is clean audio, so it cannot judge
the noisy copies, which are there for laptop microphones. Never opens the test sets.

Writes data/runs/<name>/head.pt (everything needed to classify a base-model feature vector) and
head.json (every dev result).
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
from training.head import (HeadConfig, build_inputs, layer_choices, load_inputs, predict,  # noqa: E402
                           save_head, train_head)
from training.metrics import balanced_accuracy  # noqa: E402

SETS = ("train", "train_aug", "dev", "dev_pairs")
SETTINGS = {
    "linear": HeadConfig(),
    "linear, more decay": HeadConfig(weight_decay=1e-1),
    "hidden 512, dropout 0.3": HeadConfig(hidden=512, dropout=0.3, weight_decay=1e-2),
    "hidden 512, dropout 0.5": HeadConfig(hidden=512, dropout=0.5, weight_decay=1e-2),
    "hidden 1024, dropout 0.5": HeadConfig(hidden=1024, dropout=0.5, weight_decay=1e-2),
}


def folder_name(choice: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "_" for ch in choice)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="training/<name>")
    ap.add_argument("--rebuild", action="store_true", help="write the inputs again")
    args = ap.parse_args()
    name = args.folder.resolve().name
    class_ids = [c.id for c in load_spec(args.folder / "dataset.yaml").classes]
    storage_scale = load_features_spec(args.folder / "features.yaml").storage_scale
    features = REPO / "data" / "features" / name
    runs = REPO / "data" / "runs" / name
    info = read_labels(REPO / "data" / "kaggle" / name / "local" / "items_local.tsv", class_ids)
    stats = np.load(runs / "train_stats.npz")
    mean, std = stats["mean"], stats["std"]
    choices = layer_choices([r["dev_pairs"] for r in json.loads((runs / "probe.json").read_text())["layers"]])
    inputs = runs / "inputs"

    missing = [s for s in SETS if args.rebuild or any(not (inputs / folder_name(c) / f"{s}.npy").exists()
                                                      for c in choices)]
    for s in missing:
        t0 = time.perf_counter()
        build_inputs(open_set(features, s, storage_scale), mean, std,
                     {folder_name(c): layers for c, layers in choices.items()}, inputs)
        print(f"inputs for {s} in {time.perf_counter() - t0:.0f} s", flush=True)

    def data(choice, set_names):
        ids, xs = [], []
        for s in set_names:
            i, x = load_inputs(inputs / folder_name(choice), s)
            ids += i
            xs.append(x)
        x = np.concatenate(xs)
        return x, np.array([info[i].label for i in ids]), [info[i].speaker for i in ids]

    def run(choice, label, config):
        x, y, _ = data(choice, ("train", "train_aug"))
        dx, dy, dspk = data(choice, ("dev_pairs",))
        t0 = time.perf_counter()
        model, history = train_head(x, y, dx, dy, dspk, len(class_ids), config, log=lambda m: None)
        cx, cy, cspk = data(choice, ("dev",))
        history["dev_clips"] = balanced_accuracy(cy, predict(model, cx).argmax(1), cspk, len(class_ids))
        history["seconds"] = round(time.perf_counter() - t0)
        print(f"  {choice:<14} {label:<26} dev pairs {history['best_dev']:6.1%} (epoch {history['best_epoch']:>2})"
              f"   dev clips {history['dev_clips']:6.1%}   {history['seconds']:>4} s", flush=True)
        return model, history

    results = {"choices": choices, "layer_choice": {}, "settings": {}}
    print("1. layer choices, linear classifier:", flush=True)
    for choice in choices:
        results["layer_choice"][choice] = run(choice, "linear", SETTINGS["linear"])[1]
    best_choice = max(results["layer_choice"], key=lambda c: results["layer_choice"][c]["best_dev"])
    print(f"2. classifier settings on {best_choice}:", flush=True)
    models = {}
    for label, config in SETTINGS.items():
        models[label], results["settings"][label] = run(best_choice, label, config)
    best = max(results["settings"], key=lambda s: results["settings"][s]["best_dev"])
    results["chosen"] = {"layers": choices[best_choice], "layer_choice": best_choice, "setting": best}
    config = SETTINGS[best]
    layers = choices[best_choice]
    save_head(runs / "head.pt", models[best], class_ids, layers, mean, std, config)
    (runs / "head.json").write_text(json.dumps(results, indent=1))
    print(f"chosen: {best_choice}, {best} (dev pairs {results['settings'][best]['best_dev']:.1%}); "
          f"wrote {runs / 'head.pt'}")


if __name__ == "__main__":
    main()
