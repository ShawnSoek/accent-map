"""Export the evaluated classifier for the app (stage 2, step 6).

Usage:
  python -I scripts/export_head.py training/cv27_es10 models/cv27_es10

Reads data/runs/<name>/head.pt (after scripts/evaluate_head.py has set its temperature) and
evaluation.json, and writes <model folder>/head.safetensors: tensors and a text header only, so
loading it cannot run code. Prints the SHA-256 to pin in the model's config.yaml.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from training.cv_split import load_spec  # noqa: E402
from training.features import load_features_spec  # noqa: E402
from training.head import TrainedHead, export_head  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="training/<name>")
    ap.add_argument("model_folder", type=Path, help="models/<model name>")
    args = ap.parse_args()
    name = args.folder.resolve().name
    runs = REPO / "data" / "runs" / name
    evaluation = json.loads((runs / "evaluation.json").read_text())
    head = TrainedHead(runs / "head.pt")
    if head.temperature == 1.0 or evaluation["head"]["temperature"] != head.temperature:
        sys.exit("head.pt has no fitted temperature or differs from evaluation.json: run evaluate_head.py")
    spec = load_spec(args.folder / "dataset.yaml")
    feats = load_features_spec(args.folder / "features.yaml")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    test = evaluation["sets"]["test_pairs"]
    info = {"trained_on": f"{spec.release} (Common Voice), training/{name}", "git_commit": commit,
            "base_model": f"{feats.base_model}@{feats.revision}",
            "test_pairs_balanced_accuracy": round(test["balanced_accuracy"], 4),
            "test_pairs_interval_95": [round(v, 4) for v in test["interval_95"]]}
    args.model_folder.mkdir(parents=True, exist_ok=True)
    out = args.model_folder / "head.safetensors"
    export_head(head, out, feats.storage_scale, info)
    check = TrainedHead(out)
    if (check.class_ids, check.layers, check.temperature) != (head.class_ids, head.layers, head.temperature):
        sys.exit("the exported file reads back differently")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB): layers {head.layers}, temperature {head.temperature:.2f}")
    print(f"head_sha256: {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
