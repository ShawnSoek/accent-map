"""Check the features downloaded from the Kaggle job before training uses them (stage 2, step 4).

Usage:
  python -I scripts/check_features.py training/cv27_es10 [--out data/features/cv27_es10]

Reads data/kaggle/<name>/local/items_local.tsv (which items each set should have),
upload/manifest.json (which bundle the run must have used) and the reference results inside
upload/bundle.tar, and the base model's config.json for the feature shape. See
src/training/feature_check.py for what is checked. On success it writes <out>/verified.json;
training reads only features that have it.
"""

import argparse
import json
import sys
import tarfile
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from training.feature_check import FeatureCheckError, check_output  # noqa: E402
from training.feature_items import read_split  # noqa: E402
from training.features import load_features_spec  # noqa: E402


def read_reference(bundle_tar: Path) -> tuple[list[str], np.ndarray]:
    with tarfile.open(bundle_tar) as t:
        ids = t.extractfile("bundle/reference/ids.txt").read().decode("utf-8").split()
        rows = np.load(BytesIO(t.extractfile("bundle/reference/features.npy").read()))
    return ids, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="training/<name>")
    ap.add_argument("--out", type=Path, help="the downloaded features (default data/features/<name>)")
    args = ap.parse_args()
    name = args.folder.resolve().name
    out = args.out or REPO / "data" / "features" / name
    kaggle = REPO / "data" / "kaggle" / name

    expected = defaultdict(list)
    for row in read_split(kaggle / "local" / "items_local.tsv"):
        expected[row["set"]].append(row["id"])
    manifest = json.loads((kaggle / "upload" / "manifest.json").read_text())
    feats = load_features_spec(args.folder / "features.yaml")
    model_dir = (REPO / "data" / "hf_cache" / f"models--{feats.base_model.replace('/', '--')}" / "snapshots"
                 / feats.revision)
    config = json.loads((model_dir / "config.json").read_text())
    shape = (config["num_hidden_layers"] + 1, 2 * config["hidden_size"])
    reference = read_reference(kaggle / "upload" / "bundle.tar")
    print(f"expecting {sum(map(len, expected.values()))} items of shape {shape}; "
          f"{len(reference[0])} reference items", flush=True)

    t0 = time.perf_counter()
    try:
        report = check_output(out, dict(expected), manifest["bundle_sha256"], shape, reference,
                              log=lambda m: print(m, flush=True))
    except FeatureCheckError as e:
        sys.exit(f"NOT OK: {e}")
    report["checked"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "verified.json").write_text(json.dumps(report, indent=1))
    print(f"OK in {time.perf_counter() - t0:.0f} s; largest stored value {report['largest_stored_value']}; "
          f"wrote {out / 'verified.json'}")


if __name__ == "__main__":
    main()
