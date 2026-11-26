"""Check the features the GPU job wrote (stage 2, step 4) before training uses them.

The output folder holds, per set and part, <set>.part<i>of<n>.npy / .ids / .done (see
extract_job.py) and summary.json with the SHA-256 of every .npy and .ids file. check_output
stops with FeatureCheckError unless all of this holds:
  - summary.json names the bundle we built (same SHA-256), and every file has its checksum;
  - every set has all n parts, and part i lists exactly the set's items i, i+n, i+2n, ... in the
    order of the local item table (the rule extract_job.run uses), so no item is missing or twice;
  - each array is float16 with one row per id, every row was written (.done), no row is all
    zeros (never filled) and every value is finite;
  - the full run's rows for the reference items match the laptop's results stored in the bundle.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.archives import sha256_of
from training.extract_job import compare
from training.feature_items import SETS

MIN_COSINE = 0.9999                 # the same limits as the reference check on Kaggle
MAX_RELATIVE_DIFFERENCE = 0.02
PART_NAME = re.compile(r"^(?P<set>[a-z_]+)\.part(?P<i>\d+)of(?P<n>\d+)\.(?P<ext>npy|ids|done)$")


class FeatureCheckError(Exception):
    pass


def part_files(out_dir: Path) -> dict[str, int]:
    """Set -> number of parts, after checking that every part has its .npy, .ids and .done."""
    found = defaultdict(set)
    for p in Path(out_dir).iterdir():
        m = PART_NAME.match(p.name)
        if m:
            found[(m["set"], int(m["n"]))].add((int(m["i"]), m["ext"]))
    parts = {}
    for (set_name, n), got in found.items():
        if set_name in parts:
            raise FeatureCheckError(f"{set_name}: parts of two different runs ({parts[set_name]} and {n})")
        want = {(i, ext) for i in range(n) for ext in ("npy", "ids", "done")}
        if got != want:
            raise FeatureCheckError(f"{set_name}: missing {sorted(want - got)[:5]}, extra {sorted(got - want)[:5]}")
        parts[set_name] = n
    return parts


def check_output(out_dir: Path, expected_ids: dict[str, list[str]], bundle_sha256: str, shape: tuple[int, int],
                 reference: tuple[list[str], np.ndarray], chunk: int = 512, log=print) -> dict:
    """See the module docstring. `expected_ids`: set -> item ids in the local table's order."""
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["manifest"]["bundle_sha256"] != bundle_sha256:
        raise FeatureCheckError("summary.json belongs to another bundle")

    listed = summary["files_sha256"]
    present = {p.name for p in out_dir.iterdir() if p.suffix in (".npy", ".ids")}
    if present != set(listed):
        raise FeatureCheckError(f"files differ from summary.json: missing {sorted(set(listed) - present)[:5]}, "
                                f"extra {sorted(present - set(listed))[:5]}")
    for name in sorted(listed):
        if sha256_of(out_dir / name) != listed[name]:
            raise FeatureCheckError(f"{name} has the wrong checksum")
    log(f"{len(listed)} files match the checksums in summary.json")

    parts = part_files(out_dir)
    if set(parts) != set(expected_ids):
        raise FeatureCheckError(f"sets {sorted(parts)} differ from the item table's {sorted(expected_ids)}")
    where = {}                                       # item id -> (file, row)
    largest = np.zeros(shape[0], np.float32)         # largest |value| per layer, as stored
    for set_name in [s for s in SETS if s in parts]:
        n = parts[set_name]
        for i in range(n):
            stem = out_dir / f"{set_name}.part{i}of{n}"
            ids = (stem.parent / f"{stem.name}.ids").read_text(encoding="utf-8").split()
            if ids != expected_ids[set_name][i::n]:
                raise FeatureCheckError(f"{stem.name}.ids does not list the items of this part in order")
            rows = np.load(stem.parent / f"{stem.name}.npy", mmap_mode="r")
            if rows.dtype != np.float16 or rows.shape != (len(ids), *shape):
                raise FeatureCheckError(f"{stem.name}.npy is {rows.dtype} {rows.shape}, "
                                        f"expected float16 {(len(ids), *shape)}")
            done = int((stem.parent / f"{stem.name}.done").read_text())
            if done != len(ids):
                raise FeatureCheckError(f"{stem.name}: {done} of {len(ids)} rows written")
            for start in range(0, len(ids), chunk):
                block = np.asarray(rows[start:start + chunk], np.float32)
                if not np.isfinite(block).all():
                    raise FeatureCheckError(f"{stem.name}.npy has values that are not finite")
                empty = np.flatnonzero(~block.any(axis=(1, 2)))
                if empty.size:
                    raise FeatureCheckError(f"{stem.name}.npy row {start + empty[0]} is all zeros (never filled)")
                largest = np.maximum(largest, np.abs(block).max(axis=(0, 2)))
            where.update((item, (stem.name, k)) for k, item in enumerate(ids))
        log(f"{set_name}: {len(expected_ids[set_name])} items in {n} parts")

    ref_ids, ref_rows = reference
    missing = [r for r in ref_ids if r not in where]
    if missing:
        raise FeatureCheckError(f"reference items missing from the output: {missing[:5]}")
    loaded = {}
    got = []
    for item in ref_ids:
        name, k = where[item]
        if name not in loaded:
            loaded[name] = np.load(out_dir / f"{name}.npy", mmap_mode="r")
        got.append(np.asarray(loaded[name][k]))
    ref = compare(ref_rows, np.stack(got))
    if ref["min_cosine"] < MIN_COSINE or ref["max_relative_difference"] > MAX_RELATIVE_DIFFERENCE:
        raise FeatureCheckError(f"the full run differs from the laptop's results on the reference items: {ref}")
    log(f"{len(ref_ids)} reference items match the laptop's results: {ref}")

    top = int(largest.argmax())
    return {"bundle_sha256": bundle_sha256, "files": len(listed), "items": len(where),
            "sets": {s: len(expected_ids[s]) for s in SETS if s in parts}, "parts": parts,
            "reference": ref, "largest_stored_value": {"layer": top, "value": float(largest[top])},
            "kaggle": {k: summary[k] for k in ("environment", "reference_check", "run_minutes", "total_minutes")}}
