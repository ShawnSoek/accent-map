"""Prepare the Kaggle GPU run of step 4 (base-model features).

Usage:
  python -I scripts/kaggle_bundle.py build training/cv27_es10
  python -I scripts/kaggle_bundle.py meta training/cv27_es10 --user <your kaggle username>

build (needs the extracted clips, the noise zip and the cached base model) writes, under
data/kaggle/<name>/ (not committed):
  upload/bundle.tar       clips, item list, noise collection, code, reference results (~4 GB)
  upload/manifest.json    the tar's SHA-256 and what is inside
  local/items_local.tsv   accent, speaker and gender of every item: stays on this computer
It first checks the Common Voice archive, noise zip and base model against their pinned
checksums, and computes the reference items on this CPU (about a minute).
meta writes upload/dataset-metadata.json (a private dataset) and job/ (run.py and
kernel-metadata.json: GPU and internet on, the dataset attached). run.py gets the noise
collection's tree hash, so it can check the folder Kaggle makes when it unpacks the zip.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from training.augment import NoiseBank, load_augment_spec  # noqa: E402
from training.cv_split import load_spec  # noqa: E402
from training.extract_job import run  # noqa: E402
from training.feature_items import (SETS, Job, make_items, read_split, write_local_table,  # noqa: E402
                                    write_upload_table)
from training.features import FeatureExtractor, load_features_spec  # noqa: E402
from training.kaggle_bundle import (choose_reference, kaggle_requirements, load_run_module,  # noqa: E402
                                    run_script_with_archives, write_bundle, zip_tree_hash)

DATASET_SLUG = "accent-map-{name}"
KERNEL_SLUG = "accent-map-features-{name}"


def pinned_file_ok(checksum_file: Path, expected: str) -> bool:
    return checksum_file.exists() and checksum_file.read_text().split()[0] == expected


def build(folder: Path) -> None:
    name = folder.resolve().name
    spec = load_spec(folder / "dataset.yaml")
    aug = load_augment_spec(folder / "augment.yaml")
    feats = load_features_spec(folder / "features.yaml")
    data = REPO / "data"
    if not pinned_file_ok(data / "commonvoice" / f"{spec.release}.tar.gz.sha256", spec.archive_sha256):
        sys.exit("Common Voice archive checksum missing or different from dataset.yaml")
    noise_zip = data / "noise" / aug.noise_archive
    if not pinned_file_ok(data / "noise" / f"{aug.noise_archive}.sha256", aug.noise_sha256):
        sys.exit("noise collection checksum missing or different from augment.yaml")
    model_dir = data / "hf_cache" / f"models--{feats.base_model.replace('/', '--')}" / "snapshots" / feats.revision
    from training.archives import sha256_of
    for file, digest in feats.files_sha256.items():
        if sha256_of(model_dir / file) != digest:
            sys.exit(f"base model file {file} differs from features.yaml")
    print("archive, noise collection and base model match their pinned checksums", flush=True)

    splits = {s: read_split(data / "commonvoice" / "splits" / name / f"{s}.tsv") for s in ("train", "dev", "test")}
    items = make_items(splits, spec.split.seed)
    out = data / "kaggle" / name
    (out / "local").mkdir(parents=True, exist_ok=True)
    write_local_table(items, out / "local" / "items_local.tsv")
    counts = {s: sum(it.set == s for it in items) for s in SETS}
    print(f"{len(items)} items: {counts}", flush=True)

    clips_dir = next((data / "commonvoice" / spec.release).rglob("validated.tsv")).parent / "clips"
    reference = choose_reference(items)
    t0 = time.perf_counter()
    extractor = FeatureExtractor(model_dir, "cpu", feats.storage_scale)
    bank = NoiseBank(data / "noise" / aug.noise_archive.removesuffix(".zip"), aug.reverb.simulated_rooms)
    jobs = [Job(it.id, it.set, it.clips, it.augment) for it in reference]
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        run(jobs, clips_dir, Path(tmp), extractor, (extractor.layers, 2 * extractor.width), aug, bank,
            log=lambda m: None)
        got = {}
        for ids_file in Path(tmp).glob("*.ids"):
            rows = np.load(ids_file.with_name(ids_file.name.removesuffix(".ids") + ".npy"))
            got.update(zip(ids_file.read_text().split(), rows))
    ref_rows = np.stack([got[it.id] for it in reference])
    print(f"{len(reference)} reference items computed on this CPU in {time.perf_counter() - t0:.0f} s", flush=True)

    with tempfile.TemporaryDirectory(dir=out) as tmp:
        items_tsv, ref_npy = Path(tmp) / "items.tsv", Path(tmp) / "features.npy"
        write_upload_table(items, items_tsv)
        np.save(ref_npy, ref_rows)
        import torch
        files = {"items.tsv": items_tsv, "augment.yaml": folder / "augment.yaml",
                 "features.yaml": folder / "features.yaml", f"noise/{aug.noise_archive}": noise_zip,
                 "reference/features.npy": ref_npy}
        needed = sorted({c for it in items for c in it.clips})
        files.update({f"clips/{c}": clips_dir / c for c in needed})
        for src in sorted((REPO / "src" / "training").glob("*.py")):
            files[f"code/src/training/{src.name}"] = src
        for src in ("__init__.py", "audio.py"):
            files[f"code/src/server/{src}"] = REPO / "src" / "server" / src
        texts = {"reference/ids.txt": "".join(f"{it.id}\n" for it in reference),
                 "reference/info.json": json.dumps({"device": "cpu", "torch": torch.__version__,
                                                    "threads": torch.get_num_threads()}),
                 "requirements-kaggle.txt": kaggle_requirements(REPO / "requirements.txt")}
        print(f"writing bundle.tar: {len(needed)} clips, {len(files) + len(texts) - len(needed)} other files",
              flush=True)
        digest, n_files = write_bundle(out / "upload" / "bundle.tar", files, texts)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                text=True).stdout.strip())
    manifest = {"name": f"accent-map {name} features bundle", "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git_commit": commit, "git_dirty": dirty, "items": len(items), "sets": counts,
                "files": n_files, "bundle_sha256": digest,
                "bundle_bytes": (out / "upload" / "bundle.tar").stat().st_size}
    (out / "upload" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {out / 'upload'} ({manifest['bundle_bytes'] / 1e9:.2f} GB, sha256 {digest[:16]}...)")


def meta(folder: Path, user: str) -> None:
    name = folder.resolve().name
    slug = name.replace("_", "-")
    out = REPO / "data" / "kaggle" / name
    if not (out / "upload" / "manifest.json").exists():
        sys.exit("run build first")
    dataset = f"{user}/{DATASET_SLUG.format(name=slug)}"
    (out / "upload" / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET_SLUG.format(name=slug), "id": dataset, "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    job = out / "job"
    job.mkdir(exist_ok=True)
    aug = load_augment_spec(folder / "augment.yaml")
    run_py = REPO / "scripts" / "kaggle_run.py"
    tree = zip_tree_hash(REPO / "data" / "noise" / aug.noise_archive, load_run_module(run_py))
    (job / "run.py").write_text(run_script_with_archives(run_py, {f"noise/{aug.noise_archive}": tree}),
                                encoding="utf-8")
    print(f"noise collection tree hash {tree[0][:16]}... over {tree[1]} files, written into run.py")
    kernel = KERNEL_SLUG.format(name=slug)
    (job / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{kernel}", "title": kernel, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": "", "dataset_sources": [dataset], "competition_sources": [],
        "kernel_sources": [], "model_sources": []}, indent=1))
    print(f"dataset {dataset} (private), kernel {user}/{kernel}; files in {out / 'upload'} and {job}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "meta"])
    ap.add_argument("folder", type=Path, help="training/<name>")
    ap.add_argument("--user", help="your Kaggle username (for meta)")
    args = ap.parse_args()
    if args.command == "build":
        build(args.folder)
    else:
        if not args.user:
            sys.exit("meta needs --user")
        meta(args.folder, args.user)


if __name__ == "__main__":
    main()
