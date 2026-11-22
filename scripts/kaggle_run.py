"""Runs on Kaggle (copied there as run.py by scripts/kaggle_bundle.py): base-model features on GPU.

Steps, each checked before the next:
  1. find the bundle in /kaggle/input (the uploaded bundle.tar, or a folder if Kaggle unpacked it)
     and check its SHA-256 against manifest.json
  2. unpack it safely to /tmp/work and check every file against bundle/files.sha256
  3. install the pinned libraries (torch, numpy and scipy stay Kaggle's own; their versions are logged)
  4. download the base model at the pinned revision and check its files' SHA-256
  5. unpack the noise collection (checked against augment.yaml)
  6. compute the reference items and compare them with the laptop's CPU results; stop if they differ
  7. extract everything, one process per GPU, then write features/summary.json with checksums
Output: /kaggle/working/features/ (download with `kaggle kernels output`).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

INPUT = Path("/kaggle/input")
WORK = Path("/tmp/work")
OUT = Path("/kaggle/working/features")
CHUNK = 8 << 20
MIN_COSINE = 0.9999                # reference check: GPU vs laptop CPU, after float16 storage
MAX_RELATIVE_DIFFERENCE = 0.02


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_of(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


def find_bundle(root: Path) -> tuple[Path | None, Path | None, dict]:
    """(bundle.tar or None, unpacked bundle folder or None, manifest)."""
    manifests = [p for p in root.rglob("manifest.json") if "accent-map" in p.read_text(encoding="utf-8")]
    if len(manifests) != 1:
        raise SystemExit(f"expected one accent-map manifest.json under {root}, found {manifests}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    folder = manifests[0].parent
    tar = folder / "bundle.tar"
    if tar.exists():
        return tar, None, manifest
    unpacked = [p.parent for p in folder.rglob("files.sha256")]
    if len(unpacked) != 1:
        raise SystemExit(f"no bundle.tar and no single unpacked bundle under {folder}")
    return None, unpacked[0], manifest


def safe_untar(tar_path: Path, dest: Path) -> Path:
    """Unpack only regular files and folders under bundle/, never outside `dest`."""
    with tarfile.open(tar_path) as tar:
        for member in tar:
            p = PurePosixPath(member.name)
            if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "bundle":
                raise SystemExit(f"unsafe member in bundle.tar: {member.name}")
            if member.isdir():
                continue
            if not member.isreg():
                raise SystemExit(f"not a regular file in bundle.tar: {member.name}")
            target = dest.joinpath(*p.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
    return dest / "bundle"


def check_files(bundle: Path) -> int:
    """Every file listed in files.sha256 exists with that hash, and nothing else is there."""
    listed = {}
    for line in (bundle / "files.sha256").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    present = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()} - {"files.sha256"}
    if present != set(listed):
        raise SystemExit(f"bundle files differ from files.sha256: missing {sorted(set(listed) - present)[:5]}, "
                         f"extra {sorted(present - set(listed))[:5]}")
    bad = [rel for rel, digest in listed.items() if sha256_of(bundle / rel) != digest]
    if bad:
        raise SystemExit(f"{len(bad)} bundle files have the wrong checksum, e.g. {bad[:5]}")
    return len(listed)


def main() -> None:
    t_start = time.time()
    log(f"python {sys.version.split()[0]}")
    tar, unpacked, manifest = find_bundle(INPUT)
    log(f"bundle {manifest['name']} built {manifest['built']}, {manifest['items']} items")
    WORK.mkdir(parents=True, exist_ok=True)
    if tar is not None:
        log("checking bundle.tar checksum")
        if sha256_of(tar) != manifest["bundle_sha256"]:
            raise SystemExit("bundle.tar checksum does not match manifest.json")
        bundle = safe_untar(tar, WORK)
    else:
        bundle = unpacked
    log(f"checked {check_files(bundle)} bundle files")

    log("installing pinned libraries")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(bundle / "requirements-kaggle.txt")],
                   check=True)
    sys.path.insert(0, str(bundle / "code" / "src"))
    import numpy as np
    import scipy
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    from training.archives import extract_zip
    from training.augment import load_augment_spec
    from training.extract_job import compare
    from training.features import load_features_spec

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
           "numpy": np.__version__, "scipy": scipy.__version__, "transformers": transformers.__version__, "gpus": gpus}
    log(f"environment: {env}")
    if not gpus:
        raise SystemExit("no GPU: turn on a GPU accelerator for this notebook")

    features_spec = load_features_spec(bundle / "features.yaml")
    model_dir = WORK / "model"
    for name, digest in features_spec.files_sha256.items():
        path = Path(hf_hub_download(features_spec.base_model, name, revision=features_spec.revision,
                                    local_dir=model_dir))
        if sha256_of(path) != digest:
            raise SystemExit(f"base model file {name} has the wrong checksum")
    log(f"base model {features_spec.base_model}@{features_spec.revision[:10]} downloaded and checked")

    augment_spec = load_augment_spec(bundle / "augment.yaml")
    noise_zip = bundle / "noise" / augment_spec.noise_archive
    noise = extract_zip(noise_zip, bundle / "noise", want=lambda n: "/largeroom/" not in n)
    if noise.sha256 != augment_spec.noise_sha256 or noise.skipped:
        raise SystemExit("noise collection checksum or contents wrong")
    noise_zip.unlink()
    log(f"noise collection unpacked ({len(noise.extracted)} files)")

    def extract(out: Path, part: int, parts: int, extra: list[str], gpu: int):
        env_vars = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(bundle / "code" / "src")}
        cmd = [sys.executable, "-m", "training.extract_job", "--bundle", str(bundle), "--model", str(model_dir),
               "--out", str(out), "--device", "cuda", "--part", str(part), "--parts", str(parts), *extra]
        return subprocess.Popen(cmd, env=env_vars)

    log("reference check")
    ref_out = WORK / "reference_out"
    if extract(ref_out, 0, 1, ["--ids", str(bundle / "reference" / "ids.txt")], 0).wait() != 0:
        raise SystemExit("reference extraction failed")
    ref_ids = (bundle / "reference" / "ids.txt").read_text().split()
    got = {}
    for ids_file in ref_out.glob("*.ids"):
        rows = np.load(ids_file.with_name(ids_file.name.removesuffix(".ids") + ".npy"))
        got.update(zip(ids_file.read_text().split(), rows))
    expected = np.load(bundle / "reference" / "features.npy")
    check = compare(np.stack([got[i] for i in ref_ids]), expected)
    log(f"reference: {check}")
    if check["min_cosine"] < MIN_COSINE or check["max_relative_difference"] > MAX_RELATIVE_DIFFERENCE:
        raise SystemExit("GPU results differ from the laptop's: stopping before the full run")

    log(f"full run on {len(gpus)} GPU(s)")
    OUT.mkdir(parents=True, exist_ok=True)
    t_run = time.time()
    procs = [extract(OUT, i, len(gpus), ["--workers", "2"], i) for i in range(len(gpus))]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"extraction failed: exit codes {codes}")
    files = {p.name: sha256_of(p) for p in sorted(OUT.iterdir()) if p.suffix in (".npy", ".ids")}
    summary = {"manifest": manifest, "environment": env, "reference_check": check,
               "run_minutes": round((time.time() - t_run) / 60, 1),
               "total_minutes": round((time.time() - t_start) / 60, 1), "files_sha256": files}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"done: {summary['run_minutes']} min for the full run, {len(files)} files")


if __name__ == "__main__":
    main()
