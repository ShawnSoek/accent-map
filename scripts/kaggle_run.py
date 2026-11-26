"""Runs on Kaggle (copied there as run.py by scripts/kaggle_bundle.py): base-model features on GPU.

Steps, each checked before the next:
  1. find the bundle in /kaggle/input. Kaggle unpacks uploaded archives: the bundle.tar, and the
     noise zip inside it. A packed bundle.tar is checked against manifest.json and unpacked here.
  2. check every bundle file against bundle/files.sha256. An archive Kaggle unpacked is checked
     as a whole: one SHA-256 over the sorted list of its files' checksums (UNPACKED_ARCHIVES).
  3. install the pinned libraries (torch, numpy and scipy stay Kaggle's own; their versions are logged)
  4. lay out a working view in /tmp/work (links only: /kaggle/input is read-only), unpacking the
     noise zip there if it is still packed (then checked against augment.yaml)
  5. download the base model at the pinned revision and check its files' SHA-256
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
SEARCH_DEPTH = 4                   # /kaggle/input/<dataset>/... or /kaggle/input/datasets/<user>/<dataset>/...

# Archive inside the bundle -> [SHA-256 over its sorted file checksums, file count], for when Kaggle
# has unpacked it. Filled in by `scripts/kaggle_bundle.py meta` from the pinned archive.
UNPACKED_ARCHIVES: dict[str, list] = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_of(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


def tree_listing(entries: dict[str, str]) -> tuple[str, int]:
    """SHA-256 over "sha256  path" lines in path order, and how many files: one hash for a folder."""
    text = "".join(f"{entries[name]}  {name}\n" for name in sorted(entries))
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(entries)


def tree_hash(folder: Path) -> tuple[str, int]:
    return tree_listing({p.relative_to(folder).as_posix(): sha256_of(p) for p in folder.rglob("*") if p.is_file()})


def shallow(root: Path, name: str, depth: int = SEARCH_DEPTH) -> list[Path]:
    """`name` at most `depth` folders below `root` (a full rglob of /kaggle/input takes minutes)."""
    return [p for d in range(0, depth + 1) for p in root.glob("/".join(["*"] * d + [name]))]


def find_bundle(root: Path) -> tuple[Path | None, Path | None, dict]:
    """(bundle.tar or None, unpacked bundle folder or None, manifest)."""
    manifests = [p for p in shallow(root, "manifest.json") if "accent-map" in p.read_text(encoding="utf-8")]
    if len(manifests) != 1:
        raise SystemExit(f"expected one accent-map manifest.json under {root}, found {manifests}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    folder = manifests[0].parent
    tar = folder / "bundle.tar"
    if tar.exists():
        return tar, None, manifest
    unpacked = [p.parent for p in shallow(folder, "files.sha256", 3)]
    if len(unpacked) != 1:
        raise SystemExit(f"no bundle.tar and no single unpacked bundle under {folder}: {unpacked}")
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


def check_files(bundle: Path, unpacked_archives: dict[str, list] | None = None) -> int:
    """Every file listed in files.sha256 exists with that hash, and nothing else is there. A listed
    archive that was unpacked into a folder of the same name is checked by its tree hash instead."""
    unpacked_archives = UNPACKED_ARCHIVES if unpacked_archives is None else unpacked_archives
    listed = {}
    for line in (bundle / "files.sha256").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    present = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} - {"files.sha256"}
    checked_as_tree = set()
    for archive, (tree, count) in unpacked_archives.items():
        folder = archive.removesuffix(".zip")
        inside = {p for p in present if p.startswith(folder + "/")}
        if archive in present or not inside or archive not in listed:
            continue
        if tree_hash(bundle / folder) != (tree, count):
            raise SystemExit(f"{folder}/ (Kaggle's unpacking of {archive}) differs from the pinned archive")
        present = (present - inside) | {archive}
        checked_as_tree.add(archive)
    if present != set(listed):
        raise SystemExit(f"bundle files differ from files.sha256: missing {sorted(set(listed) - present)[:5]}, "
                         f"extra {sorted(present - set(listed))[:5]}")
    bad = [rel for rel, digest in listed.items() if rel not in checked_as_tree and sha256_of(bundle / rel) != digest]
    if bad:
        raise SystemExit(f"{len(bad)} bundle files have the wrong checksum, e.g. {bad[:5]}")
    return len(listed)


def verify_bundle(input_root: Path, work: Path) -> tuple[Path, dict]:
    """Steps 1-2 (standard library only)."""
    tar, unpacked, manifest = find_bundle(input_root)
    log(f"bundle {manifest['name']} built {manifest['built']}, {manifest['items']} items")
    work.mkdir(parents=True, exist_ok=True)
    if tar is not None:
        log("checking bundle.tar checksum")
        if sha256_of(tar) != manifest["bundle_sha256"]:
            raise SystemExit("bundle.tar checksum does not match manifest.json")
        bundle = safe_untar(tar, work)
    else:
        bundle = unpacked
        log(f"Kaggle unpacked the bundle: {bundle}")
    log(f"checked {check_files(bundle)} bundle files")
    return bundle, manifest


def make_view(bundle: Path, work: Path) -> Path:
    """Step 4: a folder in `work` laid out as extract_job expects, linking to the (read-only) bundle.
    Needs the bundle's code on sys.path."""
    from training.archives import extract_zip
    from training.augment import load_augment_spec

    spec = load_augment_spec(bundle / "augment.yaml")
    packed = bundle / "noise" / spec.noise_archive
    if packed.exists():
        noise_root = work / "noise"
        noise = extract_zip(packed, noise_root, want=lambda n: "/largeroom/" not in n)
        if noise.sha256 != spec.noise_sha256 or noise.skipped:
            raise SystemExit("noise collection checksum or contents wrong")
        log(f"noise collection unpacked ({len(noise.extracted)} files)")
    else:
        noise_root = bundle / "noise" / spec.noise_archive.removesuffix(".zip")     # checked in step 2
        log(f"noise collection already unpacked by Kaggle: {noise_root}")
    view = work / "view"
    if view.exists():
        shutil.rmtree(view)
    view.mkdir(parents=True)
    for name in ("items.tsv", "augment.yaml", "features.yaml", "clips"):
        (view / name).symlink_to(bundle / name)
    (view / "noise").symlink_to(noise_root)
    return view


def fetch_model(bundle: Path, model_dir: Path) -> None:
    """Step 5."""
    from huggingface_hub import hf_hub_download

    from training.features import load_features_spec

    spec = load_features_spec(bundle / "features.yaml")
    for name, digest in spec.files_sha256.items():
        path = Path(hf_hub_download(spec.base_model, name, revision=spec.revision, local_dir=model_dir))
        if sha256_of(path) != digest:
            raise SystemExit(f"base model file {name} has the wrong checksum")
    log(f"base model {spec.base_model}@{spec.revision[:10]} downloaded and checked")


def extract(bundle: Path, view: Path, model_dir: Path, out: Path, part: int, parts: int, extra: list[str],
            gpu: int | None, device: str = "cuda") -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(bundle / "code" / "src")}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, "-m", "training.extract_job", "--bundle", str(view), "--model", str(model_dir),
           "--out", str(out), "--device", device, "--part", str(part), "--parts", str(parts), *extra]
    return subprocess.Popen(cmd, env=env, cwd=str(view.parent))


def reference_check(bundle: Path, view: Path, model_dir: Path, work: Path, device: str = "cuda") -> dict:
    """Step 6: raises SystemExit if this machine's results differ from the laptop's."""
    import numpy as np

    from training.extract_job import compare

    ref_out = work / "reference_out"
    if ref_out.exists():
        shutil.rmtree(ref_out)
    gpu = 0 if device == "cuda" else None
    if extract(bundle, view, model_dir, ref_out, 0, 1, ["--ids", str(bundle / "reference" / "ids.txt")],
               gpu, device).wait() != 0:
        raise SystemExit("reference extraction failed")
    ref_ids = (bundle / "reference" / "ids.txt").read_text().split()
    got = {}
    for ids_file in ref_out.glob("*.ids"):
        rows = np.load(ids_file.with_name(ids_file.name.removesuffix(".ids") + ".npy"))
        got.update(zip(ids_file.read_text().split(), rows))
    check = compare(np.stack([got[i] for i in ref_ids]), np.load(bundle / "reference" / "features.npy"))
    log(f"reference: {check}")
    if check["min_cosine"] < MIN_COSINE or check["max_relative_difference"] > MAX_RELATIVE_DIFFERENCE:
        raise SystemExit("results differ from the laptop's: stopping before the full run")
    return check


def main() -> None:
    t_start = time.time()
    log(f"python {sys.version.split()[0]}")
    bundle, manifest = verify_bundle(INPUT, WORK)

    log("installing pinned libraries")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(bundle / "requirements-kaggle.txt")],
                   check=True)
    sys.path.insert(0, str(bundle / "code" / "src"))
    import numpy as np
    import scipy
    import torch
    import transformers

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
           "numpy": np.__version__, "scipy": scipy.__version__, "transformers": transformers.__version__, "gpus": gpus}
    log(f"environment: {env}")
    if not gpus:
        raise SystemExit("no GPU: turn on a GPU accelerator for this notebook")

    view = make_view(bundle, WORK)
    model_dir = WORK / "model"
    fetch_model(bundle, model_dir)
    log("reference check")
    check = reference_check(bundle, view, model_dir, WORK)

    log(f"full run on {len(gpus)} GPU(s)")
    OUT.mkdir(parents=True, exist_ok=True)
    t_run = time.time()
    procs = [extract(bundle, view, model_dir, OUT, i, len(gpus), ["--workers", "2"], i) for i in range(len(gpus))]
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
