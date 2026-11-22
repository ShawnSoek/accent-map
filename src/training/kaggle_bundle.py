"""Pack everything the Kaggle GPU job needs into one bundle.tar (see scripts/kaggle_bundle.py).

Inside the tar, under bundle/: items.tsv (ids, sets and clip files only: no speaker ids, accents
or genders), augment.yaml, features.yaml, clips/, noise/<collection>.zip, code/src/..., the
reference results computed on the laptop, requirements-kaggle.txt, and files.sha256 listing the
SHA-256 of every other file.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

from training.archives import sha256_of
from training.cv_split import rank
from training.feature_items import SETS, Item

# Pinned like requirements.txt. torch, numpy and scipy stay Kaggle's own (their CUDA build).
KAGGLE_PACKAGES = ("transformers", "tokenizers", "huggingface-hub", "safetensors", "soundfile", "soxr",
                   "pydantic", "pydantic-core", "pyyaml")


def kaggle_requirements(requirements_txt: Path) -> str:
    pins = {}
    for line in Path(requirements_txt).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name = line.split("==", 1)[0].strip().lower()
            pins[name] = line
    missing = [p for p in KAGGLE_PACKAGES if p not in pins]
    if missing:
        raise ValueError(f"not pinned in {requirements_txt}: {missing}")
    return "".join(f"{pins[p]}\n" for p in KAGGLE_PACKAGES)


def choose_reference(items: list[Item], per_set: int = 8) -> list[Item]:
    """A fixed handful of items of every set, to compare the GPU's results with the laptop's."""
    out = []
    for s in SETS:
        group = [it for it in items if it.set == s]
        out += sorted(group, key=lambda it: rank("reference", it.id))[:per_set]
    return out


def write_bundle(tar_path: Path, files: dict[str, Path], texts: dict[str, str | bytes]) -> tuple[str, int]:
    """Write bundle.tar with `files` (name inside bundle/ -> source path) and `texts` (name -> content),
    plus files.sha256. Returns (SHA-256 of the tar, number of files listed)."""
    if set(files) & set(texts) or "files.sha256" in files or "files.sha256" in texts:
        raise ValueError("duplicate or reserved names")
    blobs = {name: (t.encode("utf-8") if isinstance(t, str) else t) for name, t in texts.items()}
    digests = {name: sha256_of(path) for name, path in files.items()}
    digests.update({name: hashlib.sha256(b).hexdigest() for name, b in blobs.items()})
    listing = "".join(f"{digests[n]}  {n}\n" for n in sorted(digests)).encode("utf-8")
    blobs["files.sha256"] = listing
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:                       # no compression: clips are MP3 already
        for name in sorted(files):
            tar.add(files[name], arcname=f"bundle/{name}", recursive=False)
        for name in sorted(blobs):
            info = tarfile.TarInfo(f"bundle/{name}")
            info.size = len(blobs[name])
            tar.addfile(info, io.BytesIO(blobs[name]))
    return sha256_of(tar_path), len(digests)
