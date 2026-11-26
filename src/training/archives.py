"""Read downloaded archives safely: a .tar.gz in one streaming pass (`scan`), or a .zip (`extract_zip`).

Both compute the archive's SHA-256 and extract the members the caller asks for. Archives are
downloaded data, so only regular files with plain relative paths are written, inside the output
folder: links, devices, absolute paths and ".." are skipped.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

CHUNK = 8 << 20


class _HashingReader:
    """A file wrapper that hashes every byte read through it."""

    def __init__(self, f):
        self.f = f
        self.sha = hashlib.sha256()
        self.bytes = 0

    def read(self, n: int = -1) -> bytes:
        data = self.f.read(n)
        self.sha.update(data)
        self.bytes += len(data)
        return data


@dataclass
class ScanResult:
    sha256: str
    archive_bytes: int
    members: int = 0
    extracted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)     # asked for, but unsafe or not a regular file


def safe_relative(name: str) -> PurePosixPath | None:
    """The member's path if it stays inside the output folder, else None."""
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return None
    return path


def scan(archive: Path, out_dir: Path, want: Callable[[str], bool], listing: Path | None = None,
         progress: Callable[[int], None] | None = None) -> ScanResult:
    """Hash and list `archive`, extracting members whose name passes `want` into `out_dir`.

    `listing` gets one line per member: name, size in bytes, type (f = file, d = folder, o = other).
    `progress` is called with the number of archive bytes read so far, about every 1000 members.
    """
    out_dir = Path(out_dir)
    result = ScanResult(sha256="", archive_bytes=0)
    with open(archive, "rb") as raw:
        reader = _HashingReader(raw)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar, \
                (open(listing, "w", encoding="utf-8") if listing else _Nowhere()) as out:
            for member in tar:
                result.members += 1
                kind = "f" if member.isreg() else "d" if member.isdir() else "o"
                out.write(f"{member.name}\t{member.size}\t{kind}\n")
                if progress and result.members % 1000 == 0:
                    progress(reader.bytes)
                if not want(member.name):
                    continue
                rel = safe_relative(member.name)
                if rel is None or not member.isreg():
                    if not (rel is not None and member.isdir()):    # folders are made as needed
                        result.skipped.append(member.name)
                    continue
                target = out_dir.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                result.extracted.append(str(rel))
        while reader.read(CHUNK):                     # the rest after the end marker, for the hash
            pass
    result.sha256 = reader.sha.hexdigest()
    result.archive_bytes = reader.bytes
    return result


def sha256_of(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


def extract_zip(archive: Path, out_dir: Path, want: Callable[[str], bool] = lambda name: True,
                max_bytes: int = 20 * 10**9) -> ScanResult:
    """Extract wanted regular files of a zip into `out_dir`. Refuses archives that would unpack
    to more than `max_bytes` (counting what is wanted), and checks each file's size as it goes."""
    out_dir = Path(out_dir)
    result = ScanResult(sha256=sha256_of(archive), archive_bytes=Path(archive).stat().st_size)
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        result.members = len(infos)
        wanted = [i for i in infos if want(i.filename) and not i.is_dir()]
        total = sum(i.file_size for i in wanted)
        if total > max_bytes:
            raise ValueError(f"{archive} would unpack to {total / 1e9:.1f} GB, more than {max_bytes / 1e9:.1f} GB")
        for info in wanted:
            rel = safe_relative(info.filename)
            is_link = (info.external_attr >> 16) & 0o170000 == 0o120000
            if rel is None or is_link:
                result.skipped.append(info.filename)
                continue
            target = out_dir.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                written = 0
                while chunk := src.read(CHUNK):
                    written += len(chunk)
                    if written > info.file_size:
                        raise ValueError(f"{info.filename} is larger than its header says")
                    dst.write(chunk)
            result.extracted.append(str(rel))
    return result


class _Nowhere:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, text: str) -> None:
        pass
