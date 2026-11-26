"""Reading a Common Voice archive: hash, listing, and only safe, wanted members extracted."""

import hashlib
import io
import tarfile

from training.archives import safe_relative, scan


def add(tar, name, data=b"", kind=tarfile.REGTYPE, linkname=""):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    info.size = len(data) if kind == tarfile.REGTYPE else 0
    tar.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)


def make_archive(path):
    with tarfile.open(path, "w:gz") as tar:
        add(tar, "cv-corpus-1.0/es", kind=tarfile.DIRTYPE)
        add(tar, "cv-corpus-1.0/es/validated.tsv", b"client_id\tpath\nabc\tx.mp3\n")
        add(tar, "cv-corpus-1.0/es/clips/x.mp3", b"\xff\xfb" * 1000)
        add(tar, "../evil.tsv", b"outside")
        add(tar, "/abs/evil.tsv", b"absolute")
        add(tar, "cv-corpus-1.0/es/link.tsv", kind=tarfile.SYMTYPE, linkname="/etc/passwd")


def test_extracts_only_wanted_safe_files(tmp_path):
    archive = tmp_path / "a.tar.gz"
    make_archive(archive)
    out = tmp_path / "out" / "inner"
    listing = tmp_path / "listing.tsv"
    result = scan(archive, out, lambda name: name.endswith(".tsv"), listing)

    assert result.extracted == ["cv-corpus-1.0/es/validated.tsv"]
    assert (out / "cv-corpus-1.0/es/validated.tsv").read_bytes() == b"client_id\tpath\nabc\tx.mp3\n"
    assert sorted(result.skipped) == ["../evil.tsv", "/abs/evil.tsv", "cv-corpus-1.0/es/link.tsv"]
    assert not (tmp_path / "out" / "evil.tsv").exists() and not list(out.rglob("*.mp3"))
    assert not (out / "cv-corpus-1.0/es/link.tsv").exists()

    everything = scan(archive, tmp_path / "all", lambda name: True)
    assert sorted(everything.skipped) == ["../evil.tsv", "/abs/evil.tsv", "cv-corpus-1.0/es/link.tsv"]
    assert len(everything.extracted) == 2

    lines = listing.read_text().splitlines()
    assert result.members == len(lines) == 6
    assert "cv-corpus-1.0/es/clips/x.mp3\t2000\tf" in lines and "cv-corpus-1.0/es\t0\td" in lines


def test_hash_covers_the_whole_file(tmp_path):
    archive = tmp_path / "a.tar.gz"
    make_archive(archive)
    result = scan(archive, tmp_path / "out", lambda name: False)
    data = archive.read_bytes()
    assert result.sha256 == hashlib.sha256(data).hexdigest() and result.archive_bytes == len(data)
    assert result.extracted == [] and not (tmp_path / "out").exists()


def test_safe_relative():
    assert str(safe_relative("a/b.tsv")) == "a/b.tsv"
    assert str(safe_relative("./a/b.tsv")) == "a/b.tsv"
    assert safe_relative("a/../../b") is None
    assert safe_relative("/a") is None
    assert safe_relative("") is None


def make_zip(path):
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("NOISE/README", "about")
        z.writestr("NOISE/a.wav", b"RIFF" + b"\0" * 100)
        z.writestr("NOISE/sub/", "")                              # a folder entry
        z.writestr("../evil.wav", b"outside")
        z.writestr("/abs.wav", b"absolute")
        link = zipfile.ZipInfo("NOISE/link.wav")
        link.external_attr = (0o120777 << 16)
        z.writestr(link, "/etc/passwd")


def test_zip_extracts_only_safe_files(tmp_path):
    from training.archives import extract_zip
    archive = tmp_path / "n.zip"
    make_zip(archive)
    out = tmp_path / "out"
    result = extract_zip(archive, out)
    assert sorted(result.extracted) == ["NOISE/README", "NOISE/a.wav"]
    assert sorted(result.skipped) == ["../evil.wav", "/abs.wav", "NOISE/link.wav"]
    assert (out / "NOISE/a.wav").read_bytes().startswith(b"RIFF")
    assert not (tmp_path / "evil.wav").exists() and not (out / "NOISE/link.wav").exists()
    assert result.sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    only_wav = extract_zip(archive, tmp_path / "wav", want=lambda n: n.endswith(".wav"))
    assert "NOISE/README" not in only_wav.extracted


def test_zip_size_limit(tmp_path):
    import pytest
    from training.archives import extract_zip
    archive = tmp_path / "n.zip"
    make_zip(archive)
    with pytest.raises(ValueError, match="would unpack"):
        extract_zip(archive, tmp_path / "out", max_bytes=50)
    assert not (tmp_path / "out").exists()
