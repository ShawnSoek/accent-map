"""The Kaggle bundle: contents and checksums, pinned requirements, and the checks run.py makes
before it uses anything (bundle found, safely unpacked, every file's checksum)."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

from training.feature_items import make_items
from training.kaggle_bundle import choose_reference, kaggle_requirements, write_bundle

REPO = Path(__file__).resolve().parents[1]


def load_run_script():
    spec = importlib.util.spec_from_file_location("kaggle_run", REPO / "scripts" / "kaggle_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kaggle_run = load_run_script()


def test_requirements_come_from_requirements_txt(tmp_path):
    text = kaggle_requirements(REPO / "requirements.txt")
    assert "transformers==4.57.6\n" in text and "soxr==1.1.0\n" in text
    assert "torch" not in text and "numpy" not in text            # Kaggle's own CUDA build stays
    short = tmp_path / "r.txt"
    short.write_text("transformers==1.0\n")
    with pytest.raises(ValueError, match="not pinned"):
        kaggle_requirements(short)


def test_reference_has_every_set():
    rows = {name: [{"path": f"{name}{i}.mp3", "speaker": f"s{i % 3}", "label": "a", "sentence_id": "x",
                    "seconds": "1", "gender": "f"} for i in range(12)] for name in ("train", "dev", "test")}
    ref = choose_reference(make_items(rows, "s"), per_set=2)
    assert sorted({it.set for it in ref}) == sorted(["train", "train_aug", "dev", "test", "dev_pairs", "test_pairs"])
    assert len(ref) == 12 and ref == choose_reference(make_items(rows, "s"), per_set=2)


@pytest.fixture
def bundle_tar(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.mp3").write_bytes(b"\xff\xfb" * 50)
    (src / "items.tsv").write_text("id\tset\taugment\tclips\n")
    tar = tmp_path / "upload" / "bundle.tar"
    digest, n = write_bundle(tar, {"clips/a.mp3": src / "a.mp3", "items.tsv": src / "items.tsv"},
                             {"reference/ids.txt": "train/000000\n"})
    (tmp_path / "upload" / "manifest.json").write_text(json.dumps({"name": "accent-map test", "bundle_sha256": digest}))
    return tar, digest, n


def test_bundle_lists_every_file(bundle_tar, tmp_path):
    tar, digest, n = bundle_tar
    assert n == 3 and digest == hashlib.sha256(tar.read_bytes()).hexdigest()
    with tarfile.open(tar) as t:
        names = t.getnames()
        listing = t.extractfile("bundle/files.sha256").read().decode()
    assert sorted(names) == ["bundle/clips/a.mp3", "bundle/files.sha256", "bundle/items.tsv", "bundle/reference/ids.txt"]
    assert listing.count("\n") == 3 and "  clips/a.mp3\n" in listing
    with pytest.raises(ValueError, match="reserved"):
        write_bundle(tmp_path / "x.tar", {}, {"files.sha256": "no"})


def test_run_script_finds_unpacks_and_checks(bundle_tar, tmp_path):
    tar, digest, _ = bundle_tar
    found_tar, unpacked, manifest = kaggle_run.find_bundle(tmp_path / "upload")
    assert found_tar == tar and unpacked is None and manifest["bundle_sha256"] == digest
    bundle = kaggle_run.safe_untar(tar, tmp_path / "work")
    assert kaggle_run.check_files(bundle) == 3

    (bundle / "clips" / "a.mp3").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="wrong checksum"):
        kaggle_run.check_files(bundle)
    (bundle / "clips" / "a.mp3").unlink()
    with pytest.raises(SystemExit, match="missing"):
        kaggle_run.check_files(bundle)


def test_run_script_accepts_a_bundle_kaggle_unpacked(bundle_tar, tmp_path):
    tar, _, _ = bundle_tar
    kaggle_run.safe_untar(tar, tmp_path / "upload")       # as if Kaggle had unpacked the tar
    tar.unlink()
    found_tar, unpacked, _ = kaggle_run.find_bundle(tmp_path / "upload")
    assert found_tar is None and unpacked == tmp_path / "upload" / "bundle"
    (unpacked / "extra.txt").write_text("not listed")
    with pytest.raises(SystemExit, match="extra"):
        kaggle_run.check_files(unpacked)


@pytest.mark.parametrize("name, kind", [("../evil", tarfile.REGTYPE), ("/abs", tarfile.REGTYPE),
                                        ("other/x", tarfile.REGTYPE), ("bundle/link", tarfile.SYMTYPE)])
def test_run_script_refuses_unsafe_tar_members(tmp_path, name, kind):
    tar_path = tmp_path / "bad.tar"
    with tarfile.open(tar_path, "w") as t:
        info = tarfile.TarInfo(name)
        info.type = kind
        if kind == tarfile.SYMTYPE:
            info.linkname = "/etc/passwd"
            t.addfile(info)
        else:
            info.size = 3
            t.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(SystemExit):
        kaggle_run.safe_untar(tar_path, tmp_path / "work")
    assert not (tmp_path / "evil").exists()
