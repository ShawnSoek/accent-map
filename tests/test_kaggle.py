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


def make_unpacked_bundle(tmp_path):
    """A bundle as Kaggle presents it: tar unpacked, and the noise zip inside unpacked too."""
    import zipfile

    from fakes import make_noise_folder
    from training.kaggle_bundle import zip_tree_hash

    src = tmp_path / "src"
    make_noise_folder(src / "noise_files")
    zip_path = src / "rirs_noises.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for p in sorted((src / "noise_files").rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(src / "noise_files").as_posix())
    (src / "a.mp3").write_bytes(b"\xff\xfb" * 50)
    for name in ("augment.yaml", "features.yaml"):
        (src / name).write_text((REPO / "training/cv27_es10" / name).read_text())
    (src / "items.tsv").write_text("id\tset\taugment\tclips\n")
    tar = tmp_path / "upload" / "bundle.tar"
    write_bundle(tar, {"clips/a.mp3": src / "a.mp3", "items.tsv": src / "items.tsv", "augment.yaml": src / "augment.yaml",
                       "features.yaml": src / "features.yaml", "noise/rirs_noises.zip": zip_path}, {})
    (tmp_path / "input" / "datasets" / "user" / "ds").mkdir(parents=True)
    dataset = tmp_path / "input" / "datasets" / "user" / "ds"
    (dataset / "manifest.json").write_text(json.dumps({"name": "accent-map test", "built": "now", "items": 0}))
    bundle = kaggle_run.safe_untar(tar, dataset)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(bundle / "noise" / "rirs_noises")
    (bundle / "noise" / "rirs_noises.zip").unlink()
    return bundle, {"noise/rirs_noises.zip": list(zip_tree_hash(zip_path, kaggle_run))}


def test_an_archive_kaggle_unpacked_is_checked_as_a_tree(tmp_path):
    bundle, archives = make_unpacked_bundle(tmp_path)
    assert kaggle_run.tree_hash(bundle / "noise" / "rirs_noises") == tuple(archives["noise/rirs_noises.zip"])
    assert kaggle_run.check_files(bundle, archives) == 5
    with pytest.raises(SystemExit, match="missing \\['noise/rirs_noises.zip'\\]"):
        kaggle_run.check_files(bundle, {})                     # what happened on the first Kaggle run
    wav = next((bundle / "noise" / "rirs_noises").rglob("*.wav"))
    wav.write_bytes(wav.read_bytes()[:-2] + b"xx")
    with pytest.raises(SystemExit, match="differs from the pinned archive"):
        kaggle_run.check_files(bundle, archives)


def test_bundle_is_found_a_few_levels_down_without_a_full_search(tmp_path):
    bundle, _ = make_unpacked_bundle(tmp_path)
    tar, unpacked, manifest = kaggle_run.find_bundle(tmp_path / "input")
    assert tar is None and unpacked == bundle and manifest["name"] == "accent-map test"
    deep = tmp_path / "deep" / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "manifest.json").write_text('{"name": "accent-map"}')
    with pytest.raises(SystemExit, match="expected one"):
        kaggle_run.find_bundle(tmp_path / "deep")


def test_view_links_the_read_only_bundle(tmp_path):
    from training.augment import NoiseBank
    bundle, archives = make_unpacked_bundle(tmp_path)
    view = kaggle_run.make_view(bundle, tmp_path / "work")
    assert sorted(p.name for p in view.iterdir()) == ["augment.yaml", "clips", "features.yaml", "items.tsv", "noise"]
    assert (view / "clips" / "a.mp3").read_bytes().startswith(b"\xff\xfb")
    bank = NoiseBank(view / "noise", ["smallroom", "mediumroom"])
    assert len(bank.real_rirs) == 3 and bank.noises["background"]


def test_run_script_gets_the_archive_hashes():
    from training.kaggle_bundle import run_script_with_archives
    text = run_script_with_archives(REPO / "scripts" / "kaggle_run.py", {"noise/x.zip": ("ab" * 32, 7)})
    namespace = {}
    exec(compile(text.split("\ndef log", 1)[0], "run.py", "exec"), namespace)     # just the constants
    assert namespace["UNPACKED_ARCHIVES"] == {"noise/x.zip": ["ab" * 32, 7]}
    with pytest.raises(ValueError, match="exactly once"):
        run_script_with_archives(REPO / "README.md", {})
