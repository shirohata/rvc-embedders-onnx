import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from vc_embedders.catalog import fetch, source_url
from vc_embedders.release import (SOURCE_FILES, catalog, inspect_model, inspect_zip, model_archive, model_documents, release_items,
                                  source_manifest, validate_fairseq, validate_numeric, zip_bytes)


def test_release_policy_selects_seven_models_with_pinned_license_provenance():
    data = catalog()
    items = release_items(data)
    assert len(items) == 7
    assert {item["id"] for item in items}.isdisjoint(data["held_models"])
    assert len({item["cache_id"] for item in items}) == 5
    for item in items:
        if item["id"] in {"applio-spin", "applio-spin-v2"}:
            source = data["sources"][item["cache_id"]]
            assert not item["author_source_rebuild"]
            assert item["repo"] == "IAHispano/Applio"
            assert source["license"] == "MIT" and len(source["license_statement_revision"]) == 40
            assert source["license_files"] == ["applio-LICENSE.txt", "applio-README.md"]
        else:
            assert item["author_source_rebuild"]
            assert item["repo"] != "IAHispano/Applio"
        assert len(item["revision"]) == 40
        assert item["repo"] not in {"lj1995/VoiceConversionWebUI", "wok000/weights_gpl"}
        for spec in item["files"]:
            assert len(spec["sha256"]) == 64 and spec["size"] > 0
            assert source_url(item, spec).startswith("https://")


def test_shared_author_cache_is_verified_for_each_profile(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "hubert_base.pt").write_bytes(b"author weights")
    spec = dict(name="hubert_base.pt", size=14, sha256=hashlib.sha256(b"author weights").hexdigest())
    first = dict(id="rvc-profile", cache_id="author", files=[spec])
    second = dict(id="applio-profile", cache_id="author", files=[spec])
    cache = tmp_path / "cache"
    assert fetch(first, cache, local) == fetch(second, cache, local) == cache / "author"
    with pytest.raises(ValueError, match="HTTPS"):
        source_url({}, dict(url="http://example.invalid/model.pt"))


def source_fixture(tmp_path):
    for name in SOURCE_FILES:
        (tmp_path / name).write_text("public source\n", encoding="utf-8")
    (tmp_path / "vc_embedders").mkdir()
    (tmp_path / "vc_embedders/model.py").write_text("value = 1\n")
    for name in ["cache", ".git", "dist"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "private.txt").write_text("excluded")
    (tmp_path / "vc_embedders/weights.bin").write_bytes(b"excluded")
    (tmp_path / ".env").write_text("excluded")
    return tmp_path


def test_source_archive_manifest_excludes_models_caches_git_and_private_paths(tmp_path):
    root = source_fixture(tmp_path)
    manifest = source_manifest(root)
    assert {f["path"] for f in manifest["files"]} == set(SOURCE_FILES) | {"vc_embedders/model.py"}
    (root / "vc_embedders/model.py").write_text("machine = " + repr("C:" + "/Users/" + "private/data") + "\n")
    with pytest.raises(ValueError, match="Private path"):
        source_manifest(root)


def test_archive_hash_validation_rejects_tampering_and_extra_members(tmp_path):
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as archive:
        zip_bytes(archive, "README.md", b"public")
    expected = {"README.md": hashlib.sha256(b"public").hexdigest()}
    inspect_zip(path, expected)
    with pytest.raises(ValueError, match="checksum"):
        inspect_zip(path, {"README.md": "0" * 64})
    with zipfile.ZipFile(path, "a") as archive:
        zip_bytes(archive, "unexpected.txt", b"extra")
    with pytest.raises(ValueError, match="Unexpected"):
        inspect_zip(path, expected)


def test_source_archive_includes_powershell_entrypoint_and_scans_it(tmp_path):
    root = source_fixture(tmp_path)
    (root / "scripts").mkdir()
    script = root / "scripts/dev.ps1"
    script.write_text("Write-Output 'check'\n", encoding="utf-8")
    assert "scripts/dev.ps1" in {row["path"] for row in source_manifest(root)["files"]}
    script.write_text("$path = 'C:" + "/Users/" + "private/data'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Private path"):
        source_manifest(root)


def test_source_scan_accepts_the_scanner_source_itself(tmp_path):
    import vc_embedders.release as release
    root = source_fixture(tmp_path)
    (root / "vc_embedders/model.py").write_text(Path(release.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    assert source_manifest(root)["files"]


def evidence():
    return dict(kind="legacy-fairseq-versus-onnx", passed=True, onnx_sha256="a" * 64,
                checkpoint_sha256="f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96",
                atol=5e-4, rtol=1e-4, max_mean_abs=1e-4, cases=[dict(passed=True)] * 19)


@pytest.mark.parametrize("fault", ["stale", "failed", "loose", "empty", "checkpoint"])
def test_packaging_rejects_unqualified_or_stale_fairseq_evidence(fault):
    report = evidence()
    validate_fairseq(report, "a" * 64)
    if fault == "stale": report["onnx_sha256"] = "b" * 64
    if fault == "failed": report["passed"] = False
    if fault == "loose": report["atol"] = 1e-3
    if fault == "empty": report["cases"] = []
    if fault == "checkpoint": report["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="unqualified"):
        validate_fairseq(report, "a" * 64)


def test_packaging_requires_actual_numerical_cases_and_strict_reference_thresholds():
    report = evidence()
    report["kind"] = "author-rebuild-versus-verified-app-export"
    report["atol"] = 1e-4
    validate_numeric(report, "a" * 64, report["kind"], 18, 1e-4)
    report["cases"][0] = dict(passed=False)
    with pytest.raises(ValueError, match="unqualified"):
        validate_numeric(report, "a" * 64, report["kind"], 18, 1e-4)


def test_model_metadata_and_external_tensor_data_are_checked(tmp_path):
    import onnx
    path = tmp_path / "model.onnx"
    model = onnx.helper.make_model(onnx.helper.make_graph([], "public", [], []))
    onnx.save(model, path)
    inspect_model(path)
    model.doc_string = "C:" + "/Users/" + "private/source.py"
    onnx.save(model, path)
    with pytest.raises(ValueError, match="Private path"):
        inspect_model(path)
    model.doc_string = ""
    model.graph.initializer.append(onnx.TensorProto(name="external", data_location=onnx.TensorProto.EXTERNAL))
    onnx.save(model, path)
    with pytest.raises(ValueError, match="External tensor"):
        inspect_model(path)


def test_user_documents_include_license_and_readable_use_and_source_information():
    import re
    import vc_embedders.release as release
    data = catalog()
    for item in release_items(data):
        source = data["sources"][item["cache_id"]]
        is_rvc = item["profile"] in {"rvc", "rvc-fairseq"}
        count = 38 if is_rvc else 19
        validation = dict(cases=[1] * count, atol=5e-4, rtol=1e-4, max_mean_abs=1e-4)
        docs = model_documents(item, source, {"validation": validation}, "source.zip")
        assert set(docs) == {"README.md", "README.ja.md", "LICENSE.txt"}
        license_name = next(name for name in source["license_files"] if name.endswith(".txt"))
        assert docs["LICENSE.txt"] == (release.ROOT / "licenses" / license_name).read_bytes()
        english = docs["README.md"].decode("utf-8")
        japanese = docs["README.ja.md"].decode("utf-8")
        assert "[日本語](README.ja.md)" in english
        assert "[English](README.md)" in japanese
        assert "No additional training was performed" in english
        assert "追加学習は行っていません" in japanese
        assert set(re.findall(r"https://[^)\s]+", english)) == set(re.findall(r"https://[^)\s]+", japanese))
        for text in (english, japanese):
            for value in (source["publisher"], source["license"], source["revision"], str(count),
                          "unit12", "[1, samples]", "[1, frames, channels]", "float32", "16 kHz",
                          "atol=0.0005", "rtol=0.0001", "mean_abs<=0.0001", "LICENSE.txt", "source.zip"):
                assert value in text
            assert ("unit9" in text) == is_rvc
            assert "NOTICE.md" not in text
            assert ".json" not in text


@pytest.mark.parametrize("reuse", [False, True])
def test_user_archive_replaces_machine_records_and_preserves_the_model(tmp_path, reuse):
    import os
    import shutil
    executable = shutil.which("7z") or Path(os.environ.get("ProgramFiles", "")) / "7-Zip/7z.exe"
    if reuse and not Path(executable).is_file():
        pytest.skip("Optional 7-Zip reuse path is unavailable")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"verified model bytes" * 100)
    docs = {"README.md": b"Usage and source", "README.ja.md": "使い方と出典".encode(),
            "LICENSE.txt": b"license notice"}
    old = tmp_path / "old.zip"
    with zipfile.ZipFile(old, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model, "model.onnx")
        archive.writestr("NOTICE.md", "old separate notice")
        archive.writestr("manifest.json", "old machine record")
        archive.writestr("LICENSES/source-receipts.json", "old machine record")
    old_hash = hashlib.sha256(old.read_bytes()).hexdigest()
    target = tmp_path / "new.zip"
    expected = model_archive(target, docs, model, old if reuse else None, executable)
    assert set(expected) == {"model.onnx", *docs}
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == set(expected)
        assert archive.read("model.onnx") == model.read_bytes()
        for name, value in docs.items():
            assert archive.read(name) == value
    assert hashlib.sha256(old.read_bytes()).hexdigest() == old_hash
    if reuse:
        model.write_bytes(b"different model")
        with pytest.raises(ValueError, match="model differs"):
            model_archive(tmp_path / "bad.zip", docs, model, old, executable)


def test_incremental_prepare_selection_rejects_unknown_and_keeps_release_contract():
    from vc_embedders.release import comparison_kind
    data = catalog()
    selected = release_items(data, ["applio-spin", "applio-spin-v2"])
    assert [item["id"] for item in selected] == ["applio-spin", "applio-spin-v2"]
    assert all(comparison_kind(item) == "publisher-rebuild-versus-verified-app-export" for item in selected)
    with pytest.raises(ValueError, match="Unknown release models"):
        release_items(data, ["applio-japanese-hubert-base"])
