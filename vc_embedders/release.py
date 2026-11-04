"""Author-source regeneration and allowlisted, verified release packaging."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from .catalog import check_sources, fetch, sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ["README.md", "README.ja.md", "AGENTS.md", "LICENSE", "pyproject.toml", "requirements-cpu.lock", ".gitignore", ".gitattributes"]
SOURCE_TREES = ["vc_embedders", "scripts", "tests", "docs", "licenses", ".github"]
SOURCE_SUFFIXES = {".py", ".ps1", ".json", ".md", ".txt", ".toml", ".yml", ".yaml", ".lock"}
AUDIT_FILES = {"manifest.json", "reference-comparison.json", "fairseq-parity.json"}
PRIVATE_PATTERN = r"(?i)[A-Z]:[\\/]+Users[\\/]|/(?:Users|home)/[A-Za-z0-9_.-]+/|hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"


def json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json(path, data):
    path.write_bytes(json_bytes(data))


def catalog():
    return json.loads((ROOT / "vc_embedders/release_catalog.json").read_text(encoding="utf-8"))


def release_items(data, names=None):
    items = []
    for spec in data["models"]:
        source = data["sources"][spec["source"]]
        items.append(dict(id=spec["id"], kind=source["kind"], profile=spec["profile"],
                          cache_id=spec["source"], repo=source["repo"], revision=source["revision"],
                          source_revision_scope=source["revision_scope"],
                          files=source["files"], license=source["license"],
                          license_basis=source["license_basis"],
                          author_source_rebuild=source.get("author_source_rebuild", True)))
    if names:
        unknown = set(names) - {item["id"] for item in items}
        if unknown:
            raise ValueError(f"Unknown release models: {sorted(unknown)}")
        items = [item for item in items if item["id"] in names]
    return items


def comparison_kind(item):
    # Official publisher provenance does not imply authorship of every checkpoint.
    prefix = "author" if item["author_source_rebuild"] else "publisher"
    return prefix + "-rebuild-versus-verified-app-export"


def source_files(root):
    """Only repository sources and evidence; never enumerate caches or .git."""
    paths = [root / name for name in SOURCE_FILES]
    for name in SOURCE_TREES:
        paths.extend(p for p in (root / name).rglob("*")
                     if p.is_file() and p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts)
    result = []
    for path in sorted(set(paths)):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Not a regular source file: {path.name}")
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large source file: {path.name}")
        # Return file names only on failure, never echo possible private content.
        text = path.read_text(encoding="utf-8-sig")
        if re.search(PRIVATE_PATTERN, text):
            raise ValueError(f"Private path or credential pattern in {path.relative_to(root).as_posix()}")
        result.append(path)
    return result


def source_manifest(root):
    files = [dict(path=p.relative_to(root).as_posix(), size=p.stat().st_size, sha256=sha256(p))
             for p in source_files(root)]
    fingerprint = hashlib.sha256(json_bytes(files)).hexdigest()
    return dict(format_version=1, fingerprint_sha256=fingerprint, files=files)


def validate_fairseq(report, digest):
    if (report.get("kind") != "legacy-fairseq-versus-onnx" or not report.get("passed")
            or report.get("onnx_sha256") != digest
            or report.get("checkpoint_sha256") != "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96"
            or report.get("atol", 1) > 5e-4 or report.get("rtol", 1) > 1e-4
            or report.get("max_mean_abs", 1) > 1e-4
            or len(report.get("cases", [])) < 19
            or not all(row.get("passed") for row in report.get("cases", []))):
        raise ValueError("Missing, stale, or unqualified independent fairseq evidence")


def validate_numeric(report, digest, kind, minimum_cases, atol):
    if (report.get("kind") != kind or report.get("onnx_sha256", digest) != digest
            or report.get("passed") is not True or report.get("atol", 1) > atol
            or report.get("rtol", 1) > 1e-4 or report.get("max_mean_abs", 1) > 1e-4
            or len(report.get("cases", [])) < minimum_cases
            or not all(row.get("passed") for row in report.get("cases", []))):
        raise ValueError("Missing, stale, or unqualified numerical evidence")


def reference_comparison(item, path, references, lengths, wavs, threads):
    from .validation import audio_cases, compare_arrays, create_session
    reference_dir = references / item["id"]
    reference = reference_dir / "model.onnx"
    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256(reference) != manifest["onnx_sha256"] or not manifest["validation"]["passed"]:
        raise ValueError("Unverified reference artifact")
    actual_session, reference_session = create_session(path, threads), create_session(reference, threads)
    names = [value.name for value in actual_session.get_outputs()]
    if names != [value.name for value in reference_session.get_outputs()]:
        raise ValueError("Release and reference output contracts differ")
    rows = []
    for case, audio in audio_cases(lengths, wavs):
        actual = actual_session.run(names, {actual_session.get_inputs()[0].name: audio})
        expected = reference_session.run(names, {reference_session.get_inputs()[0].name: audio})
        for name, value, ref in zip(names, actual, expected, strict=True):
            rows.append(dict(case=case, output=name, **compare_arrays(value, ref, atol=1e-4, rtol=1e-4)))
    return dict(kind=comparison_kind(item), onnx_sha256=sha256(path),
                reference_onnx_sha256=manifest["onnx_sha256"], atol=1e-4, rtol=1e-4,
                max_mean_abs=1e-4, passed=all(row["passed"] for row in rows), cases=rows)


def prepare(args):
    import torch
    from .cli import export_one
    torch.set_num_threads(args.threads)
    data = catalog()
    args.output = args.build
    args.include_applio_v1 = False
    args.atol, args.rtol, args.max_mean_abs = 5e-4, 1e-4, 1e-4
    for item in release_items(data, args.models):
        fetch(item, args.cache)
        export_one(item, args)
        directory = args.build / item["id"]
        report = reference_comparison(item, directory / "model.onnx", args.references,
                                      args.lengths, args.wav, args.threads)
        write_json(directory / "reference-comparison.json", report)
        if not report["passed"]:
            raise ValueError(f"Reference mismatch: {item['id']}")
        print(f"Compared source rebuild: {item['id']}", flush=True)
        gc.collect()


def zip_bytes(archive, name, data):
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data, compresslevel=6)


def zip_file(archive, name, path):
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with path.open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(block)


def inspect_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Unexpected or duplicate archive member")
        for name in names:
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or "\\" in name:
                raise ValueError("Unsafe archive member")
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected[name]:
                raise ValueError(f"Archive member checksum mismatch: {name}")


def inspect_model(path):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    def visit(message):
        if message.DESCRIPTOR.full_name == "onnx.TensorProto" and message.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError("External tensor data is not allowed in a single-file model asset")
        for descriptor, value in message.ListFields():
            values = value if descriptor.is_repeated else [value]
            if descriptor.type == descriptor.TYPE_MESSAGE:
                for child in values:
                    visit(child)
            elif descriptor.type == descriptor.TYPE_STRING:
                if any(re.search(PRIVATE_PATTERN, text) for text in values):
                    raise ValueError("Private path or credential pattern in ONNX metadata")
    visit(model)


def check_licenses(data):
    receipts = json.loads((ROOT / "licenses/sources.json").read_text(encoding="utf-8"))
    specs = {row["name"]: row for row in receipts if "name" in row}
    for source in data["sources"].values():
        for name in source["license_files"]:
            if name not in specs or sha256(ROOT / "licenses" / name) != specs[name]["sha256"]:
                raise ValueError(f"License evidence changed or missing: {name}")


def model_documents(item, source, manifest, source_archive):
    """Render bilingual usage and provenance together; retain license text verbatim."""
    labels = {
        "rvc-fairseq-base": ("ContentVec for RVC fairseq", "RVC fairseq版向け ContentVec"),
        "rvc-hubert-base": ("ContentVec for RVC Transformers", "RVC Transformers版向け ContentVec"),
        "applio-contentvec": ("ContentVec for Applio", "Applio向け ContentVec"),
        "applio-chinese-hubert-base": ("Chinese HuBERT for Applio", "Applio向け 中国語HuBERT"),
        "applio-korean-hubert-base": ("Korean HuBERT for Applio", "Applio向け 韓国語HuBERT"),
        "applio-spin": ("SPIN for Applio", "Applio向け SPIN"),
        "applio-spin-v2": ("SPIN v2 for Applio", "Applio向け SPIN v2"),
    }
    label_en, label_ja = labels.get(item["id"], (item["id"], item["id"]))
    is_rvc = item["profile"] in {"rvc", "rvc-fairseq"}
    support_en = ("For RVC v1/v2 voice models." if is_rvc
                  else "For Applio v2 voice models.")
    support_ja = ("RVC v1・v2の音声モデル向けです。" if is_rvc
                  else "Applio v2の音声モデル向けです。")
    outputs_en = (
        "Outputs: `unit9` (layer 9 + trained projection, 256 channels) and `unit12` (layer 12, 768 channels)."
        if is_rvc else "Output: `unit12` (final layer, 768 channels)."
    )
    outputs_ja = (
        "出力：`unit9`（第9層 + 学習済み射影、256次元）と `unit12`（第12層、768次元）。"
        if is_rvc else "出力：`unit12`（最終層、768次元）。"
    )
    if item["cache_id"] == "contentvec-original":
        origin = f"https://github.com/{source['repo']}/tree/{source['revision']}"
        basis_en = ("Based on the ContentVec author repository's MIT license and its linked pretrained weights; "
                    "no separate checkpoint license was supplied. The incomplete upstream copyright line is preserved.")
        basis_ja = ("作者リポジトリのMITと、そこから案内された学習済み重みに基づきます。"
                    "重み個別の許諾文はなく、原文の不完全な著作権行もそのまま保持しています。")
    elif source.get("license_statement_url"):
        origin = source["license_statement_url"]
        weights = f"https://huggingface.co/{source['repo']}/tree/{source['revision']}"
        basis_en = ("The official Applio README licenses code and model weights under MIT. "
                    "The AI Hispano copyright notice is retained.\n\n"
                    f"Weights: [official Applio distribution]({weights}). "
                    "No identity with a specific [Spin research checkpoint](https://github.com/vectominist/spin) "
                    "or authorship of all underlying models is claimed.")
        basis_ja = ("Applio公式READMEのコード・モデル重みのMIT表明に基づき、AI Hispanoの著作権表示を保持しています。\n\n"
                    f"重み：[Applio公式の固定配布版]({weights})。"
                    "特定の[Spin研究用チェックポイント](https://github.com/vectominist/spin)との同一性や、"
                    "元モデルすべての学習者は断定しません。")
    else:
        origin = f"https://huggingface.co/{source['repo']}/blob/{source['revision']}/README.md"
        basis_en = "Based on the license declared in the original publisher's model card."
        basis_ja = "元配布者のモデルカードに記載されたライセンスに基づきます。"
        if source["license"] == "MIT":
            basis_en += " No separate copyright line was supplied; no holder or year has been invented."
            basis_ja += "独立した著作権行は提供されておらず、著作権者や年は追加していません。"
    validation = manifest["validation"]
    tolerances = (f"atol={validation['atol']:g}, rtol={validation['rtol']:g}, "
                  f"mean_abs<={validation['max_mean_abs']:g}")
    readme_en = f"""# {label_en}

[日本語](README.ja.md)

## Use

Extract the ZIP and select `model.onnx` as the embedder in a compatible app such as vc-rs.
{support_en} Use the same embedder as during voice-model training; equal output dimensions do not imply compatibility.

## Input and output

Input: 16 kHz mono float32 waveform, `[1, samples]`, scaled to approximately `[-1, 1]` without mean/variance normalization.
{outputs_en}
Output shape: `[1, frames, channels]`; variable time axes; ONNX opset 17. The app must support these output names and tensor shapes.

## Source, license and changes

Original publisher: {source['publisher']}. License: **{source['license']}**; see [LICENSE.txt](LICENSE.txt).
[Fixed upstream source and license statement]({origin}).

{basis_en}

Re-exported pretrained weights as ONNX with app-specific feature outputs; training labels and optimizer state are omitted.
No additional training was performed. This is not an official upstream release.

## Validation and source code

PyTorch versus ONNX Runtime CPU: {len(validation['cases'])} comparisons passed (`{tolerances}`).
Comparison with the verified app-distributed-weight export also passed. Results apply only to the tested features, inputs and tolerances.
GPU execution and end-to-end audio quality/latency have not been validated.

Converted with vc-embedders 0.1.0. Source: `{source_archive}` from the same distribution.
See `docs/fairseq-compatibility_ja.md` inside the source ZIP for validation evidence and `docs/releasing_ja.md` for reproduction steps (Japanese).
"""
    readme_ja = f"""# {label_ja}

[English](README.md)

## 使い方

ZIPを展開し、vc-rsなどの対応アプリで `model.onnx` をEmbedderとして指定します。
{support_ja}音声モデルの学習時と同じEmbedderを使用してください。出力次元が同じでも互換性は保証されません。

## 入出力

入力：16 kHzモノラルのfloat32波形、`[1, samples]`。約 `[-1, 1]` にスケーリングし、平均・分散の標準化は加えません。
{outputs_ja}
出力形状：`[1, frames, channels]`。時間軸は可変、ONNX opsetは17です。アプリもこれらの出力名と入出力形状に対応する必要があります。

## 出典・ライセンス・変更内容

元の公開者：{source['publisher']}。ライセンス：**{source['license']}**（[LICENSE.txt](LICENSE.txt)）。
[固定した元配布先とライセンス表示]({origin})。

{basis_ja}

学習済み重みをONNX化し、アプリ向けの特徴抽出出力を選びました。学習用ラベル・オプティマイザー状態は含めていません。
追加学習は行っていません。元プロジェクトによる公式配布物ではありません。

## 検証・ソースコード

PyTorch対ONNX Runtime CPUの{len(validation['cases'])}比較はすべて合格しています（`{tolerances}`）。
検証済みのアプリ配布版重みから生成したONNXとの比較も合格しています。結果は検査した特徴量・入力・許容誤差の範囲に限ります。
GPU実行と、音声変換全体の音質・遅延は未検証です。

変換ツールは vc-embedders 0.1.0。同じ配布先の `{source_archive}` にソースがあります。
検証根拠はソースZIP内の `docs/fairseq-compatibility_ja.md`、再現手順は `docs/releasing_ja.md` を参照してください。
"""
    license_name = next(name for name in source["license_files"] if name.endswith(".txt"))
    return {"README.md": readme_en.encode("utf-8"), "README.ja.md": readme_ja.encode("utf-8"),
            "LICENSE.txt": (ROOT / "licenses" / license_name).read_bytes()}


def model_archive(path, contents, model_path, reuse=None, seven_zip=None):
    """Optionally retain an already verified compressed model while replacing docs.

    7-Zip edits a generated copy only. The caller still decompresses and hashes
    every final member; old metadata and archive names are never trusted.
    """
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    expected["model.onnx"] = sha256(model_path)
    if reuse:
        with zipfile.ZipFile(reuse) as old:
            names = old.namelist()
            if (len(names) != len(set(names)) or any(
                    not re.fullmatch(r"[A-Za-z0-9_.\-/]+", name) or name.startswith(("-", "/"))
                    or ".." in name.split("/") for name in names)):
                raise ValueError("Unsafe reuse archive")
            digest = hashlib.sha256()
            with old.open("model.onnx") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected["model.onnx"]:
                raise ValueError("Reuse archive model differs from the verified model")
        if not seven_zip:
            raise ValueError("7-Zip is required only when reusing compressed assets")
        shutil.copyfile(reuse, path)
        unwanted = [name for name in names if name != "model.onnx"]
        if unwanted:
            result = subprocess.run([str(seven_zip), "d", "-y", str(path.resolve()), *unwanted],
                                    capture_output=True, text=True)
            if result.returncode:
                raise ValueError("Could not replace the old model documentation")
        mode = "a"
    else:
        mode = "w"
    with zipfile.ZipFile(path, mode) as archive:
        for name, content in contents.items():
            zip_bytes(archive, name, content)
        if not reuse:
            zip_file(archive, "model.onnx", model_path)
    inspect_zip(path, expected)
    return expected


def package(args):
    data = catalog()
    check_licenses(data)
    if args.output.exists():
        raise ValueError("Release output already exists; use a new directory")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?", args.version):
        raise ValueError("Invalid release version")
    if args.version != "0.1.0":
        raise ValueError("Version must match the current project version 0.1.0")
    items = release_items(data)
    checked = []
    for item in items:
        directory = args.build / item["id"]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        digest = sha256(directory / "model.onnx")
        inspect_model(directory / "model.onnx")
        if manifest["model"] != item or manifest["onnx_sha256"] != digest or not manifest["validation"]["passed"]:
            raise ValueError("Release model identity or validation differs")
        from .hubert import output_contract
        outputs = output_contract(item["profile"])
        if manifest["outputs"] != outputs:
            raise ValueError("Release extraction contract differs")
        validate_numeric(manifest["validation"], digest, "pytorch-versus-onnx", 18 * len(outputs), 5e-4)
        comparison = json.loads((directory / "reference-comparison.json").read_text(encoding="utf-8"))
        validate_numeric(comparison, digest, comparison_kind(item), 18 * len(outputs), 1e-4)
        if item["cache_id"] == "contentvec-original":
            validate_fairseq(json.loads((directory / "fairseq-parity.json").read_text(encoding="utf-8")), digest)
        checked.append((item, directory, digest))
    # Scan and fingerprint final sources before emitting anything distributable.
    snapshot = source_manifest(ROOT)
    if args.audit.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("Audit records must stay outside the user distribution")
    if args.audit.exists():
        raise ValueError("Audit output already exists; use a new directory")
    seven_zip = shutil.which("7z")
    candidate = Path(os.environ.get("ProgramFiles", "")) / "7-Zip/7z.exe"
    if not seven_zip and candidate.is_file():
        seven_zip = candidate
    args.output.mkdir(parents=True)
    args.audit.mkdir(parents=True)
    prefix = f"vc-embedders-{args.version}"
    source_name = prefix + "-source.zip"
    expected = {}
    with zipfile.ZipFile(args.output / source_name, "w") as archive:
        for row in snapshot["files"]:
            name = prefix + "/" + row["path"]
            zip_file(archive, name, ROOT / row["path"])
            expected[name] = row["sha256"]
    write_json(args.audit / "source-manifest.json", snapshot)
    inspect_zip(args.output / source_name, expected)
    source_digest = sha256(args.output / source_name)
    assets = [dict(name=source_name, sha256=source_digest, size=(args.output / source_name).stat().st_size, kind="source")]
    for item, directory, digest in checked:
        source = data["sources"][item["cache_id"]]
        provenance = dict(version=args.version, model_id=item["id"], onnx_sha256=digest,
                          source=source, tool_source_archive=source_name,
                          tool_source_sha256=source_digest, tool_source_fingerprint=snapshot["fingerprint_sha256"],
                          changes=["Re-exported pinned source weights as ONNX opset 17; FP32; dynamic sample/frame axes.",
                                   "Selected app-specific feature layers/projection; training/optimizer state omitted.",
                                   "Waveform preprocessing follows RVC/Applio (no mean/variance normalization)."],
                          validation_scope="CPU feature tensors only; GPU and end-to-end voice conversion not qualified")
        audit_model = args.audit / item["id"]
        audit_model.mkdir()
        write_json(audit_model / "SOURCES.json", provenance)
        for member in AUDIT_FILES:
            if (directory / member).is_file():
                shutil.copyfile(directory / member, audit_model / member)
        contents = model_documents(item, source, json.loads((directory / "manifest.json").read_text(encoding="utf-8")), source_name)
        name = prefix + "-" + item["id"] + ".zip"
        reuse = args.reuse_archives / name if args.reuse_archives else None
        if reuse and not reuse.is_file():
            reuse = None
        expected = model_archive(args.output / name, contents, directory / "model.onnx", reuse, seven_zip)
        if (args.output / name).stat().st_size >= 2 * 1024 ** 3:
            raise ValueError("Asset exceeds the chosen 2 GiB limit")
        assets.append(dict(name=name, kind="model", model_id=item["id"], license=item["license"],
                           sha256=sha256(args.output / name), size=(args.output / name).stat().st_size,
                           onnx_sha256=digest, members=expected))
        print(f"Packaged and verified: {name}", flush=True)
    index = dict(schema_version=1, version=args.version, publication_performed=False,
                 source_fingerprint=snapshot["fingerprint_sha256"], assets=assets,
                 held_models=data["held_models"], archive_member_hashes_verified=True)
    write_json(args.audit / "release-index.json", index)
    shutil.copyfile(ROOT / "licenses/sources.json", args.audit / "license-receipts.json")
    print(f"Prepared {len(assets)} archives; no publication performed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "package"])
    parser.add_argument("--models", nargs="+", help="Prepare only selected release model IDs; existing artifacts are never overwritten")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--cache", type=Path, default=Path("cache/release"))
    parser.add_argument("--build", type=Path, default=Path("artifacts/release/v0.1.0"))
    parser.add_argument("--output", type=Path, default=Path("dist/v0.1.0"))
    parser.add_argument("--references", type=Path, default=Path("artifacts"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/release-audit/v0.1.0"))
    parser.add_argument("--reuse-archives", type=Path, help="Reuse unchanged compressed models from existing ZIPs; requires 7-Zip")
    parser.add_argument("--lengths", nargs="+", type=int, default=[800, 1600, 3200, 16000, 24017, 48000])
    parser.add_argument("--wav", type=Path, action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.models and args.command != "prepare":
        parser.error("--models is only valid for prepare; package always includes the full release catalog")
    if args.threads < 1 or min(args.lengths) < 800:
        parser.error("Threads must be positive and lengths >= 800")
    if args.command == "prepare":
        prepare(args)
    else:
        package(args)


if __name__ == "__main__":
    main()
