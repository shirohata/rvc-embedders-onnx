from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys

from .catalog import check_sources, fetch, select_models, sha256


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment() -> dict:
    return {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "onnx", "onnxruntime")}


def tolerances(item, args):
    policy = item.get("validation_policy", {})
    return dict(atol=args.atol if args.atol is not None else policy.get("atol", 1e-4),
                rtol=args.rtol if args.rtol is not None else policy.get("rtol", 1e-4),
                max_mean_abs=args.max_mean_abs)


def export_one(item: dict, args):
    import onnx
    import torch
    from .hubert import ExportWrapper, load_model, output_contract
    from .validation import smoke, validate

    source = check_sources(item, args.cache)
    suffix = "-applio-v1" if args.include_applio_v1 else ""
    destination = args.output / (item["id"] + suffix)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "manifest.json"
    if report_path.exists():
        raise ValueError(f"Artifact already exists: {destination}; use a new --output directory")
    temporary = destination / "model.partial.onnx"
    if item["kind"] == "onnx":
        if args.include_applio_v1:
            raise ValueError("Cannot add outputs to an upstream ONNX")
        shutil.copyfile(source / "model.onnx", temporary)
        contract = None
        validation = smoke(temporary, args.lengths, args.threads, args.wav)
    else:
        contract = output_contract(item["profile"], args.include_applio_v1)
        model = load_model(source, any(output["projection"] for output in contract), item["kind"])
        wrapper = ExportWrapper(model, contract).eval()
        dynamic = {"audio": {1: "samples"}}
        dynamic.update({output["name"]: {1: "frames"} for output in contract})
        with torch.inference_mode():
            torch.onnx.export(wrapper, (torch.zeros(1, 16000),), str(temporary),
                              input_names=["audio"], output_names=[output["name"] for output in contract],
                              dynamic_axes=dynamic, opset_version=17, dynamo=False)
        graph = onnx.load(str(temporary))
        onnx.helper.set_model_props(graph, {
            "vc_embedder.id": item["id"], "vc_embedder.source_revision": item["revision"],
            "vc_embedder.profile": item["profile"], "vc_embedder.sample_rate": "16000",
            "vc_embedder.normalization": "none; supply mono waveform scaled to [-1,1]",
            "vc_embedder.outputs": json.dumps(contract),
        })
        onnx.save(graph, str(temporary))
        del graph
        validation = validate(temporary, model, contract, args.lengths, args.wav,
                              args.threads, **tolerances(item, args))
    onnx.checker.check_model(str(temporary))
    manifest = dict(schema_version=1, model=item, outputs=contract, environment=environment(),
                    input=dict(sample_rate=16000, dtype="float32", shape=[1, "samples"], normalization="none"),
                    validation=validation, onnx_sha256=sha256(temporary))
    if not validation["passed"]:
        write_json(destination / "validation.failed.json", manifest)
        raise ValueError(f'Numerical validation failed: {destination / "validation.failed.json"}')
    temporary.replace(destination / "model.onnx")
    write_json(report_path, manifest)
    print(f'Prepared {item["id"]}: {validation["kind"]}; {len(validation["cases"])} comparisons', flush=True)


def verify_one(item: dict, args):
    from .hubert import load_model, output_contract
    from .validation import smoke, validate

    suffix = "-applio-v1" if args.include_applio_v1 else ""
    directory = args.output / (item["id"] + suffix)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "model.onnx"
    if manifest["model"] != item or sha256(path) != manifest["onnx_sha256"]:
        raise ValueError("Artifact identity/hash differs from the pinned manifest")
    if item["kind"] == "onnx":
        if sha256(path) != item["files"][0]["sha256"]:
            raise ValueError("Upstream ONNX bytes have changed")
        report = smoke(path, args.lengths, args.threads, args.wav)
    else:
        source = check_sources(item, args.cache)
        outputs = output_contract(item["profile"], args.include_applio_v1)
        if manifest["outputs"] != outputs:
            raise ValueError("Artifact extraction profile differs from the expected output contract")
        model = load_model(source, any(output["projection"] for output in outputs), item["kind"])
        report = validate(path, model, outputs, args.lengths, args.wav, args.threads, **tolerances(item, args))
    write_json(directory / "verification.json", dict(onnx_sha256=sha256(path), environment=environment(), validation=report))
    if not report["passed"]:
        raise ValueError(f"Artifact verification failed: {directory}")
    print(f'Verified {item["id"]}: {report["kind"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description="Pinned voice-conversion embedders; downloads are not bundled code")
    parser.add_argument("command", choices=["list", "fetch", "export", "prepare", "verify"])
    parser.add_argument("models", nargs="*", default=["all"], help="Model IDs or all")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--local-dir", type=Path, help="Import exact pinned source files instead of downloading")
    parser.add_argument("--lengths", nargs="+", type=int, default=[1600, 16000, 24017])
    parser.add_argument("--wav", type=Path, action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, help="Override catalog absolute tolerance (default 1e-4)")
    parser.add_argument("--rtol", type=float, help="Override catalog relative tolerance (default 1e-4)")
    parser.add_argument("--max-mean-abs", type=float, default=1e-4)
    parser.add_argument("--include-applio-v1", action="store_true")
    args = parser.parse_args()
    try:
        if args.threads < 1 or not args.lengths or min(args.lengths) < 800:
            raise ValueError("Threads must be positive and validation lengths >= 800")
        if any(value is not None and (not math.isfinite(value) or value < 0)
               for value in (args.atol, args.rtol, args.max_mean_abs)):
            raise ValueError("Tolerances must be finite and nonnegative")
        models = select_models(args.models or ["all"])
        if args.local_dir and len(models) != 1:
            raise ValueError("--local-dir requires one explicit model ID")
        if args.command == "list":
            for item in models:
                print(f'{item["id"]:36} {item["kind"]:12} {item["profile"]:7} {item["license"]}')
            return
        if args.command in ("prepare", "export", "verify"):
            import torch
            torch.set_num_threads(args.threads)
        for item in models:
            print(item["id"], flush=True)
            if args.command in ("fetch", "prepare"):
                fetch(item, args.cache, args.local_dir)
            if args.command in ("export", "prepare"):
                export_one(item, args)
            if args.command == "verify":
                verify_one(item, args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
