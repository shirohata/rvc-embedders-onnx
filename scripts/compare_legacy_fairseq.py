"""Independent comparison in an existing fairseq runtime; no ONNX export here."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np
import onnxruntime as ort
import torch


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def audio_cases(lengths, wavs):
    rng = np.random.default_rng(20261020)
    for length in lengths:
        phase = np.arange(length, dtype=np.float32) / 16000
        yield f"silence-{length}", np.zeros((1, length), np.float32)
        yield f"tone-{length}", (0.15 * np.sin(2 * np.pi * 220 * phase))[None].astype(np.float32)
        yield f"noise-{length}", (rng.standard_normal((1, length)) * 0.05).astype(np.float32)
    for index, path in enumerate(wavs):
        with wave.open(str(path), "rb") as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
                raise ValueError("Expected 16 kHz mono PCM16 WAV")
            audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if audio.size < 800:
            raise ValueError("WAV is too short")
        yield f"wav-{index}", audio[None]


def context(mode):
    if mode != "math":
        return nullcontext()
    if hasattr(torch.nn, "attention"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1,
                        help="One thread avoids the diagnosed local PyTorch 2.4 CPU runtime fault")
    parser.add_argument("--attention", choices=["default", "math", "manual"], default="default")
    parser.add_argument("--manual-attention", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lengths", nargs="+", type=int, default=[800, 1600, 3200, 16000, 24017, 48000])
    parser.add_argument("--wav", type=Path, action="append", default=[])
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--v2-only", action="store_true", help="Compare an Applio ContentVec final-layer-only export")
    args = parser.parse_args()
    if args.threads < 1 or min(args.lengths) < 800 or not np.isfinite([args.atol,args.rtol]).all() or min(args.atol,args.rtol)<0:
        parser.error("Threads must be positive and lengths >= 800")
    if args.manual_attention:
        args.attention = "manual"
    source = json.loads((Path(__file__).resolve().parents[1] / "docs/upstream-sources.json").read_text(encoding="utf-8-sig"))["legacy_reference"]
    checksum = digest(args.checkpoint)
    if checksum != source["sha256"] or args.checkpoint.stat().st_size != source["size"]:
        raise ValueError("Checkpoint is not the pinned official legacy hubert_base.pt")
    torch.set_num_threads(args.threads)
    from fairseq import checkpoint_utils, __version__ as fairseq_version
    models, cfg, _ = checkpoint_utils.load_model_ensemble_and_task([str(args.checkpoint)])
    if cfg.task.normalize:
        raise ValueError("Unexpected waveform normalization")
    model = models[0].float().eval()
    if args.attention == "manual":
        for module in model.modules():
            if hasattr(module, "onnx_trace"):
                module.onnx_trace = True
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    names = {output.name for output in session.get_outputs()}
    layer9_name = "unit9" if "unit9" in names else "units9"
    rows = []
    with torch.inference_mode(), context(args.attention):
        for case, audio in audio_cases(args.lengths, args.wav):
            outputs = ((12, "unit12"),) if args.v2_only else ((9, layer9_name), (12, "unit12"))
            for layer, name in outputs:
                expected = model.extract_features(source=torch.from_numpy(audio),
                    padding_mask=torch.zeros(audio.shape, dtype=torch.bool), mask=False, output_layer=layer)[0]
                if layer == 9:
                    expected = model.final_proj(expected)
                expected = expected.numpy()
                actual = session.run([name], {session.get_inputs()[0].name: audio})[0]
                same_shape = actual.shape == expected.shape
                finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
                error = np.abs(actual - expected) if same_shape and finite else None
                passed = (same_shape and finite and np.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
                          and error.mean() <= 1e-4)
                rows.append(dict(case=case, output=name, shape=list(actual.shape),
                    reference_shape=list(expected.shape), finite=finite,
                    max_abs=float(error.max()) if error is not None else None,
                    mean_abs=float(error.mean()) if error is not None else None, passed=bool(passed)))
    report = dict(kind="legacy-fairseq-versus-onnx", checkpoint_sha256=checksum,
        onnx_sha256=digest(args.onnx), provider="CPUExecutionProvider", atol=args.atol, rtol=args.rtol, max_mean_abs=1e-4,
        torch=torch.__version__, fairseq=fairseq_version, python=sys.version.split()[0],
        onnxruntime=ort.__version__, numpy=np.__version__, attention=args.attention, threads=args.threads,
        passed=all(row["passed"] for row in rows), cases=rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=report["passed"], comparisons=len(rows),
                         max_abs=max(row["max_abs"] or 0 for row in rows))))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
