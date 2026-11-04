from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .hubert import reference_features


def audio_cases(lengths: list[int], wav_paths: list[Path]):
    rng = np.random.default_rng(20261020)
    for length in lengths:
        if length < 800:
            raise ValueError("Validation lengths must be >= 800 samples")
        phase = np.arange(length, dtype=np.float32) / 16000
        yield f"silence-{length}", np.zeros((1, length), dtype=np.float32)
        yield f"tone-{length}", (0.15 * np.sin(2 * np.pi * 220 * phase))[None].astype(np.float32)
        yield f"noise-{length}", (rng.standard_normal((1, length)) * 0.05).astype(np.float32)
    for index, path in enumerate(wav_paths):
        with wave.open(str(path), "rb") as source:
            if source.getframerate() != 16000 or source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ValueError("--wav requires 16 kHz mono PCM16 WAV (no implicit resampling)")
            audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if audio.size < 800:
            raise ValueError("WAV is too short")
        # Keep machine-specific paths and user filenames out of reports.
        yield f"wav-{index}", audio[None]


def create_session(path: Path, threads: int):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def compare_arrays(actual, expected, *, atol: float, rtol: float, max_mean_abs: float = 1e-4) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite feature values")
    error = np.abs(actual - expected)
    return dict(shape=list(actual.shape), max_abs=float(error.max()), mean_abs=float(error.mean()),
                passed=bool(np.allclose(actual, expected, atol=atol, rtol=rtol) and error.mean() <= max_mean_abs))


def validate(path: Path, model, outputs: list[dict], lengths: list[int], wav_paths: list[Path],
             threads: int, atol: float, rtol: float, max_mean_abs: float = 1e-4) -> dict:
    session = create_session(path, threads)
    if len(session.get_inputs()) != 1 or session.get_inputs()[0].type != "tensor(float)":
        raise ValueError("Expected one float32 waveform input")
    names = [item["name"] for item in outputs]
    rows = []
    with torch.inference_mode():
        for case, audio in audio_cases(lengths, wav_paths):
            reference = reference_features(model, torch.from_numpy(audio), outputs)
            results = session.run(names, {session.get_inputs()[0].name: audio})
            for spec, actual in zip(outputs, results, strict=True):
                if actual.ndim != 3 or actual.shape[0] != 1 or actual.shape[-1] != spec["channels"]:
                    raise ValueError(f"Invalid output contract: {actual.shape}")
                expected_frames = int(model._get_feat_extract_output_lengths(audio.shape[1]))
                if actual.shape[1] != expected_frames:
                    raise ValueError("Unexpected feature frame count")
                rows.append(dict(case=case, output=spec["name"],
                                 **compare_arrays(actual, reference[spec["name"]].numpy(), atol=atol,
                                                  rtol=rtol, max_mean_abs=max_mean_abs)))
    return dict(kind="pytorch-versus-onnx", provider="CPUExecutionProvider", atol=atol, rtol=rtol,
                max_mean_abs=max_mean_abs,
                passed=all(row["passed"] for row in rows), cases=rows)


def smoke(path: Path, lengths: list[int], threads: int, wav_paths: list[Path] | None = None) -> dict:
    """Upstream ONNX without a PyTorch reference gets a smoke report, never a parity claim."""
    session = create_session(path, threads)
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 2:
        raise ValueError("Expected one float32 [1, samples] input")
    rows = []
    for case, audio in audio_cases(lengths, wav_paths or []):
        for spec, output in zip(session.get_outputs(), session.run(None, {inputs[0].name: audio}), strict=True):
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[-1] not in (256, 768):
                raise ValueError(f"Unsupported upstream output {spec.name}: {output.shape}")
            if not np.isfinite(output).all():
                raise ValueError("Non-finite upstream ONNX output")
            rows.append(dict(case=case, output=spec.name, shape=list(output.shape)))
    return dict(kind="upstream-onnx-smoke-only", provider="CPUExecutionProvider", passed=True, cases=rows)
