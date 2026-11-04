"""Reproduce local CPU weight_norm errors separately from fairseq/attention.

Run unchanged in each PyTorch environment. This only reads pinned model weights
and writes a report; it never patches the runtime or the source application.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with args.weights.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5":
        raise ValueError("Not the pinned RVC Transformers weights")
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    v = state["encoder.pos_conv_embed.conv.parametrizations.weight.original1"].float()
    g = state["encoder.pos_conv_embed.conv.parametrizations.weight.original0"].float()
    rows = []
    for threads in (1, 4):
        torch.set_num_threads(threads)
        # FP64 explicit definition is independent of the fused CPU operator.
        expected = (v.double() * g.double() / v.double().norm(dim=(0, 1), keepdim=True)).float()
        for repeat in range(3):
            actual = torch._weight_norm(v, g, 2)
            error = (actual - expected).abs()
            finite = bool(torch.isfinite(actual).all())
            rows.append(dict(threads=threads, repeat=repeat, finite=finite,
                             max_abs=float(error.max()) if finite else None,
                             mean_abs=float(error.mean()) if finite else None,
                             passed=finite and bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-4))))
    report = dict(kind="isolated-cpu-weight-norm", torch=torch.__version__,
                  checkpoint_sha256=digest.hexdigest(), comparisons=rows,
                  passed=all(row["passed"] for row in rows))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
