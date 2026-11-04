# rvc-embedders-onnx

[日本語](README.ja.md)

ONNX embedders for RVC and Applio, with a CPU tool to fetch pinned source weights,
export ONNX, and verify feature tensors. Model files are distributed as Release assets.

## Download

1. Open [GitHub Releases](https://github.com/shirohata/rvc-embedders-onnx/releases) and download `vc-embedders-0.1.0-<ID>.zip` for your model.
2. Compare its SHA-256 with the digest shown beside the asset. PowerShell: `Get-FileHash <path-to-ZIP> -Algorithm SHA256`.
3. Extract the ZIP and select `model.onnx` in an app that supports its output names and shapes, such as vc-rs. No Python setup is needed to use the prebuilt file.

Each model ZIP includes English/Japanese instructions and its model license.
`vc-embedders-0.1.0-source.zip` contains the conversion tool and supporting evidence.

## Choose a model

**Match the embedder, feature layer, and projection used to train your voice model.**
Equal output dimensions do not make different embedders interchangeable.

| ID | Profile | Outputs | Release ZIP |
| --- | --- | --- | --- |
| `rvc-fairseq-base` | fairseq-based RVC | `unit9`, `unit12` | Yes |
| `rvc-hubert-base` | Pinned Transformers-based RVC | `unit9`, `unit12` | Yes |
| `applio-contentvec` | Applio ContentVec | `unit12` | Yes |
| `applio-japanese-hubert-base` | Applio Japanese HuBERT | `unit12` | Export only |
| `applio-chinese-hubert-base` | Applio Chinese HuBERT | `unit12` | Yes |
| `applio-korean-hubert-base` | Applio Korean HuBERT | `unit12` | Yes |
| `applio-spin` | Applio SPIN | `unit12` | Yes |
| `applio-spin-v2` | Applio SPIN v2 | `unit12` | Yes |

`unit9` is layer 9 plus the trained projection (256 channels, RVC v1).
`unit12` is layer 12 (768 channels, RVC v2 / Applio default).
Applio's optional `unit12_proj` uses the final layer plus the trained projection;
export it with `--include-applio-v1`. It is a different extraction from RVC v1.
Weights without the required trained projection are rejected.

Input: float32, 16 kHz mono waveform, `[1, samples]`, scaled to approximately `[-1, 1]`
without mean/variance normalization. Output: `[1, frames, channels]`, variable time axes,
ONNX opset 17. The downstream app handles frame interpolation and voice conversion.

## Build from source

Verified environment: Windows x64, Python 3.12, CPU. In PowerShell at the repository root:

```powershell
./scripts/dev.ps1 -Task Setup
./scripts/dev.ps1 -Task Check
.\.venv\Scripts\python.exe -m vc_embedders list
.\.venv\Scripts\python.exe -m vc_embedders prepare rvc-fairseq-base
.\.venv\Scripts\python.exe -m vc_embedders verify rvc-fairseq-base
```

Normal exports need neither CUDA nor fairseq. `Setup` installs the [pinned CPU dependencies](requirements-cpu.lock);
use `-Python <python-executable>` if the Python Launcher is unavailable. The lock file pins
package versions, not wheel hashes or the OS. `Check` runs dependency, source-package,
license-hash, and contract checks. [GitHub Actions](.github/workflows/test.yml) runs the same
checks on pushes, pull requests, and manual runs; it does not generate or publish model packages.

`prepare all` selects all eight catalog models; release packaging uses seven profiles
and can use different weight sources. To separate download and export, use `fetch <ID>`
and `export <ID>`. Import local weights with `fetch <ID> --local-dir <directory>`;
local files must also match the pinned hashes.

Source weights go to `cache/<ID>/`; ONNX and validation records go to `artifacts/<ID>/`.
Completed exports are not overwritten; use a new `--output` directory to rebuild.
Add a 16 kHz mono PCM16 WAV with `verify <ID> --wav sample.wav`; no resampling is performed.

## Validation and licenses

Validation covers CPU feature tensors at the documented inputs and tolerances.
It does not establish GPU compatibility or end-to-end audio quality/latency.
The [verification report](docs/fairseq-compatibility_ja.md) distinguishes export checks,
independent fairseq comparisons, failed stricter conditions, and the saved 2026-10-20 results.

Code: [MIT](LICENSE). Model licenses are separate. Fixed application revisions and checkpoints
are in the [source references](docs/upstream-sources.json) and [catalog](vc_embedders/catalog.json).
The following supporting documents are in Japanese:

- [Sources and redistribution conditions](docs/redistribution_ja.md)
- [Feature compatibility and verification](docs/fairseq-compatibility_ja.md)
- [Rebuilding and publishing release packages](docs/releasing_ja.md)
