# rvc-embedders-onnx

[English](README.md)

RVC・Applio向けのONNX embedderと、固定した元重みの取得・ONNX変換・CPU特徴量検証を行うツールです。
モデルファイルはGitHub Releaseの添付ファイルとして配布します。

## ダウンロード

1. [GitHub Releases](https://github.com/shirohata/rvc-embedders-onnx/releases) から、下表の版に対応する `vc-embedders-<version>-<ID>.zip` を取得します。
2. ZIPのSHA-256をGitHubの該当ファイルに表示される値と照合します。PowerShell：`Get-FileHash <ZIPのパス> -Algorithm SHA256`。
3. ZIPを展開し、vc-rsなど出力名・入出力形状に対応するアプリで `model.onnx` を指定します。配布済みONNXの利用にPython環境の構築は不要です。

モデルZIPには英語・日本語の説明書とモデルライセンスを同梱しています。
同じ版の `vc-embedders-<version>-source.zip` はツールと検証・出典資料のソースです。

## モデルの選び方

**音声モデルの学習時と同じembedder・特徴層・射影を選んでください。**
出力次元が同じでも、別のembedderへの交換は互換性を意味しません。

| ID | プロファイル | 出力 | 配布ZIP |
| --- | --- | --- | --- |
| `rvc-fairseq-base` | RVC fairseq版 | `unit9`, `unit12` | v0.1.0 |
| `rvc-hubert-base` | 固定版のRVC Transformers版 | `unit9`, `unit12` | v0.1.0 |
| `applio-contentvec` | Applio ContentVec | `unit12` | v0.1.0 |
| `applio-japanese-hubert-base` | Applio 日本語HuBERT | `unit12` | v0.1.1 |
| `applio-chinese-hubert-base` | Applio 中国語HuBERT | `unit12` | v0.1.0 |
| `applio-korean-hubert-base` | Applio 韓国語HuBERT | `unit12` | v0.1.0 |
| `applio-spin` | Applio SPIN | `unit12` | v0.1.0 |
| `applio-spin-v2` | Applio SPIN v2 | `unit12` | v0.1.0 |

[v0.1.1](https://github.com/shirohata/rvc-embedders-onnx/releases/tag/v0.1.1)の添付は日本語HuBERTと対応ソースZIPの2件です。
他の7モデルは[v0.1.0](https://github.com/shirohata/rvc-embedders-onnx/releases/tag/v0.1.0)から取得します。
日本語の出典・CPU検証は[日本語HuBERTの説明](docs/japanese-hubert_ja.md)を参照してください。

`unit9` は第9層＋学習済み射影（256次元、RVC v1）、
`unit12` は第12層（768次元、RVC v2・Applio標準）です。
Applioの追加出力 `unit12_proj` は最終層＋学習済み射影で、RVC v1とは抽出層が異なります。
必要な場合は `--include-applio-v1` で生成します。必要な学習済み射影のない重みは拒否します。

入力はfloat32・16 kHzモノラルの `[1, samples]`、約 `[-1, 1]` にスケーリングした波形です。
平均・分散の標準化は加えません。出力は `[1, frames, channels]`、時間軸可変、ONNX opset 17です。
フレーム補間や音声変換は利用アプリ側で扱います。

## セットアップ

検証環境はWindows x64・Python 3.12・CPUです。リポジトリルートのPowerShellで実行します。

```powershell
./scripts/dev.ps1 -Task Setup
./scripts/dev.ps1 -Task Check
.\.venv\Scripts\python.exe -m vc_embedders list
.\.venv\Scripts\python.exe -m vc_embedders prepare rvc-fairseq-base
.\.venv\Scripts\python.exe -m vc_embedders verify rvc-fairseq-base
```

通常の変換にCUDAやfairseqは不要です。`Setup` は[固定CPU依存](requirements-cpu.lock)を導入します。
Python Launcherがない場合は `-Python <python-executable>` で指定できます。
ロックファイルはパッケージのバージョン一覧で、wheelハッシュやOS全体は固定しません。
`Check` は依存関係・ソース梱包・ライセンス原文のハッシュ・契約テストを確認します。
[GitHub Actions](.github/workflows/test.yml) もpush・PR・手動起動時に同じチェックを実行し、モデルの生成・公開は行いません。

`prepare all` は通常カタログの8件を選びます。ローカルの再配布用梱包も8プロファイルが対象で、生成元も異なる場合があります。
取得と変換を分ける場合は `fetch <ID>` と `export <ID>` を使います。
ローカル重みは `fetch <ID> --local-dir <directory>` で取り込めます。ローカルでも固定ハッシュとの一致が必要です。

元重みは `cache/<ID>/`、ONNXと検証記録は `artifacts/<ID>/` に保存します。
完成済みの出力は上書きしないため、再変換には新しい `--output` を指定してください。
追加音声の比較は `verify <ID> --wav sample.wav` で行います。16 kHzモノラルPCM16 WAVが必要で、自動リサンプリングは行いません。

## 検証・ライセンス

検証範囲は、記載した入力・許容誤差でのCPU特徴テンソルです。GPU互換性や音声変換全体の音質・遅延は未検証です。
[検証資料](docs/fairseq-compatibility_ja.md)に、変換時の比較・独立した実fairseq比較・厳しい条件での不合格理由と、
2026-10-20の保存済み結果をまとめています。

コードは [MIT](LICENSE)、モデルの利用条件は別です。固定したアプリrevision・チェックポイントは
[参照元](docs/upstream-sources.json)と[カタログ](vc_embedders/catalog.json)で確認できます。

- [出典・再配布条件](docs/redistribution_ja.md)
- [特徴量の互換性と検証](docs/fairseq-compatibility_ja.md)
- [配布パッケージの再生成・公開](docs/releasing_ja.md)
