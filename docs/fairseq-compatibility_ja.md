# 特徴量の互換性と検証条件

検証日：2026-10-20。対象はCPU FP32のembedder特徴量です。
RVC fairseq版・RVC Transformers版のv1/v2出力、およびApplio ContentVecのv2出力は、
検査した入力と明示した許容誤差の範囲で数値互換でした。
言語別HuBERT・SPIN系とContentVecの交換互換、GPU実行、音声変換全体の音質・遅延は対象外です。

[利用手順](../README.ja.md) / [出典・再配布条件](redistribution_ja.md)

## 結果と読み方

ファイルのSHA-256一致、テンソル値の一致、異なる実装間の出力比較は別の証拠です。
smoke検査は数値互換の証拠には含めません。重みの対応は[出典調査](redistribution_ja.md)、
2026-10-20に再配布用の別の取得元から生成した7成果物の結果は
[release-validation-0.1.0.json](release-validation-0.1.0.json)を参照してください。
2026-11-06の日本語HuBERT配布候補のCPU比較は[専用の検証要約](japanese-hubert_ja.md)にあります。

### 独立した実fairseqとの比較

| 実fairseqの参照環境 | 比較対象ONNX | 比較数 | 最大絶対誤差 | ケースごとの平均絶対誤差の最大 |
| --- | --- | --- | --- | --- |
| PyTorch 2.0.0+cu118 / fairseq 0.12.2、4スレッド、通常attention | [rvc-fairseq-base](compatibility/fairseq20-legacy-qualified.json) | 38 | 2.612e-4 | 1.528e-5 |
| PyTorch 2.4.0+cu121 / fairseq 0.12.2、1スレッド、通常attention | [rvc-hubert-base](compatibility/fairseq24-current-qualified.json) | 38 | 2.562e-4 | 1.731e-5 |
| PyTorch 2.0.0+cu118 / fairseq 0.12.2、4スレッド、明示attention | [applio-contentvec](compatibility/fairseq20-applio-qualified.json) のv2出力 | 19 | 2.643e-4 | 1.658e-5 |

95件すべて `atol=5e-4, rtol=1e-4`、平均絶対誤差 `1e-4` 以下で合格しました。
フレーム数・次元・有限値も確認しています。ビット単位の同一性や、任意の入力に対する誤差上限は保証しません。

入力は無音・220 Hz正弦波・固定seedノイズの800 / 1600 / 3200 / 16000 / 24017 / 48000サンプルと、
16 kHzモノラルPCM16の実音声1件です。偶数・奇数の特徴フレーム数、fairseqの末尾パディングの
有無を含みます。比較数は「入力ケース × 出力」の数で、v1/v2の2出力なら38件、v2のみなら19件です。

### PyTorchとONNX Runtime CPUの拡張比較

通常変換の8成果物・190比較で合格しました。`--atol 0.001` を明示しており、
観測した最大絶対誤差は3.942e-4以下、ケースごとの平均絶対誤差は1e-4以下でした。
モデル別の誤差・ONNXハッシュ・依存バージョンは [validation-summary.json](validation-summary.json) にあります。

対象はRVCの2系統とApplioの6種類の標準出力です。各モデル自身の参照計算と比較し、
任意custom embedderと `--include-applio-v1` の追加出力は集計に含めていません。

## 許容誤差と不合格記録

| 検証条件 | 絶対許容誤差 `atol` | 相対許容誤差 `rtol` | ケースごとの平均絶対誤差上限 |
| --- | --- | --- | --- |
| 通常変換のカタログ既定値 | `1e-4`（日本語・中国語HuBERTは `1e-3`） | `1e-4` | `1e-4` |
| 上記の実fairseq比較 | `5e-4` | `1e-4` | `1e-4` |
| 上記の8成果物拡張比較 | `1e-3` | `1e-4` | `1e-4` |

実fairseq比較では `atol=1e-4` の条件で3秒ノイズの `unit12` が不合格でした。
記録は [fairseq / RVC fairseq版](compatibility/fairseq20-legacy.json) と
[fairseq / RVC Transformers版](compatibility/fairseq24-current.json) に保持しています。
同じ入力長ではPyTorch 2.7とONNX Runtimeの間にも約2.745e-4の差が観測されました。

日本語・中国語HuBERTは、通常入力の初期比較でも `atol=1e-4` を超える差がありました。
日本語のノイズで最大約5.7e-4、中国語の短い無音で約3.9e-4です。
日本語ではORT最適化を無効にしても差が残りました。この経緯とモデル別の既定値は
[カタログ](../vc_embedders/catalog.json)の `validation_policy` に記載しています。
個々の `validation.failed.json` や中間診断ログはGitに含みません。

## 参照した抽出処理と重み

- [RVC fairseq系・20240604ブランチの固定コミット](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/d60cd3d009b7e438f92f8624fa89b710d10c6a39/infer/modules/train/extract_feature_print.py)
- [RVC Transformers版・固定コミット](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/infer/hubert.py)
- Applioを含む参照先とrevision：[upstream-sources.json](upstream-sources.json)
- 16 kHz・波形標準化なしの参照設定：[current-waveform-preprocessing.json](compatibility/current-waveform-preprocessing.json)
- 取得したRVCソースのSHA-256と確認箇所：[upstream-source-checks.json](compatibility/upstream-source-checks.json)

RVC v1は `extract_features(output_layer=9)` と学習済み `final_proj`、v2は
`output_layer=12` です。変換ツールは固定SHA-256のfairseq重みをweights-onlyで読み、
特徴抽出テンソルを対応付けます。通常の変換にfairseqは不要ですが、独立比較には実際の
fairseqローダーと推論処理を使います。

## 検証環境で観測したCPU数値異常

PyTorch 2.4.0+cu121の検証環境では、CPUの4スレッド実行で参照計算に数値異常が観測されたため、
この条件の出力は互換性の判定に使いません。
同じ環境の1スレッドと、PyTorch 2.0.0+cu118・2.7.1+cpuの検証環境の1・4スレッドでは
weight_normの診断を通過しています。

weight_norm単体でも差があり、原因となるDLL・ビルド・ハードウェアの組み合わせは未特定です。
PyTorch 2.4の全環境には一般化できません。独立比較スクリプトの既定値は1スレッドです。
参照出力にNaNや大きな差が出る場合は、同じ入力・重みで1スレッドの比較と下記のカーネル診断を行ってください。

## 再現手順

[READMEの環境](../README.ja.md#セットアップ)を用意し、通常変換の8成果物を生成します。
以下はリポジトリルートのPowerShellで実行します。

```powershell
.\.venv\Scripts\python.exe -m vc_embedders prepare all
.\.venv\Scripts\python.exe -m vc_embedders verify all --lengths 800 1600 3200 16000 24017 48000 --atol 0.001
```

実音声を含める場合は `verify` に `--wav sample.wav` を付けます。
音声はGitに含まないため、上のコマンドは合成入力の180比較です。実音声1件を加えると190比較になりますが、
別の音声で実行した結果は保存済みの実音声ケースの完全再現ではありません。

実fairseq比較には、通常変換とは別にfairseq 0.12.2が動く環境を用意します。
保存済みの合格記録ではPyTorch 2.0側がPython 3.9.13、2.4側がPython 3.10.11です。
`$fairseq20Python` と `$fairseq24Python` に各環境のPython実行ファイルを指定します。
通常変換用の `requirements-cpu.lock` はこれらの参照環境用ではありません。
ローダーがpickleを使用するため、スクリプトは固定SHA-256・サイズを確認してから呼びます。

```powershell
& $fairseq20Python scripts/compare_legacy_fairseq.py --checkpoint cache/rvc-fairseq-base/hubert_base.pt --onnx artifacts/rvc-fairseq-base/model.onnx --report artifacts/rvc-fairseq-base/fairseq-parity.json --threads 4 --atol 0.0005
& $fairseq24Python scripts/compare_legacy_fairseq.py --checkpoint cache/rvc-fairseq-base/hubert_base.pt --onnx artifacts/rvc-hubert-base/model.onnx --report artifacts/rvc-hubert-base/fairseq-parity.json --threads 1 --atol 0.0005
& $fairseq20Python scripts/compare_legacy_fairseq.py --checkpoint cache/rvc-fairseq-base/hubert_base.pt --onnx artifacts/applio-contentvec/model.onnx --v2-only --attention manual --report artifacts/applio-contentvec/fairseq-parity.json --threads 4 --atol 0.0005
```

この3コマンドも実音声なしでは合計90比較です。各コマンドに `--wav sample.wav` を付けると95比較になります。
入力ケース・許容誤差・出力だけでなく、レポート内のチェックポイントとONNXのハッシュを照合してください。

CPUカーネルの診断は、確認したいPyTorch環境のPythonを `$diagnosticPython` に指定して実行します。
対象はカタログで固定したRVC Transformers版の重みです。

```powershell
& $diagnosticPython scripts/diagnose_cpu_kernels.py --weights cache/rvc-hubert-base/pytorch_model.bin --report artifacts/cpu-kernels.json
```
