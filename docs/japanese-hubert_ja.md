# 日本語HuBERTの配布・検証

`applio-japanese-hubert-base` は、元開発貢献者のApache-2.0ミラーから取得した重みを
Applio向けのONNXに変換したモデルです。
[v0.1.1](https://github.com/shirohata/rvc-embedders-onnx/releases/tag/v0.1.1)の添付は日本語モデルZIPと対応ソースZIPの2件です。
他の7モデルはv0.1.0から取得してください。

## 選択と入出力

日本語HuBERTを使って学習したApplio音声モデル向けです。音声の言語だけで選ばず、
音声モデルの学習時と同じembedderを使ってください。

- 入力：16 kHzモノラルfloat32、`[1, samples]`、約 `[-1, 1]`。平均・分散の標準化は加えません。
- 出力：`unit12`、第12層、768次元、`[1, frames, 768]`。時間軸可変、ONNX opset 17。
- この元重みには学習済みの最終射影がありません。`unit9`・`unit12_proj` は含めません。
  RVC v1用の256次元モデルとしては使用できません。

## 固定した取得元とライセンス

元モデルはrinna Co., Ltd.の `rinna/japanese-hubert-base` です。取得には、モデルカードの貢献者一覧に
記載されたYukiya Hono氏のアカウントによる
[yky-h/japanese-hubert-baseの固定ミラー](https://huggingface.co/yky-h/japanese-hubert-base/tree/9c86c1a3424e8cddca3e59d01c2ed5477e628fcc)
を使います。revisionは `9c86c1a3424e8cddca3e59d01c2ed5477e628fcc` です。

| ファイル | bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 1,375 | `2cc68fe950504f7468f4e786a31b8bf65c922de2b9f92cd252e3919bed689fe1` |
| `pytorch_model.bin` | 377,554,841 | `6c023ccb71e4c2b5a324c94fc5ebe12403d3081c5f370df229892419996fd113` |

両ファイルは、通常カタログの固定Applio版
（`IAHispano/Applio`、revision `e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14`）とバイト一致します。
取得元の変更による別チェックポイントへの置換ではありません。ファイルサイズとSHA-256を確認してから
`weights_only=True` で読み、remote model codeは実行しません。想定外・不足のbackbone state keysは拒否します。

[ミラーのモデルカード原文](../licenses/japanese-hubert-MODEL_CARD.md)は、元モデルのApache-2.0条件と
元著作者の権利を保持すると明記しています。独立した著作権行・NOTICEは固定ミラーにありません。
日本語モデルZIPには、Apache-2.0全文の `LICENSE.txt` とカード原文の `MODEL_CARD.md` を同梱し、
英語・日本語READMEに元著作者、取得元、最終層の選択、ONNX変換を記載します。
原文の出典・SHA-256・固定ミラーのファイル一覧は [licenses/sources.json](../licenses/sources.json) にあります。
既存のGPL ONNXを流用・再ライセンスしたものではありません。

## 2026-11-06のCPU検証

[機械可読の検証要約](japanese-hubert-validation-20261106.json)に、元重み、ONNXのSHA-256、
環境、入力と許容誤差を記録しています。ONNXのSHA-256は
`97754c4452d1e263a9cc4fc4a9c2503bfe5f0892c0290d719084ecfede04e203` です。

環境はWindows x64、Python 3.12.9、PyTorch 2.7.1+cpu、Transformers 4.44.2、NumPy 2.5.4、
ONNX 1.19.1、ONNX Runtime 1.23.2、CPUExecutionProvider・4スレッドです。
無音・220 Hz正弦波・固定seedノイズについて800 / 1600 / 3200 / 16000 / 24017 / 48000サンプルを比較し、
16 kHzモノラルPCM16のローカル音声34,400サンプルを1件加えました。音声自体は同梱しません。

| 比較 | ケース数 | atol / rtol | 最大絶対誤差 | ケース平均絶対誤差の最大値 |
| --- | ---: | --- | ---: | ---: |
| PyTorch参照対ONNX CPU | 19 | `5e-4` / `1e-4` | `1.686812e-4` | `8.141367e-6` |
| 検証済みApplio版ONNX対配布候補ONNX CPU | 19 | `1e-4` / `1e-4` | `0` | `0` |

全ケースで形状・フレーム数・有限値を確認し、要素ごとの許容誤差と平均絶対誤差 `<=1e-4` の条件を満たしました。
比較参照ONNXのSHA-256は `8a09962ff8b743d887fe8784687f3c340db3310662ebbd8b2c139fd0e577754b` です。

2026-10-20の別環境での通常変換には、`atol=1e-4` で日本語のノイズが最大約 `5.7e-4` となる
不合格記録があります。通常カタログの既定値 `1e-3` とその理由は保持しています。
上表のCPUビルドで `5e-4` に合格したことは、過去の測定やすべての環境での誤差を変更するものではありません。
詳細は[歴史的な検証条件](fairseq-compatibility_ja.md#許容誤差と不合格記録)を参照してください。

対象は検査した特徴テンソルです。この日本語モデルの独立した実fairseq比較、GPU実行、
音声変換全体の音質・遅延は未検証です。

## 再生成

[セットアップ](../README.ja.md#セットアップ)後、リポジトリルートのPowerShellで実行します。
`$validationWav` は16 kHzモノラルPCM16 WAVへのパスです。異なる音声では保存済みの実音声ケースを完全再現できません。
以下の出力先は未作成である必要があります。

```powershell
$validationWav = "sample.wav"
.\.venv\Scripts\python.exe -m vc_embedders prepare applio-japanese-hubert-base --output artifacts/japanese-reference --lengths 800 1600 3200 16000 24017 48000 --wav $validationWav
.\.venv\Scripts\python.exe scripts/build_release.py prepare --models applio-japanese-hubert-base --references artifacts/japanese-reference --build artifacts/release/japanese-candidate --wav $validationWav
```

配布用の取得元は [release_catalog.json](../vc_embedders/release_catalog.json) の
`japanese-contributor-mirror` です。標準の `prepare` と配布用の `prepare` は別の固定取得元を使います。
日本語モデルとソースZIPだけを梱包するには `package --models applio-japanese-hubert-base` を使います。
ライセンス・ZIPメンバーハッシュの検査と公開手順は[パッケージ生成手順](releasing_ja.md)を参照してください。
v0.1.1は2026-11-06に検証した同じONNXを再梱包したもので、ONNXを再生成した測定ではありません。
