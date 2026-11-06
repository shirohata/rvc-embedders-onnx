# Embedderの元配布先・同一重み・再配布条件

調査日：2026-10-20。日本語HuBERTの取得元・梱包方針は2026-11-06に更新しました。
カタログにある10モデルについて、配布元、取得可否、ファイルのSHA-256、
名前を対応付けたテンソル値、ライセンス表示を照合した記録です。
以下は固定した資料に基づくパッケージ生成方針であり、配布元の現在の状態や、すべての権利を保証するものではありません。

[利用手順](../README.ja.md) / [パッケージ生成手順](releasing_ja.md) / [特徴量の互換性](fairseq-compatibility_ja.md)

## 結果

| ID | 重みの同一性・取得経路の根拠 | 採用する表示 | ローカル梱包の対象 |
| --- | --- | --- | --- |
| `rvc-fairseq-base` | RVC Transformers版と213/213テンソル一致、作者ContentVecと同値 | 作者リポジトリMIT | 対象。作者のBox重みから再生成 |
| `rvc-hubert-base` | 作者ContentVecの公式Box重みと213/213テンソル一致 | 作者リポジトリMIT | 対象。作者のBox重みから再生成 |
| `applio-contentvec` | 同じ作者チェックポイントと213/213テンソル一致 | 作者リポジトリMIT | 対象。作者のBox重みから再生成 |
| `applio-chinese-hubert-base` | TencentGameMateの元配布ファイルとSHA-256一致 | 元モデルカードMIT | 対象。元配布者から取得 |
| `applio-korean-hubert-base` | team-lucidの元配布ファイルとSHA-256一致 | 元モデルカードApache-2.0 | 対象。元配布者から取得 |
| `applio-japanese-hubert-base` | 元開発貢献者yky-hの固定ミラーとApplio版の設定・重みのSHA-256一致 | ミラーのモデルカードが元モデルのApache-2.0と権利保持を明記 | 対象。日本語ZIPはv0.1.1 |
| `applio-spin` | 作者公開HuBERT 5種と比較し、各115/213一致・98テンソル相違 | Applio公式READMEのモデル重みMIT表明 | 対象。Applio公式重みを使用 |
| `applio-spin-v2` | 同じ5種と比較し、各83/213一致・130テンソル相違 | 同上 | 対象。Applio公式重みを使用 |
| `wokada-contentvec` | 配布済みONNXを固定ハッシュで取得 | 配布者のGPL-3.0表示 | 対象外。対応ソースと通知の準備が必要 |
| `wokada-japanese-hubert` | 配布済みONNXを固定ハッシュで取得 | 配布者のGPL-3.0表示・Apache由来の表示 | 対象外。対応ソースと通知の準備が必要 |

ローカル梱包対象8件と保留2件の定義は [release_catalog.json](../vc_embedders/release_catalog.json) にあります。
2026-10-20の7成果物の検証結果は [release-validation-0.1.0.json](release-validation-0.1.0.json)、
日本語の2026-11-06の結果は[日本語HuBERTの説明](japanese-hubert_ja.md)にあります。
同じ元ContentVecでも、RVC用の2出力とApplio用の最終層出力は別の成果物として扱う。

## 1. RVC／Applio ContentVec：作者の公式配布から取得

- 作者：[auspicious3000/contentvec](https://github.com/auspicious3000/contentvec/tree/d746688a32940f4bee410ed7c87ec9cf8ff04f74)
- [作者のMIT全文](https://github.com/auspicious3000/contentvec/blob/d746688a32940f4bee410ed7c87ec9cf8ff04f74/LICENSE)
- [公式READMEの学習済みモデル一覧](https://github.com/auspicious3000/contentvec/blob/d746688a32940f4bee410ed7c87ec9cf8ff04f74/README.md) が示す **ContentVec_legacy / 500**。
- [公式Box共有リンク](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)／[取得した直接URL](https://ibm.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr.pt)
- ファイル：`checkpoint_best_legacy_500.pt`、1,330,114,945 bytes。
- SHA-256：`60d936ec5a566776fc392e69ad8b630d14eb588111233fe313436e200a7b187b`

RVC Transformers版（`cc8c20...`）とApplio ContentVec（`d8dd400...`）の全213テンソルは、
fairseq→Transformersの名前対応後、float32の全要素が `torch.equal` で一致した。
ファイル形式・state_dict名・保存dtypeは異なるため、ファイルSHA-256一致とは区別する。
比較した作者チェックポイントではRVCのfp16保存に伴う数値の追加丸めは観測されなかった。
学習用の `label_embs_concat` やoptimizer状態は比較対象の推論モデルに含めない。

作者のMITリポジトリが公式に案内するチェックポイントを直接取得し、自前のMIT変換処理で
必要なテンソルと出力を生成する経路を採る。RVC配布物内の研究用途追記を公開根拠にしない。
モデルファイル個別に独立したライセンス文書は見つからず、根拠は作者リポジトリのMITと
そこからの公式配布リンクである。この適用範囲を明示して著作権・許諾文を保持する。
原文の著作権行は不完全だが補完せず、その事情を同梱文書で説明する。

推論結果の数値互換は[別の比較](fairseq-compatibility_ja.md)で確認している。
通常カタログのRVC／Applio配布物と、作者取得を起点に再生成するパッケージは取得経路を分け、
それぞれの生成元・ハッシュを記録する。既存ONNXをテンソル一致だけでMITと再ラベルしない。

## 2. 中国語・韓国語：元開発者のファイルとバイト一致

### 中国語HuBERT

- 作者配布：[TencentGameMate/chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base)
- revision：`fce0375452b1dd6c080ac3248d423d4d037bc831`
- [固定モデルカード・MIT表示](https://huggingface.co/TencentGameMate/chinese-hubert-base/blob/fce0375452b1dd6c080ac3248d423d4d037bc831/README.md)
- `pytorch_model.bin`：377,552,987 bytes。
- SHA-256：`2fefccd26c2794a583b80f6f7210c721873cb7ebae2c1cde3baf9b27855e24d8`
- Applioで取得済みの元重みと同じバイト。MIT本文・配布者のモデルカード・元権利表示を同梱する。

原配布には独立した著作権行・LICENSEがないため、元カードと標準MIT許諾本文を保持し、著作権者や年を創作しない。

### 韓国語HuBERT

- 作者配布：[team-lucid/hubert-base-korean](https://huggingface.co/team-lucid/hubert-base-korean)
- revision：`e4d80ff0b7ae7701be69f2f3375316b1c10c4294`
- [固定モデルカード・Apache-2.0表示](https://huggingface.co/team-lucid/hubert-base-korean/blob/e4d80ff0b7ae7701be69f2f3375316b1c10c4294/README.md)
- `pytorch_model.bin`：377,554,841 bytes。
- SHA-256：`931f6232879f8eadf7dbd9e00e1fa4cac61ad269af89d509b2ed75009b1a02c5`
- Applioで取得済みの元重みと同じバイト。Apache-2.0全文・元権利表示・存在するNOTICEを保持し、
  ONNX変換、削除した出力等の変更内容を明記する。ApplioのMIT表示だけに置き換えない。

固定版にNOTICEがないことは [licenses/sources.json](../licenses/sources.json) のファイル一覧で確認している。

## 3. 日本語：元開発貢献者のApacheミラーから取得

2026-11-06に確認した [yky-h/japanese-hubert-base](https://huggingface.co/yky-h/japanese-hubert-base/tree/9c86c1a3424e8cddca3e59d01c2ed5477e628fcc)
を日本語HuBERTの取得元とします。日本語ZIPはv0.1.1の対象で、v0.1.0には含まれません。

- revision：`9c86c1a3424e8cddca3e59d01c2ed5477e628fcc`。
- [固定モデルカード](../licenses/japanese-hubert-MODEL_CARD.md)はrinnaの元モデルのミラーであり、
  Apache-2.0と元著作者の権利を保持すると明記しています。モデル貢献者としてリンクされるYukiya Hono氏のアカウントです。
- 設定と重みは固定したApplio版とバイト一致します。重みは377,554,841 bytes、
  SHA-256 `6c023ccb71e4c2b5a324c94fc5ebe12403d3081c5f370df229892419996fd113`。
- 固定ミラーのファイル一覧にNOTICEはありません。独立した著作権行は創作せず、Apache-2.0全文と
  モデルカード原文を日本語モデルZIPに同梱し、READMEに元著作者・ミラー・変換内容を記載します。
- 第12層・768次元の `unit12` のみを出力します。学習済み最終射影は元のTransformers重みに含まれず、
  `unit9` や `unit12_proj` を生成しません。

### 2026-10-20の調査記録

2026-10-20の調査で `rinna/japanese-hubert-base` のAPIとREADMEは401だった。削除・非公開・認証要件の
どれが原因かは断定しない。rinnaの[公式発表](https://rinna.co.jp/news/2024/03/20240307.html)は
HuBERT Baseを含むモデルをApache-2.0と説明している。

- [2024年のミラーREADME](https://huggingface.co/wok000/vcclient_modules/blob/012d8a6b2930015f2a2b79b9ad132efc61cde7a4/rinna_hubert/Readme.md) はrinnaを出典としてApache-2.0と明記。
- [その時点の重み](https://huggingface.co/wok000/vcclient_modules/blob/012d8a6b2930015f2a2b79b9ad132efc61cde7a4/rinna_hubert/rinna_hubert_base.pt)
  のLFSポインタは `dade3cf824ae0d214f7de8b73e70bae7c101e81f12d93577c4760bf516db4063`。
- 比較用に取得した同一SHAの378,888,853-byteチェックポイントから、Applioが使用する
  211テンソルの数値がすべて一致した。未使用のlabel embedding・final projectionは別。
- **調査時のミラーREADMEはGPL-3.0表示も含む**。過去のApache記録と混同せず、
  そのミラーのONNXをApache/MITと読み替えて再配布しない。

この調査に基づく公開済みv0.1.0の7モデルには日本語を含めていません。
上記の貢献者ミラーを使うローカル梱包とは取得元・検証日を区別します。

## 4. SPIN／SPIN v2：Applio公式のモデル重みMIT表明を採用

- [Spin作者のREADME](https://github.com/vectominist/spin/blob/73edb7ae120be67a2d584931eeb388caecda744e/README.md)
- [作者コードのMIT](https://github.com/vectominist/spin/blob/73edb7ae120be67a2d584931eeb388caecda744e/LICENSE)
- [作者の重み配布先](https://huggingface.co/datasets/vectominist/spin_ckpt/tree/e83ab02f68750a3a79c19e775b37e10816dc865e)

HuBERTの128/256/512/1024/2048 cluster版をすべて取得し、`encoder.model.*` と
`pred_head.layers.0.*` をApplioのbackbone・projectionに対応付けて比較した。
確認した5種には完全一致がなく、元重みを特定できなかった。
[Applio公式READMEの固定版](https://github.com/IAHispano/Applio/blob/5379cf261fa7d8085c3b077aa42dabca7c85310b/README.md#terms-of-use-and-commercial-usage)
はコードとモデル重みをMITで許諾し、改変・再配布・商用利用を認めると明記している。
この明記と同じ固定版のLICENSEを根拠に、Applio公式のSPIN・SPIN v2をパッケージ対象とする。
ライセンス原文・著作権表示は保持し、特定のSpin研究チェックポイントとの同一性や、
Applioが元モデルすべてを学習したとの主張はしない。
公式重みのrevisionは `e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14`。
ライセンス資料は [LICENSE原文](../licenses/applio-LICENSE.txt) と
[README原文](../licenses/applio-README.md)に保持する。

## 5. w-okadaの既存ONNX：GPL条件下で別枠

[配布元](https://huggingface.co/wok000/weights_gpl/tree/d1b578cfcdf944dd3910786f7604754c79234b36)
は含まれるモデルをGPL-3.0と明記する。固定ONNXのSHA-256は既存カタログと一致する。
GPL全文、著作権・元ライセンス表示、変更の有無、対応ソースを取得できる方法を用意する。
配布ONNXに対応する正確な変換コード一式の特定が未完了のため、v0.1.0のパッケージ対象には含まない。

ContentVecの作者MIT重みや日本語のApache元重みから**自前で生成した別ONNX**と、
w-okadaが配布する既存のGPL ONNXを区別する。テンソルが同じというだけで既存ONNXを再ライセンスしない。

## 同梱資料

モデルの許諾本文・元権利表示・出典・変更内容を保持し、コードのMITを全モデルに適用しない。
同梱構成は[パッケージ生成手順](releasing_ja.md)、原文と固定ハッシュは
[licenses/sources.json](../licenses/sources.json)にある。

## テンソル比較の再確認

`scripts/audit_checkpoint_identity.py` は、元重みのSHA-256を確認してweights-onlyで読み、
名前を対応付けた候補テンソルをfloat32で比較する。候補側は事前に通常の `fetch` でハッシュ照合したものを使う。
Python 3.12の変換環境で実行する。以下は[パッケージ生成手順](releasing_ja.md)で作者のBox重みを
`cache/release/contentvec-original/hubert_base.pt` に取得・照合済みの場合の例で、キャッシュ自体はGitに含まない。

```powershell
.\.venv\Scripts\python.exe -m vc_embedders fetch rvc-hubert-base
New-Item -ItemType Directory -Force artifacts | Out-Null
.\.venv\Scripts\python.exe scripts/audit_checkpoint_identity.py `
  --source cache/release/contentvec-original/hubert_base.pt `
  --source-sha256 60d936ec5a566776fc392e69ad8b630d14eb588111233fe313436e200a7b187b `
  --candidate cache/rvc-hubert-base/pytorch_model.bin `
  --report artifacts/author-contentvec-identity.json
```

個別テンソルの比較表はローカルに保持する。固定した取得元とハッシュは上記および
[release_catalog.json](../vc_embedders/release_catalog.json)を参照する。
