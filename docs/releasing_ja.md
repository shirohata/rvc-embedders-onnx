# モデル・ソースパッケージの生成

固定した元重みからONNXを再生成し、説明書・ライセンス・出典を添えたZIPを作る手順です。
通常変換のアプリ配布版由来ONNXを比較参照に使います。コマンドはローカル生成・梱包までを行います。

[利用手順](../README.ja.md) / [出典・再配布条件](redistribution_ja.md) / [数値検証](fairseq-compatibility_ja.md)

## 対象と配布物

v0.1.0の対象は次の7件です。定義と保留理由は
[release_catalog.json](../vc_embedders/release_catalog.json)、保存済みの数値検証結果は
[release-validation-0.1.0.json](release-validation-0.1.0.json)で確認できます。

| モデルID | 生成元 |
| --- | --- |
| `rvc-fairseq-base` / `rvc-hubert-base` / `applio-contentvec` | 作者ContentVecのBox重み |
| `applio-chinese-hubert-base` | TencentGameMateのHugging Face重み |
| `applio-korean-hubert-base` | team-lucidのHugging Face重み |
| `applio-spin` / `applio-spin-v2` | Applio公式の固定重み |

日本語HuBERTとw-okadaの既存ONNXは対象外です。ContentVecの3件は同じ作者重みを共有しますが、
アプリ向けの抽出プロファイルを個別に記録します。

モデルZIPのファイル名は `vc-embedders-0.1.0-<ID>.zip` です。

| モデルZIPの中身 | 内容 |
| --- | --- |
| `model.onnx` | CPU FP32で特徴量を検証した単一ファイルのONNX |
| `README.md` | 英語の説明書：使い方、入出力、出典・変更内容、検証範囲 |
| `README.ja.md` | 日本語の説明書（英語版と同じ内容） |
| `LICENSE.txt` | 適用するモデルライセンスの許諾本文 |

別に開発者向け `vc-embedders-0.1.0-source.zip` を生成します。配布物は合計8 ZIPです。
ソースZIPはプログラム・カタログ・テスト・本ドキュメント・検証集計・必要な上流原文を含みます。
出典・変更内容は各言語のREADMEに統合し、独自のNOTICEやJSONの生成・検証記録は同梱しません。

配布ZIPのSHA-256は、GitHubがアップロード時に計算し、Release画面やAPIの `digest` で提供します。
利用者はダウンロードしたZIPをその値と照合できます。
詳細は[GitHubの説明](https://github.blog/changelog/2025-06-03-releases-now-expose-digests-for-release-assets/)を参照してください。

## 生成先と検証記録

| 生成先（既定値） | 内容 |
| --- | --- |
| `cache/release/` | 固定revision・SHA-256・サイズを照合した元重み |
| `artifacts/<ID>/` | 通常変換で作る比較参照ONNX |
| `artifacts/release/v0.1.0/<ID>/` | 再配布用ONNX、manifest、比較結果 |
| `artifacts/release-audit/v0.1.0/` | 詳細な生成記録、ソース一覧、ZIP・メンバーハッシュ |
| `dist/v0.1.0/` | モデルZIP 7件、ソースZIP 1件 |

これらの生成先はGit対象外です。監査先の `release-index.json`・`source-manifest.json` で、
生成したZIPとソースの一覧・ハッシュを確認できます。

## 再生成手順

### 1. 変換環境と比較参照を用意する

[READMEのセットアップ](../README.ja.md#セットアップ)に従い、Python 3.12の専用環境を用意します。
コマンドはリポジトリルートのPowerShellで実行します。

通常カタログから7件の比較参照を生成します。

```powershell
.\.venv\Scripts\python.exe -m vc_embedders prepare rvc-fairseq-base rvc-hubert-base applio-contentvec applio-chinese-hubert-base applio-korean-hubert-base applio-spin applio-spin-v2
```

### 2. 再配布用ONNXを生成する

16 kHzモノラルPCM16の検証音声を用意し、`$validationWav` にパスを指定します。
音声ファイルはGitやZIPに含めません。別の音声を使った結果は保存済み実音声ケースの完全再現にはなりません。

```powershell
$validationWav = "sample.wav"
.\.venv\Scripts\python.exe scripts/build_release.py prepare --wav $validationWav
```

カタログの固定した元重みを取得・照合し、ONNX変換、PyTorch参照計算との比較、
通常変換のONNXとの比較を行います。既存成果物は上書きしません。
未生成のモデルだけ処理する場合は、例えば `prepare --models applio-spin applio-spin-v2 --wav $validationWav`
のように指定できます。`package` は常に7件すべての検証済み成果物を必要とします。

### 3. ContentVecの3成果物を実fairseqと比較する

通常変換とは別の、fairseq 0.12.2が動くPythonを `$fairseqPython` に指定します。
保存済みの再配布用比較はPython 3.9.13 / PyTorch 2.0.0+cu118の4スレッドで実施しました。
CPUで比較し、[互換性調査](fairseq-compatibility_ja.md#検証環境で観測したcpu数値異常)にある
不安定な参照環境を避けてください。スクリプト自体の既定値は1スレッドです。

```powershell
& $fairseqPython scripts/compare_legacy_fairseq.py --checkpoint cache/rvc-fairseq-base/hubert_base.pt --onnx artifacts/release/v0.1.0/rvc-fairseq-base/model.onnx --report artifacts/release/v0.1.0/rvc-fairseq-base/fairseq-parity.json --threads 4 --atol 0.0005 --wav $validationWav
& $fairseqPython scripts/compare_legacy_fairseq.py --checkpoint cache/rvc-fairseq-base/hubert_base.pt --onnx artifacts/release/v0.1.0/rvc-hubert-base/model.onnx --report artifacts/release/v0.1.0/rvc-hubert-base/fairseq-parity.json --threads 4 --atol 0.0005 --wav $validationWav
& $fairseqPython scripts/compare_legacy_fairseq.py --checkpoint cache/rvc-fairseq-base/hubert_base.pt --onnx artifacts/release/v0.1.0/applio-contentvec/model.onnx --report artifacts/release/v0.1.0/applio-contentvec/fairseq-parity.json --threads 4 --atol 0.0005 --v2-only --attention manual --wav $validationWav
```

ここで比較するONNXは `artifacts/release/v0.1.0/` のファイルです。
通常変換側の合格レポートをコピーして代用せず、レポートの `onnx_sha256` と対象ファイルを一致させます。

### 4. 契約テストと梱包を実行する

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/build_release.py package
```

梱包処理は次を確認します。

- 元重みの固定仕様・manifest・ONNXのSHA-256が一致すること。
- PyTorch・通常ONNXとの比較、およびContentVec 3件の実fairseq比較が梱包対象のハッシュと一致し、合格していること。
- 必要なライセンス文書が取得記録のSHA-256と一致すること。
- ONNXに外部重み参照や開発者パス等がなく、ソースに許可したファイルだけが入ること。
- ZIPの全メンバーを展開読み取りし、元データのSHA-256と一致すること。

これらはCPU特徴テンソルと梱包の検査です。GPU実行や音声変換全体の音質・遅延を保証しません。

## ライセンス資料の扱い

採用根拠とモデル固有の注意は[出典・再配布条件](redistribution_ja.md)、固定原文とハッシュは
[licenses/sources.json](../licenses/sources.json)にあります。原文は編集せず、モデルカード等をソースZIPに収録します。
モデルZIPのREADMEには出典と変更内容を記載します。上流が提供するNOTICE原文がある場合は別途保持します。

## 文書を変更した後の再梱包

ソースや同梱文書を変更した場合、既存ZIPには反映されません。
新しい出力先と監査先を指定して再梱包し、監査先の `release-index.json` に記録されたZIPのハッシュと生成記録を確認します。
モデルを再変換する必要がなければ、検証済みのビルド先をそのまま使えます。

```powershell
.\.venv\Scripts\python.exe scripts/build_release.py package --output dist/v0.1.0-docs --audit artifacts/release-audit/v0.1.0-docs
```

上の出力先・監査先も未作成である必要があります。
圧縮済みモデルを再利用する場合は、さらに `--reuse-archives dist/v0.1.0` を指定します。
この場合だけ7-Zipが必要で、検証済みモデルの圧縮データを再利用します。新しいZIPの全メンバーを再照合し、元のZIPは変更しません。

## GitHub Releaseへの公開

1. `./scripts/dev.ps1 -Task Check` を実行し、梱包した `source-manifest.json` の一覧・ハッシュが確定コミットと一致することを確認します。未追跡ファイルも梱包対象になるため、Gitの状態だけでは確認できません。
2. 確定コミットをpushし、同じコミットのタグでRelease下書きを作成します。8 ZIPを添付し、GitHubが示す各assetのSHA-256（APIの `digest`）を監査用 `release-index.json` と照合します。
3. リリースノート・タグ・全添付ファイルを確認して公開します。公開後は別ディレクトリに8 ZIPを再取得し、SHA-256を再照合します。不一致なら配布を止めて原因を確認します。

通常CIはチェックのみで、梱包・公開は行いません。コミット・push・タグ・アップロード・公開は、依頼で明示された範囲で実行します。
モデルはRelease assets、検証要約と出典はGit、キャッシュ・音声・詳細監査資料はGit対象外に置きます。

梱包コードはv0.1.0用です。次版では `pyproject.toml`、`release.py` のバージョン検査・既定値・保存先と文書を更新します。
ONNXを再生成した場合は、そのハッシュに対応する検証記録を作成し、過去版の結果で代用しません。
