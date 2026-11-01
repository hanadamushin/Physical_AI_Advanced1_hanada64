# Physical AI 応用1 最終課題 — hanada64

脳波計測準備を支援するジェル注入ロボットのシミュレーションです。
16電極の模擬インピーダンスを観測し、AIが処置量を選び、頭部後方270°を移動しながら注入・再観測・必要時のジェル補給を繰り返します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hanadamushin/Physical_AI_Advanced1_hanada64/blob/main/eeg_gel_demo/EEG_Gel_Assist_Colab.ipynb)

## 実行方法

上の「Open in Colab」を開き、CPUランタイムで「ランタイム → すべてのセルを実行」を選んでください。GPU・APIキーは不要です。ライブラリ導入から動画生成・結果ZIPの保存まで行います。ノートブックには実行コードを埋め込んであります。

ローカルで実行する場合：

```sh
git clone https://github.com/hanadamushin/Physical_AI_Advanced1_hanada64.git
cd Physical_AI_Advanced1_hanada64/eeg_gel_demo
python -m pip install -r requirements.txt
python demo.py --out output
```

Linuxの画面なし環境では、Colabセルと同様に `libosmesa6` を導入し、MuJoCoを読み込む前に `MUJOCO_GL=osmesa` と `PYOPENGL_PLATFORM=osmesa` を設定してください。

## ファイル

| ファイル | 内容 |
| --- | --- |
| `eeg_gel_demo/EEG_Gel_Assist_Colab.ipynb` | Colab用の自己完結ノートブック |
| `eeg_gel_demo/demo.py` | 模擬環境、AI学習、処置制御、経路、MuJoCo描画、動画出力 |
| `eeg_gel_demo/requirements.txt` | 必要ライブラリ |
| `eeg_gel_demo/verify.py` | フィードバック・配置・可動域・ピストン・残量の検証 |
| `eeg_gel_demo/visual_qa.py` | 頭部配置の4方向描画 |
| `eeg_gel_demo/montage_preview.py` | 電極配置図 |
| `eeg_gel_demo/rear_arc_preview.py` | 後方270°の可動域図 |
| `eeg_gel_demo/review_video.py` | 生成動画のフレーム確認 |
| `eeg_gel_demo/build_package.py` | ノートブック再生成とZIP作成。既存の結果があれば同梱 |
| `verification/checks.json` | 提出版で実施した検証の記録 |
| `example_results/summary.json`, `events.csv`, `refills.json` | seed=42の模擬実行ログ |

検証は `eeg_gel_demo` 内で `python verify.py` を実行します。配置図・動画確認用の各補助スクリプトは既定で `output_v07` を参照するため、使用する場合は `python demo.py --out output_v07` で結果を生成してください。`python build_package.py` はノートブックを現在の `demo.py` に同期します。

## AIの役割と実行確認

合成データ1,600例で学習したリッジ回帰モデルが、現在値と候補注入量から改善量を予測します。必要量を選択した後、再観測した結果に応じて補正・追加処置を行います。巡回順序と停止条件はルールです。AI利用を含む開発補助により実装し、段階的な設計変更と描画・数値確認を行いました。

Windows上の実行と描画を確認済みです。Colabについては作成者本人から、すべてのセルを最後まで実行できたとの報告を受けています。seed=42の既定結果は15/16電極が模擬目標内、20回処置、7回補給、動画176.5秒（2倍速）。T7は意図的な非応答条件で要確認として停止します。

## 仮定と限界

運動学シミュレーションであり、実測インピーダンス、実際のジェルの流体・圧力、把持、接触力制御、人体での有効性や安全性は検証していません。10 kOhmとmLはデモ用の仮定です。個人差のある頭部・毛髪・キャップの認識や実機の安全な動作には追加開発が必要です。

詳しいモデル設定、出力、検証範囲、参照資料は [使用コードの説明](eeg_gel_demo/README.md) に記載しています。
