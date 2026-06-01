# Core ML 圧縮・変換 ベンチマーク結果

- **ベースモデル**: `mobilenet_v3_small` (ImageNet 1000クラス)
- **入力**: RGB 画像 224×224 (前処理はモデルに埋込済み — アプリ側コード不要)
- **デプロイターゲット**: iOS18 / macOS15
- **量子化**: Post-Training Quantization / linear_symmetric / per-channel
- **計測条件**: ウォームアップ 10 回 → 計測 100 回, ComputeUnit=ALL
- **計測マシン**: Darwin arm64, coremltools 9.0, torch 2.7.0

## 1. ファイルサイズ比較

| モデル | 量子化粒度 | サイズ (MB) | 削減率 (%) | 対 FP32 比 |
|---|---|---:|---:|---:|
| PyTorch FP32 (参考) | — | 9.83 | — | 1.000× |
| Core ML FP16 | n/a | 4.97 | 49.4% | 0.506× |
| Core ML INT8 | per_channel | 2.60 | 73.6% | 0.264× |
| Core ML INT4 | per_channel | 1.39 | 85.8% | 0.142× |

## 2. 推論レイテンシ (ダミー画像, 100回連続)

| モデル | 実行デバイス | Mean (ms) | P95 (ms) | 対 PyTorch 速度比 |
|---|---|---:|---:|---:|
| PyTorch FP32 (参考) | CPU (PyTorch) | 42.879 | 43.445 | 1.00× |
| Core ML FP16 | Neural Engine (ANE) | 1.042 | 1.336 | 41.16× |
| Core ML INT8 | Neural Engine (ANE) | 0.992 | 1.235 | 43.23× |
| Core ML INT4 | Neural Engine (ANE) | 0.977 | 1.287 | 43.90× |

## 3. 計算ハードウェアの特定 (MLComputePlan)

`coremltools.models.compute_plan.MLComputePlan` で、各オペレーションが
実際に割り当てられた計算デバイスを検知した結果。

| モデル | 主デバイス | デバイス別オペレーション数 | 実計算op合計 |
|---|---|---|---:|
| Core ML FP16 | Neural Engine (ANE) | Neural Engine (ANE): 163, CPU: 2 | 165 |
| Core ML INT8 | Neural Engine (ANE) | Neural Engine (ANE): 163, CPU: 2 | 165 |
| Core ML INT4 | Neural Engine (ANE) | Neural Engine (ANE): 163, CPU: 2 | 165 |

## 補足

- **前処理の埋込**: `ImageType(scale=1/255)` で 0–255 → 0–1 に正規化し、
  per-channel の ImageNet 正規化 `(x-mean)/std` をモデル先頭層に焼き込み済み。
  さらに `ClassifierConfig` で出力を `classLabel` + 確率辞書にしているため、
  Swift 側は `model.prediction(image:)` を呼ぶだけで推論できる。
- **per-channel 量子化**: 出力チャネルごとに個別スケールを用いるため、
  一様 (per-tensor) 量子化より精度劣化を抑制できる。
- **INT4**: 重みを 4bit 化し FP32 比で大幅軽量化。`minimum_deployment_target=iOS18`
  (macOS15) 以降で利用可能。
