# Core ML 圧縮・変換 ベンチマーク結果

- **ベースモデル**: `mobilenet_v3_small` (ImageNet 1000クラス)
- **入力**: RGB 画像 224×224 (前処理はモデルに埋込済み — アプリ側コード不要)
- **デプロイターゲット**: iOS18 / macOS15
- **量子化**: Post-Training Quantization (重みのみ)。INT8=per-channel/対称、INT4=per-block(16)/非対称
- **計測条件**: レイテンシ ウォームアップ 10 回 → 計測 100 回, ComputeUnit=ALL。精度ドリフト 60 サンプル
- **計測マシン**: Darwin arm64, coremltools 9.0, torch 2.12.0

## 1. ファイルサイズ比較

| モデル | 量子化粒度 | サイズ (MB) | 削減率 (%) | 対 FP32 比 |
|---|---|---:|---:|---:|
| PyTorch FP32 (参考) | — | 9.83 | — | 1.000× |
| Core ML FP16 | n/a | 4.97 | 49.4% | 0.506× |
| Core ML INT8 | per_channel | 2.60 | 73.6% | 0.264× |
| Core ML INT4 | per_block(16) linear非対称 | 1.91 | 80.6% | 0.194× |

## 2. 推論レイテンシ (ダミー画像, 100回連続)

| モデル | 実行デバイス | Mean (ms) | P95 (ms) | 対 PyTorch 速度比 |
|---|---|---:|---:|---:|
| PyTorch FP32 (参考) | CPU (PyTorch) | 40.371 | 41.138 | 1.00× |
| Core ML FP16 | Neural Engine (ANE) | 1.031 | 1.193 | 39.16× |
| Core ML INT8 | Neural Engine (ANE) | 0.959 | 1.166 | 42.12× |
| Core ML INT4 | GPU | 2.939 | 4.118 | 13.74× |

## 3. 実画像・全1000クラス絶対精度 (ImageNet-V2, 最重要指標)

**ImageNet-V2 (matched-frequency, 全1000クラス×10枚=10,000枚)** での絶対
top-1 / top-5。前処理は torchvision 公式 (短辺256→中央224) に統一。
V2 は本来の val より難しく絶対値は低めだが、**全クラスを評価できる**。

| モデル | top-1 | top-5 | 対 FP32 top-1 差 | 判定 |
|---|---:|---:|---:|---|
| PyTorch FP32 | 54.71% | 77.19% | — (基準) | 基準 |
| Core ML FP16 | 54.84% | 76.96% | +0.1pt | ✅ 公開可 (劣化ほぼ無し) |
| Core ML INT8 | 54.68% | 76.83% | -0.0pt | ✅ 公開可 (劣化ほぼ無し) |
| Core ML INT4 | 51.91% | 75.23% | -2.8pt | ✅ 公開可 (劣化小) |

> 量子化の劣化が全クラスで確定: FP16/INT8 はほぼ無劣化、INT4 は top-1 約 -2.8pt。

## 4. (補助) 実画像・10クラス絶対精度 (Imagenette)

ImageNet の 10 クラス実画像サブセット **Imagenette** での絶対精度。
クラス数が少ないため絶対値は高め (易しいクラス構成)。

| モデル | top-1 | top-5 | 対 FP32 top-1 差 | 判定 |
|---|---:|---:|---:|---|
| PyTorch FP32 | 63.93% | 86.93% | — (基準) | 基準 |
| Core ML FP16 | 62.80% | 86.13% | -1.1pt | ✅ 公開可 (劣化ほぼ無し) |
| Core ML INT8 | 62.87% | 86.27% | -1.1pt | ✅ 公開可 (劣化ほぼ無し) |
| Core ML INT4 | 63.53% | 84.60% | -0.4pt | ✅ 公開可 (劣化ほぼ無し) |

> 10 クラスのみのため INT4 の劣化が過小評価される。全クラス評価は上表 (ImageNet-V2)。

## 5. (参考) 合成入力での相対ドリフト (FP16 基準)

低周波**合成**画像 (分布外) での FP16 との予測一致度。**絶対精度ではない**。
分布外入力では微小な重み誤差で予測が大きく揺れるため、INT4 の数値は
実画像 (上表) より悲観的に出る点に注意 (相対的な優劣判定の補助指標)。

| モデル | top-1 一致率 | top-5 ヒット率 | KL(FP16‖q) | JS |
|---|---:|---:|---:|---:|
| Core ML FP16 | 100.0% | 100.0% | 0.0000 | 0.0000 |
| Core ML INT8 | 90.0% | 100.0% | 0.0098 | 0.0025 |
| Core ML INT4 | 55.0% | 96.7% | 0.2449 | 0.0569 |

## 6. 計算ハードウェアの特定 (MLComputePlan)

`coremltools.models.compute_plan.MLComputePlan` で、各オペレーションが
実際に割り当てられた計算デバイスを検知した結果。

| モデル | 主デバイス | デバイス別オペレーション数 | 実計算op合計 |
|---|---|---|---:|
| Core ML FP16 | Neural Engine (ANE) | Neural Engine (ANE): 164, CPU: 2 | 166 |
| Core ML INT8 | Neural Engine (ANE) | Neural Engine (ANE): 164, CPU: 2 | 166 |
| Core ML INT4 | GPU | GPU: 129, Neural Engine (ANE): 35, CPU: 2 | 166 |

## 補足・運用上の結論

- **前処理の埋込**: `ImageType(scale=1/255)` で 0–255 → 0–1 に正規化し、
  per-channel の ImageNet 正規化 `(x-mean)/std` をモデル先頭層に焼き込み、
  さらに **softmax をモデル末尾に焼き込み**済み。出力は合計1の真の確率辞書で、
  Swift 側は `model.prediction(image:)` を呼ぶだけ (前処理・softmax 不要)。
- **総合推奨は INT8**: 全1000クラス (ImageNet-V2) で top-1 は FP32 と実質同等
  (-0.0pt)、サイズ FP32 比 1/4、ANE 常駐で最速。精度・サイズ・速度・ANE 常駐の
  バランスが最良で、まず INT8 を選べば間違いない。
- **INT4 (per-block16/非対称) は軽量だが軽い精度劣化あり**: 全1000クラスで top-1
  約 -2.8pt の劣化 (10クラス Imagenette では -0.4pt と過小評価される点に注意)。
  per-channel/対称では完全に破綻する (実画像 top-1 3.8%) ため必ず per-block/非対称を
  使うこと。さらに INT4 は ANE に載らず GPU 実行で INT8 より数倍遅い。
  => INT4 はサイズ最優先 (-0.7MB) で 2〜3pt の精度低下と低速化を許容できる場合のみ。
- **レイテンシの注意**: FP16 と INT8 がほぼ同値なのは ANE 実行かつ Python
  `predict()` のオーバーヘッドが支配的なため。一方 INT4(per-block) は ANE に
  載らず GPU にフォールバックするため**逆に数倍遅い** (速度・精度・ANE 常駐の
  すべてで INT8 に劣る)。厳密な実機レイテンシは Xcode の Core ML Performance
  Report / Instruments で計測すべき。「対 PyTorch 速度比」は CPU(PyTorch) 対
  Core ML の比較であり量子化単体の効果ではない。
- **デバイス検知の注意**: `MLComputePlan` の値は静的な割当**予測**であり、
  実機ランタイムの実測ではない。
