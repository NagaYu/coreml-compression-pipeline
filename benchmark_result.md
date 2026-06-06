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
| PyTorch FP32 (参考) | CPU (PyTorch) | 41.245 | 42.830 | 1.00× |
| Core ML FP16 | Neural Engine (ANE) | 1.055 | 1.368 | 39.10× |
| Core ML INT8 | Neural Engine (ANE) | 0.990 | 1.322 | 41.67× |
| Core ML INT4 | GPU | 4.767 | 5.579 | 8.65× |

## 3. 実画像での絶対精度 (Imagenette, 公開判断の最重要指標)

ImageNet の 10 クラス実画像サブセット **Imagenette** で計測した
絶対 top-1 / top-5 精度。PyTorch FP32 を真のベースラインとする。

| モデル | top-1 | top-5 | 対 FP32 top-1 差 | 判定 |
|---|---:|---:|---:|---|
| PyTorch FP32 | 63.9% | 86.9% | — (基準) | 基準 |
| Core ML FP16 | 62.8% | 86.1% | -1.1pt | ✅ 公開可 (劣化ほぼ無し) |
| Core ML INT8 | 62.9% | 86.3% | -1.1pt | ✅ 公開可 (劣化ほぼ無し) |
| Core ML INT4 | 63.5% | 84.6% | -0.4pt | ✅ 公開可 (劣化ほぼ無し) |

> Imagenette は 10 クラスのため絶対値は full-ImageNet(1000クラス) より高め。
> 量子化による相対劣化の評価には十分。full-ImageNet での確定が望ましい。

## 4. (参考) 合成入力での相対ドリフト (FP16 基準)

低周波**合成**画像 (分布外) での FP16 との予測一致度。**絶対精度ではない**。
分布外入力では微小な重み誤差で予測が大きく揺れるため、INT4 の数値は
実画像 (上表) より悲観的に出る点に注意 (相対的な優劣判定の補助指標)。

| モデル | top-1 一致率 | top-5 ヒット率 | KL(FP16‖q) | JS |
|---|---:|---:|---:|---:|
| Core ML FP16 | 100.0% | 100.0% | 0.0000 | 0.0000 |
| Core ML INT8 | 90.0% | 100.0% | 0.0098 | 0.0025 |
| Core ML INT4 | 55.0% | 96.7% | 0.2449 | 0.0569 |

## 5. 計算ハードウェアの特定 (MLComputePlan)

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
- **総合推奨は INT8**: 実画像 top-1 は FP32 とほぼ同等 (上表)、サイズ FP32 比
  1/4、ANE 常駐で最速。精度・サイズ・速度・ANE 常駐のバランスが最良で公開向き。
- **INT4 (per-block16/非対称) も公開可能な精度**: 実画像 top-1 は FP32 とほぼ同等。
  ただし per-channel/対称では破綻する (実画像 top-1 3.8%) ため必ず per-block/非対称を
  使うこと。さらに INT4 は ANE に載らず GPU 実行で INT8 より数倍遅い。
  => 極小サイズが最優先で多少の遅延を許容できる場合のみ INT4 を選ぶ。
- **レイテンシの注意**: FP16 と INT8 がほぼ同値なのは ANE 実行かつ Python
  `predict()` のオーバーヘッドが支配的なため。一方 INT4(per-block) は ANE に
  載らず GPU にフォールバックするため**逆に数倍遅い** (精度は同等だが速度と
  ANE 常駐で INT8 に劣る)。厳密な実機レイテンシは Xcode の Core ML Performance
  Report / Instruments で計測すべき。「対 PyTorch 速度比」は CPU(PyTorch) 対
  Core ML の比較であり量子化単体の効果ではない。
- **デバイス検知の注意**: `MLComputePlan` の値は静的な割当**予測**であり、
  実機ランタイムの実測ではない。
