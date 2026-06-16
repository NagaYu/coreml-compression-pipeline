# Core ML Compression Pipeline

A small, reproducible pipeline that converts a PyTorch image classifier into a
quantized **Core ML** model optimized for the **Apple Neural Engine (ANE)** — and
benchmarks it honestly, **including real-image accuracy**.

It takes `torchvision`'s `mobilenet_v3_small`, **bakes the entire preprocessing
pipeline _and_ the final softmax into the model** (so your Swift code just feeds a
raw image and reads back real probabilities), and applies **Post-Training
Quantization** to produce INT8 and INT4 variants. Companion scripts measure file
size, latency (mean / P95), the compute device each op actually runs on (via
`MLComputePlan`), and **absolute top-1 / top-5 accuracy on real images**.

> 日本語の概要は [下のセクション](#日本語概要) を参照してください。

---

## TL;DR — is it shippable?

**Yes — ship INT8.** Validated on all 1000 classes (ImageNet-V2, 10k images).

- **Ship INT8 by default** — top-1 is **-0.0pt vs FP32**, stays resident on the
  ANE, fastest, ¼ the size. Best overall balance, no measurable accuracy loss.
- **INT4 is smaller but has a real, modest loss** — **-2.8pt top-1** across all
  1000 classes, **and** it can't stay on the ANE (falls back to GPU) so it's
  several times slower. Pick it only when minimal size beats latency & 2-3pt.
- **Quantization recipe matters a lot for INT4.** Naive per-channel/symmetric INT4
  *collapses* (3.8% top-1). You must use **per-block(16) + asymmetric**.
- **Watch out for evaluation bias:** a 10-class subset (Imagenette) made INT4 look
  near-lossless (-0.4pt). Only full-1000-class evaluation revealed the true -2.8pt.

---

## Results

Measured on Apple Silicon (M2), macOS 15+, `coremltools 9.0`. Verified on both
`torch 2.7.0` (pinned) and `torch 2.12.0`. Latency is 100 sequential inferences
after 10 warmup runs, `ComputeUnit=ALL`.

### 1. File size

| Model | Quant. granularity | Size (MB) | Reduction | vs FP32 |
|---|---|---:|---:|---:|
| PyTorch FP32 (baseline) | — | 9.83 | — | 1.000× |
| Core ML FP16 | n/a | 4.97 | 49.4% | 0.506× |
| **Core ML INT8** | **per-channel / symmetric** | **2.60** | **73.6%** | **0.264×** |
| Core ML INT4 | per-block(16) / asymmetric | 1.91 | 80.6% | 0.194× |

### 2. Accuracy on real images — all 1000 classes (the metric that decides shippability)

Absolute top-1 / top-5 on **ImageNet-V2** (matched-frequency, all 1000 classes ×
10 = 10,000 images). The original ImageNet val set is access-gated, so this public,
full-1000-class test set is used; V2 is harder than the original val (absolute
numbers run lower) but covers every class. Preprocessing matches torchvision's
official transform (resize-256 → center-crop-224). PyTorch FP32 is the baseline.

| Model | top-1 | top-5 | Δ top-1 vs FP32 | Verdict |
|---|---:|---:|---:|---|
| PyTorch FP32 | 54.71% | 77.19% | — | baseline |
| Core ML FP16 | 54.84% | 76.96% | +0.1pt | ✅ shippable |
| **Core ML INT8** | **54.68%** | **76.83%** | **-0.0pt** | ✅ **shippable** |
| Core ML INT4 | 51.91% | 75.23% | -2.8pt | ⚠️ usable, real loss |

> A 10-class subset (Imagenette) reported INT4 at only **-0.4pt** — full-1000-class
> evaluation is what exposed the true **-2.8pt**. Don't judge quantization on a
> handful of easy classes. (Run `imagenetv2_accuracy.py` / `real_accuracy.py` to
> reproduce both.)

### 3. Latency & compute device

| Model | Device (MLComputePlan) | Mean (ms) | P95 (ms) | vs PyTorch |
|---|---|---:|---:|---:|
| PyTorch FP32 | CPU | 40.8 | 42.1 | 1.00× |
| Core ML FP16 | Neural Engine | 1.06 | 1.41 | 38× |
| **Core ML INT8** | **Neural Engine** | **0.98** | **1.29** | **41×** |
| Core ML INT4 | **GPU** | 4.67 | 5.53 | 9× |

`MLComputePlan` places **164 / 166 real ops on the Neural Engine** for FP16 and
INT8 (2 on CPU). INT4(per-block) is **not ANE-friendly** — 129 ops fall to the
GPU, which is why it's slower despite being smaller.

> **Honest notes.**
> - The `predict()`-based latency is dominated by Python/IPC overhead, so FP16 vs
>   INT8 look nearly identical here. Measure true on-device latency with Xcode's
>   **Core ML Performance Report** / Instruments. The "vs PyTorch" column compares
>   CPU(PyTorch) against Core ML — it is *not* a quantization speedup.
> - The earlier impression that "INT4 is broken" came from evaluating on synthetic
>   noise (out-of-distribution), which exaggerates INT4 drift. On **real** images
>   INT4(per-block) is fine. Naive per-channel INT4, however, is genuinely broken
>   (3.8% top-1) on real images too — hence the per-block/asymmetric recipe.

---

## Requirements

- **Apple Silicon** Mac (M-series), macOS 15 (Sequoia) or newer for INT4.
- **Python 3.13** (other 3.x versions work too).
- `coremltools 9.0` is **required on Python 3.13** — older builds ship without the
  native extensions (`libcoremlpython` / `libmilstoragepython`) needed for
  on-device prediction, compute-plan detection, and INT4 blob storage.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Convert + quantize. Writes artifacts/ (.pt, fp16/int8/int4 .mlpackage, manifest.json)
python compressor.py

# 2. Benchmark size / latency / hardware / synthetic drift. Writes benchmark_result.md
python benchmark.py

# 3. (recommended) Full 1000-class accuracy. Downloads ImageNet-V2 (~1.2 GB) to data/,
#    writes artifacts/imagenetv2_accuracy.json
python imagenetv2_accuracy.py

# 3b. (optional) Quick 10-class accuracy. Downloads Imagenette (~95 MB) to data/,
#     writes artifacts/real_accuracy.json
python real_accuracy.py

# Re-run benchmark.py after step 3/3b to fold the accuracy tables into the report.
```

The generated `.mlpackage` files can be dropped straight into Xcode.

## How it works

1. **Preprocessing _and_ softmax baked in.** `ImageType(scale=1/255)` rescales
   pixels to `[0, 1]`, a wrapper module bakes the per-channel ImageNet
   normalization `(x - mean) / std` into the first layer, and a final
   `nn.Softmax` is baked into the last layer (torchvision's classifier ends in a
   linear/logit layer, which `ClassifierConfig` does **not** soften). Net effect:
   the output is a `classLabel` plus a **true probability dictionary that sums to
   1**, and **the Swift side needs no preprocessing and no softmax** — just
   `model.prediction(image:)`.
2. **Weight-only PTQ** via `coremltools.optimize.coreml.linear_quantize_weights`:
   - **INT8** — `granularity="per_channel"`, `mode="linear_symmetric"`. Accuracy
     ≈ FP32, stays on the ANE. This is the recommended production config.
   - **INT4** — `granularity="per_block"`, `block_size=16`, `mode="linear"`
     (asymmetric). `mobilenet_v3_small` (depthwise convs + SE blocks + hard-swish)
     is hostile to INT4: per-channel/symmetric collapses to 3.8% top-1, while
     per-block/asymmetric recovers to within -2.8pt of FP32 (all 1000 classes).
     Layers whose channel count isn't divisible by the block size stay FP16.
3. **INT4** requires `minimum_deployment_target=iOS18` (macOS 15) and the native
   blob-storage writer in `coremltools 9.0`.

## Project layout

```
compressor.py          # convert + quantize (fp16 / int8 / int4), softmax baked in
benchmark.py           # size / latency / compute-device / synthetic drift / real-acc tables
imagenetv2_accuracy.py # absolute top-1/top-5 on all 1000 classes (ImageNet-V2)
real_accuracy.py       # quick absolute top-1/top-5 on 10 classes (Imagenette)
requirements.txt       # pinned, verified dependency set
benchmark_result.md    # generated report (sample committed)
artifacts/             # generated models + manifest (git-ignored, regenerable)
data/                  # downloaded dataset (git-ignored)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

"Apple", "Core ML", and "Neural Engine" are trademarks of Apple Inc. This is an
independent project, not affiliated with or endorsed by Apple.

---

## 日本語概要

PyTorch の画像分類モデル（`mobilenet_v3_small`）を **Apple Neural Engine (ANE)**
向けの量子化 **Core ML** モデルへ変換・圧縮し、**実画像精度まで含めて**厳密に
計測する再現可能なパイプラインです。

### 結論：INT8 で公開できます

全1000クラス（ImageNet-V2, 1万枚）で検証した top-1 劣化（対 FP32）:
**FP16 +0.1pt / INT8 -0.0pt / INT4 -2.8pt**。

- **デフォルトは INT8 推奨**：精度の実測劣化なし・ANE 常駐・最速・サイズ 1/4。
- **INT4 はさらに小さいが軽い精度低下あり**（全クラスで top-1 -2.8pt）。さらに
  ANE に載れず GPU 実行で INT8 より数倍遅い。サイズ最優先・2〜3pt 許容時のみ。
- **INT4 は量子化レシピが重要**：素朴な per-channel/対称は破綻（top-1 3.8%）。
  **per-block(16) + 非対称**が必須。
- **評価データに注意**：10クラス（Imagenette）では INT4 -0.4pt と過小評価。
  全1000クラス評価で初めて真の -2.8pt が判明。

### 主な実装ポイント

- **前処理と softmax をモデルに完全埋込**：`ImageType(scale=1/255)`、per-channel の
  ImageNet 正規化、末尾の softmax を焼き込み。出力は **合計1の真の確率**で、
  **Swift 側は画像を渡すだけ**（前処理・softmax 不要）。
- **重みのみ PTQ**：INT8=per-channel/対称、INT4=per-block(16)/非対称。
- `MLComputePlan` で実行デバイスを検知（FP16/INT8 は 166op 中 164op が ANE、
  INT4 は GPU フォールバック）。

```bash
pip install -r requirements.txt
python compressor.py          # 変換・量子化
python imagenetv2_accuracy.py # 全1000クラス絶対精度（ImageNet-V2 を data/ に DL）
python benchmark.py           # サイズ / レイテンシ / 実行HW / 精度
```

> 注：`predict()` 計測のレイテンシは Python オーバーヘッドが支配的で、FP16/INT8 は
> ほぼ同等に見えます。厳密な実機レイテンシは Xcode の Core ML Performance Report で
> 計測してください。合成画像での「ドリフト」は分布外入力ゆえ INT4 を過度に悲観的に
> 見せるため、公開判断には**実画像精度**（上表）を用いること。
