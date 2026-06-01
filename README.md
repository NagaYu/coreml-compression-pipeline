# Core ML Compression Pipeline

A small, reproducible pipeline that converts a PyTorch image classifier into a
quantized **Core ML** model optimized for the **Apple Neural Engine (ANE)** — and
benchmarks it honestly.

It takes `torchvision`'s `mobilenet_v3_small`, **bakes the entire preprocessing
pipeline into the model** (so your Swift code just feeds a raw image), and applies
**per-channel Post-Training Quantization** to produce INT8 and INT4 variants. A
companion benchmark measures file size, latency (mean / P95), and — using
`MLComputePlan` — which compute device (CPU / GPU / Neural Engine) each op actually
runs on.

> 日本語の概要は [下のセクション](#日本語概要) を参照してください。

---

## Results

Measured on Apple Silicon (arm64), macOS 15+, `coremltools 9.0`, `torch 2.7.0`.
Latency is 100 sequential inferences after 10 warmup runs, `ComputeUnit=ALL`.

### File size

| Model | Quant. granularity | Size (MB) | Reduction | vs FP32 |
|---|---|---:|---:|---:|
| PyTorch FP32 (baseline) | — | 9.83 | — | 1.000× |
| Core ML FP16 | n/a | 4.97 | 49.4% | 0.506× |
| Core ML INT8 | per-channel | 2.60 | 73.6% | 0.264× |
| **Core ML INT4** | **per-channel** | **1.39** | **85.8%** | **0.142×** |

### Latency & compute device

| Model | Device (MLComputePlan) | Mean (ms) | P95 (ms) | vs PyTorch |
|---|---|---:|---:|---:|
| PyTorch FP32 | CPU | 42.88 | 43.45 | 1.00× |
| Core ML FP16 | Neural Engine | 1.04 | 1.34 | 41× |
| Core ML INT8 | Neural Engine | 0.99 | 1.24 | 43× |
| Core ML INT4 | Neural Engine | 0.98 | 1.29 | 44× |

`MLComputePlan` confirms **163 / 165 real ops are placed on the Neural Engine**
(2 on CPU) for every variant.

> **Honest notes.** INT4 comfortably beats the "≤ ¼ the size" goal (0.142×); INT8
> lands at 0.264× because non-weight tensors (biases, etc.) stay FP16 and the
> `.mlpackage` carries metadata. On a model this small, latency is roughly flat
> across FP16/INT8/INT4 — quantization's win here is **size and memory bandwidth**,
> not raw speed. The speed advantage of INT4 grows with model size.

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

# 2. Benchmark size / latency / hardware. Writes benchmark_result.md
python benchmark.py
```

The generated `.mlpackage` files can be dropped straight into Xcode.

## How it works

1. **Preprocessing baked in.** `ImageType(scale=1/255)` rescales pixels to `[0, 1]`,
   and a wrapper module bakes the per-channel ImageNet normalization `(x - mean) / std`
   into the first layer. `ClassifierConfig` makes the output a `classLabel` plus a
   probability dictionary. Net effect: **no preprocessing code on the Swift side** —
   just `model.prediction(image:)`.
2. **Per-channel PTQ.** `coremltools.optimize.coreml.linear_quantize_weights` with
   `mode="linear_symmetric"`, `granularity="per_channel"`. Per-channel scales reduce
   accuracy loss versus a single per-tensor scale. INT8 and INT4 use the same recipe;
   INT4 falls back to grouped per-block (size 32) only if per-channel is unavailable.
3. **INT4** requires `minimum_deployment_target=iOS18` (macOS 15) and the native
   blob-storage writer in `coremltools 9.0`.

## Project layout

```
compressor.py          # convert + quantize (fp16 / int8 / int4)
benchmark.py           # size / latency (mean,P95) / compute-device detection
requirements.txt       # pinned, verified dependency set
benchmark_result.md    # generated report (sample committed)
artifacts/             # generated models (git-ignored, regenerable)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

"Apple", "Core ML", and "Neural Engine" are trademarks of Apple Inc. This is an
independent project, not affiliated with or endorsed by Apple.

---

## 日本語概要

PyTorch の画像分類モデル（`mobilenet_v3_small`）を **Apple Neural Engine (ANE)**
向けの量子化 **Core ML** モデルへ変換・圧縮し、結果を厳密に計測する再現可能な
パイプラインです。

- **前処理をモデルに完全埋込**：`ImageType(scale=1/255)` と per-channel の ImageNet
  正規化をモデル先頭層に焼き込み、`ClassifierConfig` でラベル付き出力に。
  **Swift 側は画像を渡すだけ**で推論できます。
- **per-channel PTQ** で INT8 / INT4 量子化（一様量子化より精度劣化を抑制）。
  INT4 は FP32 比 **0.142×（85.8% 削減）**。
- `MLComputePlan` で **165 op 中 163 op が ANE 実行**であることを確認。

```bash
pip install -r requirements.txt
python compressor.py   # 変換・量子化
python benchmark.py    # サイズ / レイテンシ / 実行HW を計測
```

> 注：このサイズのモデルでは FP16/INT8/INT4 のレイテンシはほぼ同等で、量子化の主効果は
> **サイズとメモリ帯域**に表れます。モデルが大きいほど INT4 の速度メリットが顕著になります。
