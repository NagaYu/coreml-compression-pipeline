#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 The Core ML Compression Pipeline Authors
# SPDX-License-Identifier: Apache-2.0
"""
benchmark.py
============
compressor.py が生成した成果物を計測し、結果を benchmark_result.md に保存する。

計測項目:
  1. ファイルサイズ比較   : PyTorch FP32 / Core ML FP16 / INT8 / INT4 の MB と削減率(%)。
  2. 推論レイテンシ       : ダミー画像で 100 回連続推論し、平均(Mean) と P95(ms)。
  3. 計算ハードウェア特定 : 各 Core ML モデルが CPU / GPU / Neural Engine の
                            どれで実行されるかを MLComputePlan で検知。

実行: python3 benchmark.py   (先に compressor.py を実行しておくこと)
"""

import json
import os
import time
import warnings
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torchvision

import coremltools as ct
from coremltools.models.utils import compile_model
from coremltools.models.compute_plan import MLComputePlan

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
MANIFEST = os.path.join(ARTIFACT_DIR, "manifest.json")

N_WARMUP = 10
N_RUNS = 100

# 人間に分かりやすいデバイス名
DEVICE_LABEL = {
    "MLCPUComputeDevice": "CPU",
    "MLGPUComputeDevice": "GPU",
    "MLNeuralEngineComputeDevice": "Neural Engine (ANE)",
}


# ----------------------------------------------------------------------------
# 計算ハードウェアの特定
# ----------------------------------------------------------------------------
def detect_compute_devices(mlpackage_path: str):
    """
    モデルをコンパイルし、MLComputePlan から各オペレーションが
    どの計算デバイスに割り当てられたかの分布を返す。
    戻り値: (dominant_label, {label: count}, total_ops)
    """
    compiled = compile_model(mlpackage_path)
    plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.ALL)
    fn = plan.model_structure.program.functions["main"]

    counter = Counter()
    for op in fn.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue  # const など実計算を伴わない op
        dev = type(usage.preferred_compute_device).__name__
        counter[DEVICE_LABEL.get(dev, dev)] += 1

    total = sum(counter.values())
    dominant = counter.most_common(1)[0][0] if total else "Unknown"
    return dominant, dict(counter), total


# ----------------------------------------------------------------------------
# 推論レイテンシ計測 (Core ML)
# ----------------------------------------------------------------------------
def bench_coreml(mlpackage_path: str, input_name: str, img_size: int):
    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)
    # 固定形状なので入力は img_size x img_size の RGB 画像
    dummy = (np.random.rand(img_size, img_size, 3) * 255).astype(np.uint8)
    img = Image.fromarray(dummy, mode="RGB")
    feed = {input_name: img}

    for _ in range(N_WARMUP):
        model.predict(feed)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        model.predict(feed)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    times = np.array(times)
    return float(times.mean()), float(np.percentile(times, 95))


# ----------------------------------------------------------------------------
# 推論レイテンシ計測 (PyTorch FP32, 参考)
# ----------------------------------------------------------------------------
def bench_pytorch(img_size: int):
    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = torchvision.models.mobilenet_v3_small(weights=weights).eval()
    x = torch.rand(1, 3, img_size, img_size)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return float(times.mean()), float(np.percentile(times, 95))


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def main():
    if not os.path.exists(MANIFEST):
        raise SystemExit("manifest.json が見つかりません。先に `python3 compressor.py` を実行してください。")

    with open(MANIFEST) as f:
        manifest = json.load(f)

    input_name = manifest["input_name"]
    img_size = manifest["img_size"]
    arts = manifest["artifacts"]

    base_mb = arts["pytorch_fp32"]["size_mb"]

    rows = []  # (表示名, サイズMB, 削減率, mean_ms, p95_ms, device, gran)

    print("計測中: PyTorch FP32 (CPU, 参考) ...")
    pt_mean, pt_p95 = bench_pytorch(img_size)
    rows.append(("PyTorch FP32 (参考)", base_mb, 0.0, pt_mean, pt_p95, "CPU (PyTorch)", "—"))

    plan_map = {
        "Core ML FP16": ("coreml_fp16", arts["coreml_fp16"]),
        "Core ML INT8": ("coreml_int8", arts["coreml_int8"]),
        "Core ML INT4": ("coreml_int4", arts["coreml_int4"]),
    }

    device_details = {}
    for label, (key, info) in plan_map.items():
        path = info["path"]
        size_mb = info["size_mb"]
        gran = info.get("granularity", "—")
        reduction = (1.0 - size_mb / base_mb) * 100.0

        print(f"計測中: {label} レイテンシ ({N_RUNS} 回) ...")
        mean_ms, p95_ms = bench_coreml(path, input_name, img_size)

        print(f"計測中: {label} 計算ハードウェア検知 ...")
        dominant, dist, total = detect_compute_devices(path)
        device_details[label] = (dominant, dist, total)

        rows.append((label, size_mb, reduction, mean_ms, p95_ms, dominant, gran))

    # ------------------------------------------------------------------
    # Markdown 生成
    # ------------------------------------------------------------------
    md = []
    md.append("# Core ML 圧縮・変換 ベンチマーク結果\n")
    md.append(f"- **ベースモデル**: `{manifest['model_name']}` (ImageNet 1000クラス)")
    md.append(f"- **入力**: RGB 画像 {img_size}×{img_size} (前処理はモデルに埋込済み — アプリ側コード不要)")
    md.append(f"- **デプロイターゲット**: {manifest['min_deployment_target']}")
    md.append(f"- **量子化**: Post-Training Quantization / linear_symmetric / per-channel")
    md.append(f"- **計測条件**: ウォームアップ {N_WARMUP} 回 → 計測 {N_RUNS} 回, ComputeUnit=ALL")

    machine = f"{os.uname().sysname} {os.uname().machine}"
    md.append(f"- **計測マシン**: {machine}, coremltools {ct.__version__}, torch {torch.__version__}\n")

    # 表1: サイズ
    md.append("## 1. ファイルサイズ比較\n")
    md.append("| モデル | 量子化粒度 | サイズ (MB) | 削減率 (%) | 対 FP32 比 |")
    md.append("|---|---|---:|---:|---:|")
    for name, size_mb, red, _, _, _, gran in rows:
        ratio = size_mb / base_mb
        red_str = "—" if name.startswith("PyTorch") else f"{red:.1f}%"
        md.append(f"| {name} | {gran} | {size_mb:.2f} | {red_str} | {ratio:.3f}× |")
    md.append("")

    # 表2: レイテンシ
    md.append("## 2. 推論レイテンシ (ダミー画像, 100回連続)\n")
    md.append("| モデル | 実行デバイス | Mean (ms) | P95 (ms) | 対 PyTorch 速度比 |")
    md.append("|---|---|---:|---:|---:|")
    for name, _, _, mean_ms, p95_ms, device, _ in rows:
        speedup = pt_mean / mean_ms if mean_ms > 0 else 0.0
        md.append(f"| {name} | {device} | {mean_ms:.3f} | {p95_ms:.3f} | {speedup:.2f}× |")
    md.append("")

    # 表3: 計算ハードウェア分布
    md.append("## 3. 計算ハードウェアの特定 (MLComputePlan)\n")
    md.append("`coremltools.models.compute_plan.MLComputePlan` で、各オペレーションが")
    md.append("実際に割り当てられた計算デバイスを検知した結果。\n")
    md.append("| モデル | 主デバイス | デバイス別オペレーション数 | 実計算op合計 |")
    md.append("|---|---|---|---:|")
    for label in plan_map.keys():
        dominant, dist, total = device_details[label]
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items(), key=lambda x: -x[1]))
        md.append(f"| {label} | {dominant} | {dist_str} | {total} |")
    md.append("")

    md.append("## 補足\n")
    md.append("- **前処理の埋込**: `ImageType(scale=1/255)` で 0–255 → 0–1 に正規化し、")
    md.append("  per-channel の ImageNet 正規化 `(x-mean)/std` をモデル先頭層に焼き込み済み。")
    md.append("  さらに `ClassifierConfig` で出力を `classLabel` + 確率辞書にしているため、")
    md.append("  Swift 側は `model.prediction(image:)` を呼ぶだけで推論できる。")
    md.append("- **per-channel 量子化**: 出力チャネルごとに個別スケールを用いるため、")
    md.append("  一様 (per-tensor) 量子化より精度劣化を抑制できる。")
    md.append("- **INT4**: 重みを 4bit 化し FP32 比で大幅軽量化。`minimum_deployment_target=iOS18`")
    md.append("  (macOS15) 以降で利用可能。")

    out_path = os.path.join(HERE, "benchmark_result.md")
    with open(out_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n結果を {out_path} に保存しました。\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
