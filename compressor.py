#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 The Core ML Compression Pipeline Authors
# SPDX-License-Identifier: Apache-2.0
"""
compressor.py
=============
Core ML 向けモデル圧縮・変換パイプライン (Apple Silicon / ANE 最適化)。

処理の流れ:
  1. torchvision から軽量画像認識モデル (mobilenet_v3_small) を読み込む。
  2. 入力前処理 (スケール / per-channel 正規化 / RGB) をモデル & ImageType に
     埋め込み、iOS/macOS アプリ側で前処理コードを書かずに「画像を入れるだけ」で
     推論できる Core ML (.mlpackage / ML Program) に変換する。
  3. coremltools.optimize.coreml の PTQ (Post-Training Quantization) を用いて
     重みを INT8 / INT4 に per-channel 量子化し、ファイルサイズを 1/4 以下へ。

成果物 (すべて ./artifacts/ 配下):
  - <name>_fp32.pt            : 元 PyTorch 重み (サイズ比較のベースライン)
  - <name>_fp16.mlpackage     : Core ML (float16, 量子化なし)
  - <name>_int8.mlpackage     : Core ML INT8 per-channel 量子化
  - <name>_int4.mlpackage     : Core ML INT4 per-channel 量子化
  - manifest.json             : benchmark.py が読むメタ情報

実行: python3 compressor.py
"""

import json
import os
import shutil
import warnings

import numpy as np
import torch
import torchvision

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------------
MODEL_NAME = "mobilenet_v3_small"
INPUT_NAME = "image"
OUTPUT_NAME = "classLabelProbs"
IMG_SIZE = 224
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

# ImageNet の前処理パラメータ (mobilenet_v3 標準)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# INT4 を使うため最低デプロイターゲットを iOS18 / macOS15 に設定
MIN_TARGET = ct.target.iOS18


# ----------------------------------------------------------------------------
# 前処理を内蔵したラッパーモデル
# ----------------------------------------------------------------------------
class PreprocessedClassifier(torch.nn.Module):
    """
    ImageType 側で pixel/255 -> [0,1] にしておき、ここで per-channel の
    ImageNet 正規化 (x-mean)/std を実施。これにより ImageType の scalar scale
    だけでは表現できない per-channel std を厳密に処理できる。
    出力はロジット (ClassifierConfig が softmax/ラベル付与を担当)。
    """

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.base(x)


def _dir_size_mb(path: str) -> float:
    """ファイル / ディレクトリ (.mlpackage) の合計サイズを MB で返す。"""
    if os.path.isfile(path):
        total = os.path.getsize(path)
    else:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
    return total / (1024.0 * 1024.0)


def _save_mlmodel(model: ct.models.MLModel, path: str) -> None:
    """既存があれば消してから保存 (.mlpackage はディレクトリ)。"""
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    model.save(path)


# ----------------------------------------------------------------------------
# 1. ベースモデルの用意
# ----------------------------------------------------------------------------
def load_base_model():
    print(f"[1/5] torchvision から {MODEL_NAME} を読み込み中 ...")
    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    base = torchvision.models.mobilenet_v3_small(weights=weights)
    base.eval()
    class_labels = list(weights.meta["categories"])
    print(f"      -> 1000 クラス分類器を読込 (例: {class_labels[:3]} ...)")
    return base, class_labels


# ----------------------------------------------------------------------------
# 2. Core ML (float16) への変換
# ----------------------------------------------------------------------------
def convert_to_coreml(base, class_labels):
    print("[2/5] Core ML (ML Program / float16) へ変換中 ...")
    wrapper = PreprocessedClassifier(base).eval()

    example = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    # ImageType: 入力画像を pixel/255 -> [0,1] に。RGB / NCHW。
    image_input = ct.ImageType(
        name=INPUT_NAME,
        shape=(1, 3, IMG_SIZE, IMG_SIZE),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB,
    )

    # ClassifierConfig: 出力を classLabel + 確率辞書 にしてアプリ即利用可能に。
    classifier_config = ct.ClassifierConfig(
        class_labels=class_labels,
        predicted_feature_name="classLabel",
    )

    mlmodel = ct.convert(
        traced,
        inputs=[image_input],
        classifier_config=classifier_config,
        convert_to="mlprogram",
        minimum_deployment_target=MIN_TARGET,
        compute_units=ct.ComputeUnit.ALL,
    )

    # メタデータ (アプリの Xcode プレビューに表示される)
    mlmodel.author = "Core ML Compression Pipeline"
    mlmodel.short_description = (
        "MobileNetV3-Small ImageNet classifier. Preprocessing "
        "(scale + per-channel ImageNet normalization) is baked in: "
        "feed a raw RGB image, no Swift preprocessing required."
    )
    mlmodel.version = "1.0"
    spec = mlmodel.get_spec()
    mlmodel.input_description[INPUT_NAME] = "Input RGB image (any size, auto-resized to 224x224)"

    print("      -> float16 変換完了")
    return mlmodel


# ----------------------------------------------------------------------------
# 3. INT8 / INT4 per-channel 量子化
# ----------------------------------------------------------------------------
def quantize(mlmodel, nbits: int):
    dtype = "int8" if nbits == 8 else "int4"
    print(f"[3/5] {dtype.upper()} per-channel 量子化を実行中 ...")
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=dtype,
        granularity="per_channel",
        # 小さい重みは量子化しない (既定 2048 要素しきい値)
        weight_threshold=2048,
    )
    config = OptimizationConfig(global_config=op_config)
    try:
        q = linear_quantize_weights(mlmodel, config=config)
        print(f"      -> {dtype.upper()} per-channel 量子化 成功")
        return q, "per_channel"
    except Exception as e:
        # INT4 が per-channel 非対応の環境では per-grouped-channel にフォールバック
        if dtype == "int4":
            print(f"      ! per_channel int4 失敗 ({e}); per-block(32) にフォールバック")
            op_config = OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=dtype,
                granularity="per_block",
                block_size=32,
                weight_threshold=2048,
            )
            config = OptimizationConfig(global_config=op_config)
            q = linear_quantize_weights(mlmodel, config=config)
            print(f"      -> INT4 per-block(32) 量子化 成功")
            return q, "per_block(32)"
        raise


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    base, class_labels = load_base_model()

    # FP32 PyTorch 重みをベースラインとして保存
    pt_path = os.path.join(ARTIFACT_DIR, f"{MODEL_NAME}_fp32.pt")
    torch.save(base.state_dict(), pt_path)
    pt_mb = _dir_size_mb(pt_path)
    print(f"      -> FP32 PyTorch 重み保存: {pt_mb:.2f} MB")

    # 変換 (float16)
    mlmodel_fp16 = convert_to_coreml(base, class_labels)
    fp16_path = os.path.join(ARTIFACT_DIR, f"{MODEL_NAME}_fp16.mlpackage")
    _save_mlmodel(mlmodel_fp16, fp16_path)
    fp16_mb = _dir_size_mb(fp16_path)
    print(f"      -> FP16 Core ML 保存: {fp16_mb:.2f} MB")

    # INT8
    mlmodel_int8, gran8 = quantize(mlmodel_fp16, 8)
    int8_path = os.path.join(ARTIFACT_DIR, f"{MODEL_NAME}_int8.mlpackage")
    _save_mlmodel(mlmodel_int8, int8_path)
    int8_mb = _dir_size_mb(int8_path)
    print(f"      -> INT8 Core ML 保存: {int8_mb:.2f} MB")

    # INT4
    mlmodel_int4, gran4 = quantize(mlmodel_fp16, 4)
    int4_path = os.path.join(ARTIFACT_DIR, f"{MODEL_NAME}_int4.mlpackage")
    _save_mlmodel(mlmodel_int4, int4_path)
    int4_mb = _dir_size_mb(int4_path)
    print(f"      -> INT4 Core ML 保存: {int4_mb:.2f} MB")

    # マニフェスト
    print("[4/5] manifest.json を書き出し中 ...")
    manifest = {
        "model_name": MODEL_NAME,
        "input_name": INPUT_NAME,
        "img_size": IMG_SIZE,
        "num_classes": len(class_labels),
        "min_deployment_target": "iOS18 / macOS15",
        "preprocessing": {
            "image_scale": 1.0 / 255.0,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "color_layout": "RGB",
            "note": "scale baked in ImageType; per-channel normalization baked in model",
        },
        "artifacts": {
            "pytorch_fp32": {"path": pt_path, "size_mb": pt_mb},
            "coreml_fp16": {"path": fp16_path, "size_mb": fp16_mb, "granularity": "n/a"},
            "coreml_int8": {"path": int8_path, "size_mb": int8_mb, "granularity": gran8},
            "coreml_int4": {"path": int4_path, "size_mb": int4_mb, "granularity": gran4},
        },
    }
    with open(os.path.join(ARTIFACT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 簡易検証: INT8 モデルで 1 回推論し動作確認
    print("[5/5] 変換済み INT8 モデルでスモーク推論 ...")
    try:
        dummy = (np.random.rand(IMG_SIZE, IMG_SIZE, 3) * 255).astype(np.uint8)
        from PIL import Image

        img = Image.fromarray(dummy)
        out = mlmodel_int8.predict({INPUT_NAME: img})
        label = out.get("classLabel", "?")
        print(f"      -> 推論 OK (top-1 = {label})")
    except Exception as e:
        print(f"      ! スモーク推論はスキップ ({type(e).__name__}: {e})")

    print("\n=== サイズ要約 ===")
    print(f"  PyTorch FP32 : {pt_mb:7.2f} MB  (baseline)")
    print(f"  Core ML FP16 : {fp16_mb:7.2f} MB  ({(1-fp16_mb/pt_mb)*100:5.1f}% 削減)")
    print(f"  Core ML INT8 : {int8_mb:7.2f} MB  ({(1-int8_mb/pt_mb)*100:5.1f}% 削減)")
    print(f"  Core ML INT4 : {int4_mb:7.2f} MB  ({(1-int4_mb/pt_mb)*100:5.1f}% 削減)")
    print("\n完了。次に `python3 benchmark.py` を実行してください。")


if __name__ == "__main__":
    main()
