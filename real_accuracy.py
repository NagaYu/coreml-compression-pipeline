#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_accuracy.py
================
Imagenette (ImageNet 10クラス実画像サブセット) で FP32/FP16/INT8/INT4 の
絶対 top-1 / top-5 精度を計測する。公開可否を判断するための実画像評価。

実行: python3 real_accuracy.py   (先に compressor.py を実行しておくこと)
"""
import os, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from PIL import Image
import torch, torchvision
import coremltools as ct

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(HERE, "data", "imagenette2-160", "val")
MAN = json.load(open(os.path.join(HERE, "artifacts", "manifest.json")))
INP, S = MAN["input_name"], MAN["img_size"]

# Imagenette の WNID -> ImageNet クラス index
WNID2IDX = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}
weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
CATS = list(weights.meta["categories"])
WNID2NAME = {w: CATS[i] for w, i in WNID2IDX.items()}
print("[クラス対応の検証]")
for w, n in WNID2NAME.items():
    print(f"  {w} -> idx {WNID2IDX[w]:3d} -> '{n}'")

# 評価サンプル収集 (クラスごと均等に上限を設けて高速化)
PER_CLASS = int(os.environ.get("PER_CLASS", "150"))
samples = []  # (path, true_idx, true_name)
for wnid in sorted(WNID2IDX):
    d = os.path.join(VAL, wnid)
    files = sorted(f for f in os.listdir(d) if f.endswith(".JPEG"))[:PER_CLASS]
    for f in files:
        samples.append((os.path.join(d, f), WNID2IDX[wnid], WNID2NAME[wnid]))
print(f"\n評価画像数: {len(samples)} ({PER_CLASS}/クラス)\n")

def load_rgb(path):
    return Image.open(path).convert("RGB").resize((S, S), Image.BILINEAR)

# --- PyTorch FP32 (真のベースライン) ---
def eval_pytorch():
    model = torchvision.models.mobilenet_v3_small(weights=weights).eval()
    prep = weights.transforms()  # 公式の前処理 (resize/crop/normalize)
    t1 = t5 = 0
    with torch.no_grad():
        for path, true_idx, _ in samples:
            img = Image.open(path).convert("RGB")
            x = prep(img).unsqueeze(0)
            logits = model(x)[0]
            top5 = torch.topk(logits, 5).indices.tolist()
            t1 += int(top5[0] == true_idx)
            t5 += int(true_idx in top5)
    n = len(samples)
    return t1 / n * 100, t5 / n * 100

# --- Core ML (前処理はモデル埋込なので生RGBを渡す) ---
def eval_coreml(path):
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
    t1 = t5 = 0
    for p, _, true_name in samples:
        out = m.predict({INP: load_rgb(p)})
        pred = out.get("classLabel")
        d = [v for v in out.values() if isinstance(v, dict)][0]
        top5 = [k for k, _ in sorted(d.items(), key=lambda x: -x[1])[:5]]
        t1 += int(pred == true_name)
        t5 += int(true_name in top5)
    n = len(samples)
    return t1 / n * 100, t5 / n * 100

arts = MAN["artifacts"]
res = {}
print("計測: PyTorch FP32 ...");  res["PyTorch FP32"] = eval_pytorch()
print("計測: Core ML FP16 ..."); res["Core ML FP16"] = eval_coreml(arts["coreml_fp16"]["path"])
print("計測: Core ML INT8 ..."); res["Core ML INT8"] = eval_coreml(arts["coreml_int8"]["path"])
print("計測: Core ML INT4 ..."); res["Core ML INT4"] = eval_coreml(arts["coreml_int4"]["path"])

print("\n=== Imagenette 実画像 絶対精度 ===")
print(f"{'モデル':<16}{'top-1':>9}{'top-5':>9}")
for k, (a, b) in res.items():
    print(f"{k:<16}{a:>8.1f}%{b:>8.1f}%")

json.dump({k: {"top1": a, "top5": b} for k, (a, b) in res.items()},
          open(os.path.join(HERE, "artifacts", "real_accuracy.json"), "w"), indent=2)
print("\n-> artifacts/real_accuracy.json に保存")
