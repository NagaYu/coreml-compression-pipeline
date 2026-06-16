#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imagenetv2_accuracy.py
======================
ImageNet-V2 (matched-frequency, 全1000クラス×10枚=10,000枚) で
FP32/FP16/INT8/INT4 の絶対 top-1 / top-5 精度を計測する。

ImageNet 本体の val は登録制で自由取得できないため、全1000クラスを
カバーする公開テストセット ImageNet-V2 を用いる。V2 は本来の val より
難しく絶対値は数pt〜10pt 程度低く出るが、**全クラスを評価できる**ため
「10クラスのみ」という批判に応えられ、量子化の相対劣化も確定できる。

実行: python3 imagenetv2_accuracy.py   (先に compressor.py を実行しておくこと)
"""
import os, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from PIL import Image
import torch, torchvision
import coremltools as ct

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(HERE, "data", "imagenetv2-matched-frequency-format-val")
MAN = json.load(open(os.path.join(HERE, "artifacts", "manifest.json")))
INP, S = MAN["input_name"], MAN["img_size"]

weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
CATS = list(weights.meta["categories"])  # index -> category name

# 評価サンプル収集: フォルダ名 = ImageNet クラス index (0..999)
MAX = int(os.environ.get("MAX_IMAGES", "0"))  # 0 = 全件
samples = []  # (path, true_idx)
for cls in sorted(os.listdir(VAL), key=lambda x: int(x) if x.isdigit() else 1e9):
    d = os.path.join(VAL, cls)
    if not os.path.isdir(d) or not cls.isdigit():
        continue
    idx = int(cls)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpeg", ".jpg", ".png")):
            samples.append((os.path.join(d, f), idx))
if MAX:
    samples = samples[:MAX]
N = len(samples)
print(f"ImageNet-V2 評価画像数: {N} (全{len(set(i for _,i in samples))}クラス)\n")


def load_rgb(path):
    # torchvision の公式前処理に合わせる: 短辺256にリサイズ -> 中央 S×S クロップ。
    # (全体を S×S に潰すとアスペクト比が崩れ精度が下がるため)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - S) // 2, (h - S) // 2
    return img.crop((left, top, left + S, top + S))


def eval_pytorch():
    model = torchvision.models.mobilenet_v3_small(weights=weights).eval()
    prep = weights.transforms()
    t1 = t5 = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        batch, idxs = [], []
        def flush():
            nonlocal t1, t5
            if not batch:
                return
            x = torch.stack(batch)
            logits = model(x)
            top5 = torch.topk(logits, 5, dim=1).indices.tolist()
            for tr, t5row in zip(idxs, top5):
                t1 += int(t5row[0] == tr)
                t5 += int(tr in t5row)
            batch.clear(); idxs.clear()
        for i, (path, tr) in enumerate(samples):
            img = Image.open(path).convert("RGB")
            batch.append(prep(img)); idxs.append(tr)
            if len(batch) == 64:
                flush()
            if (i + 1) % 2000 == 0:
                print(f"  PyTorch {i+1}/{N} ({time.perf_counter()-t0:.0f}s)")
        flush()
    return t1 / N * 100, t5 / N * 100


def eval_coreml(path, tag):
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
    t1 = t5 = 0
    t0 = time.perf_counter()
    for i, (p, tr) in enumerate(samples):
        true_name = CATS[tr]
        out = m.predict({INP: load_rgb(p)})
        pred = out.get("classLabel")
        d = [v for v in out.values() if isinstance(v, dict)][0]
        top5 = [k for k, _ in sorted(d.items(), key=lambda x: -x[1])[:5]]
        t1 += int(pred == true_name)
        t5 += int(true_name in top5)
        if (i + 1) % 2000 == 0:
            print(f"  {tag} {i+1}/{N} ({time.perf_counter()-t0:.0f}s)")
    return t1 / N * 100, t5 / N * 100


arts = MAN["artifacts"]
res = {}
print("計測: PyTorch FP32 ..."); res["PyTorch FP32"] = eval_pytorch()
print("計測: Core ML FP16 ..."); res["Core ML FP16"] = eval_coreml(arts["coreml_fp16"]["path"], "FP16")
print("計測: Core ML INT8 ..."); res["Core ML INT8"] = eval_coreml(arts["coreml_int8"]["path"], "INT8")
print("計測: Core ML INT4 ..."); res["Core ML INT4"] = eval_coreml(arts["coreml_int4"]["path"], "INT4")

print("\n=== ImageNet-V2 (matched-frequency, 全1000クラス) 絶対精度 ===")
print(f"{'モデル':<16}{'top-1':>9}{'top-5':>9}")
base = res["PyTorch FP32"][0]
for k, (a, b) in res.items():
    diff = "" if k == "PyTorch FP32" else f"  ({a-base:+.1f}pt)"
    print(f"{k:<16}{a:>8.2f}%{b:>8.2f}%{diff}")

json.dump({k: {"top1": a, "top5": b} for k, (a, b) in res.items()},
          open(os.path.join(HERE, "artifacts", "imagenetv2_accuracy.json"), "w"), indent=2)
print("\n-> artifacts/imagenetv2_accuracy.json に保存")
