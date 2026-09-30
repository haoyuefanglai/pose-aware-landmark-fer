# -*- coding: utf-8 -*-
"""头部姿态 / 光照 / 距离 / 旋转 稳定性测试（实验五 任务C）。

指导书原文要求："额外测试正面、侧转/俯仰、不同距离或光照条件下的识别稳定性"、
"记录不同头部姿态下的性能变化，这也是使用高姿态人脸关键点模型的主要价值之一"。

本脚本在 FER2013 官方留出测试集（publicTest + privateTest）上做四类测量：

1. 姿态分桶      —— 利用数据中天然存在的头部姿态变化，按 |yaw| / |pitch| / |roll|
                   分桶统计准确率（真实分布，不可控但无合成痕迹）。
2. 光照扰动      —— 受控改变亮度 / 对比度，测准确率随光照的退化。
3. 距离模拟      —— 把图像缩小到 s 倍再放大回原尺寸，模拟更远的拍摄距离
                   （等效于降低人脸的有效分辨率）。
4. 合成旋转      —— 绕图像中心旋转 ±10/20/30/40 度，模拟侧倾（roll）。

用 Deployment 说明：
  python src/experiments/pose_stability_test.py --quick     # 扰动部分每类抽 80 张，约 4 分钟
  python src/experiments/pose_stability_test.py             # 扰动部分每类抽 150 张
  python src/experiments/pose_stability_test.py --full      # 扰动部分用全量留出集（约 40 分钟）

输出：
  data/pose_stability_results.json   全部数字（可直接写入实验报告）
  docs/pose_stability_curves.png     四联图（姿态 / 光照 / 距离 / 旋转）
"""

# --- 路径引导：src/ 下 core / pipeline / experiments / apps 之间可互相 import ---
# 本段由结构重构引入。算法逻辑不依赖它，仅用于把同级子目录加入模块搜索路径，
# 使 `from feature_extractor import ...` 这类平铺导入在跨目录后依然有效。
import os as _os
import sys as _sys

_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "pipeline", "experiments", "apps"):
    _sub_path = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_sub_path) and _sub_path not in _sys.path:
        _sys.path.insert(0, _sub_path)
# --- 路径引导结束 ---

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import cv2

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from feature_extractor import FaceFeatureExtractor          # noqa: E402
from classifier import ExpressionClassifier                 # noqa: E402

SEED = 42
CLASSES = ["neutral", "smile", "surprise", "frown", "sad"]


# ---------------------------------------------------------------- 数据加载
def load_holdout_images():
    """读取 FER2013 留出测试集 (publicTest + privateTest) 的原始图像与标签。"""
    raw_dir = os.path.join(ROOT_DIR, "data", "fer2013_raw")
    parquets = []
    for dp, _dn, fn in os.walk(raw_dir):
        for f in fn:
            if f.endswith(".parquet") and ("publicTest" in f or "privateTest" in f):
                parquets.append(os.path.join(dp, f))
    if not parquets:
        raise FileNotFoundError(
            "未找到 FER2013 留出集 parquet，请先运行 python src/pipeline/extract_fer_landmarks.py")
    frames = [pd.read_parquet(p) for p in sorted(parquets)]
    df = pd.concat(frames, ignore_index=True)
    # FER2013 原始数字编码: 0=angry 1=disgust 2=fear 3=happy 4=sad 5=surprise 6=neutral
    # 目标 5 类索引: 0=neutral 1=smile 2=surprise 3=frown 4=sad；disgust/fear 丢弃
    num_map = {0: 3, 3: 1, 4: 4, 5: 2, 6: 0}
    str_map = {"neutral": 0, "happy": 1, "smile": 1, "surprise": 2,
               "angry": 3, "frown": 3, "sad": 4}
    imgs, labels = [], []
    for _, r in df.iterrows():
        lab = r.get("label")
        if isinstance(lab, (int, np.integer)):
            tgt = num_map.get(int(lab), -1)
        else:
            tgt = str_map.get(str(lab).lower(), -1)
        if tgt < 0:
            continue
        img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        imgs.append(img)
        labels.append(tgt)
    return imgs, np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------- 扰动算子
def perturb_brightness(img, beta):
    return np.clip(img.astype(np.int16) + beta, 0, 255).astype(np.uint8)


def perturb_contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def perturb_distance(img, scale):
    """缩小到 scale 倍再放大回原尺寸，模拟更远的拍摄距离。"""
    if scale >= 0.999:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(4, int(round(w * scale))), max(4, int(round(h * scale)))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def perturb_rotation(img, angle):
    if abs(angle) < 1e-6:
        return img
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- 评测
class Evaluator:
    def __init__(self):
        self.ext = FaceFeatureExtractor(static_mode=True)
        self.clf = ExpressionClassifier(os.path.join(ROOT_DIR, "models", "model_fer_svm.pkl"))

    def close(self):
        self.ext.close()

    def predict_one(self, img):
        """返回 (pred_idx or None, yaw, pitch, roll)。检出失败返回 (None, nan, nan, nan)。"""
        res = self.ext.process_frame(img)
        if not res.get("detected"):
            return None, np.nan, np.nan, np.nan
        hp = res["head_pose"]
        label, _conf, _probs = self.clf.predict(
            res["norm_vector"], geo_metrics=res["geo_metrics"],
            blendshapes=res["blendshapes"], use_ml_model=True)
        try:
            pred = CLASSES.index(label)
        except ValueError:
            pred = None
        return pred, hp["yaw"], hp["pitch"], hp["roll"]

    def batch(self, imgs, labels, desc=""):
        """批量预测，返回 (preds, yaws, pitches, rolls)。preds 元素为 -1 表示未检出。"""
        n = len(imgs)
        preds = np.full(n, -1, dtype=np.int32)
        yaws = np.full(n, np.nan)
        pitches = np.full(n, np.nan)
        rolls = np.full(n, np.nan)
        t0 = time.time()
        for i, img in enumerate(imgs):
            p, y, pt, r = self.predict_one(img)
            preds[i] = -1 if p is None else p
            yaws[i], pitches[i], rolls[i] = y, pt, r
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print("    %s %d/%d  (%.0f 张/秒)" % (desc, i + 1, n, rate), flush=True)
        return preds, yaws, pitches, rolls


def bin_stats(mask, preds, labels):
    """对一个布尔掩码内的样本统计准确率与 Macro-F1（未检出一律算错）。"""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "detect_rate": None}
    p, y = preds[mask], labels[mask]
    acc = float(np.mean(p == y))
    f1s = []
    for c in range(len(CLASSES)):
        m_c = y == c
        if m_c.sum() == 0:
            continue
        tp = np.sum((p == c) & m_c)
        fp = np.sum((p == c) & ~m_c)
        fn = np.sum((p != c) & m_c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    det = float(np.mean(p >= 0))
    return {"n": n, "accuracy": round(acc, 4), "macro_f1": round(float(np.mean(f1s)), 4),
            "detect_rate": round(det, 4)}


def abs_bins(values, edges, names):
    """按 |value| 落入的区间返回桶名。"""
    a = np.abs(values)
    out = np.full(len(values), "n/a", dtype=object)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        out[(a > lo) & (a <= hi)] = names[i]
    out[a <= edges[0]] = names[0]
    out[np.isnan(a)] = "n/a"
    return out


def run_pose_bins(ev, imgs, labels, preds, yaws, pitches, rolls):
    print("  [1/4] 头部姿态分桶 ...", flush=True)
    res = {}
    # 尾桶用 inf 开区间：修复前最后一桶有上限（如 25<|yaw|<=45），
    # 导致 |yaw|>45 的极端样本（全量约 10%）不落入任何桶 —— 而极端姿态恰是最关键的测试点。
    specs = {
        "yaw":   ([0, 5, 15, 25, np.inf], ["|yaw|<=5", "5<|yaw|<=15", "15<|yaw|<=25", "|yaw|>25"]),
        "pitch": ([0, 10, 20, 35, np.inf], ["|pitch|<=10", "10<|pitch|<=20", "20<|pitch|<=35", "|pitch|>35"]),
        "roll":  ([0, 10, 20, 35, np.inf], ["|roll|<=10", "10<|roll|<=20", "20<|roll|<=35", "|roll|>35"]),
    }
    for key, (vals,) in (("yaw", (yaws,)), ("pitch", (pitches,)), ("roll", (rolls,))):
        edges, names = specs[key]
        bins = abs_bins(vals, edges, names)
        rows = []
        for name in names:
            st = bin_stats(bins == name, preds, labels)
            st["bin"] = name
            rows.append(st)
        res[key] = rows
        for r in rows:
            print("    %-16s n=%-5d acc=%s" % (r["bin"], r["n"],
                  ("%.4f" % r["accuracy"]) if r["accuracy"] is not None else "-"), flush=True)
    return res


def run_series(ev, imgs, labels, fn, values, tag_names, desc):
    """对一组扰动档位逐一评测。返回 [{tag, n, detect_rate, accuracy, macro_f1}]。"""
    rows = []
    for v, tag in zip(values, tag_names):
        pert = [fn(im, v) for im in imgs]
        preds, _, _, _ = ev.batch(pert, labels, desc="%s %s" % (desc, tag))
        st = bin_stats(np.ones(len(imgs), dtype=bool), preds, labels)
        st["tag"] = tag
        rows.append(st)
        print("    %-10s n=%-5d det=%.3f acc=%s" % (
            tag, st["n"], st["detect_rate"],
            ("%.4f" % st["accuracy"]) if st["accuracy"] is not None else "-"), flush=True)
    return rows


def make_curves(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.suptitle("FER2013 留出测试集 · 姿态 / 光照 / 距离 / 旋转 稳定性（SVM 部署模型）",
                 fontsize=13)

    # 1) yaw / pitch / roll 分桶
    ax = axes[0][0]
    for key, marker in (("yaw", "o-"), ("pitch", "s--"), ("roll", "^:")):
        rows = [r for r in results["pose_bins"][key] if r["accuracy"] is not None]
        ax.plot(range(len(rows)), [r["accuracy"] * 100 for r in rows],
                marker, label=key, markersize=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["正面", "轻度", "中度", "大角度"], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("头部姿态分桶（真实分布）", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    # 2) 光照
    ax = axes[0][1]
    br = results["illumination"]["brightness"]
    ct = results["illumination"]["contrast"]
    ax.plot([r["tag"] for r in br], [r["accuracy"] * 100 for r in br], "o-",
            label="亮度 beta", markersize=5)
    ax.plot([r["tag"] for r in ct], [r["accuracy"] * 100 for r in ct], "s--",
            label="对比度 alpha", markersize=5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("光照扰动（受控）", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", labelsize=9, rotation=30)

    # 3) 距离
    ax = axes[1][0]
    ds = results["distance"]
    ax.plot([float(r["tag"]) for r in ds], [r["accuracy"] * 100 for r in ds], "o-",
            markersize=5)
    ax.set_xlabel("图像缩放比例 scale（模拟距离）")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("距离模拟（缩小后放大回原尺寸）", fontsize=11)
    ax.grid(alpha=0.25)

    # 4) 旋转
    ax = axes[1][1]
    ro = results["rotation"]
    ax.plot([float(r["tag"]) for r in ro], [r["accuracy"] * 100 for r in ro], "o-",
            markersize=5)
    ax.set_xlabel("合成旋转角（度，逆时针为正）")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("合成 roll 旋转（受控）", fontsize=11)
    ax.grid(alpha=0.25)

    for a in axes.flat:
        a.set_ylim(0, 100)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("  图表已保存:", out_path, flush=True)


def main():
    ap = argparse.ArgumentParser(description="头部姿态/光照/距离/旋转 稳定性测试")
    ap.add_argument("--quick", action="store_true", help="扰动实验每类抽 80 张")
    ap.add_argument("--full", action="store_true", help="扰动实验用全量留出集")
    ap.add_argument("--per-class", type=int, default=150, help="扰动实验每类抽样数（默认 150）")
    ap.add_argument("--limit", type=int, default=0, help="调试：限制留出集总数")
    ap.add_argument("--bins-only", action="store_true",
                    help="只重算头部姿态分桶（复用已有的光照/距离/旋转结果）")
    ap.add_argument("--no-charts", action="store_true", help="不生成图表")
    args = ap.parse_args()

    t0 = time.time()
    print("加载 FER2013 留出测试集 ...", flush=True)
    imgs, labels = load_holdout_images()
    if args.limit:
        imgs, labels = imgs[:args.limit], labels[:args.limit]
    print("  共 %d 张，类别分布 %s" % (len(imgs), np.bincount(labels, minlength=5).tolist()))

    ev = Evaluator()
    print("基线预测（无扰动）...", flush=True)
    preds, yaws, pitches, rolls = ev.batch(imgs, labels, desc="基线")
    base = bin_stats(np.ones(len(imgs), dtype=bool), preds, labels)
    print("  基线: n=%d acc=%.4f det=%.4f" % (base["n"], base["accuracy"], base["detect_rate"]))

    if args.bins_only:
        # 复用已有 JSON 中的扰动结果，只重算姿态分桶并重新出图
        jpath = os.path.join(ROOT_DIR, "data", "pose_stability_results.json")
        with open(jpath, encoding="utf-8") as f:
            results = json.load(f)
        results["pose_bins"] = run_pose_bins(ev, imgs, labels, preds, yaws, pitches, rolls)
        results["meta"].update(holdout_n=int(base["n"]),
                               baseline_accuracy=base["accuracy"],
                               baseline_macro_f1=base["macro_f1"],
                               date=time.strftime("%Y-%m-%d %H:%M"))
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        ev.close()
        print("分桶已更新:", jpath)
        if not args.no_charts:
            make_curves(results, os.path.join(ROOT_DIR, "docs", "pose_stability_curves.png"))
        return

    results = {"meta": {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "model": "models/model_fer_svm.pkl",
        "holdout_n": int(base["n"]),
        "baseline_accuracy": base["accuracy"],
        "baseline_macro_f1": base["macro_f1"],
        "seed": SEED,
    }, "pose_bins": {}, "illumination": {}, "distance": [], "rotation": []}

    results["pose_bins"] = run_pose_bins(ev, imgs, labels, preds, yaws, pitches, rolls)

    # 扰动实验的样本子集（保持类别均衡，固定种子可复现）
    if args.full:
        sub_idx = np.arange(len(imgs))
    else:
        k = 80 if args.quick else args.per_class
        rng = np.random.default_rng(SEED)
        picked = []
        for c in range(len(CLASSES)):
            idx_c = np.where(labels == c)[0]
            picked.append(rng.choice(idx_c, size=min(k, len(idx_c)), replace=False))
        sub_idx = np.concatenate(picked)
    sub_imgs = [imgs[i] for i in sub_idx]
    sub_labels = labels[sub_idx]
    print("扰动实验样本: %d 张" % len(sub_imgs))

    print("  [2/4] 光照扰动 ...", flush=True)
    results["illumination"]["brightness"] = run_series(
        ev, sub_imgs, sub_labels, perturb_brightness,
        [-90, -60, -30, 0, 30, 60, 90],
        ["-90", "-60", "-30", "0", "+30", "+60", "+90"], "亮度")
    results["illumination"]["contrast"] = run_series(
        ev, sub_imgs, sub_labels, perturb_contrast,
        [0.4, 0.6, 0.8, 1.0, 1.3, 1.6],
        ["0.4", "0.6", "0.8", "1.0", "1.3", "1.6"], "对比度")

    print("  [3/4] 距离模拟 ...", flush=True)
    results["distance"] = run_series(
        ev, sub_imgs, sub_labels, perturb_distance,
        [1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125],
        ["1.0", "0.75", "0.5", "0.375", "0.25", "0.1875", "0.125"], "距离")

    print("  [4/4] 合成旋转 ...", flush=True)
    results["rotation"] = run_series(
        ev, sub_imgs, sub_labels, perturb_rotation,
        [-40, -30, -20, -10, 0, 10, 20, 30, 40],
        ["-40", "-30", "-20", "-10", "0", "10", "20", "30", "40"], "旋转")

    ev.close()

    out_json = os.path.join(ROOT_DIR, "data", "pose_stability_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("结果已保存:", out_json)

    if not args.no_charts:
        out_png = os.path.join(ROOT_DIR, "docs", "pose_stability_curves.png")
        make_curves(results, out_png)

    print("总耗时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
