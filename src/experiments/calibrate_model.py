# -*- coding: utf-8 -*-
"""概率校准与低置信拒识（修复"置信度是未校准相对分数"的问题）。

问题背景：
    部署的 SVM 用 `probability=False` 训练，推理时用 `decision_function` 的输出
    手工做 softmax（classifier.py 原实现）。这个数值在类别之间是可比的相对分数，
    **没有经过任何概率校准**，因此不能解释成"有多大把握"，也不能直接拿来做阈值拒识。
    项目多处文档也承认了这一点。

本脚本做什么：
    1. 在**独立校准集**上拟合多种校准口径：
         uncalibrated —— 原样保留 decision_function + softmax（作为对照基线）
         temperature  —— 温度缩放，只调一个参数，不改排序
         sigmoid      —— Platt 标定（每类一个 sigmoid + 归一化）
         isotonic     —— 等渗回归（分片常数、单调）
    2. 在**独立选阈值集**上用指定的评分规则（默认 NLL）比较这些口径并选出胜者；
       同时把 ECE / Brier / NLL / 准确率四个指标全部打印出来 —— 包括对自己不利的数字。
    3. 在**全程未参与任何选择**的 privateTest 上报告胜者的校准质量，以及
       拒识后的"覆盖率 - 准确率"权衡表。
    4. 保存胜者模型（带校准标记与拒识阈值），供 classifier.py / realtime_demo.py 使用。

协议（延续 strict_eval.py 的两阶段严格评测，不污染最终测试集）：
    train          -> 拟合基础 SVM
    publicTest 前半 -> 拟合各校准器
    publicTest 后半 -> 比较并选定校准口径 + 选拒识阈值
    privateTest    -> 只用于最终报告

关于指标不一致的说明（实测结论，报告里应如实保留）：
    本数据集上未校准口径的 ECE 反而低于 Platt sigmoid，而等渗回归的 NLL / Brier
    最优、准确率也略高。ECE 只看"最大概率"的分箱偏差、是诊断量而非严格评分规则，
    与 NLL / Brier 这类 proper scoring rule 可能给出不同排序。因此默认用 NLL 选型，
    并把四个指标全部输出，避免"只挑好看的指标"。

输出：
    models/model_fer_svm_calibrated.pkl
    data/fer2013_calibration.json

用法：
  python src/experiments/calibrate_model.py
  python src/experiments/calibrate_model.py --metric brier
  python src/experiments/calibrate_model.py --candidates uncalibrated,temperature
  python src/experiments/calibrate_model.py --target-accuracy 0.80     # 按目标准确率选拒识阈值
"""

# --- 路径引导：src/ 下 core / pipeline / experiments / apps 之间可互相 import ---
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
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from train_fer_compare import load_features, build_xy, make_clf, CLASSES   # noqa: E402
from classifier import TemperatureScaledEstimator                        # noqa: E402

ALL_CANDIDATES = ("uncalibrated", "temperature", "sigmoid", "isotonic")


# ------------------------------------------------------------------ 概率与指标
def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = np.column_stack([-z, z])
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ece_score(probs, y_true, classes, n_bins=10):
    """期望校准误差 (ECE)：把最大概率分箱，比较平均置信度与实际准确率之差。
    越低越好，0 表示完全校准。注意它是诊断量，不是 proper scoring rule。"""
    conf = probs.max(axis=1)
    pred = np.asarray(classes)[probs.argmax(axis=1)]
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            bins.append(dict(lo=float(lo), hi=float(hi), n=0, conf=None, acc=None))
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        ece += m.mean() * abs(c - a)
        bins.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()),
                         conf=round(c, 4), acc=round(a, 4), gap=round(c - a, 4)))
    return float(ece), bins


def brier_score(probs, y_true, classes):
    """多类 Brier 分数（proper scoring rule）：one-hot 与预测概率的均方误差，越低越好。"""
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros_like(probs)
    for r, lab in enumerate(y_true):
        if lab in idx:
            Y[r, idx[lab]] = 1.0
    return float(np.mean(np.sum((probs - Y) ** 2, axis=1)))


def nll_score(probs, y_true, classes):
    """负对数似然（proper scoring rule）：越低越好。"""
    idx = {c: i for i, c in enumerate(classes)}
    p = np.array([probs[r, idx.get(lab, 0)] for r, lab in enumerate(y_true)])
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))


def evaluate_probs(probs, y_true, classes):
    """一次算出全部四个指标，避免只挑对自己有利的那个。"""
    ece, _ = ece_score(probs, y_true, classes)
    return dict(
        accuracy=round(float(accuracy_score(y_true, np.asarray(classes)[probs.argmax(1)])), 4),
        ece=round(ece, 4),
        brier=round(brier_score(probs, y_true, classes), 4),
        nll=round(nll_score(probs, y_true, classes), 4),
        mean_confidence=round(float(probs.max(axis=1).mean()), 4),
    )


# ------------------------------------------------------------------ 校准器拟合
def fit_temperature(base, X_cal, y_cal, classes):
    """在标定集上按 NLL 一维搜索最优温度。"""
    dec = base.decision_function(X_cal)
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(0.20, 6.001, 0.05):
        nll = nll_score(softmax(np.asarray(dec, dtype=np.float64) / t), y_cal, classes)
        if nll < best_nll:
            best_t, best_nll = float(round(t, 2)), nll
    return best_t, float(best_nll)


def make_ccv(base, method):
    """构造 CalibratedClassifierCV，兼容新旧 scikit-learn 的 prefit 写法。"""
    try:
        from sklearn.frozen import FrozenEstimator       # scikit-learn >= 1.6
        return CalibratedClassifierCV(FrozenEstimator(base), method=method)
    except ImportError:                                  # 旧版本
        return CalibratedClassifierCV(base, method=method, cv="prefit")


def fit_variants(base, X_cal, y_cal, classes, candidates):
    """在标定集上拟合所有候选口径。返回 {name: dict(fn, model, meta)}。"""
    variants = {}
    if "uncalibrated" in candidates:
        variants["uncalibrated"] = dict(
            fn=lambda Xq: softmax(base.decision_function(Xq)),
            model=None, meta=dict(note="decision_function + 手写 softmax，未做任何校准"))
    if "temperature" in candidates:
        t, nll = fit_temperature(base, X_cal, y_cal, classes)
        est = TemperatureScaledEstimator(base, t)
        variants["temperature"] = dict(fn=est.predict_proba, model=est,
                                       meta=dict(temperature=t, calib_nll=round(nll, 4)))
    for method in ("sigmoid", "isotonic"):
        if method in candidates:
            ccv = make_ccv(base, method)
            ccv.fit(X_cal, y_cal)
            variants[method] = dict(fn=ccv.predict_proba, model=ccv,
                                    meta=dict(method=method))
    return variants


# ------------------------------------------------------------------ 拒识
def sweep_thresholds(probs, y_true, thresholds, classes):
    """给定各阈值，返回覆盖率（被接受比例）与被接受样本的准确率、Macro-F1。

    被拒识的样本在部署时会标成 unknown，因此不计入准确率分母 —— 这正是拒识的价值：
    用覆盖率换准确率。

    classes 必须与 probs 的列顺序一致（即模型自身的 classes_），
    不能想当然用项目里那份固定的 CLASSES 常量，否则整列会错位。
    """
    conf = probs.max(axis=1)
    pred = np.asarray(classes)[probs.argmax(axis=1)]
    seen = [c for c in classes if c in set(y_true)]
    rows = []
    for th in thresholds:
        m = conf >= th
        if m.sum() == 0:
            rows.append(dict(threshold=round(float(th), 3), coverage=0.0,
                             accuracy=None, macro_f1=None, n_accepted=0))
            continue
        rows.append(dict(
            threshold=round(float(th), 3),
            coverage=round(float(m.mean()), 4),
            accuracy=round(float(accuracy_score(y_true[m], pred[m])), 4),
            macro_f1=round(float(f1_score(y_true[m], pred[m], average="macro",
                                         labels=seen, zero_division=0)), 4),
            n_accepted=int(m.sum()),
        ))
    return rows


def pick_threshold(rows, target_accuracy=None, min_coverage=0.8):
    """选拒识阈值。

    默认口径：在覆盖率不低于 min_coverage 的阈值里取准确率最高者
    （覆盖率与准确率量纲不同，直接相加会被覆盖率支配，因此用约束而不是加权和）。
    给了 target_accuracy 时：在达标阈值里取覆盖率最高者（最不浪费样本）。
    """
    usable = [r for r in rows if r["accuracy"] is not None]
    if not usable:
        return None
    if target_accuracy is not None:
        ok = [r for r in usable if r["accuracy"] >= target_accuracy]
        return max(ok, key=lambda r: r["coverage"]) if ok else max(usable, key=lambda r: r["accuracy"])
    ok = [r for r in usable if r["coverage"] >= min_coverage]
    pool = ok or usable
    return max(pool, key=lambda r: (r["accuracy"], r["coverage"]))


def print_sweep(rows, title):
    print(f"    {title}")
    print("    阈值   覆盖率   准确率   Macro-F1")
    for r in rows:
        acc = "-" if r["accuracy"] is None else f"{r['accuracy']*100:6.2f}%"
        f1 = "-" if r["macro_f1"] is None else f"{r['macro_f1']:.4f}"
        print(f"    {r['threshold']:.2f}   {r['coverage']*100:6.1f}%   {acc}   {f1}")


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="概率校准 + 低置信拒识阈值标定")
    ap.add_argument("--csv", default=os.path.join(ROOT_DIR, "data", "fer2013_landmarks.csv"))
    ap.add_argument("--out-model", default=os.path.join(ROOT_DIR, "models", "model_fer_svm_calibrated.pkl"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "data"))
    ap.add_argument("--feature-set", default="coords_bl",
                    help="特征集（与部署模型一致，默认 coords_bl）")
    ap.add_argument("--candidates", default=",".join(ALL_CANDIDATES),
                    help="参与比较的校准口径，逗号分隔: " + "/".join(ALL_CANDIDATES))
    ap.add_argument("--metric", choices=["nll", "brier", "ece"], default="nll",
                    help="选型用的评分规则（默认 nll，proper scoring rule）")
    ap.add_argument("--target-accuracy", type=float, default=None,
                    help="按目标准确率选拒识阈值（例如 0.80）")
    ap.add_argument("--min-coverage", type=float, default=0.8,
                    help="默认选阈值口径下的最低覆盖率约束（默认 0.8）")
    ap.add_argument("--no-save", action="store_true", help="只报告不保存模型")
    args = ap.parse_args()

    candidates = tuple(c.strip() for c in args.candidates.split(",") if c.strip())
    bad = [c for c in candidates if c not in ALL_CANDIDATES]
    if bad:
        raise ValueError(f"未知校准口径: {bad}")

    t0 = time.time()
    D = load_features(args.csv)
    split = D["split"]
    tr = split == "train"
    pub = np.where(split == "publicTest")[0]
    fin = split == "privateTest"

    # publicTest 一分为二：前半拟合校准器，后半比较口径 + 选拒识阈值
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pub))
    calib_idx, sel_idx = pub[perm[:len(perm) // 2]], pub[perm[len(perm) // 2:]]

    X = build_xy(D, args.feature_set)
    y = D["y"]
    print(f"[Split] train={tr.sum()}  标定集={len(calib_idx)}  选阈值集={len(sel_idx)}  "
          f"最终测试集 privateTest={fin.sum()}")

    # ---------- 1. 拟合基础 SVM（与部署模型同配置） ----------
    print(f"\n>>> [1/5] 在 train 上拟合基础 SVM（特征集 {args.feature_set}）...")
    t = time.time()
    base = make_clf("svm", scaled=True, balanced=False)
    base.fit(X[tr], y[tr])
    classes = list(base.classes_)
    print(f"    完成，用时 {time.time()-t:.1f}s；类别顺序 = {classes}")

    # ---------- 2. 拟合各校准口径 ----------
    print(f"\n>>> [2/5] 在校准集上拟合候选口径: {candidates}")
    variants = fit_variants(base, X[calib_idx], y[calib_idx], classes, candidates)
    for name, v in variants.items():
        extra = {k: val for k, val in v["meta"].items() if k != "note"}
        print(f"    {name:14s} {extra if extra else ''}")

    # ---------- 3. 在选阈值集上比较并选定口径 ----------
    print(f"\n>>> [3/5] 在选阈值集上比较（选型指标 = {args.metric}）...")
    print("    口径            Accuracy     ECE     Brier     NLL    平均置信度")
    sel_metrics = {}
    for name, v in variants.items():
        m = evaluate_probs(v["fn"](X[sel_idx]), y[sel_idx], classes)
        sel_metrics[name] = m
        print("    %-14s %8.4f  %7.4f  %7.4f  %7.4f  %8.4f" % (
            name, m["accuracy"], m["ece"], m["brier"], m["nll"], m["mean_confidence"]))
    best_name = min(sel_metrics, key=lambda n: sel_metrics[n][args.metric])
    best = variants[best_name]
    print(f"\n    [选定] 校准口径 = {best_name}（选阈值集 {args.metric.upper()} = "
          f"{sel_metrics[best_name][args.metric]:.4f}）")
    # 若换成 ECE 排序会得到另一个胜者，必须显式提示：ECE 是诊断量而非 proper scoring rule，
    # 两种指标在这份数据上排序不一致，报告里应一并给出而不是只挑好看的。
    ece_winner = min(sel_metrics, key=lambda n: sel_metrics[n]["ece"])
    if ece_winner != best_name:
        print(f"    [注意] 若按 ECE 排序，胜者会是 {ece_winner}"
              f"（ECE {sel_metrics[ece_winner]['ece']:.4f} < {sel_metrics[best_name]['ece']:.4f}）；"
              f"ECE 只看最大概率的分箱偏差，是诊断量，与 {args.metric.upper()} 排序可能不一致。")

    th_rows_sel = sweep_thresholds(best["fn"](X[sel_idx]), y[sel_idx],
                                   np.round(np.arange(0.0, 0.96, 0.05), 2), classes)
    print()
    print_sweep(th_rows_sel, "选阈值集上的覆盖率-准确率权衡:")
    chosen = pick_threshold(th_rows_sel, args.target_accuracy, args.min_coverage)
    reject_th = float(chosen["threshold"]) if chosen else 0.0
    if chosen:
        print(f"\n    [选定] 拒识阈值 = {reject_th:.2f}（选阈值集：覆盖率 "
              f"{chosen['coverage']*100:.1f}%，被接受样本准确率 {chosen['accuracy']*100:.2f}%）")
    else:
        print("\n    [选定] 不启用拒识（无可用阈值）")

    # ---------- 4. 在最终测试集上报告 ----------
    print("\n>>> [4/5] 在 privateTest（全程未参与选择）上报告 ...")
    X_fin, y_fin = X[fin], y[fin]
    report = {}
    print("    口径            Accuracy     ECE     Brier     NLL    平均置信度")
    for name, v in variants.items():
        m = evaluate_probs(v["fn"](X_fin), y_fin, classes)
        ece_bins = ece_score(v["fn"](X_fin), y_fin, classes)[1] if name == best_name else None
        m["reliability_bins"] = ece_bins
        report[name] = m
        print("    %-14s %8.4f  %7.4f  %7.4f  %7.4f  %8.4f" % (
            name, m["accuracy"], m["ece"], m["brier"], m["nll"], m["mean_confidence"]))
    bm = report[best_name]
    print(f"\n    [胜者 {best_name}] 相对未校准基线: ECE {report['uncalibrated']['ece']:.4f} → "
          f"{bm['ece']:.4f}，Brier {report['uncalibrated']['brier']:.4f} → {bm['brier']:.4f}，"
          f"NLL {report['uncalibrated']['nll']:.4f} → {bm['nll']:.4f}")

    th_rows_fin = sweep_thresholds(best["fn"](X_fin), y_fin,
                                   np.round(np.arange(0.0, 0.96, 0.05), 2), classes)
    print()
    print_sweep(th_rows_fin, "最终测试集上的覆盖率-准确率权衡:")
    at_chosen = next((r for r in th_rows_fin if abs(r["threshold"] - reject_th) < 1e-9), None)
    if at_chosen:
        print(f"\n    [部署口径] 阈值 {reject_th:.2f}：覆盖率 {at_chosen['coverage']*100:.1f}%，"
              f"被接受样本准确率 {at_chosen['accuracy']*100:.2f}%"
              f"（不拒识时全量为 {bm['accuracy']*100:.2f}%）")

    # ---------- 5. 保存 ----------
    print("\n>>> [5/5] 保存模型与报告 ...")
    final_model = dict(
        model=best["model"],
        classes=classes,
        type="svm",
        feature_set=args.feature_set,
        uses_blendshape="bl" in args.feature_set,
        scaled_pipeline=True,
        feature_schema="fer2013_68xy_geovis_blendshape_v2",
        calibrated=best["model"] is not None,
        calibration_method=best_name,
        calibration_meta=best["meta"],
        reject_threshold=reject_th,
        calibration_protocol=("train 拟合基础模型 / publicTest 前半拟合校准器 / "
                              "后半比较口径并选阈值 / privateTest 仅用于最终报告"),
        selection_metric=args.metric,
        metrics_on_select_half=sel_metrics,
        metrics_on_final_test=report,
        reliability_bins=bm["reliability_bins"],
        final_test_accuracy=bm["accuracy"],
        final_test_coverage_at_threshold=at_chosen["coverage"] if at_chosen else None,
        final_test_accuracy_at_threshold=at_chosen["accuracy"] if at_chosen else None,
        note=("概率口径以最终测试集上的 NLL/ECE/Brier 为准；未校准模型请勿把相对分数当概率使用"),
    )
    if args.no_save or best["model"] is None:
        reason = "--no-save" if args.no_save else f"胜者 {best_name} 无需额外拟合，沿用原模型"
        print(f"    [Skip] 未写出模型文件（{reason}）")
    else:
        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        joblib.dump(final_model, args.out_model)
        print(f"    [Saved] {args.out_model}  ({os.path.getsize(args.out_model)/1e6:.1f}MB)")

    out = dict(
        protocol="train 拟合基础模型 / publicTest 前半校准 / 后半选口径与阈值 / privateTest 最终报告",
        feature_set=args.feature_set,
        class_order=classes,
        candidates=list(candidates),
        selection_metric=args.metric,
        split_sizes=dict(train=int(tr.sum()), calibration=len(calib_idx),
                         threshold_select=len(sel_idx), final_test=int(fin.sum())),
        metrics_on_select_half=sel_metrics,
        metrics_on_final_test={k: {kk: vv for kk, vv in v.items() if kk != "reliability_bins"}
                               for k, v in report.items()},
        reliability_bins=bm["reliability_bins"],
        chosen_calibration=best_name,
        chosen_threshold=reject_th,
        chosen_threshold_reason=dict(target_accuracy=args.target_accuracy,
                                     min_coverage=None if args.target_accuracy else args.min_coverage),
        threshold_sweep_select=th_rows_sel,
        threshold_sweep_final=th_rows_fin,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    jpath = os.path.join(args.out_dir, "fer2013_calibration.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"    [Saved] {jpath}")
    print(f"\n[Done] 总耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
