# -*- coding: utf-8 -*-
"""两阶段严格评测协议：publicTest 选配置，privateTest 出最终数字。

背景：train_fer_compare.py 原协议把 publicTest + privateTest 合并成留出集，
8 组设置都在这个合并集上评测、再从中挑选最优 —— 最优配置的选择"看过"了测试集，
构成选择性偏差（配置是在测试集上调出来的）。

本脚本实现指导书字面要求的训练 / 验证 / 测试三分离：

  阶段1 模型选择：8 组设置在 train 上训练，publicTest（验证集）上评测 → 按 Macro-F1 选最优
  阶段2 最终报告：最优设置在 train 上重训，privateTest（最终测试集）上评测 → 最终数字

复用 train_fer_compare.py 的 SETTINGS / build_xy / make_clf / evaluate，
保证两个协议下的配置与实现完全一致，数字可以直接对比。

输出：
  data/fer2013_strict_results.json
  data/fer2013_strict_report.txt   最终测试集分类报告
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
from sklearn.metrics import classification_report, confusion_matrix

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from train_fer_compare import (load_features, SETTINGS, CLASSES,   # noqa: E402
                               build_xy, make_clf, evaluate)


def main():
    ap = argparse.ArgumentParser(description="两阶段严格评测协议")
    ap.add_argument("--csv", default=os.path.join(ROOT_DIR, "data", "fer2013_landmarks.csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "data"))
    args = ap.parse_args()

    t0 = time.time()
    D = load_features(args.csv)
    tr = D["split"] == "train"
    val = D["split"] == "publicTest"
    fin = D["split"] == "privateTest"
    y_tr, y_val, y_fin = D["y"][tr], D["y"][val], D["y"][fin]
    labels = [c for c in CLASSES if c in set(y_tr)]
    print(f"[Split] train={tr.sum()}  验证集 publicTest={val.sum()}  "
          f"最终测试集 privateTest={fin.sum()}")
    print(f"[Split] 类别: {labels}")

    # ---------- 阶段1：8 组设置在验证集上选型 ----------
    print("\n========== 阶段1 · 8 组设置在 publicTest（验证集）上选型 ==========")
    stage1 = []
    for name, mtype, fset, scaled, balanced in SETTINGS:
        X = build_xy(D, fset)
        clf = make_clf(mtype, scaled, balanced)
        m, _ = evaluate(clf, X[tr], y_tr, X[val], y_val, labels)
        m.update(dict(name=name, model=mtype, feature_set=fset,
                      scaled=scaled, balanced=balanced))
        stage1.append(m)
        print(f"  {m['acc']*100:5.2f}%  Macro-F1={m['macro_f1']:.4f}  {name}", flush=True)
    best = max(stage1, key=lambda r: r["macro_f1"])
    print(f"\n[阶段1] 按 Macro-F1 选出最优配置: {best['name']} "
          f"(验证集 Accuracy={best['acc']*100:.2f}%, Macro-F1={best['macro_f1']:.4f})")

    # ---------- 阶段2：最优配置在最终测试集上报告 ----------
    print("\n========== 阶段2 · 最优配置在 privateTest（最终测试集）上报告 ==========")
    X = build_xy(D, best["feature_set"])
    clf = make_clf(best["model"], best["scaled"], best["balanced"])
    m_fin, pred_fin = evaluate(clf, X[tr], y_tr, X[fin], y_fin, labels)
    report = classification_report(y_fin, pred_fin, labels=labels, digits=4, zero_division=0)
    print(report)
    print(f"[阶段2] 最终测试集: Accuracy={m_fin['acc']*100:.2f}%  "
          f"Macro-F1={m_fin['macro_f1']:.4f}")
    print(f"[对比]  验证集 {best['acc']*100:.2f}% / {best['macro_f1']:.4f}  →  "
          f"最终测试集 {m_fin['acc']*100:.2f}% / {m_fin['macro_f1']:.4f}")

    out = dict(
        protocol=("两阶段严格协议: 阶段1 在 publicTest 上做 8 组设置选型, "
                  "阶段2 将最优设置在 privateTest 上出最终数字; "
                  "配置选择过程未接触 privateTest"),
        labels=labels,
        split_sizes=dict(train=int(tr.sum()), val_publicTest=int(val.sum()),
                         final_privateTest=int(fin.sum())),
        stage1_validation=stage1,
        best=dict(name=best["name"], model=best["model"], feature_set=best["feature_set"],
                  scaled=best["scaled"], balanced=best["balanced"],
                  val_accuracy=best["acc"], val_macro_f1=best["macro_f1"]),
        stage2_final_test=dict(accuracy=m_fin["acc"], macro_f1=m_fin["macro_f1"],
                               recalls=m_fin["recalls"], cm=m_fin["cm"],
                               fit_seconds=m_fin["fit_seconds"]),
        old_protocol_reference=dict(  # 旧协议（合并评测）的最优结果，便于对比
            setting="SVM · 坐标+blendshape +标准化",
            holdout="publicTest+privateTest 合并 (5439)",
            accuracy=0.6896, macro_f1=0.6713),
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    jpath = os.path.join(args.out_dir, "fer2013_strict_results.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    tpath = os.path.join(args.out_dir, "fer2013_strict_report.txt")
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(f"best setting (chosen on publicTest): {best['name']}\n"
                f"final test set: privateTest ({int(fin.sum())} samples)\n\n")
        f.write(report)
        f.write("\nconfusion matrix (rows=true, order=%s):\n" % ",".join(labels))
        f.write(np.array2string(np.array(m_fin["cm"])))
    print(f"\n[Done] {jpath}\n       {tpath}\n总耗时 {time.time()-t0:.0f} 秒")


if __name__ == "__main__":
    main()
