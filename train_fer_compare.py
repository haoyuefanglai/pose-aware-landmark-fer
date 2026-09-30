"""
train_fer_compare.py
在 FER2013 关键点特征上做多设置对照实验（任务B 要求"至少两种模型或关键设置比较"）

评测协议：
  - 训练集：FER2013 官方 train 划分
  - 测试集：FER2013 官方 publicTest + privateTest 划分（完全留出，训练时未见）
  - 另对最优设置做 5 折分层交叉验证，报告稳定性

对照设置覆盖三个维度：
  1. 分类器：SVM(RBF) vs MLP
  2. 特征集：68点坐标(136维) vs 坐标+52维blendshape
  3. 预处理：原始 vs 折内 StandardScaler；以及是否做 roll 旋转对齐

说明：FER2013 不提供受试者 ID，因此无法按其人员做 GroupKFold；
     这是本数据集相对 CK+ 的已知局限，报告中需如实说明。
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["neutral", "smile", "surprise", "frown", "sad"]


def load_features(csv_path):
    df = pd.read_csv(csv_path)
    n = len(df)
    coord_cols = [f"feat_{i}" for i in range(136)]
    vis_cols = [f"vis_{i}" for i in range(68)]
    bl_cols = sorted([c for c in df.columns if c.startswith("bl_")],
                     key=lambda c: int(c.split("_")[1]))
    X_coord = df[coord_cols].to_numpy(dtype=np.float32)
    X_vis = df[vis_cols].to_numpy(dtype=np.float32) if vis_cols[0] in df.columns else None
    X_bl = df[bl_cols].to_numpy(dtype=np.float32) if bl_cols else None
    y = df["label"].to_numpy()
    split = df["split"].to_numpy()
    roll = df["roll"].to_numpy(dtype=np.float32) if "roll" in df.columns else None
    print(f"[Data] {csv_path}: {n} 行, 坐标={X_coord.shape}, "
          f"blendshape={None if X_bl is None else X_bl.shape}")
    return dict(df=df, X_coord=X_coord, X_vis=X_vis, X_bl=X_bl, y=y, split=split, roll=roll)


def roll_align(X_coord, roll_deg):
    """把归一化坐标按 PnP 估计的 roll 角做旋转对齐（绕原点=鼻尖）"""
    pts = X_coord.reshape(-1, 68, 2)
    th = np.deg2rad(-roll_deg).astype(np.float32)[:, None]
    c, s = np.cos(th), np.sin(th)
    x, y = pts[:, :, 0], pts[:, :, 1]
    out = np.stack([x * c - y * s, x * s + y * c], axis=-1)
    return out.reshape(-1, 136).astype(np.float32)


def build_xy(D, feature_set):
    if feature_set == "coords":
        return D["X_coord"]
    if feature_set == "coords_roll":
        return roll_align(D["X_coord"], D["roll"])
    if feature_set == "coords_bl":
        return np.hstack([D["X_coord"], D["X_bl"]])
    if feature_set == "coords_bl_roll":
        return np.hstack([roll_align(D["X_coord"], D["roll"]), D["X_bl"]])
    raise ValueError(feature_set)


def make_clf(model_type, scaled=False, balanced=False):
    if model_type == "svm":
        est = SVC(kernel="rbf", C=10.0, probability=False, random_state=42,
                  class_weight="balanced" if balanced else None)
    elif model_type == "mlp":
        est = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    else:
        raise ValueError(model_type)
    return make_pipeline(StandardScaler(), est) if scaled else est


def evaluate(clf, Xtr, ytr, Xte, yte, labels):
    t0 = time.time()
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    fit_s = time.time() - t0
    return dict(
        acc=float(accuracy_score(yte, pred)),
        macro_f1=float(f1_score(yte, pred, average="macro", labels=labels, zero_division=0)),
        recalls={c: float(r) for c, r in zip(
            labels, recall_score(yte, pred, average=None, labels=labels, zero_division=0))},
        cm=confusion_matrix(yte, pred, labels=labels).tolist(),
        fit_seconds=round(fit_s, 1),
    ), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "data", "fer2013_landmarks.csv"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--max-train", type=int, default=0, help="限制训练样本数（调试用）")
    args = ap.parse_args()

    D = load_features(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)
    tr_mask = D["split"] == "train"
    te_mask = np.isin(D["split"], ["publicTest", "privateTest"])

    if args.max_train:
        idx = np.where(tr_mask)[0][:args.max_train]
        tr_mask = np.zeros_like(tr_mask)
        tr_mask[idx] = True

    y_tr, y_te = D["y"][tr_mask], D["y"][te_mask]
    labels = [c for c in CLASSES if c in set(y_tr)]
    print(f"[Split] train={tr_mask.sum()}  test(留出)={te_mask.sum()}  类别={labels}")
    print("[Split] 训练集分布:", dict(zip(*np.unique(y_tr, return_counts=True))))
    print("[Split] 测试集分布:", dict(zip(*np.unique(y_te, return_counts=True))))

    settings = [
        ("SVM · 坐标136维",                       "svm", "coords",          False, False),
        ("SVM · 坐标+blendshape",                 "svm", "coords_bl",       False, False),
        ("SVM · 坐标+blendshape +标准化",          "svm", "coords_bl",       True,  False),
        ("SVM · 坐标+blendshape +标准化 +类别均衡", "svm", "coords_bl",       True,  True),
        ("SVM · roll对齐坐标+blendshape +标准化",   "svm", "coords_bl_roll",  True,  False),
        ("MLP · 坐标136维",                       "mlp", "coords",          False, False),
        ("MLP · 坐标+blendshape",                 "mlp", "coords_bl",       False, False),
        ("MLP · 坐标+blendshape +标准化",          "mlp", "coords_bl",       True,  False),
    ]

    results = []
    preds_store = {}
    for name, mtype, fset, scaled, balanced in settings:
        X = build_xy(D, fset)
        print(f"\n================ {name} ================")
        print(f"  特征维度={X.shape[1]}")
        clf = make_clf(mtype, scaled, balanced)
        m, pred = evaluate(clf, X[tr_mask], y_tr, X[te_mask], y_te, labels)
        m.update(dict(name=name, model=mtype, feature_set=fset,
                      scaled=scaled, balanced=balanced, n_features=int(X.shape[1])))
        results.append(m)
        preds_store[name] = pred
        print(f"  Accuracy={m['acc']*100:.2f}%  Macro-F1={m['macro_f1']:.4f}  拟合用时={m['fit_seconds']}s")
        print("  各类 Recall:", {k: round(v * 100, 1) for k, v in m["recalls"].items()})
        with open(os.path.join(args.out_dir, "fer2013_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- 最优设置做 5 折分层交叉验证 ----
    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\n================ 最优设置 5 折分层交叉验证: {best['name']} ================")
    X = build_xy(D, best["feature_set"])
    Xtr, ytr = X[tr_mask], y_tr
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc, cv_f1, cv_pred = [], [], np.empty_like(ytr, dtype=object)
    for k, (i_tr, i_va) in enumerate(skf.split(Xtr, ytr), 1):
        clf = make_clf(best["model"], best["scaled"], best["balanced"])
        clf.fit(Xtr[i_tr], ytr[i_tr])
        p = clf.predict(Xtr[i_va])
        cv_pred[i_va] = p
        a = accuracy_score(ytr[i_va], p)
        f1 = f1_score(ytr[i_va], p, average="macro", labels=labels, zero_division=0)
        cv_acc.append(a); cv_f1.append(f1)
        print(f"  Fold {k}: Accuracy={a*100:.2f}%  Macro-F1={f1:.4f}")
    cv_summary = dict(acc_mean=float(np.mean(cv_acc)), acc_std=float(np.std(cv_acc)),
                      f1_mean=float(np.mean(cv_f1)), f1_std=float(np.std(cv_f1)),
                      folds=[{"acc": a, "macro_f1": f} for a, f in zip(cv_acc, cv_f1)],
                      pooled_acc=float(accuracy_score(ytr, cv_pred)),
                      pooled_macro_f1=float(f1_score(ytr, cv_pred, average="macro",
                                                     labels=labels, zero_division=0)))
    print(f"  CV 平均: Accuracy={cv_summary['acc_mean']*100:.2f}%±{cv_summary['acc_std']*100:.2f}  "
          f"Macro-F1={cv_summary['f1_mean']:.4f}±{cv_summary['f1_std']:.4f}")

    # ---- 保存部署模型（附带特征集元数据，供 classifier.py 选择正确的输入向量）----
    # 部署只选不含 roll 对齐的配置：实时演示拿不到历史 roll，保持链路简单
    deploy_pool = [r for r in results if "roll" not in r["feature_set"]]
    deploy_best = max(deploy_pool, key=lambda r: r["macro_f1"])
    print(f"\n[Deploy] 部署配置按 Macro-F1 选为: {deploy_best['name']}")
    deploy_dir = args.out_dir
    for mtype, fname in (("svm", "model_fer_svm.pkl"), ("mlp", "model_fer_mlp.pkl")):
        cfg = next((r for r in deploy_pool if r["model"] == mtype
                    and r["feature_set"] == deploy_best["feature_set"]
                    and r["scaled"] == deploy_best["scaled"]
                    and r["balanced"] == deploy_best["balanced"]), None)
        if cfg is None:
            cfg = next(r for r in deploy_pool if r["model"] == mtype)
        fset, scaled = cfg["feature_set"], cfg["scaled"]
        clf = make_clf(mtype, scaled, balanced=False)
        Xd = build_xy(D, fset)
        clf.fit(Xd[tr_mask], y_tr)
        import joblib
        joblib.dump({
            "model": clf,
            "classes": labels,
            "type": mtype,
            "feature_set": fset,
            "scaled_pipeline": scaled,
            "feature_schema": "fer2013_68xy_blendshape_v1",
            "training_samples": int(tr_mask.sum()),
            "training_split": "fer2013 official train",
            "holdout_split": "fer2013 publicTest+privateTest",
            "holdout_accuracy": cfg["acc"] if cfg else None,
            "holdout_macro_f1": cfg["macro_f1"] if cfg else None,
            "dataset": "FER2013 (Aaryan333/fer2013_train_publicTest_privateTest)",
            "note": "无受试者ID，按官方划分评测；概率为未校准的相对分数",
        }, os.path.join(deploy_dir, fname))
        print(f"[Saved] {os.path.join(deploy_dir, fname)}  feature_set={fset} scaled={scaled}")

    summary = dict(labels=labels, results=results, best=best["name"], cv=cv_summary,
                   train_size=int(tr_mask.sum()), test_size=int(te_mask.sum()),
                   generated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(args.out_dir, "fer2013_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ 汇总（按 Macro-F1 排序）================")
    for r in sorted(results, key=lambda x: -x["macro_f1"]):
        print(f"  {r['macro_f1']:.4f}  {r['acc']*100:5.2f}%  {r['name']}")

    best_pred = preds_store[best["name"]]
    report_txt = classification_report(y_te, best_pred, labels=labels, digits=4, zero_division=0)
    print(f"\n--- 最优设置 {best['name']} 留出测试集分类报告 ---")
    print(report_txt)
    with open(os.path.join(args.out_dir, "fer2013_best_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"best setting: {best['name']}\n\n{report_txt}\nconfusion matrix (rows=true):\n")
        f.write(pd.DataFrame(best["cm"], index=labels, columns=labels).to_string())

    print("\n[Done] 结果 JSON:", os.path.join(args.out_dir, "fer2013_results.json"))


if __name__ == "__main__":
    main()
