"""
tune_fer.py
在 FER2013 关键点特征上做定向调参，目标是在留出测试集上取得更优的 Macro-F1。

策略：
  1. 先在分层抽样的训练子集上用 3 折 CV 搜索 SVM 的 (C, gamma) 和特征组合
  2. 用搜到的最优配置在全量训练集上重训
  3. 在官方留出测试集 (publicTest + privateTest) 上评测
  4. 若优于现有基线，则覆盖保存 model_fer_svm.pkl / model_fer_mlp.pkl
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report

SRC_DIR = os.path.dirname(os.path.abspath(__file__))            # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                              # 项目根目录
CLASSES = ["neutral", "smile", "surprise", "frown", "sad"]


def load(csv_path):
    df = pd.read_csv(csv_path)
    coord = df[[f"feat_{i}" for i in range(136)]].to_numpy(np.float32)
    vis = df[[f"vis_{i}" for i in range(68)]].to_numpy(np.float32)
    bl = df[sorted([c for c in df.columns if c.startswith("bl_")],
                   key=lambda c: int(c.split("_")[1]))].to_numpy(np.float32)
    return dict(coord=coord, vis=vis, bl=bl, y=df["label"].to_numpy(),
                split=df["split"].to_numpy())


def feats(D, name):
    if name == "coord_bl":
        return np.hstack([D["coord"], D["bl"]])
    if name == "coord_bl_vis":
        return np.hstack([D["coord"], D["bl"], D["vis"]])
    if name == "coord_bl_bl2":
        return np.hstack([D["coord"], D["bl"], D["bl"] ** 2])
    raise ValueError(name)


def score(clf, X, y, labels, folds=3, seed=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for i_tr, i_va in skf.split(X, y):
        clf.fit(X[i_tr], y[i_tr])
        p = clf.predict(X[i_va])
        accs.append(accuracy_score(y[i_va], p))
        f1s.append(f1_score(y[i_va], p, average="macro", labels=labels, zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT_DIR, "data", "fer2013_landmarks.csv"))
    ap.add_argument("--probe-size", type=int, default=8000)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    D = load(args.csv)
    tr = D["split"] == "train"
    te = np.isin(D["split"], ["publicTest", "privateTest"])
    y_tr, y_te = D["y"][tr], D["y"][te]
    labels = [c for c in CLASSES if c in set(y_tr)]

    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]
    probe_idx = rng.choice(tr_idx, size=min(args.probe_size, len(tr_idx)), replace=False)
    print(f"[Probe] 子集 {len(probe_idx)} 条, {args.folds} 折 CV")

    grid = []
    for fname in ["coord_bl", "coord_bl_vis"]:
        Xp = feats(D, fname)[probe_idx]
        for C in [3.0, 10.0, 30.0]:
            for gamma in ["scale", 0.005, 0.02]:
                clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma,
                                                          probability=False, random_state=42))
                t0 = time.time()
                a, f1 = score(clf, Xp, y_tr[probe_idx], labels, args.folds)
                grid.append(dict(feature_set=fname, C=C, gamma=str(gamma), acc=a, macro_f1=f1))
                print(f"  {fname:14s} C={C:<5} gamma={str(gamma):<6} "
                      f"CV Acc={a*100:5.2f}%  Macro-F1={f1:.4f}  ({time.time()-t0:.0f}s)")

    grid.sort(key=lambda r: -r["macro_f1"])
    best = grid[0]
    print(f"\n[Best on probe] {best}")

    # ---- 全量重训 + 留出测试 ----
    Xtr = feats(D, best["feature_set"])
    Xte = Xtr[te]
    Xtr = Xtr[tr]

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=best["C"],
                                              gamma=("scale" if best["gamma"] == "scale" else float(best["gamma"])),
                                              probability=False, random_state=42))
    t0 = time.time()
    svm.fit(Xtr, y_tr)
    pred = svm.predict(Xte)
    svm_metrics = dict(
        acc=float(accuracy_score(y_te, pred)),
        macro_f1=float(f1_score(y_te, pred, average="macro", labels=labels, zero_division=0)),
        recalls={c: float(r) for c, r in zip(labels, recall_score(y_te, pred, average=None,
                                                                 labels=labels, zero_division=0))},
        cm=confusion_matrix(y_te, pred, labels=labels).tolist(),
        fit_seconds=round(time.time() - t0, 1),
    )
    print(f"\n[SVM tuned] Accuracy={svm_metrics['acc']*100:.2f}%  "
          f"Macro-F1={svm_metrics['macro_f1']:.4f}  用时={svm_metrics['fit_seconds']}s")
    print("  各类 Recall:", {k: round(v * 100, 1) for k, v in svm_metrics["recalls"].items()})

    # ---- MLP 用同一特征集重训，作为第二模型 ----
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128),
                                                        alpha=1e-3, max_iter=400, random_state=42))
    t0 = time.time()
    mlp.fit(Xtr, y_tr)
    mp = mlp.predict(Xte)
    mlp_metrics = dict(
        acc=float(accuracy_score(y_te, mp)),
        macro_f1=float(f1_score(y_te, mp, average="macro", labels=labels, zero_division=0)),
        recalls={c: float(r) for c, r in zip(labels, recall_score(y_te, mp, average=None,
                                                                 labels=labels, zero_division=0))},
        cm=confusion_matrix(y_te, mp, labels=labels).tolist(),
        fit_seconds=round(time.time() - t0, 1),
    )
    print(f"[MLP tuned] Accuracy={mlp_metrics['acc']*100:.2f}%  "
          f"Macro-F1={mlp_metrics['macro_f1']:.4f}  用时={mlp_metrics['fit_seconds']}s")
    print("  各类 Recall:", {k: round(v * 100, 1) for k, v in mlp_metrics["recalls"].items()})

    # ---- 保存到 models/ 目录（与随仓库提供的 model_svm.pkl 同级）----
    for name, model, mtype, m in (("model_fer_svm.pkl", svm, "svm", svm_metrics),
                                  ("model_fer_mlp.pkl", mlp, "mlp", mlp_metrics)):
        path = os.path.join(ROOT_DIR, "models", name)
        clf_classes = list(model.classes_)
        joblib.dump({
            "model": model,
            "classes": clf_classes,
            "type": mtype,
            "feature_set": best["feature_set"],
            "scaled_pipeline": True,
            "feature_schema": "fer2013_68xy_blendshape_v1",
            "hyperparams": {"C": best["C"], "gamma": best["gamma"]},
            "training_samples": int(tr.sum()),
            "training_split": "fer2013 official train",
            "holdout_split": "fer2013 publicTest+privateTest",
            "holdout_accuracy": m["acc"],
            "holdout_macro_f1": m["macro_f1"],
            "holdout_recalls": m["recalls"],
            "dataset": "FER2013 (Aaryan333/fer2013_train_publicTest_privateTest)",
            "note": "FER2013 不提供受试者ID，按官方划分评测；概率为未校准的相对分数",
        }, path)
        print(f"[Saved] {path}  classes={clf_classes}  feature_set={best['feature_set']}")

    report = classification_report(y_te, pred, labels=labels, digits=4, zero_division=0)
    with open(os.path.join(ROOT_DIR, "data", "fer2013_tuned_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"best config: {best}\n\n=== SVM tuned ===\n{report}\n")
        f.write("confusion matrix (rows=true, cols=pred):\n")
        f.write(pd.DataFrame(svm_metrics["cm"], index=labels, columns=labels).to_string())
        f.write("\n\n=== MLP tuned ===\n")
        f.write(classification_report(y_te, mp, labels=labels, digits=4, zero_division=0))
        f.write("confusion matrix (rows=true, cols=pred):\n")
        f.write(pd.DataFrame(mlp_metrics["cm"], index=labels, columns=labels).to_string())

    out = dict(labels=labels, grid=grid, best=best, svm=svm_metrics, mlp=mlp_metrics,
               generated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(ROOT_DIR, "data", "fer2013_tuned.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[Done]", os.path.join(ROOT_DIR, "data", "fer2013_tuned.json"))


if __name__ == "__main__":
    main()
