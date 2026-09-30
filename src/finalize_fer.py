"""
finalize_fer.py
用已验证的最优配置重训并保存部署模型，同时做端到端链路验证。

最优配置来自 train_fer_compare.py 的留出测试集结果（FER2013 官方 publicTest+privateTest）：
  SVM : 坐标136维 + 52维blendshape + 折内 StandardScaler，C=10, gamma=scale  -> Acc 68.96% / Macro-F1 0.6713
  MLP : 坐标136维，(128,64)，max_iter=300                                    -> Acc 66.52% / Macro-F1 0.6355

另外验证：
  1. 关键点可见性列的分布（任务A 要求保存，但需确认是否有信息量）
  2. 端到端：真实图像 -> MediaPipe -> 归一化+blendshape -> 模型推理
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import joblib
import cv2
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix, classification_report)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))            # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                              # 项目根目录
sys.path.insert(0, SRC_DIR)
CSV = os.path.join(ROOT_DIR, "data", "fer2013_landmarks.csv")
CLASSES = ["neutral", "smile", "surprise", "frown", "sad"]

print(">>> [0/4] 读取特征表")
df = pd.read_csv(CSV)
coord_cols = [f"feat_{i}" for i in range(136)]
vis_cols = [f"vis_{i}" for i in range(68)]
bl_cols = sorted([c for c in df.columns if c.startswith("bl_")], key=lambda c: int(c.split("_")[1]))
X_coord = df[coord_cols].to_numpy(np.float32)
X_bl = df[bl_cols].to_numpy(np.float32)
X_cb = np.hstack([X_coord, X_bl])
y = df["label"].to_numpy()
split = df["split"].to_numpy()
tr, te = split == "train", np.isin(split, ["publicTest", "privateTest"])
labels = [c for c in CLASSES if c in set(y)]
print(f"    train={tr.sum()}  holdout={te.sum()}  类别={labels}")
print(f"    坐标={X_coord.shape[1]}维  blendshape={X_bl.shape[1]}维  合并={X_cb.shape[1]}维")

print("\n>>> 可见性列统计（任务A 要求保存可见性信息）")
V = df[vis_cols].to_numpy(np.float32)
print(f"    全局 min={V.min():.3f} max={V.max():.3f} mean={V.mean():.4f}")
print(f"    整列为常数的列数: {int(np.sum(V.std(axis=0) < 1e-9))}/{len(vis_cols)}")
print(f"    不同取值总数: {len(np.unique(np.round(V, 3)))}")

results = {}


def run(name, X, model_type, scaled, extra=None):
    if model_type == "svm":
        est = SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=42)
    else:
        est = MLPClassifier(hidden_layer_sizes=extra or (128, 64), max_iter=300, random_state=42)
    clf = make_pipeline(StandardScaler(), est) if scaled else est
    t0 = time.time()
    clf.fit(X[tr], y[tr])
    fit_s = time.time() - t0
    pred = clf.predict(X[te])
    m = dict(acc=float(accuracy_score(y[te], pred)),
             macro_f1=float(f1_score(y[te], pred, average="macro", labels=labels, zero_division=0)),
             recalls={c: float(r) for c, r in zip(labels, recall_score(y[te], pred, average=None,
                                                                      labels=labels, zero_division=0))},
             cm=confusion_matrix(y[te], pred, labels=labels).tolist(),
             fit_seconds=round(fit_s, 1), n_features=int(X.shape[1]),
             scaled=scaled, model_type=model_type)
    print(f"\n=== {name} ===")
    print(f"  Accuracy={m['acc']*100:.2f}%  Macro-F1={m['macro_f1']:.4f}  拟合={fit_s:.1f}s")
    print("  各类 Recall:", {k: round(v * 100, 1) for k, v in m["recalls"].items()})
    return m, clf, pred


svm_m, svm_clf, svm_pred = run("SVM · 坐标+blendshape+标准化（部署）", X_cb, "svm", True)
mlp_m, mlp_clf, mlp_pred = run("MLP · 坐标136维（部署）", X_coord, "mlp", False)

print("\n>>> 最终 SVM 5 折分层交叉验证（训练集内）")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = []
for k, (a, b) in enumerate(skf.split(X_cb[tr], y[tr]), 1):
    c = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale",
                                            probability=False, random_state=42))
    c.fit(X_cb[tr][a], y[tr][a])
    p = c.predict(X_cb[tr][b])
    cv.append((accuracy_score(y[tr][b], p),
               f1_score(y[tr][b], p, average="macro", labels=labels, zero_division=0)))
    print(f"  Fold {k}: Acc={cv[-1][0]*100:.2f}%  Macro-F1={cv[-1][1]:.4f}")
cv_acc = float(np.mean([c[0] for c in cv])); cv_acc_sd = float(np.std([c[0] for c in cv]))
cv_f1 = float(np.mean([c[1] for c in cv])); cv_f1_sd = float(np.std([c[1] for c in cv]))
print(f"  CV 平均: Acc={cv_acc*100:.2f}%±{cv_acc_sd*100:.2f}  Macro-F1={cv_f1:.4f}±{cv_f1_sd:.4f}")

print("\n>>> [3/4] 保存部署模型")
common = dict(
    feature_schema="fer2013_68xy_vis_blendshape_v1",
    dataset="FER2013 (Aaryan333/fer2013_train_publicTest_privateTest)",
    training_split="fer2013 official train",
    holdout_split="fer2013 publicTest+privateTest",
    training_samples=int(tr.sum()),
    holdout_samples=int(te.sum()),
    cv_accuracy=cv_acc, cv_macro_f1=cv_f1,
    note="FER2013 不提供受试者ID，按官方划分评测；概率为未校准的相对分数",
)
joblib.dump(dict(model=svm_clf, classes=list(svm_clf.classes_), type="svm",
                 feature_set="coords_bl", uses_blendshape=True, scaled_pipeline=True,
                 hyperparams={"C": 10.0, "gamma": "scale"},
                 holdout_accuracy=svm_m["acc"], holdout_macro_f1=svm_m["macro_f1"],
                 holdout_recalls=svm_m["recalls"], **common),
            os.path.join(ROOT_DIR, "models", "model_fer_svm.pkl"))
joblib.dump(dict(model=mlp_clf, classes=[str(c) for c in mlp_clf.classes_], type="mlp",
                 feature_set="coords", uses_blendshape=False, scaled_pipeline=False,
                 hyperparams={"hidden_layer_sizes": [128, 64], "max_iter": 300},
                 holdout_accuracy=mlp_m["acc"], holdout_macro_f1=mlp_m["macro_f1"],
                 holdout_recalls=mlp_m["recalls"], **common),
            os.path.join(ROOT_DIR, "models", "model_fer_mlp.pkl"))
for f in ("model_fer_svm.pkl", "model_fer_mlp.pkl"):
    print(f"  [Saved] {f}  {os.path.getsize(os.path.join(ROOT_DIR, 'models', f))/1e6:.1f}MB")

print("\n>>> [4/4] 端到端链路验证（真实图像 -> MediaPipe -> 分类器）")
from feature_extractor import FaceFeatureExtractor
from classifier import ExpressionClassifier

raw = os.path.join(ROOT_DIR, "data", "fer2013_raw")
pq = [os.path.join(dp, f) for dp, _, fs in os.walk(raw) for f in fs
      if f.endswith(".parquet") and "publicTest" in f]
if not pq:
    raise FileNotFoundError(f"未找到 publicTest parquet，请先运行 extract_fer_landmarks.py（查找目录: {raw}）")
sample = pd.read_parquet(pq[0]).head(200)
ex = FaceFeatureExtractor(static_mode=True)
svm_live = ExpressionClassifier(os.path.join(ROOT_DIR, "models", "model_fer_svm.pkl"))
mlp_live = ExpressionClassifier(os.path.join(ROOT_DIR, "models", "model_fer_mlp.pkl"))

FER_NAMES = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
LMAP = {"angry": "frown", "happy": "smile", "sad": "sad", "surprise": "surprise", "neutral": "neutral"}

ok_n = tot = 0
svm_hit = mlp_hit = 0
conf_sum = 0.0
for _, row in sample.iterrows():
    truth = LMAP.get(FER_NAMES.get(int(row["label"])))
    if truth is None:
        continue
    img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
    res = ex.process_frame(img)
    if not res["detected"]:
        continue
    tot += 1
    p1, c1, _ = svm_live.predict(res["norm_vector"], geo_metrics=res["geo_metrics"],
                                 blendshapes=res["blendshapes"], use_ml_model=True)
    p2, c2, _ = mlp_live.predict(res["norm_vector"], geo_metrics=res["geo_metrics"],
                                 blendshapes=res["blendshapes"], use_ml_model=True)
    svm_hit += (p1 == truth)
    mlp_hit += (p2 == truth)
    conf_sum += c1
ex.close()
print(f"    样本数={tot}  SVM 逐帧命中={svm_hit/tot*100:.1f}%  MLP 逐帧命中={mlp_hit/tot*100:.1f}%")
print(f"    SVM 平均置信度={conf_sum/tot*100:.1f}%（未校准）")

rules = ExpressionClassifier()
p, c, probs = rules.predict(np.zeros(136), {}, {'mouthSmileLeft': .8, 'mouthSmileRight': .8})
print(f"    规则引擎自检: 输入微笑blendshape -> {p} ({c*100:.1f}%), 类别集合={sorted(probs)}")

print("\n>>> 写出结果报告")
out = dict(labels=labels, train_size=int(tr.sum()), holdout_size=int(te.sum()),
           svm=svm_m, mlp=mlp_m,
           cv=dict(acc_mean=cv_acc, acc_std=cv_acc_sd, f1_mean=cv_f1, f1_std=cv_f1_sd,
                   folds=[{"acc": a, "macro_f1": b} for a, b in cv]),
           vis_stats=dict(min=float(V.min()), max=float(V.max()), mean=float(V.mean()),
                          constant_cols=int(np.sum(V.std(axis=0) < 1e-9)), n_dims=len(vis_cols)),
           e2e=dict(n=int(tot), svm_accuracy=float(svm_hit / tot), mlp_accuracy=float(mlp_hit / tot)),
           generated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
with open(os.path.join(ROOT_DIR, "data", "fer2013_final.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("    ", os.path.join(ROOT_DIR, "data", "fer2013_final.json"))
