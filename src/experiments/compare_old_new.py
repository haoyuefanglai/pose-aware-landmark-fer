"""
compare_old_new.py
在完全相同的 FER2013 留出图像上，对比旧 CK+ 模型与新 FER2013 模型的实际表现。

重点回答一个具体问题：旧模型没有 neutral 类，遇到平静表情时会发生什么。
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

import numpy as np
import pandas as pd
import cv2
import joblib

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                              # 项目根目录
sys.path.insert(0, SRC_DIR)
from feature_extractor import FaceFeatureExtractor
from classifier import ExpressionClassifier

FER_NAMES = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
             4: "sad", 5: "surprise", 6: "neutral"}
LMAP = {"angry": "frown", "happy": "smile", "sad": "sad",
        "surprise": "surprise", "neutral": "neutral"}

def main():
    print(">>> 旧模型元数据")
    old_meta = joblib.load(os.path.join(ROOT_DIR, "models", "model_svm.pkl"))
    print("    classes =", old_meta.get("classes"))
    print("    feature_set =", old_meta.get("feature_set", "<未声明，按坐标136维处理>"))
    print("    training_samples =", old_meta.get("training_samples"),
          " training_subjects =", old_meta.get("training_subjects"))
    print("    cv_accuracy =", old_meta.get("cv_accuracy"), " cv_macro_f1 =", old_meta.get("cv_macro_f1"))

    print("\n>>> 采样 FER2013 publicTest 图像（每类均匀取样）")
    raw = os.path.join(ROOT_DIR, "data", "fer2013_raw")
    pq = [os.path.join(dp, f) for dp, _, fs in os.walk(raw) for f in fs
          if f.endswith(".parquet") and "publicTest" in f][0]
    df = pd.read_parquet(pq)

    per_class = 80
    picked = []
    for lab in range(7):
        sub = df[df["label"] == lab].head(per_class)
        picked.append(sub)
    sample = pd.concat(picked)
    print(f"    取到 {len(sample)} 张，覆盖 FER2013 全部 7 个原始类别")

    extractor = FaceFeatureExtractor(static_mode=True)
    old = ExpressionClassifier(os.path.join(ROOT_DIR, "models", "model_svm.pkl"))
    new = ExpressionClassifier(os.path.join(ROOT_DIR, "models", "model_fer_svm.pkl"))

    rows = []
    for _, row in sample.iterrows():
        fer = int(row["label"])
        raw_label = FER_NAMES[fer]
        mapped = LMAP.get(raw_label)
        img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        res = extractor.process_frame(img)
        if not res.get("detected"):
            continue
        po, co, _ = old.predict(res["norm_vector"], geo_metrics=res["geo_metrics"],
                                blendshapes=res["blendshapes"], use_ml_model=True)
        pn, cn, _ = new.predict(res["norm_vector"], geo_metrics=res["geo_metrics"],
                                blendshapes=res["blendshapes"], use_ml_model=True)
        rows.append(dict(raw_label=raw_label, mapped=mapped, old=po, new=pn,
                         old_conf=co, new_conf=cn))
    extractor.close()

    R = pd.DataFrame(rows)
    print(f"    成功处理 {len(R)} 张")

    print("\n================ 旧 CK+ 模型 vs 新 FER2013 模型 ================")
    print("\n[1] 在 5 个可比类别上的准确率（disc/fear 类别无对应映射，单列）")
    cmp_rows = []
    for raw_label in ["neutral", "happy", "surprise", "angry", "sad"]:
        sub = R[R["raw_label"] == raw_label]
        if len(sub) == 0:
            continue
        cmp_rows.append(dict(
            真实类别=raw_label, 映射后=LMAP[raw_label], 样本数=len(sub),
            旧模型准确率=round((sub["old"] == sub["mapped"]).mean() * 100, 1),
            新模型准确率=round((sub["new"] == sub["mapped"]).mean() * 100, 1),
        ))
    cmp_df = pd.DataFrame(cmp_rows)
    print(cmp_df.to_string(index=False))

    comp = R[R["raw_label"].isin(LMAP)]
    print(f"\n  整体（{len(comp)} 张，5 个可比类别）: "
          f"旧模型 {(comp['old'] == comp['mapped']).mean()*100:.1f}%  "
          f"新模型 {(comp['new'] == comp['mapped']).mean()*100:.1f}%")

    print("\n[2] 关键现象：真实为 neutral 的样本，两个模型分别输出了什么")
    neut = R[R["raw_label"] == "neutral"]
    print(f"    样本数 {len(neut)}")
    print("    旧模型输出分布:", dict(neut["old"].value_counts()))
    print("    新模型输出分布:", dict(neut["new"].value_counts()))
    print(f"    旧模型输出 neutral 的比例: {(neut['old'] == 'neutral').mean()*100:.1f}%  "
          f"(该模型无 neutral 类，结构上不可能输出)")
    print(f"    新模型输出 neutral 的比例: {(neut['new'] == 'neutral').mean()*100:.1f}%")

    print("\n[3] 两个模型都不支持的 disgust / fear 类别样本")
    other = R[~R["raw_label"].isin(LMAP)]
    if len(other):
        print(f"    样本数 {len(other)}，旧模型输出分布:", dict(other["old"].value_counts()))
        print(f"                     新模型输出分布:", dict(other["new"].value_counts()))

    print("\n[4] 平均置信度（均为未校准的相对分数）")
    print(f"    旧模型 {R['old_conf'].mean()*100:.1f}%   新模型 {R['new_conf'].mean()*100:.1f}%")

    out = dict(n=int(len(R)), per_class=cmp_rows,
               overall=dict(old=float((comp['old'] == comp['mapped']).mean()),
                            new=float((comp['new'] == comp['mapped']).mean()),
                            n=int(len(comp))),
               neutral_behaviour=dict(
                   n=int(len(neut)),
                   old_dist={k: int(v) for k, v in neut["old"].value_counts().items()},
                   new_dist={k: int(v) for k, v in neut["new"].value_counts().items()},
                   old_neutral_rate=float((neut['old'] == 'neutral').mean()),
                   new_neutral_rate=float((neut['new'] == 'neutral').mean())),
               mean_confidence=dict(old=float(R['old_conf'].mean()), new=float(R['new_conf'].mean())))
    with open(os.path.join(ROOT_DIR, "data", "old_vs_new.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[Done]", os.path.join(ROOT_DIR, "data", "old_vs_new.json"))


if __name__ == "__main__":
    main()
