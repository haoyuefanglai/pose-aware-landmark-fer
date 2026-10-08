"""
classifier.py
表情分类器模块：支持 SVM 与 MLP 模型训练、评估（按人员划分 GroupKFold）、
混淆矩阵输出，以及开箱即用的预校准几何规则/混合分类器。
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
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# 5 类目标表情定义
EXPRESSION_CLASSES = ["neutral", "smile", "surprise", "frown", "sad"]

# 低置信拒识时的输出标签：表示"模型不确定，拒绝给出表情判定"
UNKNOWN_LABEL = "unknown"

EXPRESSION_NAMES_ZH = {
    "neutral": "自然 (Neutral)",
    "smile": "微笑 (Smile)",
    "surprise": "惊讶 (Surprise)",
    "frown": "皱眉 (Frown/Angry)",
    "sad": "难过 (Sad)",
    "disgust": "厌恶 (Disgust)",
    "unknown": "不确定/拒识 (Unknown)"
}

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                              # 项目根目录

class ExpressionClassifier:
    """表情分类推理器"""

    def __init__(self, model_path=None):
        self.classes = EXPRESSION_CLASSES
        self.model = None
        self.feature_set = "coords"
        self.uses_blendshape = False
        self.bl_names = None
        # 置信度的"性质"：False 表示输出的只是未经校准的相对分数（原 SVC
        # probability=False + decision_function 手写 softmax），不能当概率解释；
        # 由 calibrate_model.py 产出的模型会把 calibrated 置为 True。
        self.prob_is_calibrated = False
        self.reject_threshold = None

        if model_path:
            if os.path.isabs(model_path):
                resolved_path = model_path
            else:
                # 依次在 项目根目录 / models子目录 / src目录 下查找
                # 因此 "model_fer_svm.pkl" 与 "models/model_fer_svm.pkl" 两种写法都能命中
                candidates = [os.path.join(ROOT_DIR, model_path),
                              os.path.join(ROOT_DIR, "models", model_path),
                              os.path.join(SRC_DIR, model_path)]
                resolved_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

            if os.path.exists(resolved_path):
                self.load_model(resolved_path)
                self.model_path = resolved_path
            else:
                print(f"[Info] 路径 {resolved_path} 未找到模型，自动启用内置未经校准的几何规则分类器。")
        else:
            print("[Info] 未指定模型文件，自动启用内置未经校准的几何规则分类器。")

    def load_model(self, path):
        """加载已保存的机器学习模型"""
        data = joblib.load(path)
        self.model = data.get("model")
        self.classes = list(self.model.classes_)
        self.feature_set = data.get("feature_set", "coords")
        # 兼容多种命名：coords_bl / coord_bl / coords_bl_roll / ...blendshape...
        fs = str(self.feature_set).lower()
        self.uses_blendshape = bool(data.get("uses_blendshape",
                                             ("bl" in fs) or ("blendshape" in fs)))
        self.bl_names = None
        if self.uses_blendshape:
            meta = os.path.join(ROOT_DIR, "data", "fer_blendshape_names.json")
            if os.path.exists(meta):
                with open(meta, "r", encoding="utf-8") as f:
                    self.bl_names = json.load(f)
            else:
                raise FileNotFoundError(
                    f"模型声明使用 blendshape 特征，但缺少顺序元数据: {meta}"
                )
        # 校准元数据：calibrate_model.py 产出的模型带 calibrated=True 与拒识阈值
        self.prob_is_calibrated = bool(data.get("calibrated", False))
        raw_th = data.get("reject_threshold")
        self.reject_threshold = float(raw_th) if raw_th is not None else None
        conf_kind = "已校准概率" if self.prob_is_calibrated else "未校准相对分数"
        th_txt = "-" if self.reject_threshold is None else f"{self.reject_threshold:.2f}"
        print(f"[Success] 成功加载已训练模型: {path}，模型类型: {type(self.model).__name__}，"
              f"特征集: {self.feature_set}，置信度性质: {conf_kind}，拒识阈值: {th_txt}")
        return self

    def build_input(self, norm_vector, blendshapes=None):
        """按模型声明的特征集组装输入向量（坐标 / 坐标+blendshape）"""
        vec = list(np.asarray(norm_vector, dtype=np.float32).ravel())
        if self.uses_blendshape:
            bs = blendshapes or {}
            vec.extend(float(bs.get(name, 0.0)) for name in self.bl_names)
        return np.asarray(vec, dtype=np.float32).reshape(1, -1)

    def save_model(self, path, model_type="svm"):
        """保存模型"""
        joblib.dump({"model": self.model, "classes": self.classes, "type": model_type}, path)
        print(f"[Success] 模型已保存至: {path}")

    def predict(self, norm_vector, geo_metrics=None, blendshapes=None, use_ml_model=False,
                reject_threshold=None):
        """
        可切换的模型/规则表情分类器：
        1. 默认优先使用基于人脸动作编码系统 (FACS) 与 MediaPipe 52维表情基 (Blendshapes) 的精细动作解码 (未经真实视频集校准)
        2. 若启用 use_ml_model=True 且已加载机器学习模型，则使用 SVM/MLP
        返回: (pred_label, confidence, prob_dict)

        reject_threshold 低置信拒识：
          给了阈值且最高分低于阈值时，返回标签 UNKNOWN_LABEL ("unknown")，
          同时仍返回原始 confidence 与 prob_dict，调用方可以据此显示"不确定"。
          留空 (None) 时使用模型自带的 reject_threshold；传 0.0 可显式关闭拒识。

        注意 confidence 的性质取决于 prob_is_calibrated：
          - True ：Platt 标定后的概率，可以解释为"有把握的程度"
          - False：SVM decision_function 手写 softmax 的相对分数，**不是概率**，
                   阈值拒识在这种情形下只是启发式，不保证覆盖率-准确率单调
        """
        if use_ml_model and self.model is None:
            raise ValueError("ML engine requested but no model is loaded")
        if use_ml_model:
            x = self.build_input(norm_vector, blendshapes)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(x)[0]
            else:
                decision = self.model.decision_function(x)[0]
                exp_d = np.exp(decision - np.max(decision))
                probs = exp_d / np.sum(exp_d)

            best_idx = np.argmax(probs)
            pred_label = self.classes[best_idx]
            conf = float(probs[best_idx])
            prob_dict = {cls_name: float(p) for cls_name, p in zip(self.classes, probs)}
            return self._apply_rejection(pred_label, conf, prob_dict, reject_threshold)

        # 默认执行人脸生理动作编码 (FACS + Blendshapes + 几何度量)
        label, conf, prob_dict = self._predict_facs(geo_metrics, blendshapes)
        return self._apply_rejection(label, conf, prob_dict, reject_threshold)

    def _apply_rejection(self, label, conf, prob_dict, reject_threshold):
        """统一执行低置信拒识，返回新的 (label, conf, prob_dict)。"""
        threshold = self.reject_threshold if reject_threshold is None else reject_threshold
        if threshold is None:
            return label, conf, prob_dict
        threshold = float(threshold)
        if threshold > 0.0 and conf < threshold:
            return UNKNOWN_LABEL, conf, prob_dict
        return label, conf, prob_dict

    def _predict_facs(self, geo, bs):
        """
        基于人脸动作编码系统 (FACS) 与真实面部表情基 (Blendshapes) 的启发式判别逻辑:
        - 微笑 (Smile): AU12 (嘴角拉引器) + AU6 (眼周抬升)
        - 惊讶 (Surprise): AU26/27 (下颌开合) + AU1/2 (内外眉上挑) + 嘴巴开合比 (MAR)
        - 皱眉 (Frown/Angry): AU4 (双眉聚拢下压) + AU9 (鼻褶收缩)
        - 难过 (Sad): AU15 (嘴角降肌) + AU1 (内眉抬升外眉下垂)
        - 自然 (Neutral): 肌肉放松状态
        """
        if bs is None:
            bs = {}
        if geo is None:
            geo = {}

        # 基础基准分数
        classes = EXPRESSION_CLASSES
        scores = {c: 0.1 for c in classes}
        scores["neutral"] = 1.0  # 默认放松状态

        # 1. 微笑动作提取 (AU12: Lip Corner Puller)
        smile_bs = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
        smile_dimple = (bs.get("mouthDimpleLeft", 0.0) + bs.get("mouthDimpleRight", 0.0)) / 2.0
        smile_curv = geo.get("smile_curvature", 0.0)
        width_ratio = geo.get("mouth_width_ratio", 0.5)

        smile_strength = smile_bs * 1.5 + smile_dimple * 0.5
        if smile_curv > 0.02:
            smile_strength += (smile_curv - 0.01) * 10.0

        if smile_strength > 0.15:
            scores["smile"] += smile_strength * 6.0
            scores["neutral"] -= min(0.8, smile_strength * 2.0)

        # 2. 惊讶动作提取 (AU26/27: Jaw Drop & AU1/2: Brow Raiser)
        jaw_open = bs.get("jawOpen", 0.0)
        brow_inner_up = bs.get("browInnerUp", 0.0)
        mar = geo.get("mar", 0.15)

        surprise_strength = jaw_open * 1.8 + brow_inner_up * 0.5
        if mar > 0.35:
            surprise_strength += (mar - 0.25) * 6.0

        if surprise_strength > 0.20:
            scores["surprise"] += surprise_strength * 5.5
            scores["neutral"] -= min(0.8, surprise_strength * 2.0)

        # 3. 皱眉/愤怒动作提取 (AU4: Brow Lowerer)
        brow_down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
        nose_sneer = (bs.get("noseSneerLeft", 0.0) + bs.get("noseSneerRight", 0.0)) / 2.0
        brow_dist = geo.get("inner_brow_dist", 0.35)

        frown_strength = brow_down * 1.8 + nose_sneer * 0.8
        if brow_dist < 0.32:
            frown_strength += (0.35 - brow_dist) * 8.0

        if frown_strength > 0.18:
            scores["frown"] += frown_strength * 5.5
            scores["neutral"] -= min(0.8, frown_strength * 2.0)

        # 4. 难过动作提取 (AU15: Lip Corner Depressor)
        mouth_frown = (bs.get("mouthFrownLeft", 0.0) + bs.get("mouthFrownRight", 0.0)) / 2.0
        mouth_roll = bs.get("mouthRollLower", 0.0)

        sad_strength = mouth_frown * 1.6 + mouth_roll * 0.6
        if smile_curv < -0.03:
            sad_strength += (-smile_curv) * 6.0

        if sad_strength > 0.18:
            scores["sad"] += sad_strength * 5.0
            scores["neutral"] -= min(0.8, sad_strength * 2.0)

        # Softmax 概率归一化
        scores_arr = np.array([scores[c] for c in classes])
        exp_s = np.exp(scores_arr - np.max(scores_arr))  # 适度放大温差使主类别更突出
        probs = exp_s / np.sum(exp_s)
        prob_dict = {cls_name: float(p) for cls_name, p in zip(classes, probs)}

        best_cls = classes[np.argmax(probs)]
        conf = float(np.max(probs))
        return best_cls, conf, prob_dict


class TemperatureScaledEstimator:
    """温度缩放包装器：把 decision_function 输出除以温度 T 后再 softmax。

    SVC(probability=False) 没有 predict_proba，classifier 里是手写 softmax 得到相对
    分数。温度缩放只调一个参数 T，就能在**不改变分数排序**的前提下改善概率质量，
    是"最小改动"的校准方式；T 由 src/experiments/calibrate_model.py 在独立校准集上
    按 NLL 一维搜索得到。

    这个类定义在 classifier.py（而不是实验脚本里）是有意的：joblib 反序列化时需要
    按模块路径找到类定义，而实验脚本不在 classifier 的导入路径上，放那边会 UnpicklingError。
    """

    def __init__(self, estimator, temperature=1.0):
        self.estimator = estimator
        self.temperature = float(temperature)

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, X):
        d = np.asarray(self.estimator.decision_function(X), dtype=np.float64)
        if d.ndim == 1:                      # 二分类退化为单列，补成两列
            d = np.column_stack([-d, d])
        z = d / self.temperature
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.asarray(self.classes_)[self.predict_proba(X).argmax(axis=1)]


def train_and_evaluate(csv_path, model_type="svm", n_splits=5, save_path=None,
                       group_col="subject_id"):
    """
    任务 B 核心代码：
    严格按分组列做 GroupKFold 交叉验证并对比模型。

    group_col 决定"严格划分"的粒度（任务A 要求按人员或视频片段划分，防止连续帧泄漏）：
      - "subject_id"（默认）：按人员划分，同一人的样本只会出现在同一折
      - "session_id"        ：按拍摄时段划分
      - "clip_id"           ：按"人员 × 片段"划分，同时防止同一人的同一段连续帧跨折，
                              是最严格的口径（连续帧高度相关，是最隐蔽的泄漏源）
    """
    if not os.path.exists(csv_path):
        print(f"[Error] 数据集文件不存在: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"[Data] 成功加载数据集: {csv_path}, 共 {len(df)} 条样本")

    # 提取特征、标签、分组ID
    feature_cols = [f"feat_{i}" for i in range(136)]
    required = [group_col, "label", *feature_cols]
    if any(c not in df.columns for c in required):
        raise ValueError(f"CSV must contain {group_col}, label, and feat_0 through feat_135")
    if df[required].isna().any().any() or not np.isfinite(df[feature_cols].to_numpy(dtype=float)).all():
        raise ValueError("CSV contains missing or non-finite values")
    X = df[feature_cols].values
    y = df["label"].values
    groups = df[group_col].values

    unique_groups = len(np.unique(groups))
    print(f"[Data] 分组列={group_col}，分组数: {unique_groups}，特征维度: {X.shape[1]}")

    if unique_groups < 2 or n_splits < 2:
        raise ValueError(
            f"Cross-group evaluation requires at least two distinct {group_col} values and folds")
    gkf = GroupKFold(n_splits=min(n_splits, unique_groups))

    all_y_true = []
    all_y_pred = []

    print(f"\n================ 开始 {model_type.upper()} 模型 GroupKFold 严格跨组({group_col})评测 ================")
    fold = 1

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_type.lower() == "svm":
            clf = SVC(kernel="rbf", C=10.0, probability=True, random_state=42)
        elif model_type.lower() == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        held_out = np.unique(groups[test_idx])
        shown = held_out if len(held_out) <= 8 else list(held_out[:8]) + ["..."]
        print(f"Fold {fold}/{gkf.get_n_splits()} - 留出 {group_col}: {shown} -> Accuracy: {acc * 100:.2f}%")

        all_y_true.extend(y_test)
        all_y_pred.extend(preds)

        fold += 1

    print("\n---------------- 全局交叉验证评价报告 ----------------")
    print(classification_report(all_y_true, all_y_pred, digits=4))
    print("混淆矩阵 (Confusion Matrix):")
    labels = np.unique(all_y_true)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print(cm_df)

    if save_path:
        # CV measures generalization; deployment uses all available training subjects.
        clf.fit(X, y)
        joblib.dump({"model": clf, "classes": list(clf.classes_), "type": model_type,
                     "feature_schema": "legacy_68_xy_v1", "training_samples": len(df),
                     "group_col": group_col,
                     "training_subjects": unique_groups,
                     "training_groups": unique_groups,
                     "cv_accuracy": accuracy_score(all_y_true, all_y_pred),
                     "cv_macro_f1": f1_score(all_y_true, all_y_pred, average="macro")}, save_path)
        print(f"\n[Saved] 全量数据重训模型已保存至: {save_path}")

    return cm_df
