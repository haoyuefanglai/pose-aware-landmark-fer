"""
collect_and_train.py
数据构建与模型训练工具：
1. 生成基础样例数据 / 从本地图像集批量提取关键点 CSV
2. 摄像头交互式自主采集（自动打上 subject_id + session_id 标签，满足防泄漏要求）
3. 训练并对比 SVM 与 MLP 模型，输出完整的混淆矩阵与评估报告

关于划分字段（任务A 考核"训练/验证/测试按人员或视频片段划分"）：
  - subject_id 标识受试者（人）
  - session_id 标识同一位受试者的一次拍摄时段
  - clip_id    = subject_id + session_id，即"人员×片段"的最小独立单元
  按 subject_id 分组做 GroupKFold 可防止同一人跨折泄漏；按 clip_id 分组则同时
  防止同一人的同一段连续帧跨折泄漏（连续帧高度相关，是最隐蔽的泄漏源）。
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
import time
import argparse
import numpy as np
import pandas as pd
import cv2

from feature_extractor import FaceFeatureExtractor
from classifier import train_and_evaluate, EXPRESSION_CLASSES, EXPRESSION_NAMES_ZH

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                              # 项目根目录
DEFAULT_CSV = os.path.join(ROOT_DIR, "data", "dataset.csv")


def make_clip_id(subject_id, session_id):
    """clip_id = 人员 × 片段，作为分组划分的最小独立单元。"""
    return f"{subject_id}__{session_id}"


def generate_sample_dataset(csv_path=DEFAULT_CSV, n_subjects=6, samples_per_class=20, sessions=2):
    """
    生成符合真实面部解剖学分布的标准化样例数据集 (用于快速验证整个训练流水线)
    包含 subject_id / session_id / clip_id，满足 GroupKFold 跨受试者与跨片段评测标准
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    records = []

    print(f"[Generate] 正在生成跨受试者演示数据集: {n_subjects} 位受试者 × {sessions} 个拍摄时段，每类 {samples_per_class} 样本...")

    np.random.seed(42)

    for s_idx in range(1, n_subjects + 1):
        subj_id = f"Subject_{s_idx:02d}"
        # 每位受试者的人脸基础形态个体差异 (随机扰动)
        subj_bias = np.random.normal(0, 0.02, size=136)

        for sess in range(1, sessions + 1):
            session_id = f"S{sess:02d}"
            clip_id = make_clip_id(subj_id, session_id)
            # 同一时段内的光照/姿态整体漂移：模拟"连续帧高度相关"
            session_bias = np.random.normal(0, 0.01, size=136)

            for label in EXPRESSION_CLASSES:
                for _ in range(samples_per_class):
                    # 基础几何向量 (136维)
                    feat = np.random.normal(0, 0.05, size=136) + subj_bias + session_bias

                    # 针对不同表情注入显著的面部特征偏置 (模拟嘴角、眼睛、眉毛变形)
                    # 嘴唇关键点索引区域 (48-67对应向量中段)
                    if label == "smile":
                        # 嘴角上扬与横向拉伸
                        feat[48 * 2] -= 0.15     # 右嘴角向外
                        feat[54 * 2] += 0.15     # 左嘴角向外
                        feat[48 * 2 + 1] -= 0.10 # 嘴角向上
                        feat[54 * 2 + 1] -= 0.10
                    elif label == "surprise":
                        # 嘴巴纵向大开 + 挑眉
                        feat[51 * 2 + 1] -= 0.15 # 上唇向上
                        feat[57 * 2 + 1] += 0.20 # 下唇向下
                        feat[19 * 2 + 1] -= 0.10 # 眉毛上挑
                        feat[24 * 2 + 1] -= 0.10
                    elif label == "frown":
                        # 眉间距缩小
                        feat[21 * 2] += 0.08
                        feat[22 * 2] -= 0.08
                        feat[21 * 2 + 1] += 0.08
                        feat[22 * 2 + 1] += 0.08
                    elif label == "sad":
                        # 嘴角下撇
                        feat[48 * 2 + 1] += 0.12
                        feat[54 * 2 + 1] += 0.12

                    row = {"subject_id": subj_id, "session_id": session_id,
                           "clip_id": clip_id, "label": label}
                    for i in range(136):
                        row[f"feat_{i}"] = feat[i]
                    records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[Success] 样例数据集已生成至: {csv_path} (共 {len(df)} 条样本，"
          f"{n_subjects} 人 / {n_subjects * sessions} 个片段)")
    return csv_path


def interactive_collect(subject_id="Subject_01", csv_path=DEFAULT_CSV, session_id=None):
    """
    通过摄像头实时交互采集组员个人表情数据，自动写入 subject_id / session_id / clip_id
    与归一化特征。按键盘 [1-5] 录入对应表情，[N] 切换新片段，[Q] 退出

    session_id 留空时按启动时间自动生成，因此不同次启动天然是不同片段；
    同一次采集内按 [N] 可以人为切分片段（例如换姿势、换光照时切一刀）。
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    # 默认 VIDEO 模式：采集是连续视频流，跨帧跟踪比逐帧独立检测更稳定
    extractor = FaceFeatureExtractor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] 无法打开摄像头进行采集。")
        extractor.close()
        return

    if session_id is None:
        session_id = time.strftime("%Y%m%d_%H%M%S")

    print("==========================================================")
    print(f" 交互式表情数据采集工具 - 当前受试者: {subject_id}")
    print(f" 当前拍摄片段 session_id: {session_id}")
    print(" 按键说明:")
    print("   [1] 录入 自然 (Neutral)")
    print("   [2] 录入 微笑 (Smile)")
    print("   [3] 录入 惊讶 (Surprise)")
    print("   [4] 录入 皱眉 (Frown)")
    print("   [5] 录入 难过 (Sad)")
    print("   [N] 结束当前片段并开启新片段（换姿势/换光照时使用）")
    print("   [Q] 完成并退出")
    print("==========================================================")

    saved_count = 0
    records = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        res = extractor.process_frame(frame)
        h, w = frame.shape[:2]

        if res["detected"]:
            pts = res["landmarks_68"].astype(np.int32)
            for pt in pts:
                cv2.circle(frame, (pt[0], pt[1]), 2, (0, 255, 0), -1)

            cv2.putText(frame, f"Subject: {subject_id} | Session: {session_id} | Saved: {saved_count}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Press [1-5] to record expression, [N] new session, [Q] to quit", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        target_label = None
        if key == ord('1'): target_label = "neutral"
        elif key == ord('2'): target_label = "smile"
        elif key == ord('3'): target_label = "surprise"
        elif key == ord('4'): target_label = "frown"
        elif key == ord('5'): target_label = "sad"
        elif key == ord('n') or key == ord('N'):
            session_id = time.strftime("%Y%m%d_%H%M%S")
            print(f"[Session] 已切换到新拍摄片段: {session_id}")
        elif key == ord('q') or key == ord('Q') or key == 27:
            break

        if target_label and res["detected"]:
            row = {"subject_id": subject_id,
                   "session_id": session_id,
                   "clip_id": make_clip_id(subject_id, session_id),
                   "label": target_label}
            for i, val in enumerate(res["norm_vector"]):
                row[f"feat_{i}"] = val
            records.append(row)
            saved_count += 1
            print(f"[Saved] 已记录 #{saved_count}: {target_label} (session={session_id})")

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()

    if records:
        new_df = pd.DataFrame(records)
        if os.path.exists(csv_path):
            old_df = pd.read_csv(csv_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined.to_csv(csv_path, index=False)
        else:
            new_df.to_csv(csv_path, index=False)
        print(f"[Done] 成功保存 {len(records)} 条样本到: {csv_path}")
        print(f"       受试者 {new_df['subject_id'].nunique()} 人 / "
              f"片段 {new_df['clip_id'].nunique()} 个")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据采集与模型训练管理")
    parser.add_argument("--action", choices=["gen_sample", "collect", "train"], default="gen_sample",
                        help="操作类型: gen_sample(生成演示数据), collect(摄像头采集), train(训练对比模型)")
    parser.add_argument("--subject", default="Subject_01", help="采集时指定的受试者编号")
    parser.add_argument("--session", default=None,
                        help="采集时的拍摄片段编号（默认按启动时间生成，同一次采集内可用 [N] 切分）")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="数据集路径（默认 data/dataset.csv）")
    parser.add_argument("--model_type", default="svm", choices=["svm", "mlp"], help="训练模型类型: svm 或 mlp")
    parser.add_argument("--group_col", default="subject_id",
                        choices=["subject_id", "session_id", "clip_id"],
                        help="GroupKFold 分组列（默认 subject_id；clip_id 可同时防人员与片段泄漏）")
    parser.add_argument("--save", default=os.path.join(ROOT_DIR, "models", "custom_model.pkl"),
                        help="模型权重保存路径（默认存到 models/ 目录）")
    args = parser.parse_args()

    if args.action == "gen_sample":
        generate_sample_dataset(csv_path=args.csv)
    elif args.action == "collect":
        interactive_collect(subject_id=args.subject, csv_path=args.csv, session_id=args.session)
    elif args.action == "train":
        train_and_evaluate(csv_path=args.csv, model_type=args.model_type,
                           save_path=args.save, group_col=args.group_col)
