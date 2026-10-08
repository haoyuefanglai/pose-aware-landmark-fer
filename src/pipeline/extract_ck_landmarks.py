"""
extract_ck_landmarks.py
从真实 CK+ 数据集 (data/ck_raw.parquet) 中批量提取人脸 68 关键点并完成几何归一化
输出真实的学术级标准数据集: data/ck_plus_landmarks.csv
包含真实 subject_id (共 118 位受试者) 与 clip_id (人员 × 序列)，
用于严格的 GroupKFold 跨人 / 跨片段交叉验证评估

划分字段：CK+ 文件名形如 S010_004_00000017，其中
  S010 = 受试者编号 -> subject_id
  004  = 该受试者的第 4 段视频序列 -> 与受试者拼成 clip_id = "S010_004"
同一段序列内的帧在时间上连续、内容高度相关，因此按 clip_id 分组比只按 subject_id
分组更严格，可同时防住"同一人的连续帧跨折"这一最隐蔽的泄漏源。

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
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_extractor import FaceFeatureExtractor

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                              # 项目根目录

def batch_extract_ck(parquet_path=os.path.join(ROOT_DIR, "data", "ck_raw.parquet"),
                     output_csv=os.path.join(ROOT_DIR, "data", "ck_plus_landmarks.csv")):
    if not os.path.exists(parquet_path):
        print(f"[Error] 未找到数据源: {parquet_path}")
        return

    print(f">>> [1/3] 正在读取真实 CK+ 数据包: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"    共包含 {len(df)} 张真实人脸图像，分布如下:")
    print(df["label"].value_counts())

    # 选取最主要、最具代表性的 5 类基础表情 (契合课程指导书 4-5 类要求)
    # anger(愤怒/皱眉), disgust(厌恶), happy(微笑), sadness(悲伤), surprise(惊讶)
    target_emotions = ["anger", "disgust", "happy", "sadness", "surprise"]
    df_filtered = df[df["label"].isin(target_emotions)].copy()
    print(f"\n>>> [2/3] 筛选出 5 类典型表情共 {len(df_filtered)} 张图像，开始提取 MediaPipe 68 关键点与几何特征...")

    extractor = FaceFeatureExtractor(static_mode=True)
    records = []
    success_count = 0
    fail_count = 0

    for idx in tqdm(range(len(df_filtered)), desc="Extracting Landmarks"):
        row = df_filtered.iloc[idx]
        img_bytes = row["image"]["bytes"]
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            fail_count += 1
            continue

        # 图像适度双三次插值放大至 256x256，确保关键点检测达到亚像素级最高精度
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        res = extractor.process_frame(img_resized)

        if res["detected"]:
            # 解析真实人物编号与序列号 (例如 S010_004_00000017 -> S010 / 004)
            file_name = row["file"]
            parts = file_name.split("_")
            subject_id = parts[0] if "_" in file_name else "Unknown"
            # 第 2 段是序列号；缺失时退化为 subject_id 本身，保证列始终存在
            sequence = parts[1] if len(parts) > 1 else "seq"
            clip_id = f"{subject_id}_{sequence}"
            frame_no = parts[-1] if len(parts) > 2 else ""

            # 类别标签映射 (也可以保留原名)
            label_map = {
                "happy": "smile",
                "surprise": "surprise",
                "anger": "frown",
                "sadness": "sad",
                "disgust": "disgust"
            }
            mapped_label = label_map.get(row["label"], row["label"])

            record = {
                "subject_id": subject_id,
                "session_id": clip_id,
                "clip_id": clip_id,
                "frame_no": frame_no,
                "file_name": file_name,
                "raw_label": row["label"],
                "label": mapped_label
            }

            # 136 维归一化坐标特征 (feat_0 ~ feat_135)
            norm_vector = res["norm_vector"]
            for i, val in enumerate(norm_vector):
                record[f"feat_{i}"] = float(val)

            # 逐点几何可见性 (0~1)，与 FER2013 特征表同名同义
            vis68 = res["visibility_68"]
            for i, val in enumerate(vis68):
                record[f"vis_{i}"] = round(float(val), 3)
            record["vis_mean"] = round(float(np.mean(vis68)), 4)

            # 头部姿态角，便于按姿态分组评测
            hp = res["head_pose"]
            record["yaw"] = round(hp["yaw"], 3)
            record["pitch"] = round(hp["pitch"], 3)
            record["roll"] = round(hp["roll"], 3)

            # 记录关键几何指标
            geo = res.get("geo_metrics", {})
            for g_key, g_val in geo.items():
                record[f"geo_{g_key}"] = float(g_val)

            records.append(record)
            success_count += 1
        else:
            fail_count += 1

    print(f"\n>>> [3/3] 提取完成！成功率: {success_count}/{len(df_filtered)} ({success_count/len(df_filtered)*100:.2f}%)")
    result_df = pd.DataFrame(records)
    result_df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] 真实 CK+ 关键点数据集已保存至: {output_csv}")
    print(f"    受试者总数 (Subject Count): {result_df['subject_id'].nunique()}")
    print(f"    序列片段总数 (Clip Count):  {result_df['clip_id'].nunique()}")
    print("    各类别样本统计:")
    print(result_df["label"].value_counts())
    return output_csv

if __name__ == "__main__":
    batch_extract_ck()
