"""
extract_ck_landmarks.py
从真实 CK+ 数据集 (data/ck_raw.parquet) 中批量提取人脸 68 关键点并完成几何归一化
输出真实的学术级标准数据集: data/ck_plus_landmarks.csv
包含真实 subject_id (共 118 位受试者)，用于严格的 GroupKFold 跨人交叉验证评估
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_extractor import FaceFeatureExtractor

def batch_extract_ck(parquet_path="data/ck_raw.parquet", output_csv="data/ck_plus_landmarks.csv"):
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
            # 解析真实人物编号 (例如 S010_004_00000017 -> S010)
            file_name = row["file"]
            subject_id = file_name.split("_")[0] if "_" in file_name else "Unknown"

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
                "file_name": file_name,
                "raw_label": row["label"],
                "label": mapped_label
            }

            # 136 维归一化坐标特征 (feat_0 ~ feat_135)
            norm_vector = res["norm_vector"]
            for i, val in enumerate(norm_vector):
                record[f"feat_{i}"] = float(val)

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
    print("    各类别样本统计:")
    print(result_df["label"].value_counts())
    return output_csv

if __name__ == "__main__":
    batch_extract_ck()
