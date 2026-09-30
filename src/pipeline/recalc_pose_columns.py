# -*- coding: utf-8 -*-
"""重算 data/fer2013_landmarks.csv 的 yaw / pitch / roll 三列。

背景：特征表中的姿态列由旧版 solvePnP 流程生成，因 3D 模型坐标系与图像坐标系
y 轴方向不一致，pitch 有 98.6% 落在 ±150°~180°（详见 feature_extractor.py 顶部说明）。
feature_extractor.py 已改用 MediaPipe facial_transformation_matrixes 解算，
本脚本按与 extract_fer_landmarks.py 完全相同的顺序重新遍历原始 parquet，
用修复后的解算重算三列并写回。

安全措施：
1. 逐行对齐校验 —— 重新遍历产生的 (split, fer_label, label) 序列必须与 CSV 现有
   行序列完全一致才写回；任何错位立即中止，不会污染特征表。
2. 写回前先输出新旧姿态分布对比，写入临时文件后再原子替换。

用法：
  python src/pipeline/recalc_pose_columns.py            # 校验 + 重算 + 写回（约 20 分钟）
  python src/pipeline/recalc_pose_columns.py --limit 500  # 小样本验证对齐逻辑
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
import time
import argparse

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from feature_extractor import FaceFeatureExtractor               # noqa: E402
from extract_fer_landmarks import (FER_LABEL_NAMES, LABEL_MAP,    # noqa: E402
                                   SPLIT_FILES, decode_image, download_splits)


def main():
    ap = argparse.ArgumentParser(description="重算特征表姿态列（yaw/pitch/roll）")
    ap.add_argument("--csv", default=os.path.join(ROOT_DIR, "data", "fer2013_landmarks.csv"))
    ap.add_argument("--raw-dir", default=os.path.join(ROOT_DIR, "data", "fer2013_raw"))
    ap.add_argument("--limit", type=int, default=0, help="调试：只处理每个划分前 N 张")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.csv)
    print(f"[CSV] {args.csv}: {len(df)} 行")

    paths = download_splits(args.raw_dir)
    extractor = FaceFeatureExtractor(static_mode=True)

    new_pose = []
    idx = 0                      # CSV 行游标
    mismatch = None
    stats = {}

    try:
        for split, path in paths.items():
            table = pd.read_parquet(path)
            n = len(table) if args.limit <= 0 else min(args.limit, len(table))
            print(f"[Split] {split}: 处理 {n} 张", flush=True)
            ok = skipped = undetected = decode_fail = 0
            for i in range(n):
                row = table.iloc[i]
                raw_label = FER_LABEL_NAMES.get(int(row["label"]), str(row["label"]))
                mapped = LABEL_MAP.get(raw_label)
                if mapped is None:
                    skipped += 1
                    continue
                img = decode_image(row["image"])
                if img is None:
                    decode_fail += 1
                    continue
                res = extractor.process_frame(img)
                if not res.get("detected"):
                    undetected += 1
                    continue

                # ---- 对齐校验：必须与 CSV 当前行完全一致 ----
                if idx >= len(df):
                    mismatch = (idx, split, int(row["label"]), "<超出CSV行数>")
                    break
                c = df.iloc[idx]
                if (c["split"] != split or int(c["fer_label"]) != int(row["label"])
                        or c["label"] != mapped):
                    mismatch = (idx, split, int(row["label"]), mapped,
                                str((c["split"], int(c["fer_label"]), c["label"])))
                    break
                hp = res["head_pose"]
                new_pose.append((round(hp["yaw"], 3), round(hp["pitch"], 3), round(hp["roll"], 3)))
                idx += 1
                ok += 1
                if ok % 3000 == 0:
                    rate = ok / (time.time() - t0)
                    print(f"    {split} {ok} 张 ({rate:.0f} 张/秒)", flush=True)
            stats[split] = dict(ok=ok, skip_label=skipped, undetected=undetected,
                                decode_fail=decode_fail)
            print(f"    {split} 完成: 检出 {ok}，跳过类别 {skipped}，"
                  f"未检出 {undetected}，解码失败 {decode_fail}", flush=True)
            if mismatch:
                break
    finally:
        extractor.close()

    if mismatch:
        print(f"\n[中止] 第 {mismatch[0]} 行对齐失败: 重算=(split={mismatch[1]}, "
              f"fer_label={mismatch[2]}, label={mismatch[3]})  CSV={mismatch[4]}")
        print("检出集合与当年提取时不一致，为避免错位污染，本次不写回。")
        sys.exit(1)
    if idx != len(df):
        print(f"\n[中止] 重算样本数 {idx} != CSV 行数 {len(df)}，不写回。")
        sys.exit(1)

    # ---- 对齐成功：对比新旧姿态分布 ----
    pose = np.array(new_pose)
    print("\n[对齐成功] 全部 %d 行 (split, fer_label, label) 与 CSV 逐行一致" % len(df))
    print("\n姿态列前后对比（中位数 / |值|中位数）:")
    for j, name in enumerate(("yaw", "pitch", "roll")):
        old = df[name].to_numpy(dtype=np.float64)
        new = pose[:, j]
        print("  %-6s 旧: %8.2f / %8.2f   新: %8.2f / %8.2f   (|角|>150° 旧 %5.1f%% → 新 %.1f%%)" % (
            name, np.median(old), np.median(np.abs(old)),
            np.median(new), np.median(np.abs(new)),
            100 * np.mean(np.abs(old) > 150), 100 * np.mean(np.abs(new) > 150)))

    tmp = args.csv + ".tmp"
    df_out = df.copy()
    df_out["yaw"] = pose[:, 0]
    df_out["pitch"] = pose[:, 1]
    df_out["roll"] = pose[:, 2]
    df_out.to_csv(tmp, index=False)
    os.replace(tmp, args.csv)
    print(f"\n[Done] 已写回 {args.csv}，耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
