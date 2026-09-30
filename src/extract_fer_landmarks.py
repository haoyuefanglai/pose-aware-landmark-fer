"""
extract_fer_landmarks.py
从 FER2013 公开数据集批量提取人脸 68 关键点、可见性、头部姿态与 blendshape，
输出可直接用于训练的表情数据集：data/fer2013_landmarks.csv

与 extract_ck_landmarks.py 的差异：
  1. 数据来源改为 FER2013（指导书推荐的公开数据集，含 neutral 自然类）
  2. 增加关键点可见性 vis_0..vis_67（任务A 要求保存可见性信息）
  3. 增加 yaw/pitch/roll 姿态列，便于后续按头部姿态分组评测（任务C）
  4. 增加 52 维 blendshape 特征 bl_0..bl_51，用于特征集对照实验（任务B）

类别映射（FER2013 原始 7 类 -> 本项目 5 类）：
  angry -> frown, happy -> smile, sad -> sad, surprise -> surprise, neutral -> neutral
  disgust / fear 丢弃（样本极少且与规则引擎类别不一致）
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import FaceFeatureExtractor

FER_LABEL_NAMES = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
                   4: "sad", 5: "surprise", 6: "neutral"}

LABEL_MAP = {"angry": "frown", "happy": "smile", "sad": "sad",
             "surprise": "surprise", "neutral": "neutral"}

REPO_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
SPLIT_FILES = {
    "train": "data/train-00000-of-00001-5eab84e1c6a2fc27.parquet",
    "publicTest": "data/publicTest-00000-of-00001-f41bb7384b8aad6e.parquet",
    "privateTest": "data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet",
}

SRC_DIR = os.path.dirname(os.path.abspath(__file__))            # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                              # 项目根目录


def download_splits(raw_dir):
    """下载 FER2013 三个官方划分的 parquet（走 hf-mirror 镜像，huggingface.co 不可达）

    hf_hub_download 会在 local_dir 下保留仓库内的相对路径（即 local_dir/data/xxx.parquet），
    因此这里递归查找已缓存的文件，避免重复下载。
    """
    from huggingface_hub import hf_hub_download
    os.makedirs(raw_dir, exist_ok=True)
    cached = {}
    for dp, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".parquet"):
                cached[f] = os.path.join(dp, f)

    paths = {}
    for split, filename in SPLIT_FILES.items():
        base = os.path.basename(filename)
        if base in cached:
            print(f"  [cache] {split}: {cached[base]}")
            paths[split] = cached[base]
            continue
        print(f"  [下载] {split} ...")
        p = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset",
                            endpoint="https://hf-mirror.com", local_dir=raw_dir)
        paths[split] = p
    return paths


def decode_image(cell):
    if isinstance(cell, dict):
        buf = cell.get("bytes")
    elif isinstance(cell, (bytes, bytearray)):
        buf = bytes(cell)
    else:
        return None
    if not buf:
        return None
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(ROOT_DIR, "data", "fer2013_raw"))
    ap.add_argument("--out", default=os.path.join(ROOT_DIR, "data", "fer2013_landmarks.csv"))
    ap.add_argument("--limit", type=int, default=0, help="每个划分最多处理多少张（0=全部，用于快速试跑）")
    ap.add_argument("--decimals", type=int, default=5)
    args = ap.parse_args()

    print(">>> [1/4] 准备 FER2013 原始数据")
    paths = download_splits(args.raw_dir)

    print(">>> [2/4] 初始化 MediaPipe FaceLandmarker")
    extractor = FaceFeatureExtractor(static_mode=True)

    bs_names = None
    records = []
    stats = {}
    t0 = time.time()
    processed = 0

    try:
        for split, path in paths.items():
            df = pd.read_parquet(path)
            n = len(df) if args.limit <= 0 else min(args.limit, len(df))
            print(f"\n>>> [3/4] 划分 {split}: {len(df)} 张，本次处理 {n} 张")
            ok = skip_label = fail_detect = fail_decode = 0

            for i in range(n):
                row = df.iloc[i]
                raw_label = FER_LABEL_NAMES.get(int(row["label"]), str(row["label"]))
                mapped = LABEL_MAP.get(raw_label)
                if mapped is None:
                    skip_label += 1
                    continue

                img = decode_image(row["image"])
                if img is None:
                    fail_decode += 1
                    continue

                res = extractor.process_frame(img)
                if not res.get("detected"):
                    fail_detect += 1
                    continue

                if bs_names is None:
                    bs_names = sorted(res["blendshapes"].keys())
                    with open(os.path.join(ROOT_DIR, "data", "fer_blendshape_names.json"), "w",
                              encoding="utf-8") as f:
                        json.dump(bs_names, f, ensure_ascii=False, indent=2)
                    print(f"    捕获到 {len(bs_names)} 维 blendshape 名称")

                rec = {
                    "split": split,
                    "fer_label": int(row["label"]),
                    "raw_label": raw_label,
                    "label": mapped,
                    "yaw": round(res["head_pose"]["yaw"], 3),
                    "pitch": round(res["head_pose"]["pitch"], 3),
                    "roll": round(res["head_pose"]["roll"], 3),
                }
                for j, v in enumerate(res["norm_vector"]):
                    rec[f"feat_{j}"] = round(float(v), args.decimals)

                mesh = res["raw_mesh"]
                # 68 关键点对应的 MediaPipe 索引，与 feature_extractor 的映射保持一致
                from feature_extractor import MEDIAPIPE_TO_68
                for j, idx in enumerate(MEDIAPIPE_TO_68):
                    rec[f"vis_{j}"] = round(float(getattr(mesh[idx], "visibility", 0.0) or 0.0), 3)

                bl = res["blendshapes"]
                for j, name in enumerate(bs_names):
                    rec[f"bl_{j}"] = round(float(bl.get(name, 0.0)), 4)

                records.append(rec)
                ok += 1

                if ok % 2000 == 0:
                    el = time.time() - t0
                    print(f"    {split} {ok} 张完成，累计 {processed + ok} 张，用时 {el:.0f}s "
                          f"({(processed + ok) / max(el, 1e-6):.1f} 张/秒)")

            # tqdm 风格换行，避免刷屏
            print(f"    {split} 完成: 成功 {ok}，跳过类别 {skip_label}，未检出人脸 {fail_detect}，解码失败 {fail_decode}")
            stats[split] = dict(ok=ok, skip_label=skip_label,
                                fail_detect=fail_detect, fail_decode=fail_decode)
            processed += ok
            if args.limit > 0:
                break
    finally:
        extractor.close()

    print("\n>>> [4/4] 写出 CSV")
    out = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[SUCCESS] {args.out}  行数={len(out)}  列数={out.shape[1]}  "
          f"文件大小={os.path.getsize(args.out) / 1e6:.1f}MB")

    print("\n各类别样本数：")
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\n检出统计：")
    for k, v in stats.items():
        print("  ", k, v)

    meta = {
        "source": REPO_ID,
        "splits": stats,
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "classes": sorted(out["label"].unique().tolist()),
        "feature_schema": "fer2013_68xy_vis_blendshape_v1",
        "blendshape_dims": len(bs_names or []),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(ROOT_DIR, "data", "fer2013_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
