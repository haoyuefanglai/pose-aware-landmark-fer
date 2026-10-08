# -*- coding: utf-8 -*-
"""重算 data/fer2013_landmarks.csv 的可见性列 vis_0..vis_67（并新增 vis_mean）。

背景：任务A 要求"保存关键点及可见性信息"。原实现读取 MediaPipe 关键点的
`visibility` 字段，但该字段在 FaceLandmarker 中恒为 None，`getattr(..., 0.0)`
兜底之后 68 列全部被写成 0.0 —— 列名、列数、schema 都在，信息量为零
（data/README.md 与 AUDIT.md 均有记录，对照实验也确认加不加都不影响指标）。

feature_extractor.py 现已改为用 MediaPipe 输出的归一化 3D 坐标 (x, y, z)
做局部法线估计 + 背向判定，得到真实的逐点几何可见性。本脚本按与
extract_fer_landmarks.py 完全相同的遍历顺序重新提取并写回这些列。

安全措施：
1. 逐行对齐校验 —— 重新遍历产生的 (split, fer_label, label) 序列必须与 CSV 现有
   行序列完全一致才写回；任何错位立即中止，不会污染特征表。
2. 写回前先把原文件备份为 .bak，再写临时文件后原子替换。
3. 写回前输出新旧可见性分布对比（常数列数、取值范围、与 |yaw| 的相关性）。

用法：
  python src/pipeline/recalc_visibility_columns.py --limit 800   # 小样本验证对齐逻辑
  python src/pipeline/recalc_visibility_columns.py               # 全量重算（约 7~10 分钟）
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
import shutil
import argparse

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from feature_extractor import FaceFeatureExtractor, MEDIAPIPE_TO_68   # noqa: E402
from extract_fer_landmarks import (FER_LABEL_NAMES, LABEL_MAP,        # noqa: E402
                                   download_splits, decode_image)

VIS_COLS = [f"vis_{i}" for i in range(68)]


def describe(V, yaw, tag):
    """打印一组可见性列的可诊断统计（用来证明"不再是空壳"）。"""
    const_cols = int(np.sum(V.std(axis=0) < 1e-9))
    uniq = [len(np.unique(V[:, j])) for j in range(V.shape[1])]
    print(f"  {tag}: min={V.min():.3f} max={V.max():.3f} mean={V.mean():.3f} "
          f"常数列={const_cols}/{V.shape[1]} 单列唯一值中位数={int(np.median(uniq))}")
    if yaw is not None and np.std(V.mean(axis=1)) > 0:
        c = float(np.corrcoef(np.abs(yaw), V.mean(axis=1))[0, 1])
        print(f"  {tag}: corr(|yaw|, 逐行平均可见性) = {c:.3f}")


def main():
    ap = argparse.ArgumentParser(description="重算特征表可见性列（vis_0..vis_67 / vis_mean）")
    ap.add_argument("--csv", default=os.path.join(ROOT_DIR, "data", "fer2013_landmarks.csv"))
    ap.add_argument("--raw-dir", default=os.path.join(ROOT_DIR, "data", "fer2013_raw"))
    ap.add_argument("--limit", type=int, default=0, help="调试：只处理每个划分前 N 张")
    ap.add_argument("--backup", action="store_true", default=True,
                    help="写回前备份原 CSV 为 .bak（默认开启）")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.csv)
    print(f"[CSV] {args.csv}: {len(df)} 行")

    if "yaw" in df.columns:
        describe(df[VIS_COLS].to_numpy(np.float64), df["yaw"].to_numpy(np.float64), "旧值")
    else:
        describe(df[VIS_COLS].to_numpy(np.float64), None, "旧值")

    paths = download_splits(args.raw_dir)
    # 逐图独立处理：IMAGE 模式，避免 VIDEO 跟踪把相邻无关样本串起来
    extractor = FaceFeatureExtractor(static_mode=True)

    new_vis = []
    idx = 0
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

                v = np.asarray(res["visibility_68"], dtype=np.float64)
                new_vis.append(v)
                idx += 1
                ok += 1
                if ok % 2000 == 0:
                    rate = ok / (time.time() - t0)
                    print(f"    {split} {ok} 张 ({rate:.0f} 张/秒)", flush=True)
            stats[split] = dict(ok=ok, skip_label=skipped,
                                undetected=undetected, decode_fail=decode_fail)
            print(f"    {split} 完成: 检出 {ok}，跳过类别 {skipped}，"
                  f"未检出 {undetected}，解码失败 {decode_fail}", flush=True)
            if mismatch:
                break
            if args.limit > 0:
                break
    finally:
        extractor.close()

    if mismatch:
        print(f"\n[中止] 第 {mismatch[0]} 行对齐失败: 重算=(split={mismatch[1]}, "
              f"fer_label={mismatch[2]}, label={mismatch[3]})  CSV={mismatch[4]}")
        print("检出集合与当年提取时不一致，为避免错位污染，本次不写回。")
        sys.exit(1)

    V = np.asarray(new_vis, dtype=np.float64)
    if V.size == 0:
        print("[中止] 未产出任何可见性数据。")
        sys.exit(1)

    if args.limit > 0:
        print(f"\n[小样本验证] 已对齐 {idx} 行（--limit {args.limit}），本次不写回。")
        describe(V, df["yaw"].to_numpy(np.float64)[:len(V)] if "yaw" in df.columns else None, "新值(子集)")
        print("对齐逻辑通过。去掉 --limit 即可全量重算并写回。")
        return

    if idx != len(df):
        print(f"\n[中止] 重算样本数 {idx} != CSV 行数 {len(df)}，不写回。")
        sys.exit(1)

    print(f"\n[对齐成功] 全部 {len(df)} 行 (split, fer_label, label) 与 CSV 逐行一致")
    describe(V, df["yaw"].to_numpy(np.float64) if "yaw" in df.columns else None, "新值")

    if args.backup:
        bak = args.csv + ".bak"
        shutil.copy2(args.csv, bak)
        print(f"[备份] {bak}")

    tmp = args.csv + ".tmp"
    df_out = df.copy()
    for j in range(68):
        df_out[f"vis_{j}"] = np.round(V[:, j], 3)
    df_out["vis_mean"] = np.round(V.mean(axis=1), 4)
    df_out.to_csv(tmp, index=False)
    os.replace(tmp, args.csv)
    print(f"\n[Done] 已写回 {args.csv}（{df_out.shape[1]} 列），耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
