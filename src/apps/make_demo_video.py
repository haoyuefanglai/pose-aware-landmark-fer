"""
make_demo_video.py
用 FER2013 留出图像合成一段离线演示视频 example.mp4。

用途：摄像头不可用时，realtime_demo.py 可以直接读这段视频做完整演示：
    python src/apps/realtime_demo.py --source assets/example.mp4 --model models/model_fer_svm.pkl --engine ml

画面右下角烧录的是数据集真实标签 (GT)，窗口 HUD 上显示的是模型预测，
两者可以直接肉眼对照，不需要任何摄像头或外部视频素材。
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

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                              # 项目根目录
FER_NAMES = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
             4: "sad", 5: "surprise", 6: "neutral"}
LMAP = {"angry": "frown", "happy": "smile", "sad": "sad",
        "surprise": "surprise", "neutral": "neutral"}


def find_parquet(split_hint):
    raw = os.path.join(ROOT_DIR, "data", "fer2013_raw")
    hits = []
    for dp, _, fs in os.walk(raw):
        for f in fs:
            if f.endswith(".parquet") and split_hint in f:
                hits.append(os.path.join(dp, f))
    return sorted(hits)


def title_card(size, seconds, fps):
    """开场说明卡，避免演示时忘记按键功能"""
    side = size[0]
    card = np.full((size[1], size[0], 3), 24, np.uint8)
    lines = [
        ("Offline Demo Video - FER2013 samples", 0.62, (0, 255, 255), 60),
        ("No camera required", 0.5, (200, 200, 200), 110),
        ("", 0.5, (200, 200, 200), 140),
        ("Bottom-right text = dataset ground truth", 0.45, (180, 180, 180), 175),
        ("Left / right panels = model prediction", 0.45, (180, 180, 180), 205),
        ("", 0.5, (200, 200, 200), 235),
        ("[Q] Quit   [M] Mesh   [T] Engine   [S] Snapshot", 0.45, (120, 220, 120), 280),
    ]
    for text, scale, color, y in lines:
        if not text:
            continue
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(card, text, ((side - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1)
    for _ in range(int(seconds * fps)):
        yield card


def build(out_path, per_class, frames_per_image, fps, side, duration):
    sources = find_parquet("publicTest") + find_parquet("privateTest")
    if not sources:
        raise SystemExit("[Error] 未找到 FER2013 parquet，请先运行 extract_fer_landmarks.py")

    frames = []
    picked = 0
    for pq in sources:
        df = pd.read_parquet(pq)
        for lab in [3, 6, 5, 0, 4]:          # happy / neutral / surprise / angry / sad
            sub = df[df["label"] == lab].head(per_class)
            for _, row in sub.iterrows():
                img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8),
                                   cv2.IMREAD_COLOR)
                img = cv2.resize(img, (side, side), interpolation=cv2.INTER_CUBIC)
                gt = FER_NAMES[int(row["label"])]
                text = f"GT: {gt} -> {LMAP.get(gt, 'n/a')}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (img.shape[1] - tw - 14, img.shape[0] - th - 30),
                              (img.shape[1] - 6, img.shape[0] - 6), (20, 20, 20), -1)
                cv2.putText(img, text, (img.shape[1] - tw - 10, img.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                frames.extend([img] * frames_per_image)
                picked += 1
        if picked >= per_class * 5:
            break

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (side, side))
    if not writer.isOpened():
        raise SystemExit("[Error] 无法创建视频文件，检查输出路径与 mp4v 编码器")

    total = 0
    for frame in title_card((side, side), duration, fps):
        writer.write(frame)
        total += 1
    for frame in frames:
        writer.write(frame)
        total += 1
    writer.release()
    return picked, total


def verify(out_path):
    cap = cv2.VideoCapture(out_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    return ok, n, fps, w, h


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="生成免摄像头的离线演示视频")
    ap.add_argument("--out", default=os.path.join(ROOT_DIR, "assets", "example.mp4"))
    ap.add_argument("--per-class", type=int, default=8, help="每个表情类别取多少张图")
    ap.add_argument("--frames", type=int, default=12, help="每张图持续多少帧")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--side", type=int, default=480, help="输出画面边长 (48x48 原图上采样)")
    ap.add_argument("--title", type=float, default=2.5, help="开场说明卡秒数")
    a = ap.parse_args()

    print(f">>> 从 FER2013 留出图像合成演示视频 -> {a.out}")
    n_img, n_frame = build(a.out, a.per_class, a.frames, a.fps, a.side, a.title)
    ok, n, fps, w, h = verify(a.out)
    size_mb = os.path.getsize(a.out) / 1e6
    print(f"    使用图像 {n_img} 张，总帧数 {n_frame}，视频自检读出 {n} 帧")
    print(f"    分辨率 {w}x{h} @ {fps:.0f} fps，时长约 {n / max(fps, 1):.1f} 秒，"
          f"文件 {size_mb:.2f} MB，可读={ok}")
    print("\n[Done] 免摄像头运行方式：")
    print(f"    python src/apps/realtime_demo.py --source assets/{os.path.basename(a.out)} "
          f"--model models/model_fer_svm.pkl --engine ml")
