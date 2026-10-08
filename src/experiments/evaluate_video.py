# -*- coding: utf-8 -*-
"""视频逐帧评测（实验五 任务C 的"视频链路出数"环节）。

补齐的缺口：
  - realtime_demo.py 只按 [S] 存截图，不落"第几帧 / 什么姿态 / 预测成什么 / 分数多少"
    的逐帧记录；
  - pose_stability_test.py 的输入写死在 FER2013 静态图上，读不了视频文件；
  - 于是任务C 的实时链路只能"演示"，产不出可写入报告的稳定性数字。

本脚本一次遍历视频/摄像头，对每一帧：
  1. 用 MediaPipe 提取关键点、头部姿态 (yaw/pitch/roll) 与逐点几何可见性；
  2. 逐个引擎（FACS 规则 / 未校准 SVM / 校准 SVM）分别推理并做时序平滑；
  3. 把逐帧结果全部落盘成 CSV，并汇总成 JSON。

汇总口径（与 pose_stability_test.py 的姿态分桶保持一致，两处数字可直接并列）：
  - 检出率、单帧耗时（均值/P50/P95）与等效吞吐
  - 头部姿态分桶下的平均可见性、平均分数、主标签占比
  - 可见性与 |yaw| 的相关性（用于证明可见性列真的随姿态变化，不是常数）
  - 时序稳定性：标签切换次数、每百帧切换率、平均连续段长
  - 拒识比例（校准模型 + 阈值时）

用法：
  # 无摄像头/无显示器环境也能跑完并出数
  python src/experiments/evaluate_video.py --source assets/example.mp4 --no-window
  # 只跑校准模型，限制 300 帧
  python src/experiments/evaluate_video.py --source assets/example.mp4 --engines svm_calibrated --max-frames 300
  # 摄像头实时评测（按 Q 结束）
  python src/experiments/evaluate_video.py --source 0

输出：
  data/video_eval_frames_<engine>.csv
  data/video_eval_summary.json
"""

# --- 路径引导：src/ 下 core / pipeline / experiments / apps 之间可互相 import ---
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
import time
import argparse

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from feature_extractor import FaceFeatureExtractor, resolve_static_mode  # noqa: E402
from classifier import ExpressionClassifier, EXPRESSION_CLASSES, UNKNOWN_LABEL  # noqa: E402
from frame_recorder import FrameRecorder                                 # noqa: E402
from realtime_demo import TemporalSmoother, draw_hud                     # noqa: E402

# 可评测的引擎：名称 -> 模型相对路径（None 表示内置 FACS 规则引擎）
ENGINE_MODELS = {
    "rules": None,
    "svm": os.path.join("models", "model_fer_svm.pkl"),
    "mlp": os.path.join("models", "model_fer_mlp.pkl"),
    "svm_calibrated": os.path.join("models", "model_fer_svm_calibrated.pkl"),
}


def open_source(source):
    """打开摄像头或视频文件，返回 (cap, is_camera, source)。"""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    return cap, isinstance(source, int), source


def resolve_frame_mode(frame_mode, is_camera):
    """薄封装：真正的策略在 feature_extractor.resolve_static_mode（与演示脚本共用）。"""
    return resolve_static_mode(frame_mode, is_camera)


def evaluate_video(source="assets/example.mp4", engines=("rules", "svm", "svm_calibrated"),
                   window_size=7, max_frames=0, show_window=False,
                   reject_threshold=None, verbose_every=200, out_dir=None,
                   frame_mode="auto"):
    """单次遍历视频，对多个引擎逐帧评测，返回 (summary dict, recoders dict, frames)。"""
    out_dir = out_dir or os.path.join(ROOT_DIR, "data")
    cap, is_camera, src = open_source(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频源: {source}")

    static_mode, mode_desc = resolve_frame_mode(frame_mode, is_camera)
    print(f"[Mode] MediaPipe 运行模式: {mode_desc}")

    # 组装引擎
    active = []
    for name in engines:
        if name not in ENGINE_MODELS:
            print(f"[Skip] 未知引擎: {name}")
            continue
        rel = ENGINE_MODELS[name]
        if rel is None:
            clf = ExpressionClassifier()          # 内置规则引擎，无需模型文件
            active.append((name, clf, False))
            continue
        path = os.path.join(ROOT_DIR, rel)
        if not os.path.exists(path):
            print(f"[Skip] 缺少模型文件，跳过引擎 {name}: {path}")
            continue
        clf = ExpressionClassifier(model_path=path)
        active.append((name, clf, True))
    if not active:
        cap.release()
        raise ValueError("没有任何可用的评测引擎（模型文件缺失？）")

    print(f"[Eval] 视频源={source}  引擎={[a[0] for a in active]}  平滑窗口={window_size}")
    extractor = FaceFeatureExtractor(static_mode=static_mode)
    smoother = {name: TemporalSmoother(window_size) for name, _, _ in active}
    recorder = {name: FrameRecorder(engine=name, calibrated=clf.prob_is_calibrated, source=str(source))
                for name, clf, _ in active}

    t_video_start = time.time()
    frame_idx = 0
    detect_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if is_camera:
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            res = extractor.process_frame(frame)

            yaw = pitch = roll = vis_mean = vis_min = None
            if res["detected"]:
                detect_count += 1
                hp = res["head_pose"]
                yaw, pitch, roll = hp["yaw"], hp["pitch"], hp["roll"]
                vis68 = res.get("visibility_68")
                if vis68 is not None:
                    vis_mean, vis_min = float(np.mean(vis68)), float(np.min(vis68))

            latency_ms = (time.perf_counter() - t0) * 1000.0
            time_s = time.time() - t_video_start
            hud_payload = None

            for name, clf, use_ml in active:
                if res["detected"]:
                    raw_label, _raw_conf, prob_dict = clf.predict(
                        res["norm_vector"], geo_metrics=res.get("geo_metrics"),
                        blendshapes=res.get("blendshapes"), use_ml_model=use_ml,
                        reject_threshold=reject_threshold)
                    smooth_pred, smooth_conf, smoothed_probs = smoother[name].update(prob_dict)
                    thr = clf.reject_threshold if reject_threshold is None else reject_threshold
                    rejected = bool(thr is not None and float(thr) > 0 and smooth_conf < float(thr))
                    if rejected:
                        smooth_pred = UNKNOWN_LABEL
                    recorder[name].add(frame_idx, time_s, True, latency_ms, yaw=yaw, pitch=pitch,
                                       roll=roll, vis_mean=vis_mean, vis_min=vis_min,
                                       pred_raw=raw_label, pred_smooth=smooth_pred,
                                       score=smooth_conf, rejected=rejected)
                    if hud_payload is None:
                        hud_payload = (name, smooth_pred, smooth_conf, smoothed_probs, rejected, clf)
                else:
                    smoother[name].clear()
                    recorder[name].add(frame_idx, time_s, False, latency_ms)

            if show_window:
                if hud_payload is not None:
                    _n, sp, sc, sps, rej, clf = hud_payload
                    draw_hud(frame, res, sp, sc, sps, 0.0, prob_is_calibrated=clf.prob_is_calibrated,
                             rejected=rej)
                else:
                    draw_hud(frame, res, "No Face", 0.0, {}, 0.0)
                cv2.imshow("Video Eval", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q'), 27):
                    break

            frame_idx += 1
            if verbose_every and frame_idx % verbose_every == 0:
                print(f"    {frame_idx} 帧，检出 {detect_count}，平均 {latency_ms:.1f} ms/帧", flush=True)
            if max_frames and frame_idx >= max_frames:
                print(f"    达到 --max-frames {max_frames}，提前结束。")
                break
    finally:
        cap.release()
        extractor.close()
        if show_window:
            cv2.destroyAllWindows()

    if frame_idx == 0:
        raise RuntimeError("视频未读到任何帧")

    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "source": str(source),
        "is_camera": is_camera,
        "frame_mode": frame_mode,
        "frame_mode_resolved": mode_desc,
        "smoothing_window": window_size,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engines": {},
        "frame_csv": {},
    }
    for name, clf, use_ml in active:
        rec = recorder[name]
        s = rec.summary(classes=EXPRESSION_CLASSES)
        s["uses_ml_model"] = use_ml
        s["model_file"] = ENGINE_MODELS[name]
        s["reject_threshold"] = (clf.reject_threshold if reject_threshold is None
                                 else reject_threshold)
        summary["engines"][name] = s
        csv_path = os.path.join(out_dir, f"video_eval_frames_{name}.csv")
        rec.save_csv(csv_path)
        summary["frame_csv"][name] = os.path.relpath(csv_path, ROOT_DIR)

    jpath = os.path.join(out_dir, "video_eval_summary.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n================ 视频评测汇总（共 {frame_idx} 帧）================")
    for name, _, _ in active:
        print(f"\n--- 引擎 {name} ---")
        print(recorder[name].summary_text(summary["engines"][name]))
    print(f"\n[Done] JSON: {jpath}")
    for name in summary["frame_csv"]:
        print(f"       逐帧CSV({name}): {os.path.join(ROOT_DIR, summary['frame_csv'][name])}")
    return summary, recorder, frame_idx


def main():
    ap = argparse.ArgumentParser(description="视频逐帧评测（姿态/可见性/时序稳定性）")
    ap.add_argument("--source", default=os.path.join("assets", "example.mp4"),
                    help="视频文件路径或摄像头索引（默认 assets/example.mp4）")
    ap.add_argument("--engines", default="rules,svm,svm_calibrated",
                    help="要评测的引擎，逗号分隔，可选: rules, svm, mlp, svm_calibrated")
    ap.add_argument("--window", type=int, default=7, help="时序平滑窗口（默认 7 帧）")
    ap.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧（0=不限）")
    ap.add_argument("--reject-threshold", type=float, default=None,
                    help="低置信拒识阈值（默认用模型自带阈值）")
    ap.add_argument("--no-window", action="store_true", help="不弹窗（默认即无窗口，用于无显示器环境）")
    ap.add_argument("--show", action="store_true", help="弹出窗口实时查看（需要显示器）")
    ap.add_argument("--frame-mode", choices=["auto", "image", "video"], default="auto",
                    help="MediaPipe 运行模式：auto=摄像头用 video、视频文件用 image（默认）")
    ap.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "data"))
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    evaluate_video(source=args.source, engines=engines, window_size=args.window,
                   max_frames=args.max_frames, show_window=args.show,
                   reject_threshold=args.reject_threshold, out_dir=args.out_dir,
                   frame_mode=args.frame_mode)


if __name__ == "__main__":
    main()
