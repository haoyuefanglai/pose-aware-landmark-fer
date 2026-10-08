"""
realtime_demo.py
任务 C 核心：实时人脸关键点提取、头部偏转角估计与表情识别系统
支持摄像头实时输入与离线视频文件测试，集成时序概率平滑滤波以防标签闪烁。

置信度的两种口径（HUD 会明确标注是哪种）：
  - 未校准相对分数：SVM 用 probability=False 训练，输出是 decision_function 手写
    softmax 的结果，只能横向比较，不能当概率读。
  - 已校准概率：由 src/experiments/calibrate_model.py 用 Platt 标定得到，
    可以解释为"有把握的程度"，并据此做低置信拒识（显示 UNKNOWN）。

需要"逐帧落盘 + 视频级汇总"（任务C 出数）时，用
  python src/experiments/evaluate_video.py --source assets/example.mp4 --no-window
本脚本也支持 --record-csv 在演示的同时记录逐帧结果，两者共用同一套记录口径。

帧处理模式由 --frame-mode 控制（默认 auto）：摄像头用 VIDEO 跨帧跟踪，
离线视频文件用 IMAGE 逐帧独立检测。实测硬切/蒙太奇视频在 VIDEO 模式下检出率会
从 83.8% 掉到 16.7%（详见 feature_extractor.resolve_static_mode 的说明）。
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
from collections import deque
import cv2
import numpy as np

from feature_extractor import FaceFeatureExtractor, resolve_static_mode
from classifier import ExpressionClassifier, EXPRESSION_CLASSES, EXPRESSION_NAMES_ZH, UNKNOWN_LABEL
from frame_recorder import FrameRecorder

# 表情对应的显示颜色 (BGR)
CLASS_COLORS = {
    "neutral": (220, 220, 220),   # 灰色
    "smile": (50, 205, 50),       # 嫩绿
    "surprise": (0, 215, 255),    # 金黄
    "frown": (0, 69, 255),        # 橙红
    "sad": (255, 144, 30),        # 蓝紫
    "disgust": (180, 105, 255),   # 玫紫
    "unknown": (128, 128, 128)    # 灰（低置信拒识）
}


class TemporalSmoother:
    """滑动窗口概率平滑：抑制逐帧预测闪烁。

    放在这里而不是各脚本里各写一份，是为了让 realtime_demo 与
    experiments/evaluate_video 的"平滑后标签"口径完全一致，两处数字可直接比较。
    """

    def __init__(self, window_size=7):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self.queue = deque(maxlen=self.window_size)

    def clear(self):
        self.queue.clear()

    def update(self, prob_dict):
        """喂入一帧的类别->分数，返回 (平滑后标签, 平滑后分数, 平滑后分布)。"""
        self.queue.append(prob_dict)
        active = list(prob_dict.keys())
        smoothed = {}
        for cls_name in active:
            vals = [p[cls_name] for p in self.queue if cls_name in p]
            smoothed[cls_name] = float(np.mean(vals)) if vals else 0.0
        best = max(smoothed, key=smoothed.get)
        return best, smoothed[best], smoothed


def draw_hud(frame, feat_res, pred_label, conf, smoothed_probs, fps, show_mesh=True,
             use_ml_model=False, prob_is_calibrated=False, rejected=False):
    """在视频帧上绘制专业、清晰的 HUD 信息指示看板"""
    h, w = frame.shape[:2]

    # 1. 绘制 68 个关键点与面部轮廓
    if show_mesh and feat_res.get("detected"):
        pts = feat_res["landmarks_68"].astype(np.int32)
        # 绘制眼睛、眉毛、嘴唇、下巴
        for pt in pts:
            cv2.circle(frame, (pt[0], pt[1]), 2, (0, 255, 255), -1)

        # 简单连接眼睛和嘴唇轮廓
        cv2.polylines(frame, [pts[36:42]], isClosed=True, color=(255, 255, 0), thickness=1)
        cv2.polylines(frame, [pts[42:48]], isClosed=True, color=(255, 255, 0), thickness=1)
        cv2.polylines(frame, [pts[48:60]], isClosed=True, color=(0, 255, 0), thickness=1)

    # 2. 左上角信息面板背景 (半透明黑底)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, 260), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 3. 头部姿态角度 (Yaw, Pitch, Roll)
    pose = feat_res.get("head_pose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}) if feat_res.get("detected") else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

    abs_yaw = abs(yaw)
    if abs_yaw < 25:
        pose_status = "Frontal (Normal)"
        pose_color = (0, 255, 0)
    elif abs_yaw < 45:
        pose_status = "Moderate Turn"
        pose_color = (0, 215, 255)
    else:
        pose_status = "Extreme Angle (Unreliable)"
        pose_color = (0, 0, 255)

    mode_text = "SVM/MLP Model" if use_ml_model else "FACS Heuristic Rules"
    cv2.putText(frame, f"Engine: {mode_text}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Head Yaw  : {yaw:+.1f} deg", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Head Pitch: {pitch:+.1f} deg", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Pose Mode : {pose_status}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)

    # 3b. 可见性均值：真实几何自遮挡指标（0=背对相机，1=正对相机）
    vis_mean = None
    if feat_res.get("detected") and feat_res.get("visibility_68") is not None:
        vis_mean = float(np.mean(feat_res["visibility_68"]))
        vis_color = (0, 255, 0) if vis_mean > 0.6 else ((0, 215, 255) if vis_mean > 0.4 else (0, 0, 255))
        cv2.putText(frame, f"Vis Mean  : {vis_mean:.2f}", (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vis_color, 1)

    # 4. 当前表情主标签显示
    if rejected:
        color = CLASS_COLORS[UNKNOWN_LABEL]
        display_text = f"UNKNOWN ({conf * 100:.1f}% < thr)"
    else:
        color = CLASS_COLORS.get(pred_label, (255, 255, 255))
        display_text = f"{pred_label.upper()} ({conf * 100:.1f}%)"
    label_kind = "probability" if prob_is_calibrated else "score (uncalibrated)"
    cv2.putText(frame, f"Expression Result [{label_kind}]:", (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, display_text, (20, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 5. 右上角分数/概率柱状图
    right_overlay = frame.copy()
    num_classes = max(1, len(smoothed_probs))
    panel_height = min(h - 20, 40 + num_classes * 24)
    cv2.rectangle(right_overlay, (w - 250, 10), (w - 10, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(right_overlay, 0.7, frame, 0.3, 0, frame)

    panel_title = "Calibrated Probabilities" if prob_is_calibrated else "Relative Scores (uncalibrated)"
    cv2.putText(frame, panel_title, (w - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    y_offset = 55
    for cls_name, p in smoothed_probs.items():
        c = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.putText(frame, f"{cls_name[:7]:7s}", (w - 240, y_offset + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        bar_len = int(p * 110)
        cv2.rectangle(frame, (w - 170, y_offset), (w - 170 + bar_len, y_offset + 12), c, -1)
        cv2.rectangle(frame, (w - 170, y_offset), (w - 60, y_offset + 12), (80, 80, 80), 1)
        cv2.putText(frame, f"{int(p * 100)}%", (w - 55, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y_offset += 24

    # 底部快捷键提示
    cv2.putText(frame, "[Q] Quit | [M] Mesh | [T] Toggle Engine | [S] Snapshot | [R] Reset Smoothing",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)


def run_pipeline(source=0, model_path="models/model_svm.pkl", window_size=7, engine="rules",
                 record_csv=None, reject_threshold=None, show_window=True, max_frames=0,
                 frame_mode="auto"):
    """运行实时检测主循环

    record_csv:      给出路径时，把逐帧结果（帧号/时间/姿态/可见性/预测/分数）
                     落盘成 CSV，并在退出时打印视频级汇总（任务C 出数用）。
    reject_threshold: 低置信拒识阈值；留空则用模型自带阈值（未校准模型通常为 None）。
    show_window:     False 时不弹窗（无显示器/无摄像头的环境也能跑完并出数）。
    max_frames:      >0 时处理指定帧数后自动退出。
    frame_mode:      auto/image/video，决定 MediaPipe 运行模式；默认 auto ——
                     摄像头用 VIDEO 跟踪，离线视频文件用 IMAGE 逐帧独立检测
                     （实测硬切视频在 VIDEO 模式下检出率会大幅下降，详见 resolve_static_mode）。
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    print("==========================================================")
    print(" 智能科学新技术 - 实验五：人脸关键点与实时表情识别系统 DEMO")
    print(f" 视频源: {source} | 模型权重: {model_path} | 平滑窗口: {window_size} 帧")
    print(" 操作提示: [Q] 退出 | [M] 切换关键点网格 | [T] 切换引擎 | [S] 保存截图 | [R] 重置平滑")
    print("==========================================================")

    # 转换为摄像头索引或视频路径
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    is_camera = isinstance(source, int)

    static_mode, mode_desc = resolve_static_mode(frame_mode, is_camera)
    print("\n>>> [1/3] 正在加载 MediaPipe 人脸关键点检测模型...")
    print(f"    运行模式: {mode_desc}")
    extractor = FaceFeatureExtractor(static_mode=static_mode)

    print(">>> [2/3] 正在加载表情分类模型...")
    classifier = ExpressionClassifier(model_path=model_path)

    print(f">>> [3/3] 正在连接摄像头/视频源 ({source})...")
    if is_camera:
        # Windows 下优先使用 DirectShow 快速开启摄像头 (毫秒级响应，避免 MSMF 卡顿)
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        extractor.close()
        print(f"\n[Error] 无法打开输入源: {source}。")
        print("[提示] 若当前环境无物理摄像头，请传入测试视频路径，例如:")
        print("       python src/apps/realtime_demo.py --source assets/example.mp4 --model models/model_fer_svm.pkl --engine ml")
        return

    if is_camera:
        print("\n[SUCCESS] 摄像头已成功打开！实时画面已弹出，请在弹出的窗口中查看。")
        print(">>> 提示: 在画面窗口中按 [Q] 键可退出，按 [M] 键可切换关键点网格。")
    else:
        print(f"\n[SUCCESS] 视频文件已打开: {source}（离线模式，不使用摄像头）。")
        print(">>> 提示: 播放到结尾会自动退出，也可在画面窗口中按 [Q] 键提前退出。")

    # 时序概率滑动窗口队列 (防止帧间预测闪烁)
    smoother = TemporalSmoother(window_size)
    show_mesh = True
    use_ml_model = engine == "ml"
    if use_ml_model and classifier.model is None:
        cap.release()
        extractor.close()
        raise ValueError("ML engine requires a model")
    if classifier.model is not None:
        conf_kind = "已校准概率" if classifier.prob_is_calibrated else "未校准相对分数"
        thr = classifier.reject_threshold if reject_threshold is None else reject_threshold
        print(f"[Model] 置信度性质: {conf_kind}；拒识阈值: "
              f"{'-' if thr is None else format(float(thr), '.2f')}")

    recorder = None
    if record_csv:
        recorder = FrameRecorder(engine=engine, calibrated=classifier.prob_is_calibrated,
                                 source=str(source))

    fps = 0.0
    frame_count = 0
    processed = 0
    t_start = time.time()
    t_video_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Info] 视频播放完毕或读取结束。")
                break

            # 镜像翻转 (如果是摄像头自拍模式)
            if is_camera:
                frame = cv2.flip(frame, 1)

            t_frame0 = time.perf_counter()
            # 1. 提取关键点、归一化向量、头部姿态、可见性与几何指标
            res = extractor.process_frame(frame)

            yaw = pitch = roll = vis_mean = vis_min = None
            score = None
            rejected = False
            raw_label = None
            if res["detected"]:
                # 2. 分类器推理 (传入几何指标与 52 维 Blendshapes，获得毫秒级精准识别)
                raw_label, conf, prob_dict = classifier.predict(
                    res["norm_vector"],
                    geo_metrics=res.get("geo_metrics"),
                    blendshapes=res.get("blendshapes"),
                    use_ml_model=use_ml_model,
                    reject_threshold=reject_threshold
                )

                # 3. 滑动窗口平滑分数 (动态适配 FACS 模式与 CK+ 模型的不同类别，杜绝 KeyError)
                smooth_pred, smooth_conf, smoothed_probs = smoother.update(prob_dict)

                # 3b. 拒识判定放在平滑之后：逐帧分数会抖动，按平滑后的把握判定才不会闪烁
                thr = classifier.reject_threshold if reject_threshold is None else reject_threshold
                if thr is not None and float(thr) > 0 and smooth_conf < float(thr):
                    smooth_pred = UNKNOWN_LABEL
                    rejected = True

                hp = res["head_pose"]
                yaw, pitch, roll = hp["yaw"], hp["pitch"], hp["roll"]
                vis68 = res.get("visibility_68")
                if vis68 is not None:
                    vis_mean = float(np.mean(vis68))
                    vis_min = float(np.min(vis68))
                score = smooth_conf
            else:
                smoother.clear()
                smooth_pred = "No Face"
                smooth_conf = 0.0
                # 保持上一次的类别列表或默认类别置零
                fallback_classes = list(smoother.queue[-1].keys()) if smoother.queue else EXPRESSION_CLASSES
                smoothed_probs = {c: 0.0 for c in fallback_classes}

            latency_ms = (time.perf_counter() - t_frame0) * 1000.0

            if recorder is not None:
                recorder.add(processed, time.time() - t_video_start, bool(res["detected"]),
                             latency_ms, yaw=yaw, pitch=pitch, roll=roll,
                             vis_mean=vis_mean, vis_min=vis_min,
                             pred_raw=raw_label, pred_smooth=smooth_pred,
                             score=score, rejected=rejected)
            processed += 1

            # 计算实时 FPS
            frame_count += 1
            if frame_count >= 10:
                t_now = time.time()
                fps = frame_count / (t_now - t_start + 1e-6)
                frame_count = 0
                t_start = t_now

            # 4. 绘制 HUD
            draw_hud(frame, res, smooth_pred, smooth_conf, smoothed_probs, fps,
                     show_mesh=show_mesh, use_ml_model=use_ml_model,
                     prob_is_calibrated=classifier.prob_is_calibrated, rejected=rejected)

            if show_window:
                cv2.imshow(" SSPU - Real-time Facial Expression Recognition Demo", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0xFF  # 无窗口模式：不阻塞，也不响应按键

            if key == ord('q') or key == ord('Q') or key == 27:  # ESC 或 Q 退出
                break
            elif key == ord('m') or key == ord('M'):
                show_mesh = not show_mesh
            elif key == ord('r') or key == ord('R'):
                smoother.clear()
                print("[Smoothing] 已清空时序平滑窗口")
            elif key == ord('t') or key == ord('T'):
                if classifier.model is None:
                    print("[Engine] No ML model loaded")
                    continue
                use_ml_model = not use_ml_model
                smoother.clear()  # 核心：切换引擎时清空滑动窗口队列，防止跨引擎类别键冲突
                mode_name = "已加载的机器学习模型" if use_ml_model else "FACS 规则（未经校准）"
                print(f"[Engine] 已切换表情识别引擎为: {mode_name}")
            elif key == ord('s') or key == ord('S'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[Snapshot] 当前画面已保存为: {filename}")

            if max_frames and processed >= max_frames:
                print(f"[Info] 已达到 --max-frames {max_frames}，提前退出。")
                break

    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()

    if recorder is not None and recorder.rows:
        recorder.save_csv(record_csv)
        summary = recorder.summary(classes=EXPRESSION_CLASSES)
        print("\n================ 逐帧记录已保存 ================")
        print(f"  CSV: {record_csv}  ({len(recorder.rows)} 帧)")
        print(recorder.summary_text(summary))

    print("[Done] 程序已安全退出。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实时人脸关键点与表情识别系统")
    parser.add_argument("--source", default=0, help="视频输入源 (0 代表默认摄像头，也可传入 mp4 视频文件路径)")
    parser.add_argument("--model", default="models/model_svm.pkl", help="机器学习模型文件 (.pkl)，相对路径按项目根目录解析")
    parser.add_argument("--window", type=int, default=7, help="滑动平均平滑窗口大小 (默认 7 帧)")
    parser.add_argument("--engine", choices=["rules", "ml"], default="rules", help="显式选择规则或机器学习引擎")
    parser.add_argument("--record-csv", default=None, help="把逐帧结果写入该 CSV，并在退出时打印视频级汇总")
    parser.add_argument("--reject-threshold", type=float, default=None,
                        help="低置信拒识阈值（默认用模型自带阈值；0 表示关闭拒识）")
    parser.add_argument("--no-window", action="store_true", help="不弹窗（无显示器环境跑批用）")
    parser.add_argument("--max-frames", type=int, default=0, help="处理多少帧后自动退出（0=不限）")
    parser.add_argument("--frame-mode", choices=["auto", "image", "video"], default="auto",
                        help="MediaPipe 运行模式：auto=摄像头用 video、视频文件用 image（默认）")
    args = parser.parse_args()

    run_pipeline(source=args.source, model_path=args.model, window_size=args.window,
                 engine=args.engine, record_csv=args.record_csv,
                 reject_threshold=args.reject_threshold,
                 show_window=not args.no_window, max_frames=args.max_frames,
                 frame_mode=args.frame_mode)
