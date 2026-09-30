"""
realtime_demo.py
任务 C 核心：实时人脸关键点提取、头部偏转角估计与表情识别系统
支持摄像头实时输入与离线视频文件测试，集成时序概率平滑滤波以防标签闪烁。
"""

import time
import argparse
from collections import deque
import cv2
import numpy as np

from feature_extractor import FaceFeatureExtractor
from classifier import ExpressionClassifier, EXPRESSION_CLASSES, EXPRESSION_NAMES_ZH

# 表情对应的显示颜色 (BGR)
CLASS_COLORS = {
    "neutral": (220, 220, 220),   # 灰色
    "smile": (50, 205, 50),       # 嫩绿
    "surprise": (0, 215, 255),    # 金黄
    "frown": (0, 69, 255),        # 橙红
    "sad": (255, 144, 30),        # 蓝紫
    "disgust": (180, 105, 255)    # 玫紫
}

def draw_hud(frame, feat_res, pred_label, conf, smoothed_probs, fps, show_mesh=True, use_ml_model=False):
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
    cv2.rectangle(overlay, (10, 10), (330, 245), (20, 20, 20), -1)
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

    # 4. 当前表情主标签显示
    color = CLASS_COLORS.get(pred_label, (255, 255, 255))
    display_text = f"{pred_label.upper()} ({conf * 100:.1f}%)"
    cv2.putText(frame, "Expression Result:", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, display_text, (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

    # 5. 右上角置信度柱状图
    right_overlay = frame.copy()
    num_classes = max(1, len(smoothed_probs))
    panel_height = min(h - 20, 40 + num_classes * 24)
    cv2.rectangle(right_overlay, (w - 230, 10), (w - 10, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(right_overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "Relative Scores", (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_offset = 55
    for cls_name, p in smoothed_probs.items():
        c = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.putText(frame, f"{cls_name[:7]:7s}", (w - 220, y_offset + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        bar_len = int(p * 110)
        cv2.rectangle(frame, (w - 150, y_offset), (w - 150 + bar_len, y_offset + 12), c, -1)
        cv2.rectangle(frame, (w - 150, y_offset), (w - 40, y_offset + 12), (80, 80, 80), 1)
        cv2.putText(frame, f"{int(p * 100)}%", (w - 35, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y_offset += 24

    # 底部快捷键提示
    cv2.putText(frame, "[Q] Quit | [M] Mesh | [T] Toggle Engine | [S] Snapshot", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)


def run_pipeline(source=0, model_path="model_svm.pkl", window_size=7, engine="rules"):
    if window_size < 1:
        raise ValueError("window_size must be positive")
    """运行实时检测主循环"""
    print("==========================================================")
    print(" 智能科学新技术 - 实验五：人脸关键点与实时表情识别系统 DEMO")
    print(f" 视频源: {source} | 模型权重: {model_path} | 平滑窗口: {window_size} 帧")
    print(" 操作提示: [Q] 退出 | [M] 切换关键点网格 | [S] 保存当前截图")
    print("==========================================================")

    # 转换为摄像头索引或视频路径
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print("\n>>> [1/3] 正在加载 MediaPipe 人脸关键点检测模型...")
    extractor = FaceFeatureExtractor()

    print(">>> [2/3] 正在加载表情分类模型...")
    classifier = ExpressionClassifier(model_path=model_path)

    print(f">>> [3/3] 正在连接摄像头/视频源 ({source})...")
    if isinstance(source, int):
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
        print("       python realtime_demo.py --source test_video.mp4")
        return

    print("\n[SUCCESS] 摄像头已成功打开！实时画面已弹出，请在弹出的窗口中查看。")
    print(">>> 提示: 在画面窗口中按 [Q] 键可退出，按 [M] 键可切换关键点网格。")

    # 时序概率滑动窗口队列 (防止帧间预测闪烁)
    prob_queue = deque(maxlen=window_size)
    show_mesh = True
    use_ml_model = engine == "ml"
    if use_ml_model and classifier.model is None:
        cap.release()
        extractor.close()
        raise ValueError("ML engine requires a model")

    fps = 0.0
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Info] 视频播放完毕或读取结束。")
                break

            # 镜像翻转 (如果是摄像头自拍模式)
            if source == 0 or isinstance(source, int):
                frame = cv2.flip(frame, 1)

            # 1. 提取关键点、归一化向量、头部姿态与几何指标
            res = extractor.process_frame(frame)

            if res["detected"]:
                # 2. 分类器推理 (传入几何指标与 52 维 Blendshapes，获得毫秒级精准识别)
                pred_label, conf, prob_dict = classifier.predict(
                    res["norm_vector"],
                    geo_metrics=res.get("geo_metrics"),
                    blendshapes=res.get("blendshapes"),
                    use_ml_model=use_ml_model
                )

                # 3. 滑动窗口平滑概率 (动态适配 FACS 模式与 CK+ 模型的不同类别，杜绝 KeyError)
                prob_queue.append(prob_dict)
                smoothed_probs = {}
                active_classes = list(prob_dict.keys())
                for cls_name in active_classes:
                    vals = [p[cls_name] for p in prob_queue if cls_name in p]
                    smoothed_probs[cls_name] = float(np.mean(vals)) if vals else 0.0

                # 取平滑后的最优分类
                smooth_pred = max(smoothed_probs, key=smoothed_probs.get)
                smooth_conf = smoothed_probs[smooth_pred]
            else:
                prob_queue.clear()
                smooth_pred = "No Face"
                smooth_conf = 0.0
                # 保持上一次的类别列表或默认类别置零
                fallback_classes = list(prob_queue[-1].keys()) if prob_queue else EXPRESSION_CLASSES
                smoothed_probs = {c: 0.0 for c in fallback_classes}

            # 计算实时 FPS
            frame_count += 1
            if frame_count >= 10:
                t_now = time.time()
                fps = frame_count / (t_now - t_start + 1e-6)
                frame_count = 0
                t_start = t_now

            # 4. 绘制 HUD
            draw_hud(frame, res, smooth_pred, smooth_conf, smoothed_probs, fps, show_mesh=show_mesh, use_ml_model=use_ml_model)

            cv2.imshow(" SSPU - Real-time Facial Expression Recognition Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # ESC 或 Q 退出
                break
            elif key == ord('m') or key == ord('M'):
                show_mesh = not show_mesh
            elif key == ord('t') or key == ord('T'):
                if classifier.model is None:
                    print("[Engine] No ML model loaded")
                    continue
                use_ml_model = not use_ml_model
                prob_queue.clear()  # 核心：切换引擎时清空滑动窗口队列，防止跨引擎类别键冲突
                mode_name = "已加载的机器学习模型" if use_ml_model else "FACS 规则（未经校准）"
                print(f"[Engine] 已切换表情识别引擎为: {mode_name}")
            elif key == ord('s') or key == ord('S'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[Snapshot] 当前画面已保存为: {filename}")

    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
    print("[Done] 程序已安全退出。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实时人脸关键点与表情识别系统")
    parser.add_argument("--source", default=0, help="视频输入源 (0 代表默认摄像头，也可传入 mp4 视频文件路径)")
    parser.add_argument("--model", default="model_svm.pkl", help="机器学习模型文件 (.pkl)")
    parser.add_argument("--window", type=int, default=7, help="滑动平均平滑窗口大小 (默认 7 帧)")
    parser.add_argument("--engine", choices=["rules", "ml"], default="rules", help="显式选择规则或机器学习引擎")
    args = parser.parse_args()

    run_pipeline(source=args.source, model_path=args.model, window_size=args.window, engine=args.engine)
