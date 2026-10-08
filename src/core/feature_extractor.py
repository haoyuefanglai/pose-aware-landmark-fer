"""
feature_extractor.py
面向复杂头部姿态的人脸关键点提取、几何归一化、头部姿态角 (Yaw, Pitch, Roll) 估计
与逐点几何可见性估计。
基于最新 MediaPipe Tasks (Vision FaceLandmarker) 架构，内置自动模型配置。

static_mode 决定 MediaPipe 运行模式（IMAGE 逐图 / VIDEO 跨帧跟踪），
可见性由 3D 关键点的局部法线做背向判定得到（模型本身不输出 visibility 字段）。
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
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe 478 点映射到经典 68 个人脸关键点的标准索引表
# Legacy mapping: not anatomically equivalent to dlib 68; retained for model compatibility.
# 注意：本表是历史自建映射，与标准 68 点语义并不等价（详见 AUDIT.md）。
# 第 30 号点实际映射到 MediaPipe 195，并非解剖学鼻尖；为保持既有 CSV / 模型权重
# 兼容，此处不改动映射，仅由 _normalize_landmarks 用中性名字引用该锚点。
MEDIAPIPE_TO_68 = [
    # 下巴轮廓 (0-16)
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
    # 右眉毛 (17-21, 画面左侧视角)
    70, 63, 105, 66, 107,
    # 左眉毛 (22-26)
    336, 296, 334, 293, 300,
    # 鼻梁与鼻底 (27-35, 共9点)
    168, 6, 197, 195, 5, 4, 1, 19, 94,
    # 右眼 (36-41, 共6点)
    33, 160, 158, 133, 153, 144,
    # 左眼 (42-47, 共6点)
    362, 385, 387, 263, 373, 380,
    # 嘴唇外轮廓 (48-59, 共12点)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # 嘴唇内轮廓 (60-67, 共8点)
    78, 81, 13, 311, 308, 402, 14, 178
]
assert len(MEDIAPIPE_TO_68) == 68, f"MEDIAPIPE_TO_68 必须严格包含 68 个关键点，当前为 {len(MEDIAPIPE_TO_68)}"

MEDIAPIPE_TO_68_IDX = np.asarray(MEDIAPIPE_TO_68, dtype=np.int64)

# 几何可见性估计的邻域大小：对每个目标点在 MediaPipe 478 点 3D 网格中取最近的
# VIS_NEIGHBORS 个点做局部平面拟合，用拟合平面的法线判断该点的朝向是否背对相机。
VIS_NEIGHBORS = 10

# 单张图的平均可见性低于该值时，认为人脸已严重偏离正面（供调用方做预警/拒识参考）
FRONTAL_VIS_REFERENCE = 0.6


def resolve_static_mode(frame_mode="auto", is_camera=False):
    """按输入源性质决定 MediaPipe 运行模式，返回 (static_mode, 说明文字)。

    实测（assets/example.mp4，40 张互不相关的 FER2013 静态图拼接、帧间硬切）：
        IMAGE 模式（static_mode=True，逐帧独立检测）检出 444/530 = 83.8%
        VIDEO 模式（static_mode=False，跨帧跟踪）前 60 帧仅检出 16.7%
    原因是 VIDEO 模式依赖帧间连续性，遇到硬切 / 蒙太奇 / 快速跳剪会持续丢脸。

    因此 auto 策略：摄像头流（时间连续）用 VIDEO（跟踪更稳、关键点抖动更小），
    离线视频文件用 IMAGE（对剪辑内容更鲁棒）。可用 frame_mode 显式覆盖。
    """
    if frame_mode == "image":
        return True, "image（逐帧独立检测，对硬切/蒙太奇更鲁棒）"
    if frame_mode == "video":
        return False, "video（跨帧跟踪，适合时间连续的摄像头或单镜头视频）"
    if is_camera:
        return False, "video（auto：摄像头流时间连续，启用跨帧跟踪）"
    return True, "image（auto：离线视频文件按逐帧独立检测，避免硬切导致持续丢脸）"

# [已移除] 原实现用一个 6 点通用 3D 人脸模型 (MODEL_POINTS_3D) + cv2.solvePnP 解算头部姿态。
# 该模型定义在"y 轴朝上"的坐标系（眼睛 y=+170、下巴 y=-330），而图像坐标 y 轴朝下，
# solvePnP 只能引入一次绕 x 轴的 180° 翻转来补偿，于是旋转矩阵出现 R[2,2] ≈ -1，
# 使 pitch = atan2(R[2,1], R[2,2]) 被顶到 ±180° 附近。
# 实测（FER2013 留出集 73 张）：pitch 中位数 -167.5°，98.6% 的样本 |pitch| > 150°，
# 对旋转矩阵转置后仍是 163.5° / 94.5%，说明问题不在分解公式，而在模型坐标系。
# 现改为直接使用 MediaPipe FaceLandmarker 输出的 facial_transformation_matrixes
# （内部基于完整规范人脸模型拟合，且不需要自造相机内参），
# 实测 pitch 中位数回到 -6.7°，|pitch| > 150° 的比例降到 0%。

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                              # 项目根目录
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "models", "face_landmarker.task")


class FaceFeatureExtractor:
    """人脸关键点与几何特征提取器

    static_mode:
      - True  —— 逐张独立图像处理（MediaPipe IMAGE 模式，内部无跨帧跟踪），
                 适用于离线批量提取特征（FER2013 / CK+ 等）。
      - False —— 视频流处理（MediaPipe VIDEO 模式，内部做跨帧跟踪），
                 适用于摄像头 / 视频文件，帧间更稳定；调用方无需自己管时间戳。
    """

    def __init__(self, model_asset_path=None, static_mode=False, max_faces=1):
        if model_asset_path is None:
            model_asset_path = DEFAULT_MODEL_PATH
        elif not os.path.isabs(model_asset_path):
            model_asset_path = os.path.join(ROOT_DIR, model_asset_path)

        # 确保模型文件存在
        if not os.path.exists(model_asset_path):
            raise FileNotFoundError(
                f"[Error] 找不到模型文件: {model_asset_path}！请确保 models/face_landmarker.task 存在。"
            )

        # 核心优化：以二进制内存流读取模型文件，避免 Windows 下中文路径 (如 '智能科学新技术') 导致底层 C++ 报错崩溃闪退
        with open(model_asset_path, "rb") as f:
            model_bytes = f.read()

        # static_mode 真正落到 MediaPipe 的 running_mode 上：
        # 修复前该参数只被接收、从未生效，批量脚本"以为是图像模式"、实时脚本
        # 也无法使用视频跟踪，且调用方拿不到任何提示。
        self.static_mode = bool(static_mode)
        self.running_mode = (vision.RunningMode.IMAGE if self.static_mode
                             else vision.RunningMode.VIDEO)
        self._last_timestamp_ms = -1

        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=max_faces
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def close(self):
        self.detector.close()

    def _next_timestamp_ms(self, timestamp_ms=None):
        """生成严格单调递增的时间戳（VIDEO 模式强制要求，否则 MediaPipe 直接报错）。"""
        if timestamp_ms is not None:
            ts = int(timestamp_ms)
        else:
            ts = int(time.monotonic() * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def process_frame(self, frame_bgr, timestamp_ms=None):
        """
        处理输入帧，提取关键点、可见性、头部姿态与几何归一化特征

        timestamp_ms: 仅 VIDEO 模式使用。留空则按单调时钟自动生成，调用方无需关心。
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.static_mode:
            result = self.detector.detect(mp_image)
        else:
            result = self.detector.detect_for_video(mp_image, self._next_timestamp_ms(timestamp_ms))

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return {"detected": False}

        raw_landmarks = result.face_landmarks[0]

        # 1. 映射为 68 点像素坐标
        pts_68 = np.zeros((68, 2), dtype=np.float32)
        for i, idx in enumerate(MEDIAPIPE_TO_68):
            lm = raw_landmarks[idx]
            pts_68[i] = [lm.x * w, lm.y * h]

        # 2. 估计头部姿态角 (MediaPipe 面部变换矩阵)
        head_pose = self._estimate_head_pose(result)

        # 3. 几何尺度与中心归一化 (136维)
        norm_vector = self._normalize_landmarks(pts_68)

        # 4. 计算面部动作几何指标 (EAR, MAR, 微笑曲率, 眉间距等)
        geo_metrics = self._calculate_geometric_metrics(pts_68)

        # 5. 读取可选的 Blendshapes (若支持)
        blendshape_dict = {}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for cat in result.face_blendshapes[0]:
                blendshape_dict[cat.category_name] = float(cat.score)

        # 6. 逐点几何可见性 (0~1)：由 MediaPipe 输出的 3D 坐标做局部法线 + 背向判定
        visibility_68 = self._estimate_visibility(raw_landmarks, w, h)

        return {
            "detected": True,
            "landmarks_68": pts_68,
            "norm_vector": norm_vector,
            "head_pose": head_pose,
            "geo_metrics": geo_metrics,
            "blendshapes": blendshape_dict,
            "visibility_68": visibility_68,
            "raw_mesh": raw_landmarks
        }

    def _estimate_head_pose(self, result):
        """从 MediaPipe 的面部变换矩阵解算头部姿态角。

        变换矩阵把 MediaPipe 的规范人脸模型映射到相机坐标系；取其旋转部分并转置，
        即得到头部相对相机的朝向，再按标准 XYZ 顺序分解为欧拉角：
        Yaw 左右偏航、Pitch 上下俯仰、Roll 侧倾翻滚。单位为度。

        相比原来基于 solvePnP 的实现，这里复用了 MediaPipe 内部对完整 478 点规范
        人脸模型的拟合结果，既不需要自造相机内参，也避开了 3D 模型坐标系与图像
        坐标系 y 轴方向不一致造成的 180° 翻转（详见文件顶部说明）。
        """
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if not matrices:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        mat = np.asarray(matrices[0], dtype=np.float64).reshape(4, 4)
        rot_mat = mat[:3, :3].T

        sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        if sy < 1e-6:
            # 万向锁：yaw 与 roll 退化为同一自由度，按约定置 roll = 0
            pitch = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
            yaw = math.atan2(-rot_mat[2, 0], sy)
            roll = 0.0
        else:
            pitch = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
            yaw = math.atan2(-rot_mat[2, 0], sy)
            roll = math.atan2(rot_mat[1, 0], rot_mat[0, 0])

        return {
            "yaw": float(np.degrees(yaw)),
            "pitch": float(np.degrees(pitch)),
            "roll": float(np.degrees(roll)),
        }

    def _estimate_visibility(self, landmarks, w, h, target_idx=MEDIAPIPE_TO_68_IDX):
        """逐点几何可见性估计 (0 = 背对相机/被自遮挡, 1 = 正对相机)。

        为什么需要自己算：
            MediaPipe FaceLandmarker 的 478 点输出里 `visibility` / `presence`
            字段**恒为 None**（模型不产出该量）。原实现用
            `getattr(mesh[idx], "visibility", 0.0)` 兜底，于是 68 列 vis_* 全部
            被写成 0.0 —— 列在、文档在、信息量为零（详见 data/README.md 与 AUDIT.md）。

        实现原理（纯几何，不依赖额外模型权重）：
            MediaPipe 每个关键点除 x / y 外还输出归一化相对深度 z（z 越小越靠近
            相机，量级与 x 相当）。于是可以直接把这些点当作相机坐标系下的 3D 点云：
              1. 对每个目标点，在 478 点中取最近邻的 VIS_NEIGHBORS 个点，PCA 拟合
                 局部切平面，取最小特征值对应的方向作为该点法线；
              2. 用"头内参考点"定向法线（参考点 = 点云质心沿 +z 后移，即头部内部），
                 使法线指向体表外侧；
              3. 可见性 = 法线与相机视线方向夹角的余弦 max(0, -n_z)，即"背向判定"
                 (back-face culling)：正对相机的点接近 1，侧转到背面的点降到 0。

        实测（FER2013 publicTest 抽样 108 张）：
            面部左右半区平均可见性之差与头部 yaw 的相关系数为 -0.955，
            整体平均可见性随 |yaw| 单调下降（|yaw|<=5° 时 0.682 → |yaw|>25° 时 0.561），
            说明该列携带真实的遮挡信息，而非常数。

        注意：这是几何自遮挡估计，不是模型的置信度。它衡量"这个点在当前姿态下
        是否朝向相机"，不能替代关键点定位精度指标。
        """
        n_all = len(landmarks)
        if n_all < 4:
            return np.ones(len(target_idx) if target_idx is not None else n_all, dtype=np.float32)

        # 归一化坐标 -> 相机坐标系 3D 点（z 的量级与 x 一致，故按图像宽度缩放）
        pts = np.empty((n_all, 3), dtype=np.float64)
        pts[:, 0] = [lm.x for lm in landmarks]
        pts[:, 1] = [lm.y for lm in landmarks]
        pts[:, 2] = [lm.z for lm in landmarks]
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts[:, 2] *= w

        tgt = np.arange(n_all, dtype=np.int64) if target_idx is None else np.asarray(target_idx, dtype=np.int64)
        q = pts[tgt]
        k = min(VIS_NEIGHBORS, n_all - 1)
        if k < 3:
            return np.ones(len(tgt), dtype=np.float32)

        # 目标点到全部点的距离（(len(tgt), n_all)，规模很小）
        d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        nb_idx = np.argpartition(d2, k, axis=1)[:, 1:k + 1]

        # 局部切平面：邻域协方差矩阵最小特征值方向即法线
        nb = pts[nb_idx] - q[:, None, :]                 # (m, k, 3)
        cov = nb.transpose(0, 2, 1) @ nb                 # (m, 3, 3)
        _, evec = np.linalg.eigh(cov)                    # 特征值升序
        normals = evec[:, :, 0]

        # 定向：头内参考点 = 质心沿 +z 后移（远离相机，z 越小越靠近相机）
        scale = float(np.linalg.norm(pts.max(0) - pts.min(0))) + 1e-9
        ref = pts.mean(0) + np.array([0.0, 0.0, 0.4 * scale])
        orient = q - ref
        flip = (normals * orient).sum(-1) < 0
        normals[flip] *= -1.0

        nz = -normals[:, 2] / (np.linalg.norm(normals, axis=1) + 1e-12)
        return np.clip(nz, 0.0, 1.0).astype(np.float32)

    def _normalize_landmarks(self, pts_68):
        """
        几何归一化核心步骤:
        1. 中心化: 以第 30 号锚点为原点
        2. 尺度归一化: 除以左右外眼角 (第36点与第45点) 的瞳距

        命名说明：第 30 号点在历史自建映射中映射到 MediaPipe 195，并非解剖学鼻尖
        （见 AUDIT.md）。变量名从 nose_tip 改为 anchor_pt，只改名字与注释、不改数值，
        以保持既有 CSV / 模型权重兼容。
        """
        anchor_pt = pts_68[30]
        centered = pts_68 - anchor_pt

        # 瞳距 (右眼外角 36 与 左眼外角 45)
        inter_ocular_dist = np.linalg.norm(pts_68[36] - pts_68[45]) + 1e-6
        normalized = centered / inter_ocular_dist

        # 展平为 136 维特征向量
        return normalized.flatten()

    def _calculate_geometric_metrics(self, pts_68):
        """计算面部动作几何指标 (EAR, MAR, 嘴唇弧度等)"""
        # 瞳距作为基准尺度
        inter_ocular_dist = np.linalg.norm(pts_68[36] - pts_68[45]) + 1e-6

        # 1. 眼睛开合度 (EAR: Eye Aspect Ratio)
        right_ear = (np.linalg.norm(pts_68[37] - pts_68[41]) + np.linalg.norm(pts_68[38] - pts_68[40])) / (2.0 * np.linalg.norm(pts_68[36] - pts_68[39]) + 1e-6)
        left_ear = (np.linalg.norm(pts_68[43] - pts_68[47]) + np.linalg.norm(pts_68[44] - pts_68[46])) / (2.0 * np.linalg.norm(pts_68[42] - pts_68[45]) + 1e-6)
        ear = (right_ear + left_ear) / 2.0

        # 2. 嘴巴纵横比 (MAR: Mouth Aspect Ratio)
        mouth_width = np.linalg.norm(pts_68[48] - pts_68[54]) + 1e-6
        mouth_height = (np.linalg.norm(pts_68[51] - pts_68[57]) + np.linalg.norm(pts_68[62] - pts_68[66])) / 2.0
        mar = mouth_height / mouth_width

        # 3. 嘴角上扬程度 (Smile ratio: 左右嘴角相对于嘴唇中心的 y 轴偏移量)
        mouth_center_y = (pts_68[51][1] + pts_68[57][1]) / 2.0
        corners_avg_y = (pts_68[48][1] + pts_68[54][1]) / 2.0
        smile_curvature = (mouth_center_y - corners_avg_y) / inter_ocular_dist
        mouth_width_ratio = mouth_width / inter_ocular_dist

        # 4. 眉间距 (Brow distance)
        inner_brow_dist = np.linalg.norm(pts_68[21] - pts_68[22]) / inter_ocular_dist

        # 5. 眉眼距离 (Eyebrow to Eye distance)
        brow_eye_dist = ((pts_68[37][1] - pts_68[19][1]) + (pts_68[44][1] - pts_68[24][1])) / (2.0 * inter_ocular_dist)

        return {
            "ear": float(ear),
            "mar": float(mar),
            "smile_curvature": float(smile_curvature),
            "mouth_width_ratio": float(mouth_width_ratio),
            "inner_brow_dist": float(inner_brow_dist),
            "brow_eye_dist": float(brow_eye_dist)
        }
