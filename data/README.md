# Local datasets

Data and camera images are excluded from Git. Obtain datasets through their authorized channels.

## FER2013（当前默认）

- `fer2013_raw/`：从 hf-mirror 镜像获取的 FER2013 官方 parquet（train / publicTest / privateTest），约 133MB。
  `huggingface.co` 不可达，脚本自动改用 `hf-mirror.com`。
- `fer2013_landmarks.csv`：FER2013 提取结果，27,307 行 × 264 列。含
  `split`、`label`、`raw_label`、`yaw/pitch/roll`、`feat_0..135`（68 点归一化坐标）、
  `vis_0..67`（逐点几何可见性）与 `vis_mean`（每张图的平均可见性）、`bl_0..51`（blendshape）。
- `fer_blendshape_names.json`：`bl_*` 列对应的 52 个 blendshape 名称与顺序。
- `fer2013_meta.json`、`fer2013_results.json`、`fer2013_final.json`、`old_vs_new.json`：统计与评测记录。
- `fer2013_calibration.json`：概率校准与拒识阈值标定结果（ECE/Brier/覆盖率-准确率权衡）。
- `video_eval_frames_*.csv`、`video_eval_summary.json`：视频逐帧评测的逐帧记录与汇总。

### 关于可见性列（vis_0..vis_67 / vis_mean）

任务要求保存关键点"及可见性信息"。MediaPipe FaceLandmarker 的 478 点输出里
`visibility` / `presence` 字段**恒为 None**（模型不产出该量），早期实现用
`getattr(mesh[idx], "visibility", 0.0)` 兜底，导致 68 列全部被写成 0.0 ——
列名、列数、文档都在，信息量为零，且不会报错。

现在改为**几何自遮挡估计**（`feature_extractor._estimate_visibility`）：
MediaPipe 每个关键点还输出归一化相对深度 z（z 越小越靠近相机），把这些点当作
相机坐标系下的 3D 点云，对每个目标点取最近邻做 PCA 拟合局部切平面得到法线，
再用"头内参考点"定向后做背向判定，可见性 = 法线与视线夹角的余弦，取值 0~1
（0 = 背对相机/被自遮挡，1 = 正对相机）。

实测（全表 27,307 行）：68 列**无一是常数**，单列取值数达 10³ 量级；
逐行平均可见性与 |yaw| 的相关系数为负（正视时约 0.68，|yaw|>25° 时约 0.56），
说明它携带真实的遮挡信息。

注意两点：
1. 它是**几何自遮挡估计，不是模型置信度**，衡量"该点当前是否朝向相机"，
   不等价于关键点定位精度；
2. 它只新增信息，不改变原有列 —— 需要真实可见性重算时用
   `python src/pipeline/recalc_visibility_columns.py`（带逐行对齐校验与 `.bak` 备份）。

若要**模型原生**的逐点置信度，需换用输出该字段的模型（如 HRFFA 的 ONNX 方案），
本项目未内置对应权重。

## CK+（历史基线）

- `ck_plus_landmarks.csv`：本地提取 CK+ 特征，852 样本 / 106 subjects；无 neutral 类。
  含 `subject_id` 与 `clip_id`（人员 × 视频序列，从文件名 `S010_004_00000017` 解析），
  可做跨人 / 跨片段两种粒度的 GroupKFold。
- `ck_raw.parquet`：本地源图像。Not redistributed.

现有 CK+ 特征 CSV 与原始 parquet **不在本目录**（从未提交进 Git），
报告中的 CK+ 指标无法用当前仓库复现。

For a usable neutral class, collect real samples from multiple consenting subjects:
`python src/apps/collect_and_train.py --action collect --subject person01 --csv data/custom.csv`
Use a new subject ID for each person and repeat across lighting, head poses and sessions.
采集时可用 `--session` 指定拍摄时段，或按 `[N]` 在采集过程中切分新片段。
每行都会写入 `subject_id` / `session_id` / `clip_id`，因此可以按
`--group_col clip_id` 同时防住"同一人的连续帧跨折"这一最隐蔽的泄漏源：
`python src/apps/collect_and_train.py --action train --csv data/custom.csv --group_col clip_id`
