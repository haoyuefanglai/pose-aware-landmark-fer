# Local datasets

Data and camera images are excluded from Git. Obtain datasets through their authorized channels.

## FER2013（当前默认）

- `fer2013_raw/`：从 hf-mirror 镜像获取的 FER2013 官方 parquet（train / publicTest / privateTest），约 133MB。
  `huggingface.co` 不可达，脚本自动改用 `hf-mirror.com`。
- `fer2013_landmarks.csv`：FER2013 提取结果，27,307 行 × 263 列。含
  `split`、`label`、`raw_label`、`yaw/pitch/roll`、`feat_0..135`（68 点归一化坐标）、
  `vis_0..67`（可见性，见下方说明）、`bl_0..51`（blendshape）。
- `fer_blendshape_names.json`：`bl_*` 列对应的 52 个 blendshape 名称与顺序。
- `fer2013_meta.json`、`fer2013_results.json`、`fer2013_final.json`、`old_vs_new.json`：统计与评测记录。

### 关于可见性列

任务要求保存关键点及可见性信息，因此保留了 `vis_0..67`。但经核查，
MediaPipe FaceLandmarker 输出的 68 个点**全部为 0**（68/68 列恒为常数，只有 1 个取值），
不携带可用信息；对照实验也确认加入这 68 列后指标完全不变。
若要真正的可见性 / 置信度，需换用输出逐点置信度的模型（如 HRFFA 的 ONNX 方案）。

## CK+（历史基线）

- `ck_plus_landmarks.csv`：本地提取 CK+ 特征，852 样本 / 106 subjects；无 neutral 类。
- `ck_raw.parquet`：本地源图像。Not redistributed.

现有 CK+ 特征 CSV 与原始 parquet **不在本目录**（从未提交进 Git），
报告中的 CK+ 指标无法用当前仓库复现。

For a usable neutral class, collect real samples from multiple consenting subjects:
`python collect_and_train.py --action collect --subject person01 --csv data/custom.csv`
Use a new subject ID for each person and repeat across lighting, head poses and sessions.
