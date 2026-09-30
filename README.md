# Pose-aware Landmark FER

基于 MediaPipe 人脸关键点、头部姿态估计和 SVM / MLP 的实时表情演示项目。
支持摄像头与视频输入、两种识别引擎、时序平滑与 HUD。

## 安装和运行

本次验证环境：Windows、Python 3.14.3。具体依赖见 `requirements-tested.txt`。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-tested.txt
python download_model.py
python realtime_demo.py --source 0 --engine rules
```

下载脚本从 Google 官方模型存储获取 FaceLandmarker，并校验 SHA-256。
第三方模型遵循其原始条款。仓库不包含本地照片、CK+ 原始数据或特征 CSV。
仅加载你信任的 `.pkl` 模型文件。

```powershell
# 使用随仓库提供的 MLP 或 SVM 模型
python realtime_demo.py --engine ml --model model_mlp.pkl
python realtime_demo.py --engine ml --model model_svm.pkl
# 离线视频
python realtime_demo.py --source example.mp4 --engine rules
# 自检（先运行 download_model.py）
python test_pipeline.py
```

快捷键：Q / ESC 退出，M 显示关键点，T 切换引擎，S 保存截图。
也可双击 `run_demo.bat`。选择训练需要事先准备本地数据。

## 两个引擎的区别

| 引擎 | 类别 | 注意事项 |
|---|---|---|
| rules（默认） | 自然、微笑、惊讶、皱眉、难过 | 手工 blendshape / 几何规则，分数未经概率校准 |
| ml（附带权重） | 厌恶、皱眉、难过、微笑、惊讶 | **没有自然类**，平静表情也会被分入五类之一 |

规则分数和模型概率均不代表已经验证的正确率。模型没有自然类时，不能通过调整阈值补出这个类别，需要真实自然样本重新训练。

## 本地数据评测（2026-09-30）

对 `data/ck_plus_landmarks.csv` 的 852 条样本、106 个 subject_id 做 5 折 GroupKFold。
各折按人员隔离；表格为所有折的合并预测指标。评测后用全部数据重新训练部署模型。

| 模型 | Accuracy | Macro-F1 | 皱眉 Recall | 难过 Recall |
|---|---:|---:|---:|---:|
| SVM | 83.80% | 0.7773 | 58.52% | 45.24% |
| MLP | 87.21% | 0.8321 | 64.44% | 72.62% |

这些结果只描述本地关键点数据，不是摄像头实测准确率，也不是整个 CK+ 数据集的通用基准。
合成的 `dataset.csv` 仅用于流程测试，不能用来证明真实识别效果。
详细问题、混淆矩阵及改进方向见 [AUDIT.md](AUDIT.md)。

## 采集与训练

用独立文件采集真实数据，避免与合成数据混用。不同人员使用不同 ID，同一人员不同拍摄仍使用同一 ID。

```powershell
python collect_and_train.py --action collect --subject person01 --csv data/custom.csv
# 至少两位人员，建议覆盖更多人员、光照、姿态与拍摄时段
python collect_and_train.py --action train --csv data/custom.csv --model_type svm --save custom_svm.pkl
python collect_and_train.py --action train --csv data/custom.csv --model_type mlp --save custom_mlp.pkl
```

采集按键：1 自然、2 微笑、3 惊讶、4 皱眉、5 难过；Q 保存并退出。
CSV 必须包含 `subject_id`、`label`、按顺序排列的 `feat_0` 至 `feat_135`。
通过合规渠道取得原始 CK+ 数据后，可自行使用 `extract_ck_landmarks.py` 提取。

## 项目文件

- `feature_extractor.py`：MediaPipe 特征、几何指标、PnP 头部姿态。
- `classifier.py`：规则推理、SVM / MLP 训练与跨人评测。
- `realtime_demo.py`：摄像头 / 视频演示。
- `collect_and_train.py`：样本采集与训练入口。
- `extract_ck_landmarks.py`：本地 CK+ parquet 特征提取。
- `test_pipeline.py`：回归测试；不宣称验证真实表情准确率。
- `download_model.py`：官方 FaceLandmarker 模型下载和校验。
- `model_svm.pkl`、`model_mlp.pkl`：本次全部本地 CK+ 特征重训的分类器。
