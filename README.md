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
# 推荐：FER2013 重训模型（含自然类，与规则引擎类别一致）
python realtime_demo.py --engine ml --model model_fer_svm.pkl
python realtime_demo.py --engine ml --model model_fer_mlp.pkl
# 旧 CK+ 模型（无自然类，仅作对比基线）
python realtime_demo.py --engine ml --model model_svm.pkl
python realtime_demo.py --engine ml --model model_mlp.pkl
# 离线视频（不需要摄像头）
python realtime_demo.py --source example.mp4 --model model_fer_svm.pkl --engine ml
# 自检（先运行 download_model.py）
python test_pipeline.py
```

快捷键：Q / ESC 退出，M 显示关键点，T 切换引擎，S 保存截图。
也可双击 `run_demo.bat`。选择训练需要事先准备本地数据。

## 不开摄像头怎么运行

除实时样本采集（`collect_and_train.py --action collect`）必须用摄像头外，其余功能全部可以离线运行。

```powershell
# 1. 最快验证：7 项回归自检，不弹窗口、不读摄像头
python test_pipeline.py

# 2. 离线视频演示（会弹出窗口，但输入是视频文件不是摄像头）
#    画面右下角黄字是数据集真实标签，左上/右上 HUD 是模型预测，可直接肉眼对照
python realtime_demo.py --source example.mp4 --model model_fer_svm.pkl --engine ml --window 5
# 等价的一键方式：双击 run_offline_demo.bat，或 run_demo.bat 选 6

# 3. 批量图像评测，不弹窗口，直接打印准确率与类别分布
python compare_old_new.py

# 4. 数据集与训练全流程（全部离线，原始数据已缓存在 data/fer2013_raw）
python extract_fer_landmarks.py   # 首次运行会走 hf-mirror 下载约 133MB
python train_fer_compare.py       # 8 组设置对照实验
python finalize_fer.py            # 重训部署模型 + 端到端验证
```

`example.mp4` 由 `make_demo_video.py` 从 FER2013 留出图像合成（40 张图、530 帧、480×480、约 26 秒、0.67MB），
删掉后会自动重新生成，也可以手动调参重建：

```powershell
python make_demo_video.py --per-class 8 --frames 12 --fps 20
```

`--source` 传数字（如 `0`）才会打开摄像头并做镜像翻转；传视频路径按原样播放、不做翻转。

## 两个引擎的区别

| 引擎 | 类别 | 注意事项 |
|---|---|---|
| rules（默认） | 自然、微笑、惊讶、皱眉、难过 | 手工 blendshape / 几何规则，分数未经概率校准 |
| ml · `model_fer_svm.pkl` | 自然、微笑、惊讶、皱眉、难过 | FER2013 重训，与规则引擎类别一致 |
| ml · `model_svm.pkl`（旧 CK+） | 厌恶、皱眉、难过、微笑、惊讶 | **没有自然类**，平静表情也会被分入五类之一 |

两个引擎的类别集合现已一致，按 T 切换引擎不会再改变标签集合。
规则分数和模型概率均不代表已经验证的正确率。旧 CK+ 模型没有自然类，不能通过调整阈值补出这个类别。

## 数据集与评测结果

### FER2013（当前默认模型，2026-09-30 重训）

公开数据集 FER2013 经 MediaPipe FaceLandmarker 提取 68 点归一化坐标 + 52 维 blendshape。
原始 35,887 张中 27,307 张成功检出（约 10% 因未检出人脸剔除），
`disgust` / `fear` 两类样本少且与规则引擎类别不一致，已丢弃，最终 5 类：

| 划分 | 样本数 | neutral | smile | surprise | frown | sad |
|---|---:|---:|---:|---:|---:|---:|
| train | 21,868 | 4,705 | 6,838 | 2,932 | 3,398 | 3,995 |
| publicTest + privateTest（留出，训练未见） | 5,439 | 1,160 | 1,684 | 759 | 817 | 1,019 |

留出测试集上的 8 组对照（脚本 `train_fer_compare.py`）：

| 设置 | Accuracy | Macro-F1 |
|---|---:|---:|
| SVM · 坐标+blendshape +标准化 | **68.96%** | **0.6713** |
| SVM · 坐标+blendshape +标准化 +类别均衡 | 68.74% | 0.6708 |
| SVM · roll对齐坐标+blendshape +标准化 | 67.40% | 0.6517 |
| SVM · 坐标+blendshape | 66.52% | 0.6404 |
| MLP · 坐标136维 | 66.52% | 0.6355 |
| MLP · 坐标+blendshape | 65.42% | 0.6338 |
| MLP · 坐标+blendshape +标准化 | 64.44% | 0.6256 |
| SVM · 坐标136维 | 64.63% | 0.6150 |

部署 SVM 的 5 折交叉验证：68.05% ± 0.75（Macro-F1 0.6607 ± 0.0079）。
各类别 Recall（留出测试集）：

| 类别 | neutral | smile | surprise | frown | sad |
|---|---:|---:|---:|---:|---:|
| Recall | 67.2% | 84.6% | 78.9% | 54.2% | 49.5% |

三点结论：加入 blendshape 特征提升约 1.9 个点；折内 StandardScaler 提升约 2.4 个点；
roll 旋转对齐反而下降 1.6 个点（FER2013 人脸基本正立，PnP 在 48×48 小图上估角噪声大）。
frown / sad 仍是主要混淆来源。

### 与旧 CK+ 模型的同图对比

同一批 479 张 FER2013 图像逐张对比（脚本 `compare_old_new.py`）：

| 真实类别 | 旧 CK+ 模型 | 新 FER2013 模型 |
|---|---:|---:|
| neutral | 0.0%（无该类，结构上不可能输出） | 66.2% |
| smile (happy) | 59.7% | 80.5% |
| surprise | 66.7% | 83.3% |
| frown (angry) | 34.3% | 58.6% |
| sad | 19.7% | 52.5% |
| **整体（5 类）** | **36.9%** | **69.1%** |

旧模型平均置信度 74.2%、新模型 66.4%——旧模型更"自信"，但错得更多。
CK+ 上的 83.80% 来自摆拍、夸张、正面的库内数据，不能代表真实画面能力。

### CK+（历史基线）

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
- `classifier.py`：规则推理、SVM / MLP 训练与跨人评测；按模型声明的特征集自动组装输入向量。
- `realtime_demo.py`：摄像头 / 视频演示。
- `collect_and_train.py`：样本采集与训练入口。
- `extract_ck_landmarks.py`：本地 CK+ parquet 特征提取。
- `extract_fer_landmarks.py`：FER2013 特征提取（下载 → 关键点/姿态/blendshape → CSV）。
- `train_fer_compare.py`：FER2013 上 8 组设置的对照实验。
- `tune_fer.py`：SVM (C, gamma) 与特征组合的定向调参。
- `finalize_fer.py`：按最优配置重训部署模型，并做端到端链路验证。
- `compare_old_new.py`：同一批图像上旧 CK+ 模型与新 FER2013 模型的逐张对比。
- `make_demo_video.py`：用 FER2013 留出图像合成免摄像头的 `example.mp4`。
- `test_pipeline.py`：回归测试；不宣称验证真实表情准确率。
- `download_model.py`：官方 FaceLandmarker 模型下载和校验。
- `run_demo.bat`：菜单式入口；`run_offline_demo.bat`：一键离线视频演示（不需要摄像头）。
- `model_fer_svm.pkl`、`model_fer_mlp.pkl`：FER2013 5 类模型，**当前默认推荐**。
- `model_svm.pkl`、`model_mlp.pkl`：旧 CK+ 5 类模型（无 neutral），保留作对比基线。

## 复现 FER2013 全流程

```powershell
# 1. 提取特征（首次会从 hf-mirror 镜像下载 FER2013 原始 parquet，约 133MB）
python extract_fer_landmarks.py
# 2. 8 组设置对照实验（留出 publicTest+privateTest）
python train_fer_compare.py
# 3. 重训部署模型并做端到端验证
python finalize_fer.py
# 4. 与旧 CK+ 模型同图对比
python compare_old_new.py

# 用新模型跑摄像头
python realtime_demo.py --source 0 --model model_fer_svm.pkl --engine ml
```

注：`huggingface.co` 在当前网络下不可达，脚本通过 `hf-mirror.com` 镜像获取数据。
FER2013 不提供受试者 ID，因此按其官方 train / publicTest / privateTest 划分评测，
无法做到 CK+ 那样的按人员 GroupKFold，这是该数据集的已知局限。
