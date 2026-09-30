# Pose-aware Landmark FER

基于 MediaPipe 人脸关键点、头部姿态估计和 SVM / MLP 的实时表情演示项目。
支持摄像头与视频输入、两种识别引擎、时序平滑与 HUD。

## 项目结构

```
pose-aware-landmark-fer/
├── run_demo.bat                 菜单式总入口（双击即可，共 9 项）
├── run_offline_demo.bat         一键离线视频演示，不需要摄像头
├── start_camera_demo.bat        一键摄像头演示
├── 启动实时摄像头演示.bat         一键摄像头演示（中文）
├── requirements.txt
├── requirements-tested.txt
├── src/                         Python 源码，按职责分为四个子目录
│   ├── core/                        核心算法库（被其它脚本 import，不直接运行）
│   │   ├── feature_extractor.py         MediaPipe 关键点、几何指标、头部姿态角
│   │   └── classifier.py                规则推理、SVM / MLP 训练与评测、输入向量组装
│   ├── pipeline/                    数据管线：原始数据 → 特征表
│   │   ├── extract_fer_landmarks.py     FER2013 下载与特征提取
│   │   ├── extract_ck_landmarks.py      CK+ 特征提取（历史基线）
│   │   └── recalc_pose_columns.py       用修复后的姿态解算重算特征表三列
│   ├── experiments/                 训练与评测，报告里的数字都出自这里
│   │   ├── train_fer_compare.py         8 组设置对照实验
│   │   ├── strict_eval.py               两阶段严格评测（选型与出数分离）
│   │   ├── tune_fer.py                  SVM (C, gamma) 定向调参
│   │   ├── finalize_fer.py              重训部署模型 + 端到端链路验证
│   │   ├── compare_old_new.py           新旧模型同图逐张对比
│   │   └── pose_stability_test.py       姿态 / 光照 / 距离 / 旋转稳定性测试（任务C）
│   └── apps/                        可直接运行的入口脚本
│       ├── realtime_demo.py             摄像头 / 视频实时演示
│       ├── collect_and_train.py         交互式样本采集与训练
│       ├── make_demo_video.py           合成免摄像头的离线演示视频
│       ├── download_model.py            下载并校验 FaceLandmarker 模型
│       └── test_pipeline.py             7 项回归测试
├── models/                      模型权重
│   ├── face_landmarker.task         MediaPipe 关键点模型，3.7MB
│   ├── model_fer_svm.pkl            FER2013 5 类部署模型，当前默认，22MB
│   ├── model_fer_mlp.pkl            FER2013 5 类 MLP
│   ├── model_svm.pkl                旧 CK+ SVM，仅作对比基线
│   └── model_mlp.pkl                旧 CK+ MLP，仅作对比基线
├── data/                        数据集与评测结果（不进 Git）
│   ├── fer2013_landmarks.csv        27,307 条特征表，48MB
│   ├── fer2013_raw/                 FER2013 原始 parquet，133MB
│   ├── fer_blendshape_names.json    52 维 blendshape 的名称与顺序
│   ├── fer2013_strict_*.json/txt    两阶段严格评测结果
│   ├── pose_stability_results.json  稳定性测试结果
│   └── fer2013_*.json / *.txt       评测记录与混淆矩阵
├── docs/                        报告文档与图表
│   ├── AUDIT.md                     已知问题、混淆矩阵与改进方向
│   ├── RESULTS_FER2013.md           FER2013 完整结果，可直接用于实验报告
│   ├── pose_stability_curves.png    稳定性测试四联曲线图
│   └── pose_yaw_extremes_check.png  姿态角符号语义人工核验图
└── assets/
    └── example.mp4                  离线演示视频
```

**路径解析规则**：所有脚本统一以**项目根目录**为基准解析 `data/`、`models/`、`assets/`，
因此从任何工作目录调用都能找到文件；命令行传入的相对路径同样按项目根目录解释
（`--model model_fer_svm.pkl` 和 `--model models/model_fer_svm.pkl` 都能命中）。

`src/` 下四个子目录之间直接互相 `import`（例如实验脚本引用 `core/` 里的特征与分类模块）。
各脚本顶部的「路径引导」段落负责把同级子目录加入模块搜索路径，
因此**移动脚本时必须保留该段落**，否则跨目录导入会失败。

## 安装和运行

本次验证环境：Windows、Python 3.13.9。具体依赖见 `requirements-tested.txt`。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-tested.txt
python src/apps/download_model.py
python src/apps/realtime_demo.py --source 0 --engine rules
```

下载脚本从 Google 官方模型存储获取 FaceLandmarker 并校验 SHA-256，保存到 `models/`。
第三方模型遵循其原始条款。仓库不包含本地照片、CK+ 原始数据或特征 CSV。
仅加载你信任的 `.pkl` 模型文件。

```powershell
# 推荐：FER2013 重训模型（含自然类，与规则引擎类别一致）
python src/apps/realtime_demo.py --engine ml --model models\model_fer_svm.pkl
python src/apps/realtime_demo.py --engine ml --model models\model_fer_mlp.pkl
# 旧 CK+ 模型（无自然类，仅作对比基线）
python src/apps/realtime_demo.py --engine ml --model models\model_svm.pkl
python src/apps/realtime_demo.py --engine ml --model models\model_mlp.pkl
# 离线视频（不需要摄像头）
python src/apps/realtime_demo.py --source assets\example.mp4 --model models\model_fer_svm.pkl --engine ml
# 自检（先运行 src/apps/download_model.py）
python src/apps/test_pipeline.py
```

快捷键：Q / ESC 退出，M 显示关键点，T 切换引擎，S 保存截图。
也可双击 `run_demo.bat`。选择训练需要事先准备本地数据。

## 不开摄像头怎么运行

除实时样本采集（`collect_and_train.py --action collect`）必须用摄像头外，其余功能全部可以离线运行。

```powershell
# 1. 最快验证：7 项回归自检，不弹窗口、不读摄像头
python src/apps/test_pipeline.py

# 2. 离线视频演示（会弹出窗口，但输入是视频文件不是摄像头）
#    画面右下角黄字是数据集真实标签，左上/右上 HUD 是模型预测，可直接肉眼对照
python src/apps/realtime_demo.py --source assets\example.mp4 --model models\model_fer_svm.pkl --engine ml --window 5
# 等价的一键方式：双击 run_offline_demo.bat，或 run_demo.bat 选 6

# 3. 批量图像评测，不弹窗口，直接打印准确率与类别分布
python src/experiments/compare_old_new.py

# 4. 数据集与训练全流程（全部离线，原始数据已缓存在 data/fer2013_raw）
python src/pipeline/extract_fer_landmarks.py        # 首次运行会走 hf-mirror 下载约 133MB
python src/experiments/train_fer_compare.py         # 8 组设置对照实验
python src/experiments/finalize_fer.py              # 重训部署模型 + 端到端验证
python src/experiments/strict_eval.py               # 两阶段严格评测，约 13 分钟
python src/experiments/pose_stability_test.py       # 姿态/光照/距离稳定性测试，约 20 分钟
```

`assets/example.mp4` 由 `make_demo_video.py` 从 FER2013 留出图像合成（40 张图、530 帧、480×480、约 26 秒、0.67MB），
删掉后会自动重新生成，也可以手动调参重建：

```powershell
python src/apps/make_demo_video.py --per-class 8 --frames 12 --fps 20
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

留出测试集上的 8 组对照（脚本 `src/experiments/train_fer_compare.py`）：

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

### 两阶段严格评测（选型与出数分离）

上面的 8 组对照是在 publicTest + privateTest **合并**的留出集上挑最优配置，存在选择性偏差。
`src/experiments/strict_eval.py` 改用两阶段协议：8 组设置只在 publicTest（验证集）上选型，
最优配置再到 privateTest（最终测试集）上出数字，选型过程完全不接触最终测试集。

| 阶段 | 数据 | 设置 | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| 选型 | publicTest（2,726） | 8 组中选出「SVM · 坐标+blendshape +标准化」 | 68.05% | 0.6640 |
| **最终测试** | **privateTest（2,713）** | **同上** | **69.89%** | **0.6786** |

最终测试集各类 Recall：neutral 69.3% / smile 86.1% / surprise 76.3% / frown 54.1% / sad 51.5%。
最终测试集数字高于验证集，说明选型没有过拟合验证集；这是本项目**方法学上最干净的准确率**，
实验报告正文建议引用 69.89% 而非 68.96%。

### 与旧 CK+ 模型的同图对比

同一批 479 张 FER2013 图像逐张对比（脚本 `src/experiments/compare_old_new.py`）：

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

### 姿态 / 光照 / 距离稳定性（任务C）

`src/experiments/pose_stability_test.py` 对留出集做四组受控扰动。
基线为 6,043 张留出图像的端到端准确率 62.07%（含约 10% 未检出样本）；
按头部姿态角分桶时只统计检出样本（5,439 张）。

头部姿态（按 yaw 左右偏航分桶）：

| yaw 区间 | 样本数 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 绝对值 ≤5°（正面） | 2,728 | 73.02% | 0.6967 |
| 5–15° | 1,762 | 66.86% | 0.6428 |
| 15–25° | 665 | 64.21% | 0.6336 |
| >25°（大角度） | 284 | 54.23% | 0.5583 |

正面到大角度整体下降约 19 个点，且**检出率始终为 100%**——
说明大角度下的失分来自特征可分性下降，而不是检测失败。

受控扰动（每档 750 张，缩放模拟距离、偏移模拟光照、旋转模拟手机歪持）：

| 扰动 | 设置 | Accuracy | 检出率 |
|---|---|---:|---:|
| 亮度偏移 | 0（基准） | 58.13% | 89.7% |
| 亮度偏移 | −90（最暗） | 48.80% | 82.4% |
| 缩放（距离） | 1.0 | 58.13% | 89.7% |
| 缩放（距离） | 0.5 | 30.80% | 65.1% |
| 缩放（距离） | 0.125 | 0.00% | 0.0%（完全检不出） |
| 平面旋转 | 0°（基准） | 58.13% | 89.7% |
| 平面旋转 | ±40° | 31.1% / 32.8% | 80.3% / 79.2% |

**距离是最敏感的因素**：人脸缩到原尺寸 1/4 时检测基本失效；
平面旋转与头部侧转次之；亮度在 ±30 内影响很小，只有过暗（−90）时明显劣化。
四联曲线见 `docs/pose_stability_curves.png`，原始数据见 `data/pose_stability_results.json`。

### CK+（历史基线）

对 `data/ck_plus_landmarks.csv` 的 852 条样本、106 个 subject_id 做 5 折 GroupKFold。
各折按人员隔离；表格为所有折的合并预测指标。评测后用全部数据重新训练部署模型。

| 模型 | Accuracy | Macro-F1 | 皱眉 Recall | 难过 Recall |
|---|---:|---:|---:|---:|
| SVM | 83.80% | 0.7773 | 58.52% | 45.24% |
| MLP | 87.21% | 0.8321 | 64.44% | 72.62% |

这些结果只描述本地关键点数据，不是摄像头实测准确率，也不是整个 CK+ 数据集的通用基准。
合成的 `dataset.csv` 仅用于流程测试，不能用来证明真实识别效果。
详细问题、混淆矩阵及改进方向见 [docs/AUDIT.md](docs/AUDIT.md)。

## 采集与训练

用独立文件采集真实数据，避免与合成数据混用。不同人员使用不同 ID，同一人员不同拍摄仍使用同一 ID。

```powershell
python src/apps/collect_and_train.py --action collect --subject person01 --csv data/custom.csv
# 至少两位人员，建议覆盖更多人员、光照、姿态与拍摄时段
python src/apps/collect_and_train.py --action train --csv data/custom.csv --model_type svm --save models/custom_svm.pkl
python src/apps/collect_and_train.py --action train --csv data/custom.csv --model_type mlp --save models/custom_mlp.pkl
```

采集按键：1 自然、2 微笑、3 惊讶、4 皱眉、5 难过；Q 保存并退出。
CSV 必须包含 `subject_id`、`label`、按顺序排列的 `feat_0` 至 `feat_135`。
通过合规渠道取得原始 CK+ 数据后，可自行使用 `extract_ck_landmarks.py` 提取。
`--save` 不指定时默认写入 `models/custom_model.pkl`。

## 项目文件

**入口脚本（项目根目录）**

- `run_demo.bat`：菜单式总入口（摄像头 / 离线视频 / 特征提取 / 对照实验 / 同图对比 / 自检 / 下载模型 / 稳定性测试 / 严格评测）。
- `run_offline_demo.bat`：一键离线视频演示，不需要摄像头。
- `start_camera_demo.bat`、`启动实时摄像头演示.bat`：一键摄像头演示。

**源码 `src/`**（按职责分四个子目录）

`core/` —— 核心算法库，被其它脚本 import，不直接运行

- `feature_extractor.py`：MediaPipe 关键点提取、几何归一化与头部姿态角
  （角度取自 MediaPipe `facial_transformation_matrixes`，不用 solvePnP）。
- `classifier.py`：规则推理、SVM / MLP 训练与跨人评测；按模型声明的特征集自动组装输入向量。

`pipeline/` —— 原始数据到特征表

- `extract_fer_landmarks.py`：FER2013 特征提取（下载 → 关键点/姿态/blendshape → CSV）。
- `extract_ck_landmarks.py`：本地 CK+ parquet 特征提取。
- `recalc_pose_columns.py`：用修复后的姿态解算重算特征表 yaw/pitch/roll，带逐行对齐校验。

`experiments/` —— 训练与评测

- `train_fer_compare.py`：FER2013 上 8 组设置的对照实验。
- `strict_eval.py`：两阶段严格协议，publicTest 选型、privateTest 出最终数字。
- `tune_fer.py`：SVM (C, gamma) 与特征组合的定向调参。
- `finalize_fer.py`：按最优配置重训部署模型，并做端到端链路验证。
- `compare_old_new.py`：同一批图像上旧 CK+ 模型与新 FER2013 模型的逐张对比。
- `pose_stability_test.py`：姿态 / 光照 / 距离 / 旋转稳定性测试，输出 JSON 与四联曲线。

`apps/` —— 可直接运行的入口

- `realtime_demo.py`：摄像头 / 视频演示。
- `collect_and_train.py`：样本采集与训练入口。
- `make_demo_video.py`：用 FER2013 留出图像合成免摄像头的 `assets/example.mp4`。
- `test_pipeline.py`：回归测试；不宣称验证真实表情准确率。
- `download_model.py`：官方 FaceLandmarker 模型下载和校验。

**模型 `models/`**

- `face_landmarker.task`：MediaPipe 关键点检测模型。
- `model_fer_svm.pkl`、`model_fer_mlp.pkl`：FER2013 5 类模型，**当前默认推荐**。
- `model_svm.pkl`、`model_mlp.pkl`：旧 CK+ 5 类模型（无 neutral），保留作对比基线。

**数据与文档**

- `data/`：特征 CSV、FER2013 原始 parquet、评测 JSON；说明见 [data/README.md](data/README.md)。
- `docs/AUDIT.md`、`docs/RESULTS_FER2013.md`：问题清单与完整结果记录。
- `assets/example.mp4`：离线演示视频。

## 复现 FER2013 全流程

```powershell
# 1. 提取特征（首次会从 hf-mirror 镜像下载 FER2013 原始 parquet，约 133MB）
python src/pipeline/extract_fer_landmarks.py
# 2. 8 组设置对照实验（留出 publicTest+privateTest）
python src/experiments/train_fer_compare.py
# 3. 重训部署模型并做端到端验证
python src/experiments/finalize_fer.py
# 4. 与旧 CK+ 模型同图对比
python src/experiments/compare_old_new.py
# 5. 两阶段严格评测：publicTest 选型 → privateTest 出最终数字
python src/experiments/strict_eval.py
# 6. 任务C 稳定性测试：姿态分桶 + 光照/距离/旋转受控扰动
python src/experiments/pose_stability_test.py

# 用新模型跑摄像头
python src/apps/realtime_demo.py --source 0 --model models\model_fer_svm.pkl --engine ml
```

注：`huggingface.co` 在当前网络下不可达，脚本通过 `hf-mirror.com` 镜像获取数据。
FER2013 不提供受试者 ID，因此按其官方 train / publicTest / privateTest 划分评测，
无法做到 CK+ 那样的按人员 GroupKFold，这是该数据集的已知局限。
