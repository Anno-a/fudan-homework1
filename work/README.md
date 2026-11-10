# 任务二 代码说明

全部命令都在 **`小组期末作业/`** 目录下执行（不是 `chest_xray/`，也不是 `work/`）。

## 两种用法

| 用法 | 适合 | 入口 |
|---|---|---|
| **Jupyter notebook**（推荐，也是最终交付物） | 在 Anaconda 里逐格跑、看图、写报告 | `AIT410020_PJ2_任务二.ipynb` |
| 命令行脚本 | 长时间批量训练、后台跑 | `python work/train.py ...` |

notebook 是自包含的：数据索引、划分、预处理、两个模型、训练、评估、t-SNE、Grad-CAM
全部在里面，不依赖 `work/*.py`（脚本和 notebook 是两套独立可用的实现，内容等价）。

## 0. 目录结构

```
小组期末作业/
├── AIT410020_PJ2_任务二.ipynb  ★ 最终交付的 notebook（自包含，从头跑到尾）
├── 任务清单.md                  待办与时间安排
├── chest_xray/                 原始数据（train / test，不要改）
├── AIT_PJ2/                    题目与任务一示例
├── figures/                    notebook 输出的图（ROC/混淆矩阵/t-SNE/Grad-CAM…）
└── work/
    ├── requirements.txt        依赖清单（已按本机版本锁定）
    ├── data_pipeline.py        数据索引 + 病人级划分
    ├── build_cache.py          建立 256×256 letterbox 预处理缓存（可选，可提前跑）
    ├── train.py                命令行训练脚本（和 notebook 等价的另一套实现）
    ├── notebook_src.py         notebook 的源文件（jupytext 格式，可直接忽略）
    ├── cache/                  预处理缓存（img256.npy，384 MB）
    ├── splits/                 划分文件（csv）
    │   ├── index.csv           全量索引：路径/标签/亚型/病人号/尺寸/md5
    │   ├── official_train.csv  官方训练集，含 fold 列（5 折，病人级分层）
    │   ├── official_test.csv   官方测试集（病人与训练集完全不重叠）
    │   └── pooled_70_30.csv    按题目 70/30 重新划分（病人级，part 列）
    └── runs/<tag>/             每次训练的输出
        ├── best.pt             验证集 AUC 最好的一轮权重
        ├── metrics.json        全部指标 + 训练曲线数据
        ├── preds_val.csv       验证集逐样本预测（做 ROC / Grad-CAM / t-SNE 用）
        ├── preds_test.csv      测试集逐样本预测
        └── log.txt             完整日志
```

## 1. 环境

```bash
pip install -r work/requirements.txt
```

**只有两个坑：**

1. **torchvision 不能单独装**。本机是 `torch 2.12.0+cpu` + `torchvision 0.27.0+cpu`，单独 `pip install torchvision`
   会把 torch 一起升级或降级。要重装请用
   `pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu`。
2. **`download.pytorch.org` 在这台机器上不通**（连接被重置），所以方案 B 的预训练权重走 HuggingFace 镜像。
   `train.py` 里已经写了 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`，不用手动设。
   第一次跑方案 B 会自动下载 resnet18（约 45MB，6 秒左右），之后走本地缓存。

确认环境没问题：

```bash
python -c "import torch, torchvision, timm; print(torch.__version__, torchvision.__version__, timm.__version__)"
# 期望：2.12.0+cpu 0.27.0+cpu 1.0.31
```

## 2. 在 Anaconda 里用 notebook

1. 打开 Anaconda Navigator → Jupyter Notebook（或 Anaconda Prompt 里 `jupyter notebook`）；
2. 进到 `小组期末作业/` 目录，打开 `AIT410020_PJ2_任务二.ipynb`；
3. Kernel 选 `base (Python 3.12)`（就是本机 `D:\Anaconda\python.exe`）；
4. 从第一格开始逐格运行。

**第一次运行前建议先建好预处理缓存**（notebook 里也会自动建，提前跑一次更省事）：

```bash
python work/build_cache.py
```

notebook 顶部有几个开关：

| 变量 | 含义 | 默认 |
|---|---|---|
| `RUN_TRAINING` | `True` = 强制重新训练所有交叉验证折（约 3.5 小时） | `False`（读已有结果） |
| `RUN_FINAL` | `True` = 用全部训练集重训并在测试集评估 | `False` |
| `FOLDS` | 跑哪几折，调试时可只留 `[0]` | `[0,1,2,3,4]` |
| `EPOCHS_A / EPOCHS_B` | 两个方案的最大轮数（有早停，实际通常更少） | 25 / 15 |

训练结果会写到 `work/runs/<tag>/`，所以**关掉 notebook 再打开不用重新训练**，
后面的评估格会直接读磁盘上的预测文件。

> `work/notebook_src.py` 是 notebook 的源文件（jupytext py:percent 格式），只用于版本管理。
> 如果你直接在 Jupyter 里改了 `.ipynb`，忽略这个文件即可。

## 3. 先跑数据管线（只跑一次，约 40 秒）

```bash
python work/data_pipeline.py
```

会生成 `work/splits/` 下的 4 个 csv，并打印病人级泄漏检查结果。

## 4. 训练

```bash
# 冒烟测试：200 张图 1 轮，30 秒确认流程通
python work/train.py --scheme resnet18 --fold 0 --epochs 1 --smoke

# 方案 B：ImageNet 预训练 ResNet18，224×224
python work/train.py --scheme resnet18 --fold 0 --epochs 15 --tag B_fold0

# 方案 A：自建 SmallCNN，128×128，从零训练
python work/train.py --scheme smallcnn --fold 0 --epochs 25 --tag A_fold0

# 5 折交叉验证（for 循环即可）
for k in 0 1 2 3 4; do
  python work/train.py --scheme resnet18 --fold $k --epochs 15 --tag B_fold$k
done

# 全量训练集训练 + 官方测试集评估
python work/train.py --scheme resnet18 --fold all --epochs 15 --tag B_full --eval-test

# 题目要求的 70/30 重划分（敏感性分析）
python work/train.py --scheme resnet18 --split pooled --epochs 15 --tag B_pool70
```

### 实测耗时（32 核 CPU，无 GPU）

| 方案 | 输入 | 单轮 | 建议轮数 | 单折 | 5 折 |
|---|---|---|---|---|---|
| B ResNet18 | 224 | **83 s** | 15 | ~21 min | ~1.7 h |
| A SmallCNN | 128 | **54 s** | 25 | ~22 min | ~1.9 h |

两个方案、5 折、加消融，整晚足够跑完。CPU 线程数用 `--threads`（默认 12），
和 `--workers`（默认 4）搭配，别两个都给满，否则会抢核反而变慢。

### 常用参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--split` | `official` / `pooled`（70/30 重划分） | official |
| `--fold` | 0–4，或 `all` | 0 |
| `--freeze-epochs` | 方案 B 先冻结 backbone 的轮数，0 表示一开始就全网络微调 | 3 |
| `--aug` | `full` / `light`（只翻转）/ `none` | full |
| `--class-weight` | `none` / `balanced`（按类频率反比加权） | none |
| `--eval-test` | 训练完在测试集上评估一次 | 关 |
| `--tag` | 输出目录名 | 自动生成 |

消融对照建议直接改参数、换 tag：

```bash
# 关掉水平翻转等增强
python work/train.py --scheme resnet18 --aug light --tag B_noaug
# 类别加权
python work/train.py --scheme resnet18 --class-weight balanced --tag B_cw
# 不冻结，一开始就全网络微调
python work/train.py --scheme resnet18 --freeze-epochs 0 --tag B_nofreeze
```

## 5. 结果文件怎么读

`metrics.json` 里：

```json
{
  "history": [{"epoch":1, "train_loss":..., "val_auc":..., "val_sensitivity":...}, ...],
  "best": {"auc": 0.9838, "epoch": 1},
  "val_metrics": {"accuracy":..., "sensitivity":..., "specificity":..., "f1":..., "auc":...,
                  "tn":..., "fp":..., "fn":..., "tp":...},
  "test_metrics": {...}
}
```

`preds_*.csv` 每行一个样本，字段 `relpath / patient / subtype / y_true / y_prob`，
后面画 ROC、混淆矩阵、做细菌-病毒亚组分析、Grad-CAM、t-SNE 都从这两个文件出发。

## 6. 已实测的参考数字

第 0 折、**只训练 1 轮**的结果（不是最终结果，只用来确认流程和量级）：

| 方案 | val AUC | 准确率 | 灵敏度 | 特异度 |
|---|---|---|---|---|
| B ResNet18（全网络微调） | 0.984 | 0.943 | 0.965 | 0.880 |
| A SmallCNN（从零） | 0.945 | 0.775 | 0.712 | 0.964 |

方案 A 训练 1 轮时明显偏向"全判正常"（特异度 0.96 / 灵敏度 0.71），
这正是后面要分析的现象：不平衡数据下从零训练的模型更容易退化成多数类。

## 7. 已知需要注意的地方

- **划分必须按病人**：`data_pipeline.py` 已处理。官方 train/test 病人交集为 0；
  但肺炎文件里 `personN` 的细菌、病毒两套编号是独立的，去重时必须带亚型前缀，
  否则会误判出 170 个"重叠病人"。
- **32 张完全重复的图片**已自动去重（保留每组第一张），所以总数是 5824 而不是 5856。
- **letterbox 而不是直接 resize**：长宽比跨度 0.84~3.38，直接拉伸会把肺野压扁。
- 训练日志是中文，Windows 控制台如果乱码是编码问题，`log.txt` 里是正常的 UTF-8。
