# AIT410020 PJ2 · 任务二：胸部 X 光肺炎分类

课程期末作业（小组）。本仓库包含代码、实验结果、图表和论文；**原始数据集与模型权重不在仓库内**（体积原因），获取方式见「准备数据」。

- **任务**：胸部 X 光影像的正常 / 肺炎二分类，含文献综述、两个模型的对比、中间特征的可解释性分析
- **数据**：Kaggle Chest X-Ray（Kermany et al., *Cell* 2018），5856 张 JPEG，去重后 5824 张，来自 4096 名 1–5 岁患儿
- **环境**：Python 3.12.7，**纯 CPU**（无 GPU），32 核

---

## 结论速览

三条主要发现，完整数据见 `work/实验结果汇总.md`。

1. **同分布下两个方案打平**：五折验证集 AUC 0.9981（自建 SmallCNN，1.17 M 参数）对 0.9972（ResNet18，11.18 M 参数），差异小于折间标准差。预训练没有带来同分布精度收益。
2. **差距只在跨队列时出现**：官方测试集 AUC 0.9655 对 0.9297，特异度 0.662 对 0.455。迁移学习的价值是泛化。
3. **预处理引入了捷径**：数据集两类图像的宽高比不同（1.25 对 1.52），仅凭宽高比单项即可达 AUC 0.867；等比缩放补黑边使其变成可见线索。改为中心裁剪后，方案 A 的测试准确率由 0.791 升至 0.903、特异度由 0.455 升至 0.801，两方案差距由 0.036 缩至 0.015。

---

## 目录结构

```
.
├── README.md                      本文件
├── AIT410020_PJ2_任务二.ipynb     ★ 全部代码与运行结果（已执行，含图表输出）
├── AIT410020_PJ2_任务二_报告.docx  ★ 课程论文（1499 字，6 表 4 图，9 篇文献）
├── 报告正文.md                     论文的纯文本源
├── 报告初稿框架.md                 写作框架与字数分配
├── 任务清单.md                     待办与时间安排
├── figures/                       5 张 300 dpi 图（ROC / 混淆矩阵 / t-SNE / Grad-CAM / 样本）
├── AIT_PJ2/                       课程下发的原始题目与任务一示例
└── work/
    ├── README.md                  代码层面的详细说明（脚本用法、参数、排错）
    ├── requirements.txt           依赖清单
    ├── data_pipeline.py           数据索引、病人级划分、去重
    ├── build_cache.py             256×256 letterbox 预处理缓存
    ├── train.py                   训练脚本（SmallCNN / ResNet18）
    ├── run_all.py                 一键跑完全部实验（可中断续跑）
    ├── build_report_docx.py       由 报告正文.md + 结果 + 图 生成 docx
    ├── notebook_src.py            notebook 的源文件（jupytext 格式，可忽略）
    ├── splits/                    划分文件（index.csv / official_train.csv / official_test.csv / pooled_70_30.csv）
    ├── runs/<tag>/                16 个训练 run：metrics.json + preds_*.csv + log.txt（权重未提交）
    ├── logs/                      训练日志与总控日志
    ├── 实验结果汇总.md             ★ 结果主汇总（7 节，带结论与消融）
    └── results_summary.md         自动生成的原始表（按 run 罗列，不含分析）
```

---

## 准备数据（仓库中没有）

数据集 1.2 GB，未提交。需要自行下载后按下列结构放置：

```
chest_xray/
├── train/
│   ├── NORMAL/      1349 张
│   └── PNEUMONIA/   3883 张
└── test/
    ├── NORMAL/       234 张
    └── PNEUMONIA/    390 张
```

来源：Kaggle *Chest X-Ray Images (Pneumonia)*。放置后先跑 `python work/data_pipeline.py` 重建划分文件。

---

## 运行

```bash
pip install -r work/requirements.txt

# 1) 数据索引与划分（约 40 秒）
python work/data_pipeline.py

# 2) 预处理缓存（约 40 秒，384 MB）
python work/build_cache.py

# 3) 全部实验（约 6 小时 CPU；已有结果会自动跳过）
python work/run_all.py
```

或者直接用 Jupyter 打开 `AIT410020_PJ2_任务二.ipynb` 从头执行。notebook 顶部有 `RUN_TRAINING` / `RUN_FINAL` 两个开关，默认 `False`，即读取 `work/runs/` 里已有的结果，只重画图表；要重新训练才改成 `True`。

### 没有模型权重时能做什么

仓库只提交了预测结果（`preds_*.csv`）和指标，没有 `.pt` 权重：

| 可以 | 需要权重 |
|---|---|
| ROC、混淆矩阵、全部指标表格 | t-SNE |
| 亚组分析、阈值分析、70/30 对比 | Grad-CAM |
| 所有图（`figures/` 已包含） | 重新推理 |

权重可由 `python work/run_all.py` 复现（训练是确定性的，种子 2026）。

---

## 环境

| 包 | 版本 |
|---|---|
| Python | 3.12.7（支持 3.10–3.12） |
| torch / torchvision | 2.12.0+cpu / 0.27.0+cpu |
| timm / huggingface_hub | 1.0.31 / 2.4.0 |
| scikit-learn / pandas / numpy | 1.5.1 / 2.2.2 / 1.26.4 |

预训练权重走 HuggingFace 镜像（`HF_ENDPOINT=https://hf-mirror.com`），脚本内已设默认值。

---

## 已知限制

- 单中心儿科数据（1–5 岁），模型在成人胸片上的表现未经验证
- 官方测试集仅 618 张，两方案的差异未做显著性检验
- 黑边捷径已定位并消除，但"成像来源"这一混淆因素本身未完全剥离
- 报告中的文献卷期页码已核对（2026-11-10）
