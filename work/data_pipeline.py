# -*- coding: utf-8 -*-
"""
AIT410020 PJ2 - 任务二：胸部 X 光二分类
数据管线：建立索引、解析病人 ID、去重、生成可复现的划分文件。

运行：
    python work/data_pipeline.py

产物：
    work/splits/index.csv            全部 5856 张图的索引与元信息
    work/splits/official_train.csv   官方训练集（含 5 折标记）
    work/splits/official_test.csv    官方测试集
    work/splits/pooled_70_30.csv     按题目要求重新划分的 70% / 30%（病人级）
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]          # 小组期末作业/
DATA = ROOT / "chest_xray"
SPLITS = ROOT / "work" / "splits"
SEED = 2026


# --------------------------------------------------------------------------- #
# 文件名解析
# --------------------------------------------------------------------------- #
# NORMAL  : IM-0115-0001.jpeg / NORMAL2-IM-0150-0001.jpeg  -> 病人号 IM-0115 / NORMAL2-IM-0150
# PNEUMONIA: person1000_bacteria_2931.jpeg                 -> 病人号 (bacteria, person1000)
# 注意：细菌与病毒两套 person 编号相互独立，必须带亚型前缀才能唯一确定病人。
_RE_NORMAL = re.compile(r"^(NORMAL2-IM-\d+|IM-\d+)")
_RE_PNEU = re.compile(r"^(person\d+)_(bacteria|virus)_")


def parse_patient(filename: str) -> str:
    stem = Path(filename).stem
    m = _RE_NORMAL.match(stem)
    if m:
        return m.group(1)
    m = _RE_PNEU.match(stem)
    if m:
        return f"{m.group(2)}-{m.group(1)}"
    raise ValueError(f"无法解析病人编号: {filename}")


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. 建立索引
# --------------------------------------------------------------------------- #
def build_index() -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            for p in sorted((DATA / split / cls).glob("*.jpeg")):
                with Image.open(p) as im:
                    w, h = im.size
                    mode = im.mode
                stem = p.stem
                subtype = "normal" if cls == "NORMAL" else (
                    "bacteria" if "bacteria" in stem else "virus")
                rows.append(dict(
                    relpath=p.relative_to(DATA).as_posix(),
                    split=split,
                    label=int(cls == "PNEUMONIA"),
                    label_name=cls,
                    subtype=subtype,
                    patient=parse_patient(p.name),
                    width=w,
                    height=h,
                    mode=mode,
                    md5=md5(p),
                ))
    df = pd.DataFrame(rows)
    df["dup_keep"] = ~df.duplicated("md5", keep="first")
    return df


# --------------------------------------------------------------------------- #
# 2. 划分
# --------------------------------------------------------------------------- #
def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("patient")
              .agg(label=("label", "first"), n=("relpath", "size"))
              .reset_index())


def stratified_patient_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """按病人（整组图片一起）分层 K 折。返回与 df 对齐的 fold 列（0..K-1）。"""
    from sklearn.model_selection import StratifiedKFold

    pts = _patient_table(df).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of: dict[str, int] = {}
    for k, (_, va) in enumerate(skf.split(pts, pts["label"])):
        for pid_ in pts.loc[va, "patient"]:
            fold_of[pid_] = k
    return df["patient"].map(fold_of).astype(int)


def stratified_patient_split(df: pd.DataFrame, frac: float, seed: int) -> pd.Series:
    """按病人分层随机抽 frac 的病人划入 part=0，其余 part=1。返回 0/1 列。"""
    from sklearn.model_selection import train_test_split

    pts = _patient_table(df)
    idx0, _ = train_test_split(pts.index, train_size=frac, stratify=pts["label"],
                               random_state=seed)
    side = pd.Series(1, index=pts["patient"].to_numpy())
    side.loc[pts.loc[idx0, "patient"].to_numpy()] = 0
    return df["patient"].map(side).astype(int)


def main() -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    df = build_index()

    print("=" * 68)
    print(f"总图片数: {len(df)}   完全重复(md5)副本: {(~df.dup_keep).sum()}")
    print(df.groupby(["split", "label_name", "subtype"]).size().to_string())
    print("-" * 68)
    print("图像尺寸: 宽 %d~%d, 高 %d~%d, 模式 %s"
          % (df.width.min(), df.width.max(), df.height.min(), df.height.max(),
             df["mode"].unique().tolist()))

    # --- 病人级泄漏检查 -------------------------------------------------- #
    tr_pt = set(df.loc[df.split == "train", "patient"])
    te_pt = set(df.loc[df.split == "test", "patient"])
    print(f"病人数: train {len(tr_pt)}, test {len(te_pt)}, 交集 {len(tr_pt & te_pt)}")
    assert not (tr_pt & te_pt), "官方划分存在病人重叠，请检查 parse_patient！"

    # 每个病人只属于一个类别？
    assert df.groupby("patient")["label"].nunique().max() == 1

    # 去重：保留每组第一张
    clean = df[df.dup_keep].copy()

    # --- 官方划分 -------------------------------------------------------- #
    train = clean[clean.split == "train"].copy()
    test = clean[clean.split == "test"].copy()
    train["fold"] = stratified_patient_folds(train, n_folds=5, seed=SEED)
    train["val"] = (train["fold"] == 0).astype(int)
    train.to_csv(SPLITS / "official_train.csv", index=False, encoding="utf-8")
    test.to_csv(SPLITS / "official_test.csv", index=False, encoding="utf-8")

    # --- 70/30 病人级重划分（敏感性分析） -------------------------------- #
    pooled = clean.copy()
    pooled["part"] = stratified_patient_split(pooled, frac=0.7, seed=SEED)
    pooled.to_csv(SPLITS / "pooled_70_30.csv", index=False, encoding="utf-8")

    clean.to_csv(SPLITS / "index.csv", index=False, encoding="utf-8")

    # --- 校验输出 -------------------------------------------------------- #
    print("-" * 68)
    print("官方训练集 5 折（图片数 / 病人数 / 肺炎占比）:")
    for k in range(5):
        f = train[train.fold == k]
        print(f"  fold{k}: {len(f):5d} 张 / {f.patient.nunique():4d} 人 / "
              f"肺炎 {f.label.mean():.3f}")
    print("训练集整体肺炎占比: %.3f" % train.label.mean())
    print("测试集整体肺炎占比: %.3f" % test.label.mean())
    p = pooled[pooled.part == 0]
    q = pooled[pooled.part == 1]
    print(f"70/30 重划分: 训练 {len(p)} 张 ({len(set(p.patient))} 人, 肺炎 {p.label.mean():.3f}) | "
          f"测试 {len(q)} 张 ({len(set(q.patient))} 人, 肺炎 {q.label.mean():.3f})")
    assert not (set(p.patient) & set(q.patient))
    print("=" * 68)


if __name__ == "__main__":
    main()
