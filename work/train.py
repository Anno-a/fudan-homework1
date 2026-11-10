# -*- coding: utf-8 -*-
"""
AIT410020 PJ2 任务二 —— 训练脚本

两个方案：
    方案 A  smallcnn   : 自建 4 块 CNN，从零训练，输入 128×128
    方案 B  resnet18   : ImageNet 预训练 ResNet18，先冻结 backbone 再全网络微调，输入 224×224

用法（在「小组期末作业」目录下执行）：

    # 方案B 第0折
    python work/train.py --scheme resnet18 --fold 0 --epochs 15 --tag B_fold0

    # 方案A 第0折
    python work/train.py --scheme smallcnn --fold 0 --epochs 25 --tag A_fold0

    # 全量训练集训练 + 官方测试集评估（最好先用交叉验证定 epoch 数）
    python work/train.py --scheme resnet18 --fold all --epochs 15 --tag B_full --eval-test

    # 题目要求的 70/30 病人级重划分（敏感性分析）
    python work/train.py --scheme resnet18 --split pooled --epochs 15 --tag B_pool70

    # 30 秒冒烟测试（确认环境没问题）
    python work/train.py --scheme resnet18 --fold 0 --epochs 1 --smoke

产物：work/runs/<tag>/{best.pt, metrics.json, preds_val.csv, preds_test.csv, log.txt}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "chest_xray"
SPLITS = ROOT / "work" / "splits"
RUNS = ROOT / "work" / "runs"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
class Letterbox:
    """等比缩放 + 居中黑边填充到 size×size。

    本数据集长宽比 0.84~3.38，直接 Resize 成正方形会把肺野压扁；
    胸片四周本来就是黑的，用黑边填充不引入伪影。
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, im: Image.Image) -> Image.Image:
        im = im.convert("L")
        w, h = im.size
        s = self.size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("L", (self.size, self.size), 0)
        canvas.paste(im, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


class CropSquare:
    """中心裁剪成正方形后缩放，完全不产生黑边（用于检验 letterbox 黑边是否被当成捷径）。"""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, im: Image.Image) -> Image.Image:
        im = im.convert("L")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        return im.crop((left, top, left + s, top + s)).resize((self.size, self.size), Image.BILINEAR)


def build_transforms(img_size: int, train: bool, aug: str = "full", prep: str = "letterbox"):
    ops = [CropSquare(img_size) if prep == "crop" else Letterbox(img_size)]
    if train:
        if aug == "full":
            ops += [
                T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.11)),
                T.RandomAffine(degrees=7, translate=(0.05, 0.05)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        elif aug == "light":
            ops += [T.RandomHorizontalFlip(p=0.5)]
    ops += [T.Grayscale(num_output_channels=3), T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class ChestXRay(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = [DATA / p for p in df["relpath"]]
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.meta = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im)
        return x, int(self.labels[i]), i


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """自建基线：4 个 conv block（Conv-BN-ReLU ×2 + MaxPool）+ GAP + FC，约 1.2M 参数。"""

    def __init__(self, in_ch: int = 3, num_classes: int = 2, width: tuple = (32, 64, 128, 256),
                 p_drop: float = 0.3):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        layers, ci = [], in_ch
        for co in width:
            layers.append(block(ci, co))
            ci = co
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(p_drop), nn.Linear(ci, num_classes))

    def forward(self, x):
        return self.head(self.features(x))

    def param_groups(self, lr_head: float, lr_backbone: float):
        return [{"params": self.parameters(), "lr": lr_head}]


def build_model(scheme: str) -> nn.Module:
    if scheme == "smallcnn":
        return SmallCNN()
    if scheme == "resnet18":
        import timm
        # 国内网络走镜像；已缓存或能直连 huggingface.co 时该设置无害
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        m = timm.create_model("resnet18", pretrained=True, num_classes=2)
        return m
    raise ValueError(scheme)


def set_backbone_frozen(model: nn.Module, scheme: str, frozen: bool) -> None:
    """冻结/解冻特征提取部分，只保留最后的分类头可训练。"""
    if scheme == "smallcnn":
        return
    for name, p in model.named_parameters():
        if not name.startswith("fc.") and not name.startswith("head."):
            p.requires_grad = not frozen


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_hat = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "sensitivity": float(recall_score(y_true, y_hat, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, ys, idxs = [], [], []
    for x, y, i in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
        idxs.append(i.numpy())
    return np.concatenate(ys), np.concatenate(probs), np.concatenate(idxs)


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def split_by_patient(df: pd.DataFrame, val_frac: float, seed: int):
    pts = df.groupby("patient")["label"].first()
    tr_idx, va_idx = train_test_split(pts.index, test_size=val_frac,
                                      stratify=pts.values, random_state=seed)
    return (df[df.patient.isin(tr_idx)].copy(),
            df[df.patient.isin(va_idx)].copy())


def resolve_frames(args):
    if args.split == "pooled":
        df = pd.read_csv(SPLITS / "pooled_70_30.csv")
        pool_tr, test = df[df.part == 0].copy(), df[df.part == 1].copy()
        tr, va = split_by_patient(pool_tr, 0.1, args.seed)
        return tr, va, test
    tr_all = pd.read_csv(SPLITS / "official_train.csv")
    test = pd.read_csv(SPLITS / "official_test.csv")
    if args.fold == "all":
        return tr_all.copy(), None, test
    k = int(args.fold)
    return tr_all[tr_all.fold != k].copy(), tr_all[tr_all.fold == k].copy(), test


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["smallcnn", "resnet18"], default="resnet18")
    ap.add_argument("--split", choices=["official", "pooled"], default="official")
    ap.add_argument("--fold", default="0", help="0-4，或 all 表示用全部官方训练集")
    ap.add_argument("--img", type=int, default=None, help="输入尺寸，默认 A=128 / B=224")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="默认 A=1e-3 / B(head)=1e-3")
    ap.add_argument("--backbone-lr", type=float, default=1e-4, help="解冻后 backbone 的学习率")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--freeze-epochs", type=int, default=3, help="方案B 先冻结 backbone 的轮数")
    ap.add_argument("--aug", choices=["full", "light", "none"], default="full")
    ap.add_argument("--prep", choices=["letterbox", "crop"], default="letterbox")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--eval-test", action="store_true", help="训练结束后在测试集上评估一次")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：200 张图、1 轮")
    args = ap.parse_args()

    img_size = args.img or (128 if args.scheme == "smallcnn" else 224)
    lr = args.lr or 1e-3
    tag = args.tag or f"{args.scheme}_{args.split}_fold{args.fold}_{img_size}"
    if args.smoke:
        args.epochs, tag = 1, tag + "_smoke"
    out = RUNS / tag
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_df, va_df, te_df = resolve_frames(args)
    if args.smoke:
        tr_df = tr_df.groupby("label").head(100)
        va_df = va_df.groupby("label").head(50) if va_df is not None else None

    log_lines = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[{tag}] scheme={args.scheme} split={args.split} fold={args.fold} "
        f"img={img_size} epochs={args.epochs} device={device} threads={args.threads}")
    log(f"train {len(tr_df)} 张/{tr_df.patient.nunique()} 人 | "
        f"val {len(va_df) if va_df is not None else 0} 张 | test {len(te_df)} 张")

    # --- 数据 ---------------------------------------------------------------
    tr_tf = build_transforms(img_size, True, args.aug, args.prep)
    ev_tf = build_transforms(img_size, False, prep=args.prep)
    ds_tr = ChestXRay(tr_df, tr_tf)
    ds_va = ChestXRay(va_df, ev_tf) if va_df is not None else None
    ds_te = ChestXRay(te_df, ev_tf)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       drop_last=False, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers) \
        if ds_va is not None else None
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    # --- 模型 ---------------------------------------------------------------
    model = build_model(args.scheme).to(device)
    frozen = args.scheme == "resnet18" and args.freeze_epochs > 0
    set_backbone_frozen(model, args.scheme, frozen)
    log(f"参数量: {count_params(model)}  冻结backbone={frozen}")

    if args.class_weight == "balanced":
        cnt = np.bincount(tr_df.label.to_numpy(), minlength=2).astype(float)
        w = torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float32, device=device)
        log(f"类别权重: 正常{w[0]:.3f} 肺炎{w[1]:.3f}")
    else:
        w = None
    criterion = nn.CrossEntropyLoss(weight=w)

    head = [p for n, p in model.named_parameters() if n.startswith(("fc.", "head."))]
    opt = torch.optim.AdamW([
        {"params": head, "lr": lr},
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith(("fc.", "head."))], "lr": args.backbone_lr},
    ], weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # --- 训练循环 -----------------------------------------------------------
    best = {"auc": -1.0, "epoch": -1}
    bad = 0
    hist = []
    for ep in range(1, args.epochs + 1):
        if frozen and ep == args.freeze_epochs + 1:
            set_backbone_frozen(model, args.scheme, False)
            log(f"  epoch {ep}: 解冻 backbone，参数量 {count_params(model)}")
            frozen = False

        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            nb += len(y)
        sched.step()

        rec = {"epoch": ep, "train_loss": tot / nb, "lr": sched.get_last_lr()[0]}
        if dl_va is not None:
            yv, pv, _ = evaluate(model, dl_va, device)
            m = compute_metrics(yv, pv)
            rec.update({f"val_{k}": v for k, v in m.items()})
            if m["auc"] > best["auc"]:
                best = {"auc": m["auc"], "epoch": ep}
                torch.save(model.state_dict(), out / "best.pt")
                bad = 0
            else:
                bad += 1
        else:
            # fold=all 没有验证集，只保留最后一轮，避免 15×45MB 的权重堆积
            torch.save(model.state_dict(), out / "last.pt")
        hist.append(rec)
        log(f"  epoch {ep:2d} | loss {rec['train_loss']:.4f} | "
            + (f"val AUC {m['auc']:.4f} acc {m['accuracy']:.4f} sen {m['sensitivity']:.4f} "
               f"spe {m['specificity']:.4f} | {time.time() - t0:.0f}s" if dl_va is not None
               else f"{time.time() - t0:.0f}s"))

        if dl_va is not None and bad >= args.patience:
            log(f"  early stop @ epoch {ep} (best AUC {best['auc']:.4f} @ ep{best['epoch']})")
            break

    # --- 评估 ---------------------------------------------------------------
    result = {"args": vars(args), "tag": tag, "img_size": img_size,
              "params": count_params(model), "history": hist, "best": best}

    if dl_va is not None and (out / "best.pt").exists():
        model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
        yv, pv, iv = evaluate(model, dl_va, device)
        result["val_metrics"] = compute_metrics(yv, pv)
        pd.DataFrame({
            "relpath": ds_va.meta.loc[iv, "relpath"].values,
            "patient": ds_va.meta.loc[iv, "patient"].values,
            "subtype": ds_va.meta.loc[iv, "subtype"].values,
            "y_true": yv, "y_prob": pv,
        }).to_csv(out / "preds_val.csv", index=False, encoding="utf-8")
        log(f"best val: {json.dumps(result['val_metrics'], ensure_ascii=False)}")

    if args.eval_test:
        if not (out / "best.pt").exists():
            # fold=all 时没有 val，用最后一轮权重
            model.load_state_dict(
                torch.load(out / "last.pt", map_location=device, weights_only=True))
        yt, pt, it = evaluate(model, dl_te, device)
        result["test_metrics"] = compute_metrics(yt, pt)
        pd.DataFrame({
            "relpath": ds_te.meta.loc[it, "relpath"].values,
            "patient": ds_te.meta.loc[it, "patient"].values,
            "subtype": ds_te.meta.loc[it, "subtype"].values,
            "y_true": yt, "y_prob": pt,
        }).to_csv(out / "preds_test.csv", index=False, encoding="utf-8")
        log(f"test: {json.dumps(result['test_metrics'], ensure_ascii=False)}")

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(out / "log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"结果已保存到 {out}")


if __name__ == "__main__":
    main()
