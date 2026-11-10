# -*- coding: utf-8 -*-
"""一次性跑完全部实验（方案 A / B，5 折 + 全量重训 + 70/30 敏感性分析）。

    python work/run_all.py

· 已有 work/runs/<tag>/metrics.json 的步骤自动跳过，可以反复运行、中断后续跑。
· 全部日志写到 work/logs/run_all.log。
· 跑完自动生成 work/results_summary.md。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "runs"
LOGS = ROOT / "work" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
TRAIN = ROOT / "work" / "train.py"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOG_PATH = LOGS / "run_all.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(tag: str) -> bool:
    return (RUNS / tag / "metrics.json").exists()


def run_step(tag: str, args: list[str], force: bool = False) -> bool:
    if done(tag) and not force:
        log(f"跳过 {tag}（已完成）")
        return True
    log(f"开始 {tag}: {' '.join(args)}")
    t0 = time.time()
    cmd = [sys.executable, str(TRAIN)] + args
    with open(LOGS / f"{tag}.log", "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    if p.returncode != 0:
        log(f"!! {tag} 失败（退出码 {p.returncode}），用时 {dt:.1f} 分钟，详见 logs/{tag}.log")
        return False
    if done(tag):
        with open(RUNS / tag / "metrics.json", encoding="utf-8") as f:
            m = json.load(f)
        auc = m.get("val_metrics", {}).get("auc")
        tauc = m.get("test_metrics", {}).get("auc")
        extra = f" | val AUC {auc:.4f}" if auc else ""
        extra += f" | test AUC {tauc:.4f}" if tauc else ""
        log(f"完成 {tag}，用时 {dt:.1f} 分钟{extra}")
    else:
        log(f"完成 {tag}，用时 {dt:.1f} 分钟（无 metrics.json）")
    return True


def best_epoch(tag: str, default: int) -> int:
    p = RUNS / tag / "metrics.json"
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return int(json.load(f)["best"]["epoch"])


def main() -> None:
    log("=" * 60)
    log("开始跑全部实验")

    schemes = [
        ("A", "smallcnn", 128, 25),
        ("B", "resnet18", 224, 15),
    ]

    # --- 1. 5 折交叉验证 -------------------------------------------------- #
    for key, scheme, img, epochs in schemes:
        for k in range(5):
            run_step(f"{key}_fold{k}",
                     ["--scheme", scheme, "--fold", str(k), "--epochs", str(epochs),
                      "--img", str(img), "--tag", f"{key}_fold{k}"])

    # --- 2. 全量重训 + 官方测试集 ------------------------------------------ #
    # 注意：--fold all 的 run 没有验证集，metrics.json 里的 best.epoch 会是 -1，
    # 所以轮数必须从 5 折结果里取中位数，不能从 <key>_test 里读。
    final_eps: dict[str, int] = {}
    for key, scheme, img, epochs in schemes:
        eps = []
        for k in range(5):
            p = RUNS / f"{key}_fold{k}" / "metrics.json"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    eps.append(int(json.load(f)["best"]["epoch"]))
        e = int(sorted(eps)[len(eps) // 2]) if eps else epochs
        final_eps[key] = e
        log(f"{key}: 5 折最优轮数 {eps}，全量重训取 {e} 轮")
        run_step(f"{key}_test",
                 ["--scheme", scheme, "--fold", "all", "--epochs", str(e),
                  "--img", str(img), "--tag", f"{key}_test", "--eval-test"])

    # --- 3. 70/30 敏感性分析 ------------------------------------------------ #
    for key, scheme, img, _ in schemes:
        run_step(f"{key}_pool70",
                 ["--scheme", scheme, "--split", "pooled", "--epochs", str(final_eps[key]),
                  "--img", str(img), "--tag", f"{key}_pool70", "--eval-test"])

    log("全部步骤结束")
    make_summary()


def make_summary() -> None:
    rows = []
    for tag in sorted(p.name for p in RUNS.iterdir() if (p / "metrics.json").exists()):
        with open(RUNS / tag / "metrics.json", encoding="utf-8") as f:
            m = json.load(f)
        row = {"tag": tag, "方案": "A 自建 SmallCNN" if tag.startswith("A") else "B ResNet18",
               "类型": ("5折" if "_fold" in tag else
                        "测试集" if tag.endswith("_test") else "70/30"),
               "最优轮": m.get("best", {}).get("epoch")}
        for split in ("val", "test"):
            mm = m.get(f"{split}_metrics")
            if mm:
                for k in ("auc", "accuracy", "sensitivity", "specificity", "f1"):
                    row[f"{split}_{k}"] = round(mm.get(k, float("nan")), 4)
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / "work" / "results_summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 实验结果汇总\n\n")
        f.write(f"生成时间：{datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## 5 折均值 ± 标准差\n\n")
        for key, name in [("A", "A 自建 SmallCNN"), ("B", "B ResNet18")]:
            sub = df[df["类型"] == "5折"]
            sub = sub[sub["tag"].str.startswith(key)]
            if len(sub) == 0:
                continue
            f.write(f"\n**{name}**\n\n")
            for k in ("val_auc", "val_accuracy", "val_sensitivity", "val_specificity", "val_f1"):
                if k in sub:
                    f.write(f"- {k}: {sub[k].mean():.4f} ± {sub[k].std(ddof=0):.4f}\n")
    log(f"汇总已写入 {out}")


if __name__ == "__main__":
    main()
