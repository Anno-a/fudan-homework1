# -*- coding: utf-8 -*-
"""建立 256×256 letterbox 预处理缓存（notebook 第一次运行会自动做，也可以提前跑）。

    python work/build_cache.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "chest_xray"
CACHE = ROOT / "work" / "cache"
SIZE = 256


def letterbox(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("L")
    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rels = sorted(p.relative_to(DATA).as_posix() for p in DATA.glob("*/*/*.jpeg"))
    print(f"共 {len(rels)} 张图，输出到 {CACHE}")
    arr = np.zeros((len(rels), SIZE, SIZE), dtype=np.uint8)
    t0 = time.time()
    for i, rel in enumerate(rels):
        with Image.open(DATA / rel) as im:
            arr[i] = np.asarray(letterbox(im, SIZE))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(rels)}  {time.time() - t0:.0f}s", flush=True)
    np.save(CACHE / f"img{SIZE}.npy", arr)
    pd.DataFrame({"relpath": rels, "row": np.arange(len(rels))}).to_csv(
        CACHE / f"img{SIZE}_index.csv", index=False, encoding="utf-8")
    print(f"完成，用时 {time.time() - t0:.0f}s，"
          f"文件大小 {(CACHE / f'img{SIZE}.npy').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
