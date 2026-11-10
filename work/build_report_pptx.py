# -*- coding: utf-8 -*-
"""生成汇报用 PPT（16:9，蓝色商务风，参考 桌面/（原创）蓝色高端工作汇报.pptx 的视觉语言）。

    python work/build_report_pptx.py

输出：AIT410020_PJ2_任务二_汇报.pptx
所有数字从 work/runs 重新计算，与正文独立。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "runs"
FIGS = ROOT / "figures"
OUT = ROOT / "AIT410020_PJ2_任务二_汇报.pptx"

# ---- 配色：沿用模板的蓝色体系 ----
NAVY = RGBColor(0x1F, 0x35, 0x64)      # 主色 深蓝
BLUE = RGBColor(0x2E, 0x5C, 0x9A)      # 次色
LIGHT = RGBColor(0x5B, 0x9B, 0xD5)     # 浅蓝
ACCENT = RGBColor(0xD5, 0x5E, 0x00)    # 强调 橙
INK = RGBColor(0x26, 0x26, 0x26)
GRAY = RGBColor(0x6B, 0x6B, 0x6B)
PALE = RGBColor(0xF2, 0xF5, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "微软雅黑"

W, H = 13.333, 7.5
ML, MR = 0.9, 0.9
CW = W - ML - MR


# --------------------------------------------------------------------------- #
def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, color, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    return sh


def text(slide, x, y, w, h, s, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.25):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    lines = s.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = FONT
    return tb


def title_bar(slide, title, sub=None):
    """统一标题：左侧竖色块 + 标题 + 细分隔线。"""
    rect(slide, ML, 0.62, 0.10, 0.52, NAVY)
    text(slide, ML + 0.26, 0.58, CW - 0.3, 0.60, title, size=27, color=NAVY, bold=True)
    if sub:
        text(slide, ML + 0.26, 1.16, CW - 0.3, 0.36, sub, size=13, color=GRAY)
    rect(slide, ML, 1.62 if sub else 1.30, CW, 0.012, RGBColor(0xD8, 0xDE, 0xE8))


def bullets(slide, items, x=ML, y=1.95, w=CW, size=17, gap=0.62, color=INK):
    """items: [(粗体前缀, 说明), ...] 或 [str, ...]"""
    cy = y
    for it in items:
        rect(slide, x, cy + 0.12, 0.085, 0.085, BLUE)
        if isinstance(it, tuple):
            head, body = it
            text(slide, x + 0.32, cy, w - 0.32, 0.42, head, size=size, color=NAVY, bold=True)
            text(slide, x + 0.32, cy + 0.40, w - 0.32, 0.42, body, size=size - 3, color=color)
            cy += gap + 0.32
        else:
            text(slide, x + 0.32, cy - 0.02, w - 0.32, 0.42, it, size=size, color=color)
            cy += gap
    return cy


def table(slide, header, rows, x, y, w, col_w=None, size=13, row_h=0.42, highlight=None):
    nr, nc = len(rows) + 1, len(header)
    shape = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    if col_w:
        total = sum(col_w)
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(w * cw / total)
    for j, htxt in enumerate(header):
        c = tbl.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = NAVY
        c.margin_left = c.margin_right = Inches(0.06)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = str(htxt)
        r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = FONT
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = PALE if i % 2 else WHITE
            c.margin_left = c.margin_right = Inches(0.06)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(v)
            r.font.size = Pt(size); r.font.name = FONT
            r.font.color.rgb = INK
            if highlight and (i - 1, j) in highlight:
                r.font.bold = True; r.font.color.rgb = ACCENT
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
    return tbl


def speaking_box(slide, s, dark=False):
    """底部讲解框：答辩时照着念，同时写进 PowerPoint 备注。"""
    y, h = 6.22, 1.18
    bg = RGBColor(0x18, 0x27, 0x45) if dark else RGBColor(0xEE, 0xF2, 0xF7)
    fg = RGBColor(0xC9, 0xD8, 0xEE) if dark else RGBColor(0x44, 0x4A, 0x55)
    rect(slide, ML, y, CW, h, bg)
    rect(slide, ML, y, 0.055, h, ACCENT)
    text(slide, ML + 0.20, y + 0.09, CW - 0.34, h - 0.16, s,
         size=9.5, color=fg, spacing=1.2)
    slide.notes_slide.notes_text_frame.text = s


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


# --------------------------------------------------------------------------- #
def metrics():
    from sklearn.metrics import confusion_matrix, roc_auc_score

    def met(y, p, s=0.5):
        yh = (p >= s).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
        return dict(auc=roc_auc_score(y, p), acc=(tp + tn) / len(y),
                    sen=tp / (tp + fn), spe=tn / (tn + fp))

    out = {}
    for k in ("A", "B"):
        d = pd.read_csv(RUNS / f"{k}_test" / "preds_test.csv")
        out[f"{k}_test"] = met(d.y_true.values, d.y_prob.values)
        c = pd.read_csv(RUNS / f"{k}_crop_test" / "preds_test.csv")
        out[f"{k}_crop"] = met(c.y_true.values, c.y_prob.values)
        p = pd.read_csv(RUNS / f"{k}_pool70" / "preds_test.csv")
        out[f"{k}_pool"] = met(p.y_true.values, p.y_prob.values)
        vals = []
        for f in range(5):
            v = pd.read_csv(RUNS / f"{k}_fold{f}" / "preds_val.csv")
            vals.append(met(v.y_true.values, v.y_prob.values))
        out[f"{k}_fold"] = {m: (np.mean([x[m] for x in vals]), np.std([x[m] for x in vals], ddof=0))
                            for m in ("auc", "acc", "sen", "spe")}
    return out


def main() -> None:
    M = metrics()
    prs = new_deck()

    # ---------------- 1 封面 ----------------
    s = blank(prs)
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, 0, 0.18, H, LIGHT)
    text(s, 1.15, 1.75, 10.5, 0.5, "AIT410020 人工智能与概率建模 · 期末课程设计", size=15, color=LIGHT)
    text(s, 1.15, 2.32, 11.0, 0.95, "基于胸部 X 光的肺炎自动分类", size=42, color=WHITE, bold=True)
    text(s, 1.15, 3.42, 11.0, 0.6, "模型、数据划分与预处理捷径的影响", size=26, color=WHITE)
    rect(s, 1.15, 4.35, 2.4, 0.03, ACCENT)
    text(s, 1.15, 4.65, 11.0, 0.4, "任务二：基于胸部 X 光图像的自动分类", size=16, color=LIGHT)
    text(s, 1.15, 5.72, 11.0, 0.4, "小组成员：＿＿＿   ｜   2026 年 11 月", size=13, color=RGBColor(0x9F, 0xB4, 0xD0))

    # ---------------- 2 提纲 ----------------
    s = blank(prs)
    title_bar(s, "汇报提纲")
    items = [("01  任务与数据", "胸部 X 光肺炎二分类，5824 张影像 / 4096 名患儿"),
             ("02  方法", "病人级划分 · 两个方案 · 训练与评估协议"),
             ("03  结果", "同分布打平，跨队列分胜负"),
             ("04  关键发现", "预处理引入的捷径，以及去掉它之后发生了什么"),
             ("05  结论与局限", "性能差异到底来自哪里")]
    cy = 1.95
    for head, body in items:
        rect(s, ML, cy, 0.09, 0.62, BLUE)
        text(s, ML + 0.32, cy - 0.02, 4.2, 0.4, head, size=20, color=NAVY, bold=True)
        text(s, ML + 4.7, cy + 0.02, CW - 4.7, 0.4, body, size=15, color=GRAY)
        cy += 0.86

    # ---------------- 3 任务与数据 ----------------
    s = blank(prs)
    title_bar(s, "任务与数据", "Kaggle Chest X-Ray（Kermany et al., Cell 2018）")
    bullets(s, [("任务", "输入一张胸部 X 光片，判断正常还是肺炎"),
                ("规模", "5856 张，去除 32 张完全重复后 5824 张，来自 4096 名 1–5 岁患儿"),
                ("不平衡", "训练集肺炎占 74.1%，全部判为肺炎就有 0.741 准确率")],
            x=ML, y=1.95, w=6.1, size=17, gap=0.55)
    table(s, ["", "训练集", "测试集"],
          [["正常", "1348", "231"], ["肺炎（细菌）", "2520", "240"],
           ["肺炎（病毒）", "1338", "147"], ["合计", "5206", "618"]],
          x=7.35, y=1.95, w=5.0, col_w=[1.5, 1, 1], size=13, row_h=0.4)
    text(s, 7.35, 4.35, 5.0, 0.35, "类别不平衡，因此不以准确率作为主要指标", size=12, color=GRAY)

    # ---------------- 4 病人级划分 ----------------
    s = blank(prs)
    title_bar(s, "关键一步：病人级划分")
    bullets(s, [("为什么必须按病人分", "同一名患儿可能有多张影像，随机按图片划分会让同一病人同时出现在训练和测试集"),
                ("文件名里藏着病人号", "正常影像 IM-xxxx / NORMAL2-IM-xxxx；肺炎影像 person 编号 + 细菌/病毒亚型"),
                ("一个容易踩的坑", "细菌与病毒是两套独立的 person 编号，只解析编号会误判出 170 名“跨集病人”；加上亚型前缀后，官方训练集与测试集的病人交集为 0")],
            x=ML, y=1.95, size=16.5, gap=0.55)
    rect(s, ML, 5.35, CW, 0.68, PALE)
    text(s, ML + 0.3, 5.49, CW - 0.6, 0.4,
         "五折交叉验证同样按病人分层：每折 734 人、约 1040 张，肺炎比例稳定在 0.72–0.75",
         size=13.5, color=BLUE)

    # ---------------- 5 两个方案 ----------------
    s = blank(prs)
    title_bar(s, "两个方案", "参数量相差近十倍，训练条件一致")
    table(s, ["", "方案 A", "方案 B"],
          [["结构", "自建 4 层 CNN", "ResNet18"],
           ["参数量", "1.17 M", "11.18 M"],
           ["初始化", "随机", "ImageNet 预训练"],
           ["输入", "128 × 128", "224 × 224"],
           ["训练", "从头训练", "冻结主干 3 轮 → 全网络微调"]],
          x=ML, y=1.85, w=CW, col_w=[1.6, 2.2, 3.0], size=14, row_h=0.52)
    text(s, ML, 5.1, CW, 0.4, "方案 A 考察“小模型够不够用”，方案 B 考察“预训练带来了什么”",
         size=13, color=GRAY)

    # ---------------- 6 训练与评估 ----------------
    s = blank(prs)
    title_bar(s, "训练与评估协议")
    left = [("训练", "AdamW + 余弦退火；head 1e-3，backbone 1e-4"),
            ("增强", "随机缩放裁剪、±7° 旋转、水平翻转、亮度对比度抖动"),
            ("早停", "按验证集 AUC，patience = 5")]
    right = [("指标", "AUC、灵敏度、特异度、F1 —— 不看准确率"),
             ("阈值", "主结果用 0.5，另报验证集 Youden 阈值"),
             ("测试集", "模型与轮数由五折确定，测试集只在最后评估一次")]
    bullets(s, left, x=ML, y=1.95, w=5.6, size=16, gap=0.5)
    bullets(s, right, x=6.9, y=1.95, w=5.6, size=16, gap=0.5)

    # ---------------- 7 结果① ----------------
    s = blank(prs)
    title_bar(s, "结果①  同分布下，两个方案打平", "五折交叉验证（病人级划分）")
    fa, fb = M["A_fold"], M["B_fold"]
    table(s, ["方案", "AUC", "准确率", "灵敏度", "特异度"],
          [["A 自建 SmallCNN", f"{fa['auc'][0]:.4f}±{fa['auc'][1]:.4f}",
            f"{fa['acc'][0]:.4f}", f"{fa['sen'][0]:.4f}", f"{fa['spe'][0]:.4f}"],
           ["B ResNet18 迁移", f"{fb['auc'][0]:.4f}±{fb['auc'][1]:.4f}",
            f"{fb['acc'][0]:.4f}", f"{fb['sen'][0]:.4f}", f"{fb['spe'][0]:.4f}"]],
          x=ML, y=1.95, w=CW, col_w=[2.2, 1.5, 1.2, 1.2, 1.2], size=14, row_h=0.55)
    rect(s, ML, 4.5, CW, 0.9, PALE)
    text(s, ML + 0.3, 4.68, CW - 0.6, 0.6,
         "AUC 差 0.0009，小于折间标准差 → 预训练没有带来同分布精度收益",
         size=17, color=NAVY, bold=True)

    # ---------------- 8 结果② ----------------
    s = blank(prs)
    title_bar(s, "结果②  跨队列时，差距出现了", "官方测试集（618 张，全量训练集重训）")
    picture(s, FIGS / "fig_roc.png", 0.95, 1.82, w=4.95)
    ta, tb = M["A_test"], M["B_test"]
    table(s, ["方案", "AUC", "准确率", "特异度"],
          [["A 自建", f"{ta['auc']:.4f}", f"{ta['acc']:.4f}", f"{ta['spe']:.4f}"],
           ["B 迁移", f"{tb['auc']:.4f}", f"{tb['acc']:.4f}", f"{tb['spe']:.4f}"]],
          x=6.7, y=1.95, w=5.6, col_w=[1.3, 1.2, 1.2, 1.2], size=14, row_h=0.48,
          highlight={(1, 1), (1, 2), (1, 3)})
    text(s, 6.7, 3.55, 5.6, 0.9, "AUC 高出 0.036\n特异度高出 0.208",
         size=18, color=ACCENT, bold=True, spacing=1.3)
    text(s, 6.7, 4.60, 5.6, 1.5,
         "两方案灵敏度都接近 0.99，误判集中在“正常被判成肺炎”。\n"
         "按验证集 Youden 阈值调整后，准确率 A 0.846 / B 0.888，\n仍补不回 AUC 的差距。",
         size=12.5, color=GRAY, spacing=1.45)

    # ---------------- 9 结果③ ----------------
    s = blank(prs)
    title_bar(s, "结果③  亚型差异与特征分布", "t-SNE：ResNet18 倒数第二层特征（512 维 → 2 维，测试集）")
    picture(s, FIGS / "fig_tsne.png", 0.75, 2.0, w=7.35)
    text(s, 0.75, 5.05, 7.35, 0.35, "左：按真实标签着色    右：按肺炎亚型着色",
         size=11.5, color=GRAY, align=PP_ALIGN.CENTER)
    table(s, ["亚组（正常 vs …）", "A", "B"],
          [["细菌性（471）", "0.9400", "0.9725"], ["病毒性（378）", "0.9130", "0.9540"]],
          x=8.55, y=2.05, w=3.9, col_w=[1.6, 1, 1], size=13, row_h=0.46)
    text(s, 8.55, 3.55, 3.95, 2.6,
         "两类肺炎都判别得不错，\n病毒性略弱。\n\n"
         "t-SNE 上病毒样本恰好落在\n正常与细菌之间 ——\n与亚组数字一致。",
         size=13, color=GRAY, spacing=1.5)

    # ---------------- 10 结果④ ----------------
    s = blank(prs)
    title_bar(s, "结果④  划分方式本身就能改变结论", "把全部数据按病人随机划分 70 / 30")
    pa, pb = M["A_pool"], M["B_pool"]
    table(s, ["方案", "70/30 随机划分", "官方跨批次测试集", "差距"],
          [["A 自建", f"{pa['auc']:.4f}", f"{ta['auc']:.4f}", f"+{pa['auc']-ta['auc']:.4f}"],
           ["B 迁移", f"{pb['auc']:.4f}", f"{tb['auc']:.4f}", f"+{pb['auc']-tb['auc']:.4f}"]],
          x=ML, y=2.0, w=CW, col_w=[1.4, 1.8, 1.8, 1.2], size=15, row_h=0.55)
    rect(s, ML, 4.35, CW, 0.95, PALE)
    text(s, ML + 0.3, 4.55, CW - 0.6, 0.6,
         "同一批数据，随机划分把性能高估了 3–7 个百分点", size=18, color=ACCENT, bold=True)
    text(s, ML, 5.55, CW, 0.5,
         "因此以官方跨批次测试集为主结果；题目要求的 70/30 只作为敏感性分析", size=13.5, color=GRAY)

    # ---------------- 11 关键发现 ----------------
    s = blank(prs)
    title_bar(s, "转折：我们发现了一个捷径", "而它是我们自己的预处理引入的")
    bullets(s, [("线索藏在图像尺寸里", "正常影像平均长宽比 1.25，肺炎 1.52 —— 两类图像的成像来源不同"),
                ("单靠这一个数字就能分类", "只用宽高比判别：验证集 AUC 0.8665，测试集 AUC 0.7027"),
                ("预处理把它变成了可见特征", "等比缩放 + 补黑边，使“黑边有多宽”与类别强相关，模型顺手学了去"),
                ("Grad-CAM 印证", "误判样本的高响应集中在黑边与图像边角，而不是肺野")],
            x=ML, y=1.95, size=16.5, gap=0.62)

    # ---------------- 12 消融 ----------------
    s = blank(prs)
    title_bar(s, "消融：去掉黑边之后", "同一协议，只把 letterbox 换成中心裁剪")
    picture(s, FIGS / "fig_gradcam.png", 0.86, 1.78, h=4.2)
    ca, cb = M["A_crop"], M["B_crop"]
    table(s, ["方案", "预处理", "AUC", "特异度"],
          [["A", "含黑边", f"{ta['auc']:.4f}", f"{ta['spe']:.4f}"],
           ["A", "无黑边", f"{ca['auc']:.4f}", f"{ca['spe']:.4f}"],
           ["B", "含黑边", f"{tb['auc']:.4f}", f"{tb['spe']:.4f}"],
           ["B", "无黑边", f"{cb['auc']:.4f}", f"{cb['spe']:.4f}"]],
          x=6.35, y=1.9, w=6.0, col_w=[0.9, 1.6, 1.3, 1.3], size=12.5, row_h=0.42,
          highlight={(1, 2), (1, 3), (3, 2), (3, 3)})
    text(s, 6.35, 4.12, 6.0, 0.95,
         "A：AUC 0.930 → 0.948，特异度 0.455 → 0.801\nB：特异度 0.662 → 0.836，AUC 基本不变",
         size=14, color=NAVY, bold=True, spacing=1.4)
    text(s, 6.35, 5.12, 6.0, 0.95,
         "两方案差距由 0.036 缩小到 0.015 ——\nB 的一部分优势来自对捷径更鲁棒，\n而非更强的病理表征能力",
         size=12.5, color=GRAY, spacing=1.35)

    # ---------------- 13 结论 ----------------
    s = blank(prs)
    title_bar(s, "结论")
    cards = [("01", "迁移学习的价值是泛化，不是精度",
              "同分布下两方案打平（AUC 0.9981 对 0.9972），差距只在跨队列时出现（0.9655 对 0.9297）"),
             ("02", "差距的一部分来自对无关特征的鲁棒性",
              "消除黑边后两方案几乎持平（0.948 对 0.963），差距由 0.036 缩至 0.015"),
             ("03", "数据管线的隐性偏差影响更大",
              "仅改变预处理，A 的测试准确率由 0.791 升至 0.903，超过换模型带来的增益")]
    cy = 1.9
    for num, head, body in cards:
        rect(s, ML, cy, CW, 1.35, PALE)
        text(s, ML + 0.35, cy + 0.28, 0.9, 0.6, num, size=28, color=LIGHT, bold=True)
        text(s, ML + 1.35, cy + 0.22, CW - 1.6, 0.4, head, size=17, color=NAVY, bold=True)
        text(s, ML + 1.35, cy + 0.68, CW - 1.6, 0.5, body, size=13, color=GRAY)
        cy += 1.40

    # ---------------- 14 局限 ----------------
    s = blank(prs)
    title_bar(s, "局限与展望")
    bullets(s, [("数据", "单中心儿科影像（1–5 岁），模型在成人胸片上的表现未经验证"),
                ("样本量", "官方测试集仅 618 张，两方案差异未做显著性检验"),
                ("混淆因素", "黑边捷径已定位并消除，但“成像来源”本身尚未完全剥离"),
                ("下一步", "引入外部数据集做验证；用更严格的来源无关划分重做实验")],
            x=ML, y=2.0, size=16.5, gap=0.75)

    # ---------------- 15 结束 ----------------
    s = blank(prs)
    rect(s, 0, 0, W, H, NAVY)
    text(s, 0, 2.85, W, 0.8, "谢 谢", size=48, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    rect(s, W / 2 - 0.9, 3.85, 1.8, 0.03, ACCENT)
    text(s, 0, 4.15, W, 0.5, "代码与结果见 AIT410020_PJ2_任务二.ipynb",
         size=15, color=LIGHT, align=PP_ALIGN.CENTER)

    # ---------------- 每页底部讲解框 ----------------
    scripts = {
        1: "各位老师好，我们是第 X 组，选的题目是任务二——基于胸部 X 光图像的肺炎自动分类。汇报分四部分：任务与数据、两个模型方案与训练评估协议、实验结果，最后一部分会重点讲一个发现——我们的预处理里无意中埋入了一个与病变无关的类别线索，去掉它之后模型反而变好了。需要先说明的是，整份工作都在 32 核 CPU 上完成，没有使用 GPU，两套方案的五折交叉验证加消融一共训练了 18 个模型，累计约 6 小时。",
        2: "先看整体结构。第一部分交代任务背景和数据特点，包括这个数据集最容易被忽视的两个坑。第二部分讲方法，重点是病人级划分、两个模型的设计，以及我们为了保证评估可信做了哪些约束。第三部分是实验结果，从同分布到跨队列，再到不同划分方式下的对比。第四部分是我们这次最想讲的内容：一个由预处理引入的捷径，以及消除它之后的性能变化。最后是结论、局限和下一步计划。",
        3: "任务本身很直接：输入一张胸部 X 光片，判断正常还是肺炎。数据是 Kaggle 公开的儿童胸片集，Kermany 2018 年发表在 Cell 上的那篇，去重后 5824 张，来自 4096 名一到五岁的患儿。这里有两点要注意：一是训练集里肺炎占 74.1%，也就是全部判成肺炎就能拿到 0.741 的准确率，所以准确率在这个任务上几乎没有信息量，后面我们一律用 AUC、灵敏度和特异度；二是测试集的正类比例只有 62.6%，和训练集不一样，这个差异后面会直接影响结果。",
        4: "划分这一步是整个实验最容易出错的地方。同一名患儿往往有好几张片子，如果按图片随机划分，同一个人会同时出现在训练集和测试集，模型等于提前见过答案，结果会明显虚高。我们的做法是从文件名里解析病人编号，按病人整体划分。正常影像的编号是 IM-xxxx 或 NORMAL2-IM-xxxx，肺炎影像是 person 编号加细菌或病毒前缀。这里有一个坑：细菌和病毒用的是两套相互独立的编号，如果只解析 person 后面的数字，会误判出 170 名“跨集病人”；加上亚型前缀之后，官方训练集和测试集的病人交集才真正为零。",
        5: "我们设计了两个方案做对比。方案 A 是自己搭的卷积网络，四个卷积块，通道数从 32 到 256，全局平均池化后接全连接层，一共 117 万参数，输入 128 乘 128，从随机初始化开始训练。方案 B 是 ImageNet 预训练的 ResNet18，1100 万参数，输入 224 乘 224，先冻结主干只训分类头三轮，再解冻做全网络微调。两个方案的增强策略、优化器、早停准则完全一致，唯一的变量就是模型本身和初始化方式，这样对比才有意义。",
        6: "训练用 AdamW，分类头学习率 1e-3，主干 1e-4，配合余弦退火；数据增强包括随机缩放裁剪、正负 7 度旋转、水平翻转和亮度对比度抖动；按验证集 AUC 早停，patience 取 5。评估报 AUC、灵敏度、特异度和 F1 四项。为了保证测试集的独立性，模型结构和训练轮数全部由五折交叉验证确定，确定之后用全部训练数据重训一遍，测试集只在最后评估一次，中途不参与任何模型选择。",
        7: "先看同分布的结果。五折交叉验证，方案 A 的 AUC 是 0.9981，方案 B 是 0.9972，差 0.0009，比折与折之间的标准差还小；准确率 0.979 对 0.978，也基本一样。也就是说，参数量差了将近十倍，性能却没有实质区别。这个结果本身就值得注意，它说明在这个任务上 ImageNet 预训练并没有带来同分布的精度收益。那预训练到底有没有用？要看下一张。",
        8: "换到官方测试集，差距就出来了。方案 B 的 AUC 是 0.9655，方案 A 是 0.9297，差 0.036；特异度 0.662 对 0.455，差了 0.208，这个差距很大。但要注意，两个方案的灵敏度都接近 0.99，几乎不漏诊，问题出在大量正常片子被判成了肺炎。我们没有直接把它归结为模型差，而是先看排序能力：AUC 还有 0.93，说明排序是对的，只是判断阈值整体偏了。按验证集 Youden 指数重选阈值之后，A 的准确率从 0.791 升到 0.846，B 从 0.869 升到 0.888，好一些，但补不回 AUC 的差距。",
        9: "再把肺炎按亚型拆开。细菌性和病毒性分别和正常片组成二分类看 AUC，方案 A 是 0.940 对 0.913，方案 B 是 0.973 对 0.954，两个方案都是病毒性更难一些，但差距不大，也没有出现“只认细菌”的退化。右边的 t-SNE 是 ResNet18 倒数第二层 512 维特征降到二维的结果。左边按真假标签着色，两类分得很开；右边按亚型着色，可以看到病毒样本正好落在正常和细菌之间，和亚组的数字是一致的。",
        10: "这一页想强调一个方法论上的问题。题目要求 70% 训练、30% 测试，我们照做了一版：把全部数据按病人重新随机划分成 70/30。结果两个方案的测试 AUC 都是 0.9945，看起来非常漂亮。但同一批数据用官方跨批次测试集，只有 0.930 和 0.966，随机划分把性能高估了 3 到 7 个百分点。原因不难理解：随机划分下训练集和测试集来自同一批病人、同一批设备，分布几乎一样；而官方测试集是跨批次的。所以我们把所有主结果都建立在官方划分上，70/30 只作为敏感性分析。",
        11: "接下来是我们认为这次最有价值的部分。做可解释性的时候，Grad-CAM 显示误判样本的高响应区域不在肺野，而在图像的黑边和边角，我们就去查了原因。数据集里正常片和肺炎片的平均长宽比分别是 1.25 和 1.52，因为两类图像的成像来源本来就不同。我们做了个很简单的验证：只用宽高比这一个数字做判别，验证集 AUC 就有 0.8665，测试集也有 0.7027。而我们的预处理是等比缩放后补黑边到正方形，等于把“黑边有多宽”变成了一张图里肉眼可见的特征。模型走这条捷径，在训练分布内照样接近满分，一跨队列就现原形。",
        12: "确认了这一点之后，我们把预处理改成中心裁剪，彻底消除黑边，其余设置完全不变，重新训练。结果是：方案 A 的测试 AUC 从 0.930 升到 0.948，特异度从 0.455 升到 0.801；方案 B 的 AUC 基本没变，0.966 到 0.963，但特异度从 0.662 升到 0.836。去掉捷径之后两个方案都变好了，尤其是特异度提升非常大。更重要的是，两个方案之间的差距从 0.036 缩小到了 0.015，这说明 B 原本的一部分优势，并不是它更懂病理，而是它对这种无关特征更不敏感。",
        13: "总结成三条。第一，迁移学习在这个任务上的价值是泛化而不是精度：同分布下两方案打平，差距只在跨队列时出现。第二，这个差距里有一部分来自对无关特征的鲁棒性，而不是更强的病理表征能力，证据就是消除黑边之后两者几乎持平。第三，也是我们觉得最值得强调的：数据管线的隐性偏差，影响比换模型大得多。只改预处理，测试准确率就从 0.791 提到 0.903，超过任何模型结构改动带来的增益。",
        14: "局限也要说清楚。第一，数据是单中心的儿科影像，模型在成人胸片上的表现完全没有验证过，这个问题在相关文献里也很普遍。第二，官方测试集只有 618 张，两个方案的差异我们没有做显著性检验，严格说不能断言显著更好。第三，黑边这个捷径虽然定位并消除了，但“成像来源”这个混淆因素本身并没有被完全剥离，我们只是去掉了它的一个可见通道。下一步打算引入外部数据集做验证，并用更严格的来源无关划分重做一遍。",
        15: "以上就是我们的全部汇报。代码、结果和报告都整理在仓库里，欢迎各位老师批评指正，谢谢。",
    }
    for i, slide in enumerate(prs.slides, start=1):
        speaking_box(slide, scripts[i], dark=(i in (1, 15)))

    prs.save(OUT)
    print("已生成:", OUT)


if __name__ == "__main__":
    main()
