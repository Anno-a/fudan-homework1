# -*- coding: utf-8 -*-
"""把 报告正文.md + work/runs 的结果 + figures/ 的图，生成 Word 报告。

    python work/build_report_docx.py

输出：AIT410020_PJ2_任务二_报告.docx
所有数字都从 work/runs 下的预测文件重新计算，与正文手写内容独立校验。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "runs"
FIGS = ROOT / "figures"
SRC = ROOT / "报告正文.md"
OUT = ROOT / "AIT410020_PJ2_任务二_报告.docx"

SONG, HEI, LATIN = "宋体", "黑体", "Times New Roman"


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def met(y, p, s=0.5):
    yh = (p >= s).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    return dict(auc=roc_auc_score(y, p), acc=(tp + tn) / len(y),
                sen=tp / (tp + fn), spe=tn / (tn + fp),
                f1=2 * tp / (2 * tp + fp + fn))


def fold_stats(key):
    rows = []
    for k in range(5):
        d = pd.read_csv(RUNS / f"{key}_fold{k}" / "preds_val.csv")
        rows.append(met(d.y_true.values, d.y_prob.values))
    out = {}
    for c in ("auc", "acc", "sen", "spe", "f1"):
        v = [r[c] for r in rows]
        out[c] = f"{np.mean(v):.4f}±{np.std(v, ddof=0):.4f}"
    return out


def test_metrics(tag):
    d = pd.read_csv(RUNS / tag / "preds_test.csv")
    return met(d.y_true.values, d.y_prob.values)


def youden(key):
    v = pd.concat([pd.read_csv(RUNS / f"{key}_fold{k}" / "preds_val.csv") for k in range(5)])
    fpr, tpr, thr = roc_curve(v.y_true, v.y_prob)
    return float(thr[np.argmax(tpr - fpr)])


# --------------------------------------------------------------------------- #
# docx 工具
# --------------------------------------------------------------------------- #
def set_run(run, size=10.5, bold=False, ascii_font=LATIN, cjk_font=SONG, color=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = ascii_font
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), cjk_font)
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_body_par(doc, text, indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(0)
    if indent:
        p.paragraph_format.first_line_indent = Pt(21)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_run(p.add_run(text))
    return p


def add_heading_par(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    set_run(p.add_run(text), size=12, bold=(level == 1), cjk_font=HEI)
    return p


def add_caption(doc, text, before=4, after=4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    set_run(p.add_run(text), size=9, cjk_font=SONG)
    return p


def _border(el_tag, sz, color="000000"):
    el = OxmlElement(f"w:{el_tag}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), color)
    return el


def three_line_table(doc, header, rows, widths=None):
    """三线表：顶线、栏目线、底线，无竖线。"""
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True

    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    borders.append(_border("top", 12))
    borders.append(_border("bottom", 12))
    for e in ("left", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{e}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        borders.append(el)
    # OOXML 里 tblBorders 必须排在 tblLayout / tblLook 之前，直接 append 会让 Word 报文件损坏
    tblPr.insert_element_before(
        borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
        "w:tblCaption", "w:tblDescription")

    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        set_run(p.add_run(str(h)), size=9, bold=False, cjk_font=SONG)
        # 栏目线
        tcPr = cell._tc.get_or_add_tcPr()
        tcB = OxmlElement("w:tcBorders")
        tcB.append(_border("bottom", 6))
        tcPr.insert_element_before(
            tcB, "w:shd", "w:noWrap", "w:tcMar", "w:textDirection",
            "w:tcFitText", "w:vAlign", "w:hideMark")

    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            set_run(p.add_run(str(v)), size=9)
    return t


def add_figure(doc, path, width_cm, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    add_caption(doc, caption, before=0, after=8)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    A_fold, B_fold = fold_stats("A"), fold_stats("B")
    A_test, B_test = test_metrics("A_test"), test_metrics("B_test")
    A_crop, B_crop = test_metrics("A_crop_test"), test_metrics("B_crop_test")
    A_pool, B_pool = test_metrics("A_pool70"), test_metrics("B_pool70")
    tA, tB = youden("A"), youden("B")

    # 用 Youden 阈值重算测试集指标
    dA = pd.read_csv(RUNS / "A_test" / "preds_test.csv")
    dB = pd.read_csv(RUNS / "B_test" / "preds_test.csv")
    A_t = met(dA.y_true.values, dA.y_prob.values, tA)
    B_t = met(dB.y_true.values, dB.y_prob.values, tB)

    # 亚组
    def sub(tag, s):
        d = pd.read_csv(RUNS / tag / "preds_test.csv")
        dd = d[(d.y_true == 0) | (d.subtype == s)]
        return met(dd.y_true.values, dd.y_prob.values)["auc"]

    tables = {
        "表 1": ("表 1  五折交叉验证结果",
                 ["方案", "AUC", "准确率", "灵敏度", "特异度", "F1"],
                 [["A 自建 SmallCNN"] + [A_fold[c] for c in ("auc", "acc", "sen", "spe", "f1")],
                  ["B ResNet18"] + [B_fold[c] for c in ("auc", "acc", "sen", "spe", "f1")]]),
        "表 2": ("表 2  官方测试集性能（阈值 0.5）",
                 ["方案", "AUC", "准确率", "灵敏度", "特异度", "F1"],
                 [["A 自建 SmallCNN"] + [f"{A_test[c]:.4f}" for c in ("auc", "acc", "sen", "spe", "f1")],
                  ["B ResNet18"] + [f"{B_test[c]:.4f}" for c in ("auc", "acc", "sen", "spe", "f1")]]),
        "表 3": ("表 3  阈值调整前后的测试指标",
                 ["方案", "Youden 阈值", "准确率@0.5", "准确率@阈值", "灵敏度@阈值", "特异度@阈值"],
                 [["A 自建 SmallCNN", f"{tA:.3f}", f"{A_test['acc']:.4f}", f"{A_t['acc']:.4f}",
                   f"{A_t['sen']:.4f}", f"{A_t['spe']:.4f}"],
                  ["B ResNet18", f"{tB:.3f}", f"{B_test['acc']:.4f}", f"{B_t['acc']:.4f}",
                   f"{B_t['sen']:.4f}", f"{B_t['spe']:.4f}"]]),
        "表 4": ("表 4  不同亚型的判别能力",
                 ["方案", "亚组", "样本数", "AUC"],
                 [["A", "正常 vs 细菌性", 471, f"{sub('A_test', 'bacteria'):.4f}"],
                  ["A", "正常 vs 病毒性", 378, f"{sub('A_test', 'virus'):.4f}"],
                  ["B", "正常 vs 细菌性", 471, f"{sub('B_test', 'bacteria'):.4f}"],
                  ["B", "正常 vs 病毒性", 378, f"{sub('B_test', 'virus'):.4f}"]]),
        "表 5": ("表 5  70/30 重划分结果",
                 ["方案", "AUC", "准确率", "灵敏度", "特异度", "F1"],
                 [["A 自建 SmallCNN"] + [f"{A_pool[c]:.4f}" for c in ("auc", "acc", "sen", "spe", "f1")],
                  ["B ResNet18"] + [f"{B_pool[c]:.4f}" for c in ("auc", "acc", "sen", "spe", "f1")]]),
        "表 6": ("表 6  预处理对照",
                 ["方案", "预处理", "AUC", "准确率", "灵敏度", "特异度"],
                 [["A 自建 SmallCNN", "letterbox（含黑边）"] + [f"{A_test[c]:.4f}" for c in ("auc", "acc", "sen", "spe")],
                  ["A 自建 SmallCNN", "中心裁剪（无黑边）"] + [f"{A_crop[c]:.4f}" for c in ("auc", "acc", "sen", "spe")],
                  ["B ResNet18", "letterbox（含黑边）"] + [f"{B_test[c]:.4f}" for c in ("auc", "acc", "sen", "spe")],
                  ["B ResNet18", "中心裁剪（无黑边）"] + [f"{B_crop[c]:.4f}" for c in ("auc", "acc", "sen", "spe")]]),
    }
    figures = {
        "图 1": (FIGS / "fig_roc.png", 12.5, "图 1  五折验证集 ROC 曲线"),
        "图 2": (FIGS / "fig_confusion.png", 14.0, "图 2  五折验证集混淆矩阵"),
        "图 3": (FIGS / "fig_tsne.png", 15.5, "图 3  ResNet18 特征的 t-SNE 可视化"),
        "图 4": (FIGS / "fig_gradcam.png", 12.0, "图 4  典型样本 Grad-CAM"),
    }

    # --- 文档 ---------------------------------------------------------------
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, attr, Cm(2.5))

    normal = doc.styles["Normal"]
    normal.font.name = LATIN
    normal.font.size = Pt(10.5)
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), SONG)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    used_tbl, used_fig = set(), set()
    in_refs = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if in_refs:
            rp = doc.add_paragraph()
            rp.paragraph_format.line_spacing = 1.25
            rp.paragraph_format.space_after = Pt(2)
            rp.paragraph_format.left_indent = Pt(21)
            rp.paragraph_format.first_line_indent = Pt(-21)
            set_run(rp.add_run(line), size=9)
            continue
        if line.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(14)
            set_run(p.add_run(line[2:]), size=15, bold=True, cjk_font=HEI)
            continue
        if line.startswith("## "):
            title = line[3:]
            if title == "参考文献":
                add_heading_par(doc, "参考文献", level=1)
                in_refs = True
                continue
            if title == "摘要":
                # 摘要用 run-in 形式：加粗标签 + 正文
                while i < len(lines) and not lines[i].strip():
                    i += 1
                body = lines[i].strip()
                i += 1
                p = doc.add_paragraph()
                p.paragraph_format.line_spacing = 1.5
                p.paragraph_format.space_after = Pt(10)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                set_run(p.add_run("摘要　"), size=10.5, bold=True, cjk_font=HEI)
                set_run(p.add_run(body), size=10.5)
                continue
            add_heading_par(doc, title, level=1)
            continue

        add_body_par(doc, line)

        # 根据段落里引用的图表编号，按出现顺序插入
        for token in re.findall(r"[表图] [1-6]", line):
            if token.startswith("表") and token in tables and token not in used_tbl:
                cap, hdr, rows = tables[token]
                add_caption(doc, cap, before=8, after=2)
                three_line_table(doc, hdr, rows)
                used_tbl.add(token)
            elif token.startswith("图") and token in figures and token not in used_fig:
                path, w, cap = figures[token]
                add_figure(doc, path, w, cap)
                used_fig.add(token)

    # 参考文献从 报告正文.md 里读（脚本不再硬编码，保证三处一致）

    doc.save(OUT)
    print("已生成:", OUT)
    print("表格:", sorted(used_tbl), "| 图:", sorted(used_fig))


if __name__ == "__main__":
    main()
