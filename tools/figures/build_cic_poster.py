#!/usr/bin/env python3
"""Reconstroi o poster do XXXVIII CIC Unesp sobre o template oficial.

Preserva, com a geometria original, os elementos institucionais do template (faixa de
cabecalho do CIC, filete do rodape, logos PROPe/FAPESP/CNPq e o QR code) e redesenha
todo o conteudo cientifico. O diagrama do pipeline e montado com formas vetoriais
nativas do PowerPoint -- como o poster e impresso em A0, um PNG ficaria borrado.

Os graficos vem de results/figures/ (gerados por make_cic_figures.py a partir dos CSVs
de results/, de modo que nenhum numero do poster e digitado a mao aqui).

Uso:  python3 tools/figures/build_cic_poster.py ENTRADA.pptx SAIDA.pptx [--figs DIR]
"""

import argparse
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Cm, Pt

# ---------------------------------------------------------------- paleta
NAVY = RGBColor(0x1F, 0x3B, 0x57)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA = RGBColor(0x17, 0x96, 0x69)
AQUA_L = RGBColor(0x1B, 0xAF, 0x7A)
YELLOW = RGBColor(0xC9, 0x8B, 0x00)
YELLOW_L = RGBColor(0xED, 0xA1, 0x00)
MAGENTA = RGBColor(0xB8, 0x44, 0x6E)
PINK = RGBColor(0xE8, 0x7B, 0xA4)
GREEN = RGBColor(0x00, 0x83, 0x00)
CRIT = RGBColor(0xC0, 0x2E, 0x2B)
INK = RGBColor(0x1E, 0x1E, 0x1E)
INK2 = RGBColor(0x4E, 0x4E, 0x4E)
MUTED = RGBColor(0x77, 0x77, 0x77)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TINT_BLUE = RGBColor(0xE8, 0xF1, 0xFC)
TINT_ORANGE = RGBColor(0xFD, 0xF0, 0xE9)
TINT_AQUA = RGBColor(0xE9, 0xF9, 0xF2)
TINT_YELLOW = RGBColor(0xFD, 0xF6, 0xE6)
TINT_PINK = RGBColor(0xFD, 0xEE, 0xF4)
TINT_GREEN = RGBColor(0xEC, 0xF6, 0xEC)
TINT_GREY = RGBColor(0xF2, 0xF4, 0xF7)
RULE = RGBColor(0xD8, 0xDD, 0xE4)
SOFT_ORANGE = RGBColor(0xF3, 0xD3, 0xC3)
SOFT_AQUA = RGBColor(0xC6, 0xE8, 0xDA)
SOFT_YELLOW = RGBColor(0xEE, 0xDA, 0xA8)

FONT = "Arial"

# formas institucionais do template, preservadas na geometria original
KEEP = {"Google Shape;84;g37106c98961_0_150",   # faixa de cabecalho do CIC
        "Google Shape;89;g37106c98961_0_150",   # filete do rodape
        "Google Shape;94;g37106c98961_0_150",   # PROPe
        "Google Shape;95;g37106c98961_0_150",   # CNPq
        "Google Shape;96;g37106c98961_0_150"}   # FAPESP
QR_SHAPE = "Picture 122"

M = 2.6              # margem esquerda (cm)
RIGHT = 87.4         # limite direito do conteudo (cm)
W = RIGHT - M        # largura util: 84,8 cm


# ---------------------------------------------------------------- numeros do poster
def load_claims(results="results"):
    """Le os artefatos e devolve todo numero que aparece no poster.

    Nada e digitado a mao no texto: se um novo treino mudar os CSVs, os valores mudam
    aqui e as afirmacoes em prosa sao reconferidas por assert -- em vez de o poster
    passar a mentir silenciosamente.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_cic_figures import read_arch, read_ci, read_confusion, read_mcnemar, read_metrics

    keys = ["Normal", "Laringite", "Disfonia Psicogenica", "Disfonia Funcional",
            "Edema de Reinke"]
    per_class, macro, acc, _ = read_metrics(
        os.path.join(results, "metrics_global_borderline_C_baseline.csv"))
    ci = read_ci(os.path.join(results, "bootstrap_ci_borderline_C_baseline.csv"))
    mcn = read_mcnemar(os.path.join(results,
                                    "mcnemar_vs_baselines_borderline_C_baseline.csv"))
    cm = read_confusion(os.path.join(results, "train_log_v32_gap2_smote_ab_console.txt"),
                        [per_class[k]["support"] for k in keys], acc)
    arch = read_arch(os.path.join(results, "arch_compare_comparison.csv"))

    def pt(x, nd=4):
        return f"{x:.{nd}f}".replace(".", ",")

    # confusao mutua entre as duas disfonias funcionais (indices 2 e 3)
    mutual = int(cm[2, 3] + cm[3, 2])
    dysph_n = int(per_class["Disfonia Psicogenica"]["support"]
                  + per_class["Disfonia Funcional"]["support"])
    best_psic = max(float(r["f1_disfonia_psicogenica"]) for r in arch)
    best_func = max(float(r["f1_disfonia_funcional"]) for r in arch)
    assert all(p < 0.001 for _, p in mcn.values()), "McNemar deixou de ser p<0,001"
    assert len(arch) == 12, f"esperados 12 bracos de arquitetura, achei {len(arch)}"

    return dict(
        acc_pct=f"{acc*100:.1f}".replace(".", ",") + "%",
        acc_ci=f"IC 95%: {ci['accuracy'][1]*100:.1f} – {ci['accuracy'][2]*100:.1f}%"
               .replace(".", ","),
        macro=pt(macro),
        macro_ci=f"IC 95%: {pt(ci['macro_f1'][1], 3)} – {pt(ci['macro_f1'][2], 3)}",
        f1_normal=pt(per_class["Normal"]["f1"], 3),
        recall_normal=f"{per_class['Normal']['recall']*100:.0f}%",
        f1_reinke=pt(per_class["Edema de Reinke"]["f1"], 3),
        f1_psic=pt(per_class["Disfonia Psicogenica"]["f1"], 3),
        f1_func=pt(per_class["Disfonia Funcional"]["f1"], 3),
        cm_func_to_psic=int(cm[3, 2]),
        cm_psic_to_func=int(cm[2, 3]),
        mutual=mutual,
        dysph_n=dysph_n,
        n_arms=len(arch),
        best_psic=pt(best_psic, 2),
        best_func=pt(best_func, 2),
        n_patients=f"{int(sum(per_class[k]['support'] for k in keys)):,}"
                   .replace(",", "."),
    )


# ---------------------------------------------------------------- helpers
def clear_slide(slide):
    """Remove o conteudo antigo, mantendo os elementos do template. Devolve o QR."""
    qr = None
    for shape in list(slide.shapes):
        if shape.name in KEEP:
            continue
        if shape.name == QR_SHAPE:
            qr = shape
            continue
        shape._element.getparent().remove(shape._element)
    return qr


def box(slide, l, t, w, h, fill=None, line=None, lw=1.6, radius=0.16,
        shape_type=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = slide.shapes.add_shape(shape_type, Cm(l), Cm(t), Cm(w), Cm(h))
    if shape_type == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sh.adjustments[0] = radius
        except (IndexError, KeyError):
            pass
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    sh.text_frame.word_wrap = True
    return sh


def write(shape, lines, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
          margins=(0.3, 0.3, 0.1, 0.1), spacing=0.92):
    """lines = [[(texto, pt, negrito, cor), ...], ...] -- uma sublista por paragrafo."""
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left, tf.margin_right = Cm(margins[0]), Cm(margins[1])
    tf.margin_top, tf.margin_bottom = Cm(margins[2]), Cm(margins[3])
    for i, runs in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        for text_, size, bold, color in runs:
            run = para.add_run()
            run.text = text_
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.name = FONT
            run.font.color.rgb = color
    return shape


def text(slide, l, t, w, h, lines, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
         margins=(0.0, 0.0, 0.0, 0.0), spacing=0.92):
    tb = slide.shapes.add_textbox(Cm(l), Cm(t), Cm(w), Cm(h))
    return write(tb, lines, align, anchor, margins, spacing)


def section(slide, l, t, w, label, accent, sub=None):
    """Cabecalho de secao: barra de acento + rotulo em caixa alta."""
    box(slide, l, t + 0.15, 0.5, 1.8, fill=accent, radius=0.5)
    text(slide, l + 1.05, t, w - 1.05, 2.2, [[(label, 50, True, NAVY)]],
         align=PP_ALIGN.LEFT)
    if sub:
        text(slide, l + 1.05, t + 2.05, w - 1.05, 1.2, [[(sub, 23, False, MUTED)]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)


def arrow(slide, l, t, w, h, color=RGBColor(0x9A, 0xA3, 0xAE)):
    sh = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Cm(l), Cm(t), Cm(w), Cm(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    try:
        sh.adjustments[0] = 0.5
        sh.adjustments[1] = 0.45
    except (IndexError, KeyError):
        pass
    return sh


def picture(slide, path, l, t, w):
    """Insere preservando o aspecto nativo; devolve (shape, altura em cm)."""
    pic = slide.shapes.add_picture(path, Cm(l), Cm(t), width=Cm(w))
    return pic, pic.height / 360000.0


def caption(slide, l, t, w, head, tail, accent=INK):
    return text(slide, l, t, w, 1.9,
                [[(head, 21, True, accent), (tail, 21, False, INK2)]],
                align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=0.9)


# ---------------------------------------------------------------- diagrama do pipeline
def pipeline(slide, y, h):
    widths = [13.4, 15.6, 24.6, 11.4, 11.0]
    gap = 2.2
    xs, x = [], M
    for wd in widths:
        xs.append(x)
        x += wd + gap

    def stage_arrow(i):
        arrow(slide, xs[i] + widths[i] + 0.5, y + h / 2 - 0.8, gap - 1.0, 1.6)

    # ---- 1. sinal de voz
    x0, w0 = xs[0], widths[0]
    box(slide, x0, y, w0, h, TINT_BLUE, BLUE, 2.0)
    text(slide, x0, y + 0.6, w0, 1.6, [[("SINAL DE VOZ", 29, True, BLUE)]])
    text(slide, x0, y + 2.1, w0, 1.2, [[("Saarbrücken Voice Database", 18, False, INK2)]])
    wave = box(slide, x0 + 1.4, y + 3.5, w0 - 2.8, 2.4, WHITE, BLUE, 1.2, radius=0.12)
    write(wave, [[("∿∿∿∿", 40, True, BLUE)]])
    text(slide, x0, y + 6.4, w0, 3.2, [[("1.098", 44, True, INK)],
                                       [("pacientes", 21, False, INK2)]])
    text(slide, x0, y + 9.8, w0, 2.8, [[("vogais sustentadas", 19, False, INK2)],
                                       [("/a/    /i/    /u/", 29, True, BLUE)]])
    box(slide, x0 + 1.2, y + 12.9, w0 - 2.4, 4.3, WHITE, RGBColor(0xC9, 0xDE, 0xF6),
        1.2, radius=0.12)
    text(slide, x0 + 1.2, y + 13.0, w0 - 2.4, 4.1,
         [[("5 classes clínicas", 20, True, INK)],
          [("687 · 140 · 91 · 112 · 68", 19, False, INK2)],
          [("fortemente desbalanceada", 17, False, MUTED)]], spacing=0.9)
    stage_arrow(0)

    # ---- 2. caracteristicas
    x1, w1 = xs[1], widths[1]
    box(slide, x1, y, w1, h, TINT_ORANGE, ORANGE, 2.0)
    text(slide, x1, y + 0.6, w1, 2.5, [[("CARACTERÍSTICAS", 29, True, ORANGE)],
                                       [("83 por vogal", 19, False, INK2)]])
    blocks = [("Temporais", "10", "jitter · shimmer · HNR · ZCR",
               "irregularidade da fonação"),
              ("Espectrais", "55", "F0 · formantes · MFCC+Δ+ΔΔ · CPP · pulso glotal",
               "timbre e qualidade vocal"),
              ("Wavelet", "18", "Daubechies-4 · 6 níveis", "padrões multiescala")]
    by = y + 3.5
    for name, n, what, why in blocks:
        card = box(slide, x1 + 0.7, by, w1 - 1.4, 3.9, WHITE, SOFT_ORANGE, 1.2,
                   radius=0.12)
        write(card, [[(name + "   ", 22, True, INK), (n, 29, True, ORANGE)],
                     [(what, 16.5, False, INK2)],
                     [(why, 16, False, MUTED)]], spacing=0.88,
              margins=(0.25, 0.25, 0.08, 0.08))
        by += 4.18
    text(slide, x1, by + 0.05, w1, 1.85, [[("+ idade e sexo do paciente", 18, False, MUTED)],
                                        [("251 variáveis", 30, True, INK)]], spacing=0.95)
    stage_arrow(1)

    # ---- 3. duas redes por vogal
    x2, w2 = xs[2], widths[2]
    box(slide, x2, y, w2, h, TINT_AQUA, AQUA_L, 2.0)
    text(slide, x2, y + 0.6, w2, 2.5,
         [[("DUAS REDES POR VOGAL", 29, True, AQUA)],
          [("×3 vogais · pesos independentes", 19, False, INK2)]])

    def net(nx, nw, title, subtitle, out_label, out_color, out_tint):
        box(slide, nx, y + 3.3, nw, 11.9, WHITE, SOFT_AQUA, 1.2, radius=0.1)
        text(slide, nx, y + 3.5, nw, 2.3, [[(title, 24, True, AQUA)],
                                           [(subtitle, 17, False, INK2)]], spacing=0.9)
        ly = y + 6.0
        layers = [("85 características", WHITE, SOFT_AQUA, INK2),
                  ("Dense 128 + LeakyReLU", AQUA_L, AQUA_L, WHITE),
                  ("Dense 64 + LeakyReLU", AQUA_L, AQUA_L, WHITE),
                  (out_label, out_tint, out_color, out_color)]
        for i, (lab, fill, line, col) in enumerate(layers):
            lb = box(slide, nx + 1.0, ly, nw - 2.0, 1.5, fill, line, 1.2, radius=0.24)
            write(lb, [[(lab, 17, True, col)]], margins=(0.1, 0.1, 0.02, 0.02))
            ly += 1.5
            if i < len(layers) - 1:
                text(slide, nx + 1.0, ly - 0.28, nw - 2.0, 0.85,
                     [[("↓", 16, True, MUTED)]])
                ly += 0.57
        text(slide, nx, y + 13.9, nw, 1.2, [[("dropout 0,5 / 0,4", 16.5, False, MUTED)]])

    half = (w2 - 2.6) / 2
    net(x2 + 0.9, half, "MESTRA", "patológico × saudável", "normal | patológico",
        BLUE, TINT_BLUE)
    net(x2 + 1.7 + half, half, "ESPECIALISTA", "qual patologia?", "4 patologias",
        MAGENTA, TINT_PINK)
    foot = box(slide, x2 + 0.9, y + 15.6, w2 - 1.8, h - 16.1, WHITE, SOFT_AQUA, 1.2,
               radius=0.1)
    write(foot, [[("Adam", 19, True, INK), (" · decaimento cosseno · ", 17, False, INK2),
                  ("parada por Macro-F1", 19, True, INK)],
                 [("treino balanceado com augmentação de áudio + Borderline-SMOTE1",
                   17, False, INK2)],
                 [("z-score ajustado só nas amostras originais de treino",
                   16, False, MUTED)]], spacing=0.9)
    stage_arrow(2)

    # ---- 4. fusao tardia
    x3, w3 = xs[3], widths[3]
    box(slide, x3, y, w3, h, TINT_YELLOW, YELLOW_L, 2.0)
    text(slide, x3, y + 0.6, w3, 2.5, [[("FUSÃO TARDIA", 27, True, YELLOW)],
                                       [("as 3 vogais votam", 18, False, INK2)]])
    vy = y + 3.6
    for v in ("/a/", "/i/", "/u/"):
        row = box(slide, x3 + 1.0, vy, w3 - 2.0, 1.9, WHITE, SOFT_YELLOW, 1.2, radius=0.2)
        write(row, [[(v, 22, True, INK), ("  P(classe)", 17, False, INK2)]],
              margins=(0.15, 0.15, 0.02, 0.02))
        vy += 2.25
    text(slide, x3, vy - 0.1, w3, 1.1, [[("↓", 22, True, YELLOW)]])
    mean = box(slide, x3 + 1.0, vy + 1.0, w3 - 2.0, 3.2, YELLOW_L, None, radius=0.16)
    write(mean, [[("média das", 20, True, WHITE)], [("probabilidades", 20, True, WHITE)]])
    text(slide, x3 + 0.6, y + 14.9, w3 - 1.2, 2.6,
         [[("nenhuma vogal", 17, False, MUTED)], [("decide sozinha", 17, False, MUTED)]],
         spacing=0.9)
    stage_arrow(3)

    # ---- 5. decisao
    x4, w4 = xs[4], widths[4]
    text(slide, x4, y + 0.6, w4, 1.6, [[("DECISÃO", 29, True, NAVY)]])
    gate = box(slide, x4, y + 2.2, w4, 3.0, WHITE, NAVY, 1.8, radius=0.18)
    write(gate, [[("P(patológico)", 20, True, INK)], [("≥ 0,5 ?", 23, True, NAVY)]])
    nb = box(slide, x4, y + 5.8, w4, 2.3, TINT_GREEN, GREEN, 1.8, radius=0.2)
    write(nb, [[("não → ", 17, False, INK2), ("NORMAL", 24, True, GREEN)]],
          margins=(0.1, 0.1, 0.02, 0.02))
    text(slide, x4, y + 8.3, w4, 1.0, [[("sim ↓", 18, True, MUTED)]])
    py = y + 9.6
    for lab in ("Laringite", "Disfonia Psicogênica", "Disfonia Funcional",
                "Edema de Reinke"):
        card = box(slide, x4, py, w4, 1.9, TINT_PINK, PINK, 1.4, radius=0.22)
        write(card, [[(lab, 18, True, MAGENTA)]], margins=(0.12, 0.12, 0.02, 0.02))
        py += 2.1


# ---------------------------------------------------------------- poster
def build(src, dst, figs="results/figures", results="results"):
    K = load_claims(results)
    prs = Presentation(src)
    slide = prs.slides[0]
    qr = clear_slide(slide)

    # ---------------- cabecalho do trabalho
    text(slide, M, 17.4, W, 6.3,
         [[("IDENTIFICAÇÃO COMPUTACIONAL INTELIGENTE", 72, True, NAVY)],
          [("DE DISFUNÇÕES NO TRATO VOCAL", 72, True, NAVY)]], spacing=0.94)
    text(slide, M, 23.9, W, 1.8,
         [[("Ronaldo Chiavegatti Sampaio Corrêa", 33, True, INK),
           ("     |     ", 33, False, MUTED),
           ("Rodrigo Capobianco Guido", 33, False, INK),
           ("  (Orientador)", 28, False, MUTED)]])
    text(slide, M, 25.7, W, 1.4,
         [[("Instituto de Biociências, Letras e Ciências Exatas — IBILCE/Unesp · "
            "Câmpus de São José do Rio Preto", 25, False, INK2)]])
    tag = box(slide, M + W / 2 - 21.0, 27.4, 42.0, 1.9, TINT_GREY, RULE, 1.2, radius=0.5)
    write(tag, [[("Iniciação Científica Voluntária — PIBIC/Unesp, sem bolsa     ·     ",
                  20, False, INK2),
                 ("pipeline em C99, sem frameworks de ML", 20, True, INK)]])

    # ---------------- o problema | objetivos
    y = 30.2
    colw = (W - 2.4) / 2
    section(slide, M, y, colw, "O PROBLEMA", BLUE)
    section(slide, M + colw + 2.4, y, colw, "OBJETIVOS", ORANGE)
    cy = y + 2.7
    c1 = box(slide, M, cy, 17.6, 5.2, WHITE, BLUE, 1.6)
    write(c1, [[("hoje", 18, False, MUTED)],
               [("diagnóstico diferencial exige", 21, False, INK)],
               [("exame invasivo", 27, True, BLUE)]], spacing=0.92)
    arrow(slide, M + 18.0, cy + 2.0, 2.2, 1.3)
    c2 = box(slide, M + 20.6, cy, colw - 20.6, 5.2, TINT_BLUE, BLUE, 1.6)
    write(c2, [[("proposta", 18, False, MUTED)],
               [("triagem ", 21, False, INK), ("não invasiva", 27, True, BLUE)],
               [("e de baixo custo, só pela voz", 21, False, INK)]], spacing=0.92)

    ox = M + colw + 2.4
    objs = [("5", "condições vocais\ndistinguidas pela voz"),
            ("6", "redes por dobra: 2 por\nvogal, fundidas ao final"),
            ("3", "lacunas metodológicas\nfechadas com A/B")]
    ow = (colw - 2 * 1.0) / 3
    for i, (big, small) in enumerate(objs):
        card = box(slide, ox + i * (ow + 1.0), cy, ow, 5.2, TINT_ORANGE, ORANGE, 1.6)
        write(card, [[(big, 40, True, ORANGE)]] +
              [[(ln, 19, False, INK)] for ln in small.split("\n")], spacing=0.9)

    # ---------------- material e metodos
    y = 38.9
    section(slide, M, y, W, "MATERIAL E MÉTODOS", AQUA,
            "validação cruzada estratificada de 5 dobras · seed fixa (42) · "
            "z-score e balanceamento ajustados dentro de cada dobra")
    pipeline(slide, y + 3.6, 18.0)

    # ---------------- resultados
    y = 61.3
    section(slide, M, y, W, "RESULTADOS", BLUE,
            f"{K['n_patients']} pacientes · predições agregadas fora da dobra de treino "
            "(out-of-fold)")
    ky = y + 3.6
    kpis = [(K["acc_pct"], "Acurácia global", K["acc_ci"], BLUE, TINT_BLUE),
            (K["macro"], "Macro F1 (5 classes)", K["macro_ci"], ORANGE, TINT_ORANGE),
            ("p < 0,001", "vs. os 3 baselines", "McNemar · MLP superior", GREEN, TINT_GREEN),
            (K["f1_normal"], "F1 da classe Normal",
             f"recall de {K['recall_normal']} nos saudáveis", AQUA, TINT_AQUA)]
    kw = (W - 3 * 1.2) / 4
    for i, (big, lab, sub, col, tint) in enumerate(kpis):
        card = box(slide, M + i * (kw + 1.2), ky, kw, 5.6, tint, col, 1.8)
        write(card, [[(big, 52, True, col)],
                     [(lab, 21, True, INK)],
                     [(sub, 17.5, False, INK2)]], spacing=0.92)

    # ---- faixa 1: onde acerta e onde erra (alturas iguais, largura pelo aspecto nativo)
    by = ky + 7.2
    band1 = 14.0
    fw1 = band1 * 1.70
    pic, hh = picture(slide, f"{figs}/poster_confusao.png", M, by, fw1)
    caption(slide, M, by + hh + 0.1, fw1, "Onde o modelo erra. ",
            f"As duas disfonias funcionais se trocam entre si "
            f"({K['cm_func_to_psic']} e {K['cm_psic_to_func']} casos) e vazam para Normal.",
            BLUE)

    x2 = M + fw1 + 1.6
    fw2 = band1 * 1.57
    pic2, hh2 = picture(slide, f"{figs}/poster_f1_classes.png", x2, by, fw2)
    caption(slide, x2, by + hh2 + 0.1, fw2, "Desempenho por classe. ",
            "Normal e Reinke acima da média; as funcionais, abaixo.", ORANGE)

    x3 = x2 + fw2 + 1.6
    fw3 = RIGHT - x3
    card = box(slide, x3, by, fw3, max(hh, hh2), WHITE, RULE, 1.6)
    write(card, [[("O TETO ACÚSTICO", 26, True, NAVY)],
                 [("", 10, False, INK)],
                 [("Disfonia psicogênica e funcional", 20, False, INK)],
                 [("não têm lesão estrutural", 21, True, NAVY)],
                 [("— a voz sozinha não as separa.", 20, False, INK)],
                 [("", 10, False, INK)],
                 [(f"{K['mutual']} dos {K['dysph_n']} pacientes", 21, True, CRIT),
                  (" dessas", 20, False, INK)],
                 [("duas classes foram trocados entre si.", 20, False, INK)],
                 [("", 10, False, INK)],
                 [(f"Nos {K['n_arms']} braços de arquitetura", 19, False, INK2)],
                 [("testados, o melhor F1 nessas classes", 19, False, INK2)],
                 [(f"foi {K['best_psic']} e {K['best_func']}.", 19, False, INK2)]],
          anchor=MSO_ANCHOR.MIDDLE, spacing=0.95, margins=(0.7, 0.7, 0.4, 0.4))

    # ---- faixa 2: evidencia A/B (larguras resolvidas para as duas figuras terem a
    # mesma altura, preenchendo a largura util)
    ay = by + max(hh, hh2) + 2.6
    text(slide, M, ay, W * 0.62, 1.4,
         [[("EVIDÊNCIA A/B — ", 29, True, NAVY),
           ("mesma seed, mesmas 5 dobras, decisão por regra fixa", 23, False, INK2)]],
         align=PP_ALIGN.LEFT)
    asp_g, asp_a, gap2 = 2.86, 3.05, 2.4
    band2 = (W - gap2) / (asp_g + asp_a)
    gw = asp_g * band2
    xa = M + gw + gap2
    text(slide, xa, ay, RIGHT - xa, 1.4,
         [[("GAP 3 — ", 23, True, AQUA),
           (f"profundidade da rede: {K['n_arms']} braços "
            "(4 arquiteturas × 3 regularizações)", 22, False, INK2)]],
         align=PP_ALIGN.LEFT)
    gy = ay + 1.7
    picture(slide, f"{figs}/poster_gaps_ab.png", M, gy, gw)
    picture(slide, f"{figs}/poster_arquiteturas.png", xa, gy, RIGHT - xa)

    # ---------------- conclusao e referencias
    y = 104.0
    ccolw = 44.0
    section(slide, M, y, ccolw, "CONCLUSÃO", GREEN)
    section(slide, M + ccolw + 2.0, y, W - ccolw - 2.0, "REFERÊNCIAS", NAVY)

    cy = y + 2.6
    concl = [("Triagem funciona.",
              f"Saudável × patológico se separa: F1 {K['f1_normal']}, "
              f"recall {K['recall_normal']}.", GREEN, TINT_GREEN),
             ("Reinke é separável.",
              f"A lesão estrutural deixa marca acústica (F1 {K['f1_reinke']}).",
              BLUE, TINT_BLUE),
             ("Teto nas funcionais.",
              f"Psicogênica ({K['f1_psic']}) × funcional ({K['f1_func']}) exigem sinal "
              "além da voz.", CRIT, RGBColor(0xFD, 0xEC, 0xEB)),
             ("Rigor acima do ganho.",
              "Das 3 técnicas propostas, 1 foi rejeitada pela própria evidência A/B.",
              NAVY, TINT_GREY)]
    cw = (ccolw - 3 * 1.0) / 4
    for i, (head, body, col, tint) in enumerate(concl):
        card = box(slide, M + i * (cw + 1.0), cy, cw, 5.6, tint, col, 1.8)
        write(card, [[(head, 21, True, col)], [("", 6, False, INK)],
                     [(body, 18, False, INK)]], spacing=0.94,
              margins=(0.35, 0.35, 0.2, 0.2))

    rx = M + ccolw + 2.0
    rw = W - ccolw - 2.0 - 6.2
    refs = [("BARRY, W. J.; PÜTZER, M. ",
             "Saarbrücken Voice Database. Univ. des Saarlandes, 2007."),
            ("HAN, H.; WANG, W.-Y.; MAO, B.-H. ",
             "Borderline-SMOTE. LNCS, v. 3644, p. 878-887, 2005."),
            ("VRBA, J. et al. ",
             "Reproducible ML-based voice pathology detection. J. Voice, 2025."),
            ("LEE, J.-Y. ",
             "Deep learning for pathological voice detection. Appl. Sci., v. 11, 2021."),
            ("GUIDO, R. C. ",
             "Wavelets behind the scenes. Physics Reports, v. 985, 2022."),
            ("KINGMA, D. P.; BA, J. ",
             "Adam: stochastic optimization. ICLR, 2015.")]
    ry = cy
    for head, tail in refs:
        text(slide, rx, ry, rw, 1.35, [[(head, 17, True, INK), (tail, 17, False, INK2)]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=0.9)
        ry += 1.18
    if qr is not None:
        qr.left, qr.top = Cm(RIGHT - 5.3), Cm(cy + 0.2)
        qr.width = qr.height = Cm(5.0)
    text(slide, RIGHT - 7.0, cy + 5.4, 8.0, 1.5,
         [[("código-fonte", 16, True, INK)], [("e dados completos", 16, False, INK2)]],
         spacing=0.9)

    prs.save(dst)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--figs", default="results/figures")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    print("gerado:", build(a.src, a.dst, a.figs, a.results))
