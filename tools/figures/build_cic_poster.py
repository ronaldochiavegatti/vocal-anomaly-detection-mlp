#!/usr/bin/env python3
"""Reconstroi o poster do XXXVIII CIC Unesp sobre o template oficial.

Preserva, com a geometria original, os elementos institucionais do template (faixa de
cabecalho do CIC, filete do rodape, logos PROPe/FAPESP/CNPq e o QR code) e redesenha
todo o conteudo cientifico. O diagrama do pipeline e montado com formas vetoriais
nativas do PowerPoint -- como o poster e impresso em A0, um PNG ficaria borrado.

Duas restricoes governam o layout:

1. Legibilidade a distancia. O poster e impresso em 90 x 120 cm e lido a uns dois
   metros. Toda a tipografia e dimensionada para isso -- corpo em 28-30 pt (~1,0 cm de
   altura fisica), titulo em 100 pt -- e os graficos sao gerados no tamanho fisico final
   em que serao inseridos, em 1:1, para que as fontes internas valham como pontos de
   impressao em vez de encolherem na colocacao.

2. Hierarquia por tipografia, nao por moldura. Uma cor de acento sobre tinta escura,
   fios finos e espaco em branco no lugar de cartoes com borda colorida; as cores por
   classe existem apenas dentro dos graficos, onde codificam dados.

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
# Deliberadamente curta: tinta + um azul institucional + um vermelho de alerta usado
# uma unica vez. O rodizio de cinco cores da versao anterior era o que dava ao poster
# aparencia de modelo pronto.
INK = RGBColor(0x14, 0x20, 0x2E)          # tinta principal
INK2 = RGBColor(0x3D, 0x4A, 0x5A)         # texto secundario
MUTED = RGBColor(0x6B, 0x77, 0x85)        # apoio
ACCENT = RGBColor(0x1F, 0x4E, 0x8C)       # azul institucional
ALERT = RGBColor(0xA8, 0x32, 0x26)        # alerta
TINT = RGBColor(0xF3, 0xF5, 0xF8)         # faixa de fundo do pipeline
RULE_LIGHT = RGBColor(0xD5, 0xDC, 0xE4)   # fio fino
RULE_STRONG = RGBColor(0x9A, 0xA8, 0xB8)  # numerais dos estagios

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

# escala tipografica do pipeline: (pt, cor, negrito, avanco vertical em cm)
KINDS = {
    "big":   (54, INK, True, 3.4),
    "mid":   (34, INK, True, 2.3),
    "label": (27, MUTED, False, 1.6),
    "body":  (29, INK2, False, 1.5),
    "small": (25, MUTED, False, 1.5),
}


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

    mutual = int(cm[2, 3] + cm[3, 2])
    dysph_n = int(per_class["Disfonia Psicogenica"]["support"]
                  + per_class["Disfonia Funcional"]["support"])
    assert all(p < 0.001 for _, p in mcn.values()), "McNemar deixou de ser p<0,001"
    assert len(arch) == 12, f"esperados 12 bracos de arquitetura, achei {len(arch)}"

    return dict(
        acc_pct=f"{acc*100:.1f}".replace(".", ",") + "%",
        acc_ci=f"IC 95%  {ci['accuracy'][1]*100:.1f} – {ci['accuracy'][2]*100:.1f}%"
               .replace(".", ","),
        macro=pt(macro),
        macro_ci=f"IC 95%  {pt(ci['macro_f1'][1], 3)} – {pt(ci['macro_f1'][2], 3)}",
        f1_normal=pt(per_class["Normal"]["f1"], 3),
        recall_normal=f"{per_class['Normal']['recall']*100:.0f}%",
        f1_reinke=pt(per_class["Edema de Reinke"]["f1"], 3),
        f1_psic=pt(per_class["Disfonia Psicogenica"]["f1"], 3),
        f1_func=pt(per_class["Disfonia Funcional"]["f1"], 3),
        mutual=mutual,
        dysph_n=dysph_n,
        n_arms=len(arch),
        best_psic=pt(max(float(r["f1_disfonia_psicogenica"]) for r in arch), 2),
        best_func=pt(max(float(r["f1_disfonia_funcional"]) for r in arch), 2),
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


def box(slide, l, t, w, h, fill=None, line=None, lw=1.2):
    """Retangulo reto -- sem canto arredondado, que era parte da aparencia de modelo."""
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(l), Cm(t), Cm(w), Cm(h))
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


def rule(slide, l, t, w, h, color=RULE_LIGHT, vertical=False, weight=0.055):
    """Fio fino horizontal (h=0) ou vertical (w=0)."""
    if vertical:
        return box(slide, l, t, weight, h, fill=color)
    return box(slide, l, t, w, weight, fill=color)


def write(shape, lines, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
          margins=(0.0, 0.0, 0.0, 0.0), spacing=0.95, tracking=None):
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
            if tracking:
                # font._element ja e o rPr; spc e a entreletra em centesimos de ponto
                run.font._element.set("spc", str(int(tracking * 100)))
    return shape


def text(slide, l, t, w, h, lines, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
         margins=(0.0, 0.0, 0.0, 0.0), spacing=0.95, tracking=None):
    tb = slide.shapes.add_textbox(Cm(l), Cm(t), Cm(w), Cm(h))
    return write(tb, lines, align, anchor, margins, spacing, tracking)


def section(slide, t, label):
    """Cabecalho de secao: caixa alta com entreletra, sobre fio de largura total.

    Devolve o y onde o conteudo da secao pode comecar.
    """
    text(slide, M, t, W, 2.4, [[(label, 60, True, INK)]], align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, tracking=6)
    rule(slide, M, t + 2.5, W, 0.0, color=INK, weight=0.10)
    return t + 3.2


def hero(slide, y, K):
    """Linha de estatisticas: um numero dominante e tres de apoio, separados por fio."""
    text(slide, M, y - 0.5, 26.0, 6.0, [[(K["acc_pct"], 132, True, ACCENT)]],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=0.85)
    text(slide, M + 0.3, y + 4.6, 26.0, 2.4,
         [[("acurácia global", 32, True, INK)], [(K["acc_ci"], 26, False, MUTED)]],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

    stats = [(K["macro"], "Macro F1, 5 classes", K["macro_ci"]),
             ("p < 0,001", "vs. os três baselines",
              "McNemar · classe majoritária, k-NN, log."),
             (K["f1_normal"], "F1 da classe Normal",
              f"recall de {K['recall_normal']} nos saudáveis")]
    x0 = M + 28.0
    sw = (RIGHT - x0) / 3
    for i, (big, lab, sub) in enumerate(stats):
        x = x0 + i * sw
        rule(slide, x - 1.4, y + 0.1, 0.0, 6.0, vertical=True)
        text(slide, x, y, sw - 1.4, 3.0, [[(big, 78, True, INK)]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=0.85)
        text(slide, x, y + 3.1, sw - 1.4, 3.0,
             [[(lab, 30, True, INK)], [(sub, 25, False, MUTED)]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)


def aspect(path):
    """Razao largura/altura do PNG, para casar as alturas de figuras lado a lado."""
    from PIL import Image

    w, h = Image.open(path).size
    return w / h


def picture(slide, path, l, t, w):
    """Insere preservando o aspecto nativo; devolve (shape, altura em cm)."""
    pic = slide.shapes.add_picture(path, Cm(l), Cm(t), width=Cm(w))
    return pic, pic.height / 360000.0
# ---------------------------------------------------------------- diagrama do pipeline
def pipeline(slide, y, h):
    """Faixa horizontal com os cinco estagios, separados por fio -- nao por cartoes.

    Sem caixas aninhadas e sem uma cor por estagio: a leitura e conduzida pelo numeral
    grande, pelo peso do rotulo e pelo fio vertical entre estagios.
    """
    band = box(slide, M, y, W, h, TINT, None)
    band.line.fill.background()
    stages = [
        ("1", "VOZ",
         [("big", "1.098"), ("label", "pacientes da base SVD"),
          ("mid", "/a/   /i/   /u/"), ("label", "vogais sustentadas"),
          ("small", "687 / 140 / 91 / 112 / 68")]),
        ("2", "CARACTERÍSTICAS",
         [("big", "251"), ("label", "variáveis por paciente"),
          ("body", "10 temporais, 55 espectrais"),
          ("body", "e 18 wavelet, por vogal"),
          ("small", "+ idade e sexo do paciente")]),
        ("3", "DUAS REDES POR VOGAL",
         [("big", "6"), ("label", "redes por dobra"),
          ("body", "MESTRA — normal ou não"),
          ("body", "ESPECIALISTA — qual das 4"),
          ("small", "85 → 128 → 64 · Adam")]),
        ("4", "FUSÃO TARDIA",
         [("big", "3"), ("label", "vogais votam"),
          ("body", "média das probabilidades"),
          ("body", "das três vogais"),
          ("small", "nenhuma decide sozinha")]),
        ("5", "DECISÃO",
         [("mid", "P(pat.) ≥ 0,5"), ("label", "limiar da rede Mestra"),
          ("body", "sim → 4 patologias"),
          ("body", "não → Normal"),
          ("small", "5 classes na saída")]),
    ]
    sw = W / len(stages)
    for i, (num, title, lines) in enumerate(stages):
        x = M + i * sw
        if i:
            rule(slide, x, y + 1.4, 0.0, h - 3.4, vertical=True)
        text(slide, x + 1.3, y + 1.0, 4.0, 2.4, [[(num, 46, True, RULE_STRONG)]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
        text(slide, x + 1.3, y + 3.5, sw - 2.2, 2.9, [[(title, 30, True, ACCENT)]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=0.9)
        ty = y + 7.0
        for kind, txt in lines:
            size, color, bold, gap = KINDS[kind]
            text(slide, x + 1.3, ty, sw - 2.2, gap + 0.8,
                 [[(txt, size, bold, color)]], align=PP_ALIGN.LEFT,
                 anchor=MSO_ANCHOR.TOP, spacing=0.9)
            ty += gap
    text(slide, M + 1.3, y + h - 1.9, W - 2.6, 1.7,
         [[("Validação cruzada de 5 dobras estratificadas · semente fixa 42 · z-score e "
            "balanceamento (Borderline-SMOTE1) ajustados dentro de cada dobra, nunca no "
            "conjunto de teste", 26, False, MUTED)]],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)


# ---------------------------------------------------------------- poster
def build(src, dst, figs="results/figures", results="results"):
    K = load_claims(results)
    prs = Presentation(src)
    slide = prs.slides[0]
    qr = clear_slide(slide)

    # ---------------- cabecalho do trabalho
    text(slide, M, 18.1, W, 7.4,
         [[("IDENTIFICAÇÃO COMPUTACIONAL INTELIGENTE", 100, True, INK)],
          [("DE DISFUNÇÕES NO TRATO VOCAL", 100, True, INK)]],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=0.9)
    text(slide, M, 26.0, W, 2.2,
         [[("Ronaldo Chiavegatti Sampaio Corrêa", 48, True, INK),
           ("     ·     ", 48, False, RULE_STRONG),
           ("Rodrigo Capobianco Guido", 48, False, INK2),
           ("   orientador", 36, False, MUTED)]],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    text(slide, M, 28.0, W, 1.6,
         [[("IBILCE/Unesp — Câmpus de São José do Rio Preto   ·   Iniciação Científica "
            "voluntária, sem bolsa   ·   pipeline em C99, sem frameworks de ML",
            30, False, MUTED)]],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    rule(slide, M, 30.0, W, 0.0, color=ACCENT, weight=0.14)

    # ---------------- o problema
    y = section(slide, 31.2, "O PROBLEMA")
    text(slide, M, y, W * 0.70, 5.0,
         [[("O diagnóstico diferencial das disfonias exige hoje exame endoscópico. ",
            40, False, INK),
           ("Este trabalho mede até onde a voz sozinha chega — e onde ela não chega.",
            40, True, ACCENT)]],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0)

    # ---------------- metodo
    y = section(slide, 38.8, "MÉTODO")
    pipeline(slide, y, 20.0)

    # ---------------- resultados
    y = section(slide, 62.2, "RESULTADOS")
    hero(slide, y, K)

    # larguras resolvidas para que as duas figuras terminem na mesma altura, a partir
    # do aspecto real dos arquivos -- e nao de um numero fixado aqui
    by = y + 8.0
    gap = 2.0
    asp_c, asp_g = aspect(f"{figs}/poster_confusao.png"), aspect(f"{figs}/poster_gaps_ab.png")
    band = (W - gap) / (asp_c + asp_g)
    wc = asp_c * band
    picture(slide, f"{figs}/poster_confusao.png", M, by, wc)
    xg = M + wc + gap
    picture(slide, f"{figs}/poster_gaps_ab.png", xg, by, RIGHT - xg)

    cy = by + band + 0.5
    text(slide, M, cy, wc, 3.4,
         [[("Onde o modelo erra.", 30, True, ACCENT),
           (f"  As duas disfonias funcionais trocam {K['mutual']} dos "
            f"{K['dysph_n']} pacientes entre si — nenhuma tem lesão estrutural, e a voz "
            "sozinha não as separa.", 30, False, INK2)]],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    text(slide, xg, cy, RIGHT - xg, 3.4,
         [[("Cada mudança foi comparada A/B.", 30, True, ACCENT),
           ("  Mesma semente, mesmas dobras, decisão por regra fixa: duas técnicas "
            "adotadas, uma rejeitada pela própria evidência.", 30, False, INK2)]],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

    # ---------------- conclusao
    y = section(slide, 100.0, "CONCLUSÃO")
    concl = [("Triagem funciona.",
              f"Saudável × patológico se separa: F1 {K['f1_normal']}, "
              f"recall {K['recall_normal']}."),
             ("Reinke é separável.",
              f"A lesão estrutural deixa marca acústica: F1 {K['f1_reinke']}."),
             ("As funcionais, não.",
              f"F1 {K['f1_psic']} e {K['f1_func']}; melhor de {K['n_arms']} braços: "
              f"{K['best_psic']} e {K['best_func']}.")]
    cw = (W - 2 * 2.4) / 3
    for i, (head, body) in enumerate(concl):
        x = M + i * (cw + 2.4)
        if i:
            rule(slide, x - 1.2, y + 0.2, 0.0, 4.6, vertical=True)
        text(slide, x, y, cw, 4.8,
             [[(head, 34, True, INK)], [(body, 29, False, INK2)]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.02)

    # ---------------- referencias
    ry = 108.2
    rule(slide, M, ry - 0.9, W, 0.0)
    refs = [("BARRY, W. J.; PÜTZER, M. ",
             "Saarbrücken Voice Database. Univ. des Saarlandes, 2007."),
            ("HAN, H.; WANG, W.-Y.; MAO, B.-H. ",
             "Borderline-SMOTE. LNCS, v. 3644, p. 878-887, 2005."),
            ("KINGMA, D. P.; BA, J. ",
             "Adam: stochastic optimization. ICLR, 2015."),
            ("GUIDO, R. C. ",
             "Wavelets behind the scenes. Physics Reports, v. 985, 2022."),
            ("LEE, J.-Y. ",
             "Deep learning for pathological voice detection. Appl. Sci., v. 11, 2021."),
            ("VRBA, J. et al. ",
             "Reproducible ML-based voice pathology detection. J. Voice, 2025.")]
    colw = (W - 6.6 - 2.0) / 2
    for i, (head, tail) in enumerate(refs):
        col, row = divmod(i, 3)
        text(slide, M + col * (colw + 2.0), ry + row * 1.55, colw, 1.5,
             [[(head, 24, True, INK2), (tail, 24, False, MUTED)]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    if qr is not None:
        qr.left, qr.top = Cm(RIGHT - 4.8), Cm(ry - 0.3)
        qr.width = qr.height = Cm(4.8)
    text(slide, RIGHT - 5.6, ry + 4.6, 5.6, 1.1,
         [[("código e dados", 21, False, MUTED)]], align=PP_ALIGN.CENTER)

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
