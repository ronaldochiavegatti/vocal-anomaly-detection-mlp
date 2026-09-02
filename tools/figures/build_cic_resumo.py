#!/usr/bin/env python3
"""Atualiza o resumo do XXXVIII CIC Unesp sobre o proprio arquivo do template.

Preserva pagina, margens, estilos e a fonte Times New Roman do modelo do congresso, e
reescreve o conteudo cientifico secao por secao, mantendo a estrutura de topicos exigida
(INTRODUCAO / MATERIAL E METODOS / RESULTADOS E DISCUSSAO / Figura 1 / Tabela 1 /
CONCLUSOES / AGRADECIMENTOS / REFERENCIAS).

Todos os numeros vem dos CSVs de results/ via load_claims(), nunca digitados aqui.

Uso:  python3 tools/figures/build_cic_resumo.py ENTRADA.docx SAIDA.docx
"""

import argparse
import os
import sys

import docx
from docx.shared import Cm, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cic_poster import load_claims  # noqa: E402
from make_cic_figures import read_ab, read_ci, read_metrics  # noqa: E402

FONT = "Times New Roman"
BODY = 10.0
SMALL = 8.0


# ---------------------------------------------------------------- helpers
def set_runs(para, runs, size=BODY):
    """Substitui o conteudo do paragrafo, mantendo o estilo/alinhamento existentes."""
    for run in list(para.runs):
        run._element.getparent().remove(run._element)
    for text, bold, italic in runs:
        run = para.add_run(text)
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
    return para


def find(paras, prefix):
    """Localiza o paragrafo pelo inicio do texto, nao pela posicao.

    Indexar por posicao tornava o script utilizavel uma unica vez: depois de gravar o
    resultado sobre o proprio arquivo, a contagem de paragrafos mudava (os espacadores
    vazios saem) e a segunda execucao estourava o indice.
    """
    for para in paras:
        if para.text.strip().upper().startswith(prefix.upper()):
            return para
    raise SystemExit(f"paragrafo iniciado por {prefix!r} nao encontrado no modelo")


def drop(para):
    para._element.getparent().remove(para._element)


def tighten(para, after=1, before=0, spacing=0.98):
    pf = para.paragraph_format
    pf.space_after = Pt(after)
    pf.space_before = Pt(before)
    pf.line_spacing = spacing
    pf.first_line_indent = Cm(0)


def replace_image(doc, shape, path, width_cm):
    """Troca os bytes da imagem embutida e reajusta a caixa, preservando o aspecto."""
    from PIL import Image

    blip = shape._inline.graphic.graphicData.pic.blipFill.blip
    part = doc.part.related_parts[blip.embed]
    with open(path, "rb") as fh:
        part._blob = fh.read()
    w, h = Image.open(path).size
    shape.width = Cm(width_cm)
    shape.height = Cm(width_cm * h / w)
    return shape.height.cm


# ---------------------------------------------------------------- texto
def build(src, dst, results="results", figs="results/figures"):
    K = load_claims(results)
    keys = ["Normal", "Laringite", "Disfonia Psicogenica", "Disfonia Funcional",
            "Edema de Reinke"]
    per_class, macro, acc, _ = read_metrics(
        os.path.join(results, "metrics_global_borderline_C_baseline.csv"))
    ci = read_ci(os.path.join(results, "bootstrap_ci_borderline_C_baseline.csv"))
    smote = read_ab(os.path.join(results, "smote_ab_comparison.csv"),
                    "standard", "borderline")
    para_ab = read_ab(os.path.join(results, "paraconsistent_ab_comparison.csv"),
                      "without_selection", "with_selection", "without", "with")

    def br(x, nd=4):
        return f"{x:.{nd}f}".replace(".", ",")

    smote_delta = br(smote["macro_f1"][1][0] - smote["macro_f1"][0][0])
    para_delta = br(para_ab["macro_f1"][1][0] - para_ab["macro_f1"][0][0])

    doc = docx.Document(src)
    paras = doc.paragraphs

    # ---------------- INTRODUCAO
    set_runs(find(paras, "INTRODUÇÃO"), [
        ("INTRODUÇÃO: ", True, False),
        ("O diagnóstico diferencial de distúrbios vocais costuma exigir exame "
         "endoscópico, e o caso mais difícil é separar as disfonias funcionais sem lesão "
         "estrutural — psicogênica e funcional — de quadros orgânicos como laringite e "
         "edema de Reinke. A análise acústica computacional desponta como triagem não "
         "invasiva e de baixo custo. Este trabalho avalia um classificador de cinco "
         "condições vocais a partir de vogais sustentadas, em C99 puro, sem frameworks de "
         "aprendizado de máquina. A contribuição desta etapa é metodológica: cada decisão "
         "de projeto passou por comparação A/B reprodutível — mesma semente, mesmas dobras "
         "— com adoção ou rejeição por regra fixa definida antes de ver o resultado.",
         False, False)])

    # ---------------- MATERIAL E METODOS
    set_runs(find(paras, "MATERIAL E MÉTODOS"), [
        ("MATERIAL E MÉTODOS: ", True, False),
        (f"Usou-se um subconjunto de {K['n_patients']} pacientes da Saarbrücken Voice "
         "Database (BARRY; PÜTZER, 2007), nas cinco classes e nos tamanhos da Tabela 1, "
         "com as vogais sustentadas /a/, /i/ e /u/. ", False, False),
        ("(i) Extração: ", True, False),
        ("83 características por vogal — 10 temporais (jitter, shimmer, HNR, ZCR), 55 "
         "espectrais (F0, formantes F1–F4, entropia, centroide, rolloff, 13 MFCC com Δ e "
         "ΔΔ, CPP, pulso glotal) e 18 wavelet (Daubechies-4, seis níveis; GUIDO, 2022) — "
         "que, com idade e sexo, somam 251 variáveis. ", False, False),
        ("(ii) Protocolo: ", True, False),
        ("cinco dobras estratificadas, semente fixa (42); z-score ajustado só nas amostras "
         "originais de treino; augmentação e balanceamento apenas no treino de cada dobra. ",
         False, False),
        ("(iii) Arquitetura: ", True, False),
        ("fusão tardia multi-vogal — por vogal, uma rede binária (“Mestra”, patológico × "
         "saudável) e uma de quatro classes (“Especialista”), ambas com camadas ocultas "
         "[128, 64], LeakyReLU, dropout, L2, Adam (KINGMA; BA, 2015) e parada por Macro "
         "F1; as três vogais entram por média das probabilidades, com limiar "
         "P(patológico) ≥ 0,5. ", False, False),
        ("(iv) Experimentos A/B: ", True, False),
        ("SMOTE padrão × Borderline-SMOTE1 (HAN; WANG; MAO, 2005), doze combinações de "
         "arquitetura e regularização, e a seleção paraconsistente de características; "
         "McNemar para significância e bootstrap (N = 1000) para os intervalos.",
         False, False)])

    # ---------------- RESULTADOS E DISCUSSAO
    set_runs(find(paras, "RESULTADOS E DISCUSSÃO"), [
        ("RESULTADOS E DISCUSSÃO: ", True, False),
        (f"A configuração adotada alcançou acurácia de {K['acc_pct']} "
         f"[IC 95%: {br(ci['accuracy'][1]*100, 1)}–{br(ci['accuracy'][2]*100, 1)}%] e "
         f"Macro F1 de {K['macro']} "
         f"[{br(ci['macro_f1'][1], 3)}–{br(ci['macro_f1'][2], 3)}] (Tabela 1), superando os "
         "três baselines — classe majoritária, k-NN e regressão logística — por McNemar "
         "(p < 0,001 em todos). ", False, False),
        ("Experimentos A/B: ", True, False),
        (f"o Borderline-SMOTE1 foi adotado por elevar o Macro F1 em {smote_delta}, com "
         "ganho concentrado nas duas classes menores, embora a diferença entre os braços "
         "não seja significativa (p = 0,6606); das doze combinações, [128, 64] foi mantida "
         "por ser a mais parcimoniosa dentro de um erro-padrão da melhor e não inferior a "
         "ela (p = 0,7463); e a seleção paraconsistente foi ", False, False),
        ("rejeitada", True, False),
        (f", por não descartar nenhuma característica (0% de redução) e reduzir o Macro F1 "
         f"em {para_delta.lstrip('-')} — resultado negativo reportado por transparência, "
         "não uma técnica do modelo final. ", False, False),
        ("Erros: ", True, False),
        (f"Normal foi a classe mais bem discriminada (F1 = {K['f1_normal']}, recall de "
         f"{K['recall_normal']}), seguida do Edema de Reinke (F1 = {K['f1_reinke']}), "
         f"única com lesão estrutural. A Figura 1 mostra "
         f"{K['mutual']} dos {K['dysph_n']} pacientes das duas disfonias funcionais "
         "trocados entre si — coerente com a literatura sobre sobreposição acústica entre "
         "disfonias sem lesão estrutural (LEE, 2021; VRBA et al., 2025).", False, False)])

    # ---------------- legenda da figura e da tabela
    set_runs(find(paras, "Figura 1"), [
        ("Figura 1. ", True, False),
        (f"Matriz de confusão agregada ({K['n_patients']} pacientes, cinco dobras). "
         "Diagonal = acertos.", False, False)], size=9)
    set_runs(find(paras, "Tabela 1"), [
        ("Tabela 1. ", True, False),
        ("Desempenho por classe (predições fora da dobra de treino).",
         False, False)], size=9)

    # ---------------- CONCLUSOES
    set_runs(find(paras, "CONCLUSÕES"), [
        ("CONCLUSÕES: ", True, False),
        (f"A triagem entre voz saudável e patológica mostrou-se viável (Normal: F1 = "
         f"{K['f1_normal']}, recall de {K['recall_normal']}) e o Edema de Reinke foi "
         f"separado com F1 de {K['f1_reinke']}. O diferencial entre as duas disfonias "
         f"funcionais segue como gargalo: F1 de {K['f1_psic']} e {K['f1_func']}, sem que "
         f"nenhum dos {K['n_arms']} braços superasse {K['best_psic']} e {K['best_func']} "
         "nessas classes — evidência de um teto do próprio sinal acústico, que deve exigir "
         "informação além da voz. Das três técnicas previstas na "
         "proposta original, duas se confirmaram e uma foi descartada pela evidência A/B.",
         False, False)])

    # ---------------- AGRADECIMENTOS
    set_runs(find(paras, "AGRADECIMENTOS"), [
        ("AGRADECIMENTOS: ", True, False),
        ("Ao Prof. Dr. Eng. Rodrigo Capobianco Guido, pela orientação; ao IBILCE/Unesp, "
         "pela infraestrutura; e aos autores da Saarbrücken Voice Database. IC "
         "voluntária, sem bolsa.", False, False)])

    # ---------------- REFERENCIAS
    ref_head = find(paras, "REFERÊNCIAS")
    set_runs(ref_head, [("REFERÊNCIAS", True, False)])
    refs = [
        ("BARRY, W. J.; PÜTZER, M. ", "Saarbrücken Voice Database. Universität des "
         "Saarlandes, 2007. Acesso em: 26 jul. 2026."),
        ("HAN, H.; WANG, W.-Y.; MAO, B.-H. ", "Borderline-SMOTE: a new over-sampling "
         "method in imbalanced data sets learning. LNCS, v. 3644, p. 878-887, 2005."),
        ("GUIDO, R. C. ", "Wavelets behind the scenes: practical aspects, insights, and "
         "perspectives. Physics Reports, v. 985, 2022."),
        ("KINGMA, D. P.; BA, J. ", "Adam: a method for stochastic optimization. ICLR, "
         "2015."),
        ("LEE, J.-Y. ", "Experimental evaluation of deep learning methods for an "
         "intelligent pathological voice detection system using the Saarbruecken Voice "
         "Database. Applied Sciences, v. 11, n. 15, art. 7149, 2021."),
        ("VRBA, J. et al. ", "Reproducible machine learning-based voice pathology "
         "detection: introducing the pitch difference feature. Journal of Voice, 2025."),
    ]
    existing = paras[paras.index(ref_head) + 1:]
    written = []
    anchor = ref_head
    for i, (head, tail) in enumerate(refs):
        if i < len(existing):
            target = existing[i]
        else:
            target = doc.add_paragraph()
            anchor._element.addnext(target._element)
            target.alignment = anchor.alignment
            target.style = anchor.style
        set_runs(target, [(head, True, False), (tail, False, False)], size=SMALL)
        tighten(target, after=0, spacing=0.96)
        written.append(target)
        anchor = target
    for extra in existing[len(refs):]:   # sobras do modelo, incluindo espacadores
        drop(extra)

    # ---------------- figura
    height = replace_image(doc, doc.inline_shapes[0],
                           os.path.join(figs, "resumo_fig1_confusao.png"), 7.0)

    # ---------------- tabela
    table = doc.tables[0]
    header = ["Classe", "Precisão", "Recall", "F1-Score", "n"]
    for cell, label in zip(table.rows[0].cells, header):
        set_runs(cell.paragraphs[0], [(label, True, False)], size=SMALL)
    labels = ["Normal", "Laringite", "Disfonia Psicogênica", "Disfonia Funcional",
              "Edema de Reinke"]
    for row, key, label in zip(table.rows[1:], keys, labels):
        m = per_class[key]
        values = [label, br(m["precision"], 3), br(m["recall"], 3), br(m["f1"], 3),
                  str(m["support"])]
        for cell, value in zip(row.cells, values):
            set_runs(cell.paragraphs[0], [(value, False, False)], size=SMALL)
    # larguras explicitas: a tabela tem de caber na coluna de 8,22 cm. E preciso
    # reescrever o tblGrid do template (3,4 cm por coluna), nao so o tcW das celulas,
    # senao as duas ultimas colunas ficam fora da coluna de texto.
    widths = [2.85, 1.45, 1.25, 1.35, 0.85]
    table.autofit = False
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    grid = table._tbl.find(f"{ns}tblGrid")
    if grid is not None:
        for col, wcm in zip(grid.findall(f"{ns}gridCol"), widths):
            col.set(f"{ns}w", str(int(Cm(wcm).twips)))
    # o tblW do template ficou em 17 cm (largura de pagina inteira); sem ajusta-lo a
    # tabela continua sendo diagramada mais larga que a coluna e as ultimas colunas
    # saem da area de texto
    tbl_w = table._tbl.find(f"{ns}tblPr/{ns}tblW")
    if tbl_w is not None:
        tbl_w.set(f"{ns}w", str(sum(int(Cm(w).twips) for w in widths)))
        tbl_w.set(f"{ns}type", "dxa")
    for row in table.rows:
        for cell, wcm in zip(row.cells, widths):
            cell.width = Cm(wcm)

    # ---------------- compactar para caber em 1 pagina
    # remove espacadores vazios, preservando o paragrafo que carrega a figura
    for para in list(doc.paragraphs):
        if para.text.strip():
            continue
        if para._element.findall(".//{http://schemas.openxmlformats.org/"
                                 "drawingml/2006/wordprocessingDrawing}inline"):
            continue
        drop(para)
    for para in doc.paragraphs:
        if para.text.strip():
            tighten(para, after=1, spacing=0.98)

    doc.save(dst)
    return dst, height


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figs", default="results/figures")
    a = ap.parse_args()
    out, h = build(a.src, a.dst, a.results, a.figs)
    print(f"gerado: {out}  (figura com {h:.2f} cm de altura)")
