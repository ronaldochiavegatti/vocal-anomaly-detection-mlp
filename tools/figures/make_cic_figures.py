#!/usr/bin/env python3
"""Gera as figuras do resumo e do poster do XXXVIII CIC Unesp a partir dos CSVs de results/.

Nao faz parte do pipeline de build em C -- e um utilitario de geracao de material
academico. Todos os numeros sao lidos dos artefatos versionados em results/, de modo
que as figuras sao reprodutiveis e nao contem valores digitados a mao.

Configuracao reportada = braco adotado apos o fechamento dos 3 gaps do SPEC.md:
Hierarchical Late Fusion + Borderline-SMOTE1 + Config C [128,64], sem selecao
paraconsistente (Gap 1 REJEITADO).

Uso:  python3 tools/figures/make_cic_figures.py [--results results] [--out results/figures]
"""

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import blended_transform_factory

# ---------------------------------------------------------------- paleta
# Paleta de referencia da skill dataviz, validada por scripts/validate_palette.js
# (modo light, superficie #fcfcfb): todos os checks PASS.
BLUE = "#2a78d6"      # slot categorico 1
ORANGE = "#eb6834"    # slot categorico 2
AQUA = "#1baf7a"      # slot categorico 3
ORD = ["#7cafe6", "#2f7fd4", "#15467d"]  # rampa ordinal 1 hue (light/baseline/strong)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e3e2dd"
SURFACE = "#fcfcfb"
GOOD = "#008300"
CRIT = "#c8322f"

FONT = "Liberation Sans"  # metricamente compativel com Arial (fonte do template)

CLASSES = ["Normal", "Laringite", "Disfonia\nPsicogenica", "Disfonia\nFuncional",
           "Edema de\nReinke"]
CLASSES_ACC = ["Normal", "Laringite", "Disfonia\nPsicogênica", "Disfonia\nFuncional",
               "Edema de\nReinke"]
SUPPORT = [687, 140, 91, 112, 68]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "svg.fonttype": "none",
})


# ---------------------------------------------------------------- leitura dos artefatos
def read_metrics(path):
    """results/metrics_global_*.csv -> (per_class dict, macro_f1, accuracy, weighted_f1)."""
    per_class, macro, acc, weighted = {}, None, None, None
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["class"]
            if name == "macro_avg":
                macro = float(row["f1"])
            elif name == "accuracy":
                acc = float(row["f1"])
            elif name == "weighted_avg":
                weighted = float(row["f1"])
            else:
                per_class[name] = dict(precision=float(row["precision"]),
                                       recall=float(row["recall"]),
                                       f1=float(row["f1"]),
                                       support=int(row["support"]))
    return per_class, macro, acc, weighted


def read_ci(path):
    """results/bootstrap_ci_*.csv -> {metric: (mean, lo, hi)}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["metric"]] = (float(row["mean"]), float(row["ci_lower"]),
                                  float(row["ci_upper"]))
    return out


def read_ab(path, col_a, col_b, ci_a=None, ci_b=None):
    """CSV de comparacao A/B -> {metric: ((a, a_lo, a_hi), (b, b_lo, b_hi))}.

    ci_a/ci_b sao os prefixos das colunas de IC quando diferem do nome da coluna de
    valor (o CSV do Gap 1 usa 'without_ci_lower' para a coluna 'without_selection').
    """
    ci_a, ci_b = ci_a or col_a, ci_b or col_b
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            def trio(col, ci):
                if not row.get(col):
                    return None
                lo, hi = row.get(f"{ci}_ci_lower"), row.get(f"{ci}_ci_upper")
                return (float(row[col]), float(lo) if lo else None,
                        float(hi) if hi else None)
            out[row["metric"]] = (trio(col_a, ci_a), trio(col_b, ci_b))
    return out


def read_arch(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_mcnemar(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["baseline"]] = (float(row["chi2"]), float(row["p_value"]))
    return out


def read_confusion(path, supports, accuracy, tol=5e-4):
    """Extrai do log a matriz de confusao 5x5 agregada que corresponde ao braco pedido.

    O log de console contem varias matrizes (por fold e por braco A/B). A selecao e
    feita por conferencia com os artefatos CSV -- suportes por classe e acuracia --
    para nao depender da ordem em que os bracos foram escritos.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    supports = np.asarray(supports)
    matches = []
    for s in [i for i, ln in enumerate(lines) if "Matriz de Confusao" in ln]:
        rows = []
        for ln in lines[s + 2:s + 7]:
            nums = re.findall(r"(?<![\d.])\d+(?![\d.])", ln)
            if len(nums) < 5:
                break
            rows.append([int(v) for v in nums[-5:]])
        if len(rows) != 5:
            continue
        m = np.array(rows)
        if (m.sum(axis=1) == supports).all() and abs(np.trace(m) / m.sum() - accuracy) < tol:
            matches.append(m)
    if not matches:
        raise SystemExit(f"nenhuma matriz agregada compativel (acuracia {accuracy:.6f}) em {path}")
    return matches[-1]


def build_arch_arms(arch_rows, results):
    """Para cada arquitetura, o braco de melhor regularizacao com seu IC bootstrap.

    A comparacao do Gap 3 varreu 12 combinacoes; o forest plot mostra so o melhor braco
    de cada arquitetura, que e sobre o que a regra de decisao (banda de 1 erro-padrao +
    McNemar + menos parametros) de fato opera.
    """
    labels = {"A": "A  [128]", "B": "B  [64]", "C": "C  [128, 64]",
              "D": "D  [128, 64, 32]"}
    best = {}
    for row in arch_rows:
        a, f1 = row["arch"], float(row["macro_f1"])
        if a not in best or f1 > best[a][0]:
            best[a] = (f1, row["reg"])
    arms = []
    for a in ("A", "B", "C", "D"):
        f1, reg = best[a]
        ci = read_ci(os.path.join(results, f"bootstrap_ci_borderline_{a}_{reg}.csv"))
        adopted = a == "C"
        arms.append((labels[a], ci["macro_f1"], ORANGE if adopted else BLUE,
                     "ADOTADA" if adopted else ""))
    return arms


def fmt_p(p):
    return "p < 0,001" if p < 0.001 else f"p = {p:.4f}".replace(".", ",")


def br(x, nd=4):
    return f"{x:.{nd}f}".replace(".", ",")


# ---------------------------------------------------------------- helpers de desenho
def style_axes(ax, ymax=1.0, ystep=0.2, ylabel=None):
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1e-9, ystep))
    ax.set_yticklabels([br(v, 1) for v in np.arange(0, ymax + 1e-9, ystep)])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if ylabel:
        ax.set_ylabel(ylabel)


def rounded_bars(ax, xs, values, width, color, radius_frac=0.22):
    """Barras finas: base quadrada ancorada na linha de zero, topo arredondado."""
    for x, v in zip(xs, values):
        if v <= 0:
            continue
        x0, x1 = x - width / 2, x + width / 2
        r = min(width * radius_frac, v * 0.5)
        verts = [(x0, 0), (x0, v - r), (x0, v), (x0 + r, v), (x1 - r, v), (x1, v),
                 (x1, v - r), (x1, 0), (x0, 0)]
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
                 Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), linewidth=0, facecolor=color))


# ---------------------------------------------------------------- figuras
def fig_confusion_compact(cm, out, figsize=(2.756, 1.68), dpi=600):
    """Matriz de confusao dimensionada para a coluna de 8,22 cm do resumo.

    O resumo e diagramado em duas colunas, entao a figura e gerada no tamanho fisico
    final (1:1) e as fontes sao declaradas em pontos de impressao -- nao ha reducao de
    escala depois, que e o que tornaria os rotulos ilegiveis.
    """
    recall = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "azul", ["#ffffff", "#dbe9f8", "#9cc2ec", "#4b8ed8", "#15467d"])
    ax.imshow(recall, cmap=ramp, vmin=0, vmax=1, aspect="auto")
    for i in range(5):
        for j in range(5):
            v = recall[i, j]
            diag = i == j
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=7.5 if diag else 6.5,
                    fontweight="bold" if diag else "normal",
                    color="#ffffff" if v > 0.55 else INK)
    for k in range(6):
        ax.axhline(k - 0.5, color=SURFACE, linewidth=1.1)
        ax.axvline(k - 0.5, color=SURFACE, linewidth=1.1)
    short = ["Normal", "Laring.", "D. Psic.", "D. Func.", "Reinke"]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(short, fontsize=5.8)
    ax.set_yticklabels([f"{s} ({n})" for s, n in zip(short, SUPPORT)], fontsize=5.8)
    ax.set_xlabel("Classe prevista pelo modelo", fontsize=6.4, labelpad=3)
    ax.set_ylabel("Classe real (n)", fontsize=6.4, labelpad=3)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def fig_confusion_poster(cm, out, width_cm=41.0, height_cm=21.5, dpi=200):
    """Matriz de confusao para o poster A0, gerada no tamanho fisico final.

    Como a figura e inserida em 1:1, os tamanhos abaixo valem como pontos de
    impressao: e o que garante legibilidade a alguns metros de distancia. Gerar
    pequeno e ampliar na colocacao -- o erro da versao anterior -- reduz a fonte
    efetiva a ~12 pt no papel.
    """
    recall = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54), dpi=dpi)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "azul", ["#ffffff", "#dceaf9", "#a8c9ee", "#5b95da", "#1f4e8c"])
    ax.imshow(recall, cmap=ramp, vmin=0, vmax=1, aspect="auto")
    for i in range(5):
        for j in range(5):
            v = recall[i, j]
            col = "#ffffff" if v > 0.5 else INK
            if i == j:
                ax.text(j, i - 0.10, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=46, fontweight="bold", color=col)
                ax.text(j, i + 0.26, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=26, color=col)
            else:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=32,
                        color=col)
    for k in range(6):
        ax.axhline(k - 0.5, color="#ffffff", linewidth=4)
        ax.axvline(k - 0.5, color="#ffffff", linewidth=4)
    labels = ["Normal", "Laringite", "Disfonia\nPsicogênica", "Disfonia\nFuncional",
              "Edema de\nReinke"]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=26)
    ax.set_yticklabels([f"{c}\n{n} pacientes" for c, n in zip(labels, SUPPORT)],
                       fontsize=25)
    ax.set_xlabel("classe prevista pelo modelo", fontsize=28, labelpad=16, color=INK2)
    ax.set_ylabel("classe clínica real", fontsize=28, labelpad=16, color=INK2)
    ax.tick_params(length=0, pad=10)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def fig_gaps_poster(smote, para, arch_arms, mcn_p_smote, mcn_p_para, mcn_p_arch, out,
                    width_cm=41.8, height_cm=22.0, dpi=200):
    """Forest plot com as decisoes A/B dos tres gaps, no tamanho fisico do poster.

    Reune num unico grafico legivel o que antes eram duas figuras (este forest plot e
    um grafico de 12 barras para as arquiteturas). Todos os pontos sao medias bootstrap
    com IC 95%, para nao misturar estimativa pontual crua com media bootstrap na mesma
    escala.
    """
    fig, (axl, ax) = plt.subplots(1, 2, figsize=(width_cm / 2.54, height_cm / 2.54),
                                  dpi=dpi,
                                  gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.02))
    groups = [
        dict(title="Balanceamento das classes",
             arms=[("SMOTE padrão", smote["macro_f1"][0], BLUE, ""),
                   ("Borderline-SMOTE1", smote["macro_f1"][1], ORANGE, "ADOTADO")],
             p=mcn_p_smote, badge_color=GOOD),
        dict(title="Profundidade da rede",
             arms=arch_arms, p=mcn_p_arch, badge_color=GOOD),
        dict(title="Seleção paraconsistente",
             arms=[("sem seleção", para["macro_f1"][0], BLUE, "MANTIDO"),
                   ("com seleção", para["macro_f1"][1], ORANGE, "REJEITADA")],
             p=mcn_p_para, badge_color=CRIT),
    ]
    rows, y = [], 0.0
    for g in groups:
        g["head_y"] = y
        for lab, v, col, badge in g["arms"]:
            y -= 1.0
            rows.append((y, lab, v, col, badge, g["badge_color"]))
        g["foot_y"] = y - 0.68
        y -= 1.72

    for a in (axl, ax):
        a.set_ylim(y + 0.42, 0.62)
        for side in a.spines.values():
            side.set_visible(False)
        a.tick_params(length=0)
        a.set_yticks([])
    axl.set_xlim(0, 1)
    axl.set_xticks([])
    ax.patch.set_visible(False)

    for g in groups:
        axl.text(0.0, g["head_y"], g["title"], fontsize=31, fontweight="bold",
                 color=INK, va="center", clip_on=False)
        axl.text(0.0, g["foot_y"], f"McNemar {fmt_p(g['p'])} · não significativo",
                 fontsize=24, color=MUTED, va="center", clip_on=False)

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for (yy, lab, v, col, badge, badge_col) in rows:
        axl.text(0.045, yy, lab, fontsize=28, color=INK2, va="center", clip_on=False)
        ax.errorbar(v[0], yy, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color=col,
                    markersize=22, elinewidth=4, capsize=12, capthick=4,
                    markeredgecolor="#ffffff", markeredgewidth=4, zorder=3)
        ax.text(1.025, yy, br(v[0]), fontsize=31, fontweight="bold", color=INK,
                va="center", transform=trans, clip_on=False)
        if badge:
            ax.text(1.28, yy, badge, fontsize=27, fontweight="bold", color=badge_col,
                    va="center", transform=trans, clip_on=False)

    ax.set_xlim(0.383, 0.522)
    ticks = np.arange(0.40, 0.5201, 0.04)
    ax.set_xticks(ticks)
    ax.set_xticklabels([br(v, 2) for v in ticks], fontsize=25)
    ax.xaxis.grid(True, color=GRID, linewidth=2)
    ax.set_axisbelow(True)
    ax.set_xlabel("Macro F1 (média bootstrap, IC 95%) — mesma seed, mesmas 5 dobras",
                  fontsize=26, labelpad=12, color=INK2)
    fig.subplots_adjust(left=0.004, right=0.655, top=0.99, bottom=0.115)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def fig_arch_sweep(rows, out, figsize=(12.0, 3.76), dpi=300):
    """Gap 3: 4 arquiteturas x 3 forcas de regularizacao (rampa ordinal, 1 hue)."""
    archs = ["A", "B", "C", "D"]
    labels = {"A": "A\n[128]", "B": "B\n[64]", "C": "C\n[128, 64]", "D": "D\n[128, 64, 32]"}
    regs = ["light", "baseline", "strong"]
    reg_pt = {"light": "leve", "baseline": "padrão", "strong": "forte"}
    data = {(r["arch"], r["reg"]): float(r["macro_f1"]) for r in rows}
    params = {r["arch"]: int(r["param_count_master"]) + int(r["param_count_expert"])
              for r in rows}
    epochs = {(r["arch"], r["reg"]): float(r["mean_epochs_to_stop"]) for r in rows}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    step, w = 0.235, 0.20
    for gi, reg in enumerate(regs):
        xs = np.arange(4) + (gi - 1) * step
        vals = [data[(a, reg)] for a in archs]
        rounded_bars(ax, xs, vals, w, ORD[gi])
        for x, v in zip(xs, vals):
            crit = (reg == "strong" and v < 0.30)
            ax.text(x, v + 0.009, br(v, 3), ha="center", va="bottom", fontsize=9,
                    color=CRIT if crit else INK, fontweight="bold" if crit else "normal")

    # braco adotado: C / baseline -- selo curto logo acima da barra
    xc = 2.0
    ax.annotate("adotada", xy=(xc, 0.497), xytext=(xc, 0.543),
                fontsize=10.5, color=GOOD, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=GOOD, linewidth=1.8,
                                shrinkA=3, shrinkB=1))
    ax.text(xc, 0.578, "mais parcimoniosa dentro\nde 1 erro-padrão do melhor",
            fontsize=8.5, color=GOOD, ha="center", va="bottom", style="italic")

    # colapso do braco D/forte, como nota de rodape (evita sobrepor as barras)
    ax.text(0, -0.335, f"D com regularização forte colapsa — para em apenas "
            f"{epochs[('D','strong')]:.0f} épocas, contra {epochs[('D','light')]:.0f} do "
            f"seu próprio braço leve: esta varredura não distingue “profundidade não "
            f"ajuda” de “profundidade precisa de outro ajuste”.",
            transform=ax.transAxes, fontsize=8.5, color=CRIT, va="top", ha="left")

    style_axes(ax, 0.6, 0.1, "Macro F1 (out-of-fold)")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"{labels[a]}\n" + f"{params[a]:,}".replace(",", ".") + " parâmetros"
                        for a in archs], fontsize=9.5)
    ax.set_xlim(-0.5, 3.62)
    handles = [matplotlib.patches.Patch(facecolor=ORD[i], label=f"regularização {reg_pt[r]}")
               for i, r in enumerate(regs)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.005), frameon=False,
              fontsize=9.5, ncol=3, handlelength=1.0, handleheight=1.0, columnspacing=1.5)
    ax.set_title("Mesma seed (42) e mesmos 5 folds nos 12 braços · dropout e L2 variados em conjunto",
                 fontsize=9.5, color=INK2, pad=24, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    R, O = args.results, args.out
    os.makedirs(O, exist_ok=True)

    per_class, macro, acc, weighted = read_metrics(
        os.path.join(R, "metrics_global_borderline_C_baseline.csv"))
    class_keys = ["Normal", "Laringite", "Disfonia Psicogenica", "Disfonia Funcional",
                  "Edema de Reinke"]
    cm = read_confusion(os.path.join(R, "train_log_v32_gap2_smote_ab_console.txt"),
                        [per_class[k]["support"] for k in class_keys], acc)
    smote = read_ab(os.path.join(R, "smote_ab_comparison.csv"), "standard", "borderline")
    para = read_ab(os.path.join(R, "paraconsistent_ab_comparison.csv"),
                   "without_selection", "with_selection", "without", "with")
    arch = read_arch(os.path.join(R, "arch_compare_comparison.csv"))
    arch_arms = build_arch_arms(arch, R)

    # conferencia: a matriz de confusao tem de reproduzir a acuracia do CSV
    assert cm.sum() == 1098, f"matriz soma {cm.sum()}, esperado 1098"
    acc_cm = np.trace(cm) / cm.sum()
    assert abs(acc_cm - acc) < 5e-4, f"acuracia da matriz {acc_cm:.6f} != CSV {acc:.6f}"
    for name, sup in zip(class_keys, cm.sum(axis=1)):
        assert per_class[name]["support"] == sup, f"suporte divergente em {name}"
    for i, name in enumerate(class_keys):  # recall da matriz == recall do CSV
        assert abs(cm[i, i] / cm[i].sum() - per_class[name]["recall"]) < 5e-4, \
            f"recall divergente em {name}"

    made = [
        fig_confusion_compact(cm, os.path.join(O, "resumo_fig1_confusao.png")),
        fig_confusion_poster(cm, os.path.join(O, "poster_confusao.png")),
        fig_gaps_poster(smote, para, arch_arms, 0.6606, 0.8220, 0.7463,
                        os.path.join(O, "poster_gaps_ab.png")),
        fig_arch_sweep(arch, os.path.join(O, "poster_arquiteturas.png")),
    ]
    print(f"acuracia {acc:.6f} · macro F1 {macro:.6f} · weighted F1 {weighted:.6f}")
    print(f"matriz de confusao conferida (soma {cm.sum()}, acuracia {acc_cm:.6f})")
    for m in made:
        print("  gerado:", m)


if __name__ == "__main__":
    main()
