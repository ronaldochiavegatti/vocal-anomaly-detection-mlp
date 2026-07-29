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
from matplotlib.patches import FancyBboxPatch, PathPatch
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
def fig_confusion(cm, per_class, out, figsize=(8.45, 5.0), dpi=300, title=True):
    recall = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "azul", ["#ffffff", "#dbe9f8", "#9cc2ec", "#4b8ed8", "#15467d"])
    ax.imshow(recall, cmap=ramp, vmin=0, vmax=1, aspect="auto")

    for i in range(5):
        for j in range(5):
            v = recall[i, j]
            col = "#ffffff" if v > 0.55 else INK
            diag = i == j
            ax.text(j, i - (0.13 if diag else 0), f"{cm[i, j]}",
                    ha="center", va="center", fontsize=13 if diag else 11,
                    fontweight="bold" if diag else "normal", color=col)
            if diag:
                ax.text(j, i + 0.22, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=9, color=col)
    # separadores de 2px na cor da superficie
    for k in range(6):
        ax.axhline(k - 0.5, color=SURFACE, linewidth=2)
        ax.axvline(k - 0.5, color=SURFACE, linewidth=2)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASSES_ACC, fontsize=9.5)
    ax.set_yticklabels([f"{c}\n(n={n})" for c, n in zip(CLASSES_ACC, SUPPORT)], fontsize=9.5)
    ax.set_xlabel("Classe prevista pelo modelo", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Classe clínica real", fontsize=10.5, labelpad=6)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    if title:
        ax.set_title("Diagonal = acertos; % = recall da classe", fontsize=9.5,
                     color=INK2, pad=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


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


def fig_f1_classes(per_class, ci, macro, out, figsize=(7.8, 5.0), dpi=300):
    keys = ["Normal", "Laringite", "Disfonia Psicogenica", "Disfonia Funcional",
            "Edema de Reinke"]
    ci_keys = ["f1_normal", "f1_laringite", "f1_disfonia_psicogenica",
               "f1_disfonia_funcional", "f1_reinke"]
    vals = [per_class[k]["f1"] for k in keys]
    los = [ci[k][1] for k in ci_keys]
    his = [ci[k][2] for k in ci_keys]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    xs = np.arange(5)
    # serie unica sobre categorias nominais -> um unico hue (slot 1)
    rounded_bars(ax, xs, vals, 0.40, BLUE)
    ax.errorbar(xs, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                fmt="none", ecolor=INK2, elinewidth=1.4, capsize=5, capthick=1.4, zorder=5)
    for x, v, hi in zip(xs, vals, his):
        ax.text(x, hi + 0.035, br(v, 3), ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=INK)

    ax.axhline(macro, color=ORANGE, linewidth=2, linestyle=(0, (5, 3)), zorder=4)
    ax.text(2.5, macro + 0.018, f"Macro F1 = {br(macro)}", ha="center", va="bottom",
            fontsize=10, color=ORANGE, fontweight="bold")

    style_axes(ax, 1.0, 0.2, "F1-Score (out-of-fold)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(CLASSES, SUPPORT)], fontsize=9.5)
    ax.set_xlim(-0.55, 4.55)
    ax.set_title("Barras = F1 pontual · hastes = IC 95% bootstrap (N=1000)",
                 fontsize=9.5, color=INK2, pad=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
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


def fig_gaps_ab(smote, para, mcn_p_smote, mcn_p_para, out, figsize=(12.0, 3.76), dpi=300):
    """Gap 2 e Gap 1 como forest plot: um braco por linha, IC 95% e decisao.

    Layout em duas colunas de eixos (rotulos | grafico) para que nenhum texto
    dependa de posicao em coordenada de dados -- foi assim que a versao anterior
    desta figura colidia.
    """
    fig, (axl, ax) = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                  gridspec_kw=dict(width_ratios=[1, 1.55], wspace=0.02))

    groups = [
        dict(title="GAP 2 — Balanceamento de classes",
             arms=[("SMOTE padrão", smote["macro_f1"][0], BLUE, ""),
                   ("Borderline-SMOTE1", smote["macro_f1"][1], ORANGE, "ADOTADO")],
             p=mcn_p_smote, badge_color=GOOD,
             note="ganho concentrado nas duas disfonias funcionais"),
        dict(title="GAP 1 — Seleção paraconsistente (LPA2v)",
             arms=[("sem seleção", para["macro_f1"][0], BLUE, "MANTIDO"),
                   ("com seleção", para["macro_f1"][1], ORANGE, "REJEITADA")],
             p=mcn_p_para, badge_color=CRIT,
             note="0% de redução: nenhuma característica passou do limiar"),
    ]

    # posicoes verticais: 2 bracos por grupo, com folga entre grupos
    rows, y = [], 0.0
    for gi, g in enumerate(groups):
        head = y
        for lab, v, col, badge in g["arms"]:
            y -= 1.0
            rows.append((y, lab, v, col, badge, g["badge_color"]))
        g["head_y"], g["foot_y"] = head, y - 0.72
        y -= 1.95
    top, bottom = 0.55, y + 1.1

    for a in (axl, ax):
        a.set_ylim(bottom, top)
        for side in a.spines.values():
            side.set_visible(False)
        a.tick_params(length=0)
        a.set_yticks([])
    axl.set_xlim(0, 1)
    axl.set_xticks([])
    ax.patch.set_visible(False)   # deixa os titulos da coluna esquerda transbordarem

    for g in groups:
        axl.text(0.0, g["head_y"], g["title"], fontsize=10.5, fontweight="bold",
                 color=INK, va="center", clip_on=False)
        axl.text(0.0, g["foot_y"], f"McNemar {fmt_p(g['p'])} · não significativo",
                 fontsize=9, color=INK2, va="center", clip_on=False)
        axl.text(0.0, g["foot_y"] - 0.42, g["note"], fontsize=8.5, color=MUTED,
                 va="center", style="italic", clip_on=False)

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for (yy, lab, v, col, badge, badge_col) in rows:
        axl.text(0.055, yy, lab, fontsize=10, color=INK, va="center", clip_on=False)
        ax.errorbar(v[0], yy, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color=col,
                    markersize=10, elinewidth=1.6, capsize=5, capthick=1.6,
                    markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)
        # valor e decisao a direita do grafico (as hastes ja mostram o IC)
        ax.text(1.02, yy, br(v[0]), fontsize=10.5, fontweight="bold", color=INK,
                va="center", transform=trans, clip_on=False)
        if badge:
            ax.text(1.17, yy, badge, fontsize=9.5, fontweight="bold", color=badge_col,
                    va="center", transform=trans, clip_on=False)

    ax.set_xlim(0.39, 0.505)
    ticks = np.arange(0.40, 0.5001, 0.02)
    ax.set_xticks(ticks)
    ax.set_xticklabels([br(v, 2) for v in ticks], fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Macro F1 (média bootstrap, N=1000) com IC 95% — mesma seed, mesmos 5 folds",
                  fontsize=9.5)
    fig.subplots_adjust(left=0.005, right=0.735, top=0.97, bottom=0.13)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_mcnemar(mcn, out, figsize=(6.4, 3.5), dpi=300):
    """Painel de significancia: MLP vs os 3 baselines (chi2 e p de McNemar)."""
    order = [("MajorityClass", "Classe majoritária"), ("kNN", "k-NN (k=5)"),
             ("LogisticRegression", "Regressão logística")]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0, 9.4, "Teste de McNemar — MLP hierárquico vs. baselines", fontsize=11.5,
            fontweight="bold", color=INK, va="center")
    ax.text(0, 8.3, "mesmas predições out-of-fold, correção de Edwards", fontsize=9.5,
            color=MUTED, va="center", style="italic")
    y = 6.6
    for key, label in order:
        chi2, p = mcn[key]
        ax.add_patch(FancyBboxPatch((0, y - 0.95), 10, 1.75,
                                    boxstyle="round,pad=0,rounding_size=0.25",
                                    linewidth=0, facecolor="#f1f5f9"))
        ax.text(0.4, y, label, fontsize=10.5, color=INK, va="center", fontweight="bold")
        ax.text(5.5, y, f"χ² = {br(chi2, 2)}", fontsize=10, color=INK2, va="center")
        ax.text(7.5, y, fmt_p(p), fontsize=10, color=GOOD, va="center",
                fontweight="bold")
        y -= 2.15
    ax.text(0, 0.2, "MLP superior aos três baselines com significância estatística (p < 0,05)",
            fontsize=9.5, color=GOOD, va="center", fontweight="bold")
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
    ci = read_ci(os.path.join(R, "bootstrap_ci_borderline_C_baseline.csv"))
    mcn = read_mcnemar(os.path.join(R, "mcnemar_vs_baselines_borderline_C_baseline.csv"))
    class_keys = ["Normal", "Laringite", "Disfonia Psicogenica", "Disfonia Funcional",
                  "Edema de Reinke"]
    cm = read_confusion(os.path.join(R, "train_log_v32_gap2_smote_ab_console.txt"),
                        [per_class[k]["support"] for k in class_keys], acc)
    smote = read_ab(os.path.join(R, "smote_ab_comparison.csv"), "standard", "borderline")
    para = read_ab(os.path.join(R, "paraconsistent_ab_comparison.csv"),
                   "without_selection", "with_selection", "without", "with")
    arch = read_arch(os.path.join(R, "arch_compare_comparison.csv"))

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
        fig_confusion(cm, per_class, os.path.join(O, "poster_confusao.png")),
        fig_f1_classes(per_class, ci, macro, os.path.join(O, "poster_f1_classes.png")),
        fig_arch_sweep(arch, os.path.join(O, "poster_arquiteturas.png")),
        fig_gaps_ab(smote, para, 0.6606, 0.8220, os.path.join(O, "poster_gaps_ab.png")),
        fig_mcnemar(mcn, os.path.join(O, "poster_mcnemar.png")),
    ]
    print(f"acuracia {acc:.6f} · macro F1 {macro:.6f} · weighted F1 {weighted:.6f}")
    print(f"matriz de confusao conferida (soma {cm.sum()}, acuracia {acc_cm:.6f})")
    for m in made:
        print("  gerado:", m)


if __name__ == "__main__":
    main()
