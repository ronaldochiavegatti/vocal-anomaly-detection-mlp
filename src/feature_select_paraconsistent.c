/*
 * feature_select_paraconsistent.c - Selecao Paraconsistente de Caracteristicas (Gap 1)
 *
 * Computa (mu, lambda, Gc, Gct) por feature via Logica Paraconsistente Anotada
 * de dois valores (LPA2v/PAL2v):
 *   - mu (grau de evidencia favoravel): eta-quadrado de ANOVA de uma via
 *     (SSB/SST), naturalmente limitado em [0,1] -- substitui deliberadamente
 *     o Fisher-ratio nao-padrao do SPEC.md original (PARA-01).
 *   - lambda (grau de evidencia desfavoravel): media NAO-ponderada da razao
 *     sigma_por_classe/sigma_global -- substitui deliberadamente o CV=std/mean
 *     do SPEC.md original, que explode para features com media proxima de
 *     zero (delta-MFCCs) (PARA-01).
 *   - Gc = mu - lambda (grau de certeza), Gct = mu + lambda - 1 (grau de
 *     contradicao). Selecao: Gc >= gc_thresh E |Gct| <= gct_max, com
 *     relaxamento de gc_thresh limitado a PARA_MAX_RELAX_ITERS iteracoes e
 *     fallback garantido para todas as features (nunca retorna 0) (PARA-02).
 */

#include "feature_select_paraconsistent.h"
#include "config.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Piso de variancia (SST), analogo (ao quadrado) ao MIN_STD de normalize.c */
#define MIN_VAR 1e-16f
/* Piso de desvio-padrao global, mesmo valor/nome de normalize.c, redefinido
 * localmente por convencao deste projeto (nao importado entre arquivos) */
#define MIN_STD 1e-8f

/*
 * Eta-quadrado (ANOVA de uma via) da feature feat_idx: SSB/SST.
 * Retorna 0.0f (sem evidencia de separabilidade) se SST < MIN_VAR (feature
 * constante), evitando divisao por quase-zero.
 */
static float feature_eta_squared(const float *x, const int *y, int n, int nf,
                                  int feat_idx, int n_classes)
{
    float global_mean = 0.0f;
    for (int i = 0; i < n; i++) global_mean += x[i * nf + feat_idx];
    global_mean /= (float)n;

    float *class_sum = (float *)safe_calloc(n_classes, sizeof(float));
    int   *class_n   = (int *)safe_calloc(n_classes, sizeof(int));
    for (int i = 0; i < n; i++) {
        class_sum[y[i]] += x[i * nf + feat_idx];
        class_n[y[i]]++;
    }

    float ssb = 0.0f;
    for (int c = 0; c < n_classes; c++) {
        if (class_n[c] == 0) continue;
        float class_mean = class_sum[c] / (float)class_n[c];
        float diff = class_mean - global_mean;
        ssb += (float)class_n[c] * diff * diff;
    }

    float ssw = 0.0f;
    for (int i = 0; i < n; i++) {
        int c = y[i];
        float class_mean = class_sum[c] / (float)class_n[c];
        float diff = x[i * nf + feat_idx] - class_mean;
        ssw += diff * diff;
    }

    free(class_sum);
    free(class_n);

    float sst = ssb + ssw;
    if (sst < MIN_VAR) return 0.0f; /* feature constante: sem evidencia de separabilidade */
    return ssb / sst;               /* eta-quadrado, ja em [0,1] */
}

/*
 * Lambda da feature feat_idx: media NAO-ponderada, entre classes, da razao
 * (desvio-padrao da classe / desvio-padrao global), limitada em [0,1] por
 * classe. Deliberadamente nao-ponderada por contagem de amostras (ao
 * contrario de SSB/SSW acima) -- uma versao ponderada colapsaria lambda para
 * uma funcao fixa de mu (lambda=sqrt(1-mu)), destruindo a premissa de duas
 * fontes de evidencia independentes que sustenta o par Gc/Gct.
 */
static float feature_lambda(const float *x, const int *y, int n, int nf,
                             int feat_idx, int n_classes, float global_std)
{
    if (global_std < MIN_STD) global_std = MIN_STD;

    float *class_sum = (float *)safe_calloc(n_classes, sizeof(float));
    float *class_sq  = (float *)safe_calloc(n_classes, sizeof(float));
    int   *class_n   = (int *)safe_calloc(n_classes, sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = y[i];
        float v = x[i * nf + feat_idx];
        class_sum[c] += v;
        class_sq[c] += v * v;
        class_n[c]++;
    }

    float lambda_sum = 0.0f;
    int active_classes = 0;
    for (int c = 0; c < n_classes; c++) {
        if (class_n[c] < 2) continue; /* precisa de >=2 amostras para estimar std */
        float mean_c = class_sum[c] / (float)class_n[c];
        float var_c = class_sq[c] / (float)class_n[c] - mean_c * mean_c;
        if (var_c < 0.0f) var_c = 0.0f; /* guarda numerica */
        float std_c = sqrtf(var_c);
        float ratio = std_c / global_std;
        if (ratio > 1.0f) ratio = 1.0f; /* clip para [0,1] */
        lambda_sum += ratio;
        active_classes++;
    }

    free(class_sum);
    free(class_sq);
    free(class_n);

    return (active_classes > 0) ? (lambda_sum / (float)active_classes) : 0.0f;
}

int paraconsistent_select(const float *x, const int *y, int n, int nf, int n_classes,
                           float gc_thresh, float gct_max, int *selected,
                           float *mu_out, float *lambda_out,
                           float *gc_out, float *gct_out)
{
    float *mu = (float *)safe_malloc(nf * sizeof(float));
    float *lambda = (float *)safe_malloc(nf * sizeof(float));
    float *gc = (float *)safe_malloc(nf * sizeof(float));
    float *gct = (float *)safe_malloc(nf * sizeof(float));

    for (int j = 0; j < nf; j++) {
        float global_mean = 0.0f, global_sq = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = x[i * nf + j];
            global_mean += v;
            global_sq += v * v;
        }
        global_mean /= (float)n;
        float global_var = global_sq / (float)n - global_mean * global_mean;
        if (global_var < 0.0f) global_var = 0.0f;
        float global_std = sqrtf(global_var);

        mu[j] = feature_eta_squared(x, y, n, nf, j, n_classes);
        lambda[j] = feature_lambda(x, y, n, nf, j, n_classes, global_std);
        gc[j] = mu[j] - lambda[j];
        gct[j] = mu[j] + lambda[j] - 1.0f;
    }

    int n_selected = 0;
    float cur_gc_thresh = gc_thresh;
    for (int iter = 0; iter <= PARA_MAX_RELAX_ITERS; iter++) {
        n_selected = 0;
        for (int j = 0; j < nf; j++) {
            if (gc[j] >= cur_gc_thresh && fabsf(gct[j]) <= gct_max) selected[n_selected++] = j;
        }
        if (n_selected > 0) break;
        if (iter < PARA_MAX_RELAX_ITERS) {
            log_warn("paraconsistent_select: 0 features selecionadas com gc_thresh=%.3f -- "
                     "relaxando para %.3f (iter %d/%d)",
                     cur_gc_thresh, cur_gc_thresh - PARA_GC_RELAX_STEP, iter + 1, PARA_MAX_RELAX_ITERS);
            cur_gc_thresh -= PARA_GC_RELAX_STEP;
        }
    }
    if (n_selected == 0) {
        log_warn("paraconsistent_select: relaxamento esgotado (%d iteracoes) sem selecionar "
                 "nenhuma feature -- fallback para todas as %d features (mesmo padrao de "
                 "fallback do SMOTE-03)",
                 PARA_MAX_RELAX_ITERS, nf);
        for (int j = 0; j < nf; j++) selected[j] = j;
        n_selected = nf;
    }

    if (mu_out) memcpy(mu_out, mu, nf * sizeof(float));
    if (lambda_out) memcpy(lambda_out, lambda, nf * sizeof(float));
    if (gc_out) memcpy(gc_out, gc, nf * sizeof(float));
    if (gct_out) memcpy(gct_out, gct, nf * sizeof(float));

    free(mu);
    free(lambda);
    free(gc);
    free(gct);

    return n_selected;
}
