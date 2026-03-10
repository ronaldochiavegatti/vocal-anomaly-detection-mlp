/*
 * metrics.c - Metricas de avaliacao
 *
 * Calcula matriz de confusao 3x3, precisao, recall, F1-score
 * por classe, acuracia, F1 macro e F1 weighted.
 */

#include "metrics.h"
#include "utils.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ROC_N_THRESHOLDS 101

static const char *class_names[NUM_CLASSES] = {
    CLASS_NAME_NORMAL, CLASS_NAME_LARYNGITIS, CLASS_NAME_DYSPHONIA
};

void metrics_compute(const int *y_true, const int *y_pred, int n,
                     MetricsResult *result)
{
    memset(result, 0, sizeof(MetricsResult));
    result->total = n;

    /* Montar matriz de confusao */
    int correct = 0;
    for (int i = 0; i < n; i++) {
        int t = y_true[i];
        int p = y_pred[i];
        if (t >= 0 && t < NUM_CLASSES && p >= 0 && p < NUM_CLASSES) {
            result->confusion[t][p]++;
        }
        if (t == p) correct++;
    }
    result->accuracy = (n > 0) ? (float)correct / n : 0.0f;

    /* Precisao, recall, F1 por classe */
    float f1_sum = 0.0f;
    float f1_weighted_sum = 0.0f;
    int total_support = 0;

    for (int c = 0; c < NUM_CLASSES; c++) {
        int tp = result->confusion[c][c];

        /* Soma da coluna c (total predito como c) */
        int pred_sum = 0;
        for (int i = 0; i < NUM_CLASSES; i++) pred_sum += result->confusion[i][c];

        /* Soma da linha c (total real da classe c) */
        int true_sum = 0;
        for (int j = 0; j < NUM_CLASSES; j++) true_sum += result->confusion[c][j];

        result->precision[c] = (pred_sum > 0) ? (float)tp / pred_sum : 0.0f;
        result->recall[c] = (true_sum > 0) ? (float)tp / true_sum : 0.0f;

        float p = result->precision[c];
        float r = result->recall[c];
        result->f1[c] = (p + r > 0.0f) ? 2.0f * p * r / (p + r) : 0.0f;

        f1_sum += result->f1[c];
        f1_weighted_sum += result->f1[c] * true_sum;
        total_support += true_sum;
    }

    result->macro_f1 = f1_sum / NUM_CLASSES;
    result->weighted_f1 = (total_support > 0) ? f1_weighted_sum / total_support : 0.0f;
}

void metrics_print(const MetricsResult *result, FILE *out)
{
    fprintf(out, "\n=== Matriz de Confusao ===\n");
    fprintf(out, "%22s", "");
    for (int j = 0; j < NUM_CLASSES; j++) {
        fprintf(out, " %12s", class_names[j]);
    }
    fprintf(out, "\n");

    for (int i = 0; i < NUM_CLASSES; i++) {
        fprintf(out, "  %-20s", class_names[i]);
        for (int j = 0; j < NUM_CLASSES; j++) {
            fprintf(out, " %12d", result->confusion[i][j]);
        }
        fprintf(out, "\n");
    }

    fprintf(out, "\n=== Metricas por Classe ===\n");
    fprintf(out, "%-22s %10s %10s %10s\n", "Classe", "Precisao", "Recall", "F1-Score");
    for (int c = 0; c < NUM_CLASSES; c++) {
        fprintf(out, "%-22s %10.4f %10.4f %10.4f\n",
                class_names[c], result->precision[c], result->recall[c], result->f1[c]);
    }

    fprintf(out, "\n=== Metricas Globais ===\n");
    fprintf(out, "Acuracia:     %.4f\n", result->accuracy);
    fprintf(out, "Macro F1:     %.4f\n", result->macro_f1);
    fprintf(out, "Weighted F1:  %.4f\n", result->weighted_f1);
    fprintf(out, "Total amostras: %d\n", result->total);
}

void metrics_permutation_importance(const float *X_val, const int *y_true,
                                    int n_samples, int n_features,
                                    MLP *net,
                                    const int *selected, int n_selected,
                                    float *importance_acc_out,
                                    float *importance_f1_out)
{
    float *X = (float *)safe_malloc((size_t)n_samples * n_features * sizeof(float));
    int   *y_pred = (int *)safe_malloc((size_t)n_samples * sizeof(int));
    float *col    = (float *)safe_malloc((size_t)n_samples * sizeof(float));
    float  output[MLP_OUTPUT_SIZE];

    memcpy(X, X_val, (size_t)n_samples * n_features * sizeof(float));

    /* Baseline */
    for (int i = 0; i < n_samples; i++) {
        mlp_forward(net, &X[i * n_features], output, 0);
        int pred = 0;
        for (int c = 1; c < MLP_OUTPUT_SIZE; c++)
            if (output[c] > output[pred]) pred = c;
        y_pred[i] = pred;
    }
    MetricsResult base;
    metrics_compute(y_true, y_pred, n_samples, &base);

    /* Permutacao por feature */
    for (int j = 0; j < n_selected; j++) {
        /* Backup */
        for (int i = 0; i < n_samples; i++)
            col[i] = X[i * n_features + j];

        /* Fisher-Yates in-place shuffle dos valores da coluna j */
        for (int i = n_samples - 1; i > 0; i--) {
            int k = (int)(rng_uniform() * (i + 1));
            if (k > i) k = i;
            float tmp = X[i * n_features + j];
            X[i * n_features + j] = X[k * n_features + j];
            X[k * n_features + j] = tmp;
        }

        /* Predizer com feature j permutada */
        for (int i = 0; i < n_samples; i++) {
            mlp_forward(net, &X[i * n_features], output, 0);
            int pred = 0;
            for (int c = 1; c < MLP_OUTPUT_SIZE; c++)
                if (output[c] > output[pred]) pred = c;
            y_pred[i] = pred;
        }
        MetricsResult perm;
        metrics_compute(y_true, y_pred, n_samples, &perm);

        importance_acc_out[j] = base.accuracy  - perm.accuracy;
        importance_f1_out[j]  = base.macro_f1  - perm.macro_f1;

        /* Restaurar */
        for (int i = 0; i < n_samples; i++)
            X[i * n_features + j] = col[i];

        (void)selected; /* usado em main.c para mapeamento de grupos */
    }

    free(X); free(y_pred); free(col);
}

static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

void metrics_bootstrap_ci(const int *y_true, const int *y_pred, int n_samples,
                          int n_bootstrap, unsigned int seed,
                          ConfidenceInterval results[CI_N_METRICS])
{
    float *dist[CI_N_METRICS];
    for (int m = 0; m < CI_N_METRICS; m++)
        dist[m] = (float *)safe_malloc((size_t)n_bootstrap * sizeof(float));

    int *bt = (int *)safe_malloc((size_t)n_samples * sizeof(int));
    int *bp = (int *)safe_malloc((size_t)n_samples * sizeof(int));

    rng_seed(seed);

    for (int b = 0; b < n_bootstrap; b++) {
        /* Reamostrar com reposicao */
        for (int i = 0; i < n_samples; i++) {
            int idx = (int)(rng_uniform() * n_samples);
            if (idx >= n_samples) idx = n_samples - 1;
            bt[i] = y_true[idx];
            bp[i] = y_pred[idx];
        }

        MetricsResult mr;
        metrics_compute(bt, bp, n_samples, &mr);

        dist[CI_ACCURACY][b]     = mr.accuracy;
        dist[CI_MACRO_F1][b]     = mr.macro_f1;
        dist[CI_F1_NORMAL][b]    = mr.f1[0];
        dist[CI_F1_LARYNGITE][b] = mr.f1[1];
        dist[CI_F1_DISFONIA][b]  = mr.f1[2];
    }

    free(bt);
    free(bp);

    int lo = (int)(0.025f * n_bootstrap);
    int hi = (int)(0.975f * n_bootstrap);
    if (hi >= n_bootstrap) hi = n_bootstrap - 1;

    for (int m = 0; m < CI_N_METRICS; m++) {
        qsort(dist[m], (size_t)n_bootstrap, sizeof(float), cmp_float);

        float sum = 0.0f;
        for (int b = 0; b < n_bootstrap; b++) sum += dist[m][b];

        results[m].mean  = sum / n_bootstrap;
        results[m].lower = dist[m][lo];
        results[m].upper = dist[m][hi];

        free(dist[m]);
    }
}

void metrics_roc_auc(const int *y_true, const float *y_prob, int n_samples,
                     int n_classes, float *auc_out, const char *roc_csv_path)
{
    FILE *f = fopen(roc_csv_path, "w");
    if (f) fprintf(f, "class,threshold,tpr,fpr\n");

    for (int c = 0; c < n_classes; c++) {
        int n_pos = 0, n_neg = 0;
        for (int i = 0; i < n_samples; i++) {
            if (y_true[i] == c) n_pos++;
            else n_neg++;
        }

        float prev_tpr = 0.0f, prev_fpr = 0.0f;
        float auc = 0.0f;

        for (int t = 0; t < ROC_N_THRESHOLDS; t++) {
            float thresh = 1.0f - (float)t / (ROC_N_THRESHOLDS - 1);
            int tp = 0, fp = 0;
            for (int i = 0; i < n_samples; i++) {
                float prob_c = y_prob[i * n_classes + c];
                if (prob_c >= thresh) {
                    if (y_true[i] == c) tp++;
                    else fp++;
                }
            }
            float tpr = (n_pos > 0) ? (float)tp / n_pos : 0.0f;
            float fpr = (n_neg > 0) ? (float)fp / n_neg : 0.0f;

            if (f) fprintf(f, "%d,%.4f,%.6f,%.6f\n", c, thresh, tpr, fpr);

            if (t > 0)
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5f;
            prev_tpr = tpr;
            prev_fpr = fpr;
        }
        auc_out[c] = fabsf(auc);
    }

    if (f) fclose(f);
}

void metrics_pr_curve(const int *y_true, const float *y_prob, int n_samples,
                      int n_classes, const char *pr_csv_path)
{
    FILE *f = fopen(pr_csv_path, "w");
    if (f) fprintf(f, "class,threshold,precision,recall\n");

    for (int c = 0; c < n_classes; c++) {
        int n_pos = 0;
        for (int i = 0; i < n_samples; i++) {
            if (y_true[i] == c) n_pos++;
        }

        for (int t = 0; t < ROC_N_THRESHOLDS; t++) {
            float thresh = 1.0f - (float)t / (ROC_N_THRESHOLDS - 1);
            int tp = 0, fp = 0;
            for (int i = 0; i < n_samples; i++) {
                float prob_c = y_prob[i * n_classes + c];
                if (prob_c >= thresh) {
                    if (y_true[i] == c) tp++;
                    else fp++;
                }
            }
            float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 1.0f;
            float recall    = (n_pos > 0)   ? (float)tp / n_pos      : 0.0f;

            if (f) fprintf(f, "%d,%.4f,%.6f,%.6f\n", c, thresh, precision, recall);
        }
    }

    if (f) fclose(f);
}

int metrics_export_csv(const MetricsResult *result, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "class,precision,recall,f1,support\n");
    for (int c = 0; c < NUM_CLASSES; c++) {
        int support = 0;
        for (int j = 0; j < NUM_CLASSES; j++) support += result->confusion[c][j];
        fprintf(f, "%s,%.6f,%.6f,%.6f,%d\n",
                class_names[c], result->precision[c], result->recall[c],
                result->f1[c], support);
    }
    fprintf(f, "macro_avg,,,%.6f,%d\n", result->macro_f1, result->total);
    fprintf(f, "weighted_avg,,,%.6f,%d\n", result->weighted_f1, result->total);
    fprintf(f, "accuracy,,,%.6f,%d\n", result->accuracy, result->total);

    fclose(f);
    return 0;
}
