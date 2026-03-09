/*
 * metrics.c - Metricas de avaliacao
 *
 * Calcula matriz de confusao 3x3, precisao, recall, F1-score
 * por classe, acuracia, F1 macro e F1 weighted.
 */

#include "metrics.h"
#include <string.h>
#include <stdio.h>

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
