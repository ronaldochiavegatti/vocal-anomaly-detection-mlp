/*
 * metrics.h - Metricas de avaliacao
 *
 * Matriz de confusao, precisao, recall, F1-score por classe,
 * e medias macro/weighted.
 */

#ifndef METRICS_H
#define METRICS_H

#include "config.h"
#include <stdio.h>

/* Resultado de avaliacao completa */
typedef struct {
    int confusion[NUM_CLASSES][NUM_CLASSES];  /* [real][predito] */
    float precision[NUM_CLASSES];
    float recall[NUM_CLASSES];
    float f1[NUM_CLASSES];
    float accuracy;
    float macro_f1;
    float weighted_f1;
    int total;
} MetricsResult;

/*
 * Calcula todas as metricas a partir de predicoes.
 * y_true: labels verdadeiros [n]
 * y_pred: labels preditos [n]
 * n: numero de amostras
 */
void metrics_compute(const int *y_true, const int *y_pred, int n,
                     MetricsResult *result);

/*
 * Imprime as metricas formatadas.
 */
void metrics_print(const MetricsResult *result, FILE *out);

/*
 * Exporta metricas para arquivo CSV.
 */
int metrics_export_csv(const MetricsResult *result, const char *path);

#endif /* METRICS_H */
