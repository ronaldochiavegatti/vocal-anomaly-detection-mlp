/*
 * metrics.h - Metricas de avaliacao
 *
 * Matriz de confusao, precisao, recall, F1-score por classe,
 * e medias macro/weighted.
 */

#ifndef METRICS_H
#define METRICS_H

#include "config.h"
#include "mlp.h"
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
    float auc[NUM_CLASSES];   /* AUC one-vs-rest por classe (preenchido por metrics_roc_auc) */
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

/* Intervalo de confianca (media + IC 95% via bootstrap) */
typedef struct {
    float mean;
    float lower;  /* percentil 2.5 */
    float upper;  /* percentil 97.5 */
} ConfidenceInterval;

/* Indices dos metricas no array de CI */
#define CI_ACCURACY          0
#define CI_MACRO_F1          1
#define CI_F1_NORMAL         2
#define CI_F1_LARYNGITE      3
#define CI_F1_DISFONIA       4
#define CI_F1_FUNC_DISFONIA  5
#define CI_F1_REINKE         6
#define CI_N_METRICS         7

/*
 * Calcula intervalos de confianca 95% via bootstrap sobre predicoes acumuladas.
 * y_true/y_pred: labels verdadeiros e preditos [n_samples]
 * n_bootstrap: numero de reamostras (recomendado: 1000)
 * seed: semente para reproducibilidade
 * results: array de CI_N_METRICS ConfidenceInterval (saida)
 */
void metrics_bootstrap_ci(const int *y_true, const int *y_pred, int n_samples,
                          int n_bootstrap, unsigned int seed,
                          ConfidenceInterval results[CI_N_METRICS]);

/*
 * Calcula importancia de features por permutacao (Breiman, 2001).
 * X_val: matriz de validacao normalizada e selecionada [n_samples × n_features]
 * selected: indices originais das features selecionadas [n_selected]
 * importance_acc_out/importance_f1_out: queda de acuracia/macro-F1 ao permutar [n_selected]
 */
void metrics_permutation_importance(const float *X_val, const int *y_true,
                                    int n_samples, int n_features,
                                    MLP *net,
                                    const int *selected, int n_selected,
                                    float *importance_acc_out,
                                    float *importance_f1_out);

/*
 * Calcula curvas ROC one-vs-rest e AUC por classe via regra do trapezio.
 * y_prob: probabilidades softmax [n_samples * n_classes] (linha-maior)
 * auc_out: array de saida com AUC por classe [n_classes]
 * roc_csv_path: arquivo CSV de saida (class,threshold,tpr,fpr)
 */
void metrics_roc_auc(const int *y_true, const float *y_prob, int n_samples,
                     int n_classes, float *auc_out, const char *roc_csv_path);

/*
 * Calcula curvas Precision-Recall por classe.
 * pr_csv_path: arquivo CSV de saida (class,threshold,precision,recall)
 */
void metrics_pr_curve(const int *y_true, const float *y_prob, int n_samples,
                      int n_classes, const char *pr_csv_path);

/*
 * Teste de McNemar para comparar dois classificadores (Edwards, 1948).
 * Calcula chi-quadrado com correcao de continuidade sobre a tabela 2x2:
 *   b = A certo, B errado   c = A errado, B certo
 * chi2 = (|b - c| - 1)^2 / (b + c)
 * p_value = erfc(sqrt(chi2 / 2))  [chi2 com 1 grau de liberdade]
 *
 * Retorna 0 se b+c == 0 (classificadores identicos).
 * Saida: chi2_out e p_value_out.
 */
void metrics_mcnemar(const int *y_true,
                     const int *y_pred_a, const int *y_pred_b,
                     int n_samples,
                     float *chi2_out, float *p_value_out);

#endif /* METRICS_H */
