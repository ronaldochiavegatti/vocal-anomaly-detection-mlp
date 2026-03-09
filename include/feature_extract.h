/*
 * feature_extract.h - Orquestrador de extracao de features
 *
 * Coordena a leitura de WAVs e extracao de features temporais,
 * espectrais e wavelet para cada paciente, gerando o vetor
 * final de 87 features.
 */

#ifndef FEATURE_EXTRACT_H
#define FEATURE_EXTRACT_H

#include "dataset.h"
#include "config.h"

/* Conjunto de features extraidas para todos os pacientes */
typedef struct {
    float *features;   /* matriz [count x TOTAL_FEATURES] em row-major */
    int *labels;       /* vetor de labels [count] */
    int count;         /* numero de pacientes */
    int num_features;  /* TOTAL_FEATURES (87) */
} FeatureMatrix;

/*
 * Extrai features de todos os pacientes do dataset.
 * Para cada paciente, le as 3 vogais (a_n, i_n, u_n),
 * aplica pre-enfase e extrai 29 features de cada.
 *
 * ds: dataset carregado
 * fm: ponteiro para struct de saida
 *
 * Retorna 0 em sucesso, -1 em erro.
 */
int features_extract_all(const Dataset *ds, FeatureMatrix *fm);

/*
 * Exporta a matriz de features para um arquivo CSV.
 * Formato: feature_0, feature_1, ..., feature_86, label
 */
int features_export_csv(const FeatureMatrix *fm, const char *path);

/*
 * Libera a memoria da matriz de features.
 */
void features_free(FeatureMatrix *fm);

#endif /* FEATURE_EXTRACT_H */
