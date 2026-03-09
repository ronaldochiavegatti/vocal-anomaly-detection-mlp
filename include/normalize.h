/*
 * normalize.h - Normalizacao Z-score
 *
 * Calcula media e desvio-padrao por feature no conjunto de treino
 * e aplica a transformacao (x - media) / std.
 */

#ifndef NORMALIZE_H
#define NORMALIZE_H

/* Parametros de normalizacao (media e std por feature) */
typedef struct {
    float *mean;
    float *std;
    int num_features;
} NormParams;

/*
 * Calcula parametros de normalizacao a partir dos dados de treino.
 * features: matriz [n x num_features] em row-major
 * n: numero de amostras
 * num_features: numero de features
 */
void norm_fit(const float *features, int n, int num_features, NormParams *params);

/*
 * Aplica normalizacao Z-score in-place.
 * features: matriz [n x num_features] em row-major
 */
void norm_transform(float *features, int n, const NormParams *params);

/*
 * Salva parametros de normalizacao em arquivo binario.
 */
int norm_save(const NormParams *params, const char *path);

/*
 * Carrega parametros de normalizacao de arquivo binario.
 */
int norm_load(NormParams *params, const char *path);

/*
 * Libera memoria dos parametros.
 */
void norm_free(NormParams *params);

#endif /* NORMALIZE_H */
