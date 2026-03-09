/*
 * feature_wavelet.h - Extracao de features via Wavelet (DWT Daubechies-4)
 *
 * Decomposicao em 6 niveis e extracao de estatisticas
 * (media, variancia, energia) dos coeficientes de detalhe.
 */

#ifndef FEATURE_WAVELET_H
#define FEATURE_WAVELET_H

#include "config.h"

/* Estrutura com as 18 features wavelet (6 niveis x 3 estatisticas) */
typedef struct {
    float mean[WAVELET_LEVELS];      /* media dos coeficientes de detalhe */
    float variance[WAVELET_LEVELS];  /* variancia dos coeficientes de detalhe */
    float energy[WAVELET_LEVELS];    /* energia dos coeficientes de detalhe */
} WaveletFeatures;

/*
 * Extrai features wavelet de um sinal de audio.
 *
 * signal: amostras do sinal
 * n: numero de amostras
 * out: ponteiro para struct de saida
 *
 * Retorna 0 em sucesso, -1 em erro.
 */
int wavelet_extract(const float *signal, int n, WaveletFeatures *out);

#endif /* FEATURE_WAVELET_H */
