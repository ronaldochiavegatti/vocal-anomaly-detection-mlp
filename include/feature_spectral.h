/*
 * feature_spectral.h - Extracao de features espectrais
 *
 * Frequencia fundamental (f0), formantes (F1-F4) via LPC,
 * e entropia espectral.
 */

#ifndef FEATURE_SPECTRAL_H
#define FEATURE_SPECTRAL_H

/* Estrutura com as 22 features espectrais */
typedef struct {
    float f0_mean;            /* frequencia fundamental media (Hz) */
    float f0_std;             /* desvio-padrao de f0 (Hz) */
    float formants[4];        /* F1, F2, F3, F4 (Hz) */
    float spectral_entropy;   /* entropia espectral normalizada */
    float spectral_centroid;  /* centroide espectral (Hz) */
    float spectral_rolloff;   /* rolloff espectral 85% (Hz) */
    float mfcc[13];           /* coeficientes mel-cepstrais */
} SpectralFeatures;

/*
 * Extrai features espectrais de um sinal de audio.
 *
 * signal: amostras (ja com pre-enfase aplicada)
 * n: numero de amostras
 * sample_rate: taxa de amostragem
 * out: ponteiro para struct de saida
 *
 * Retorna 0 em sucesso, -1 em erro.
 */
int spectral_extract(const float *signal, int n, int sample_rate,
                     SpectralFeatures *out);

#endif /* FEATURE_SPECTRAL_H */
