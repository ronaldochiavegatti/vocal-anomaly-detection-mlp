/*
 * feature_temporal.h - Extracao de features temporais
 *
 * Jitter (local, RAP), Shimmer (local) e energia media.
 * Baseado em deteccao de periodos via picos do sinal.
 */

#ifndef FEATURE_TEMPORAL_H
#define FEATURE_TEMPORAL_H

/* Estrutura com as 10 features temporais */
typedef struct {
    float jitter_local;   /* perturbacao de frequencia local (%) */
    float jitter_rap;     /* Relative Average Perturbation (%) */
    float jitter_ppq5;    /* 5-point Period Perturbation Quotient (%) */
    float shimmer_local;  /* perturbacao de amplitude local (%) */
    float shimmer_apq3;   /* 3-point Amplitude Perturbation Quotient (%) */
    float shimmer_apq5;   /* 5-point Amplitude Perturbation Quotient (%) */
    float shimmer_apq11;  /* 11-point Amplitude Perturbation Quotient (%) */
    float energy_mean;    /* energia media do sinal */
    float hnr;            /* Harmonics-to-Noise Ratio (dB) */
    float zcr;            /* Zero Crossing Rate */
} TemporalFeatures;

/*
 * Extrai features temporais de um sinal de audio.
 *
 * signal: amostras normalizadas [-1, 1]
 * n: numero de amostras
 * sample_rate: taxa de amostragem (ex: 44100)
 * out: ponteiro para struct de saida
 *
 * Retorna 0 em sucesso, -1 se nao foi possivel detectar periodos.
 */
int temporal_extract(const float *signal, int n, int sample_rate,
                     TemporalFeatures *out);

#endif /* FEATURE_TEMPORAL_H */
