/*
 * feature_spectral.h - Extracao de features espectrais
 *
 * Frequencia fundamental (f0), formantes (F1-F4) via LPC,
 * e entropia espectral.
 */

#ifndef FEATURE_SPECTRAL_H
#define FEATURE_SPECTRAL_H

/* Estrutura com as 51 features espectrais (US-011: +Delta/Delta-Delta MFCCs; +CPP) */
typedef struct {
    float f0_mean;            /* frequencia fundamental media (Hz) */
    float f0_std;             /* desvio-padrao de f0 (Hz) */
    float formants[4];        /* F1, F2, F3, F4 (Hz) */
    float spectral_entropy;   /* entropia espectral normalizada */
    float spectral_centroid;  /* centroide espectral (Hz) */
    float spectral_rolloff;   /* rolloff espectral 85% (Hz) */
    float mfcc[13];           /* coeficientes mel-cepstrais estaticos */
    float delta_mfcc[13];     /* delta MFCCs (std dev da derivada temporal 1a ordem) */
    float delta2_mfcc[13];    /* delta-delta MFCCs (std dev da derivada temporal 2a ordem) */
    float cpp_mean;           /* CPP medio (Cepstral Peak Prominence) - Boersma/Hillenbrand */
    float cpp_std;            /* desvio-padrao do CPP (variabilidade da regularidade glotal) */
    float cpp_slope;          /* inclinacao temporal do CPP (tendencia de regularizacao) */
    
    /* Features da Fonte Glotica (Glottal Source) - US-028 */
    float glottal_oq;         /* Open Quotient (tempo de glote aberta / periodo) */
    float glottal_sq;         /* Speed Quotient (tempo subida / tempo descida) */
    float glottal_naq;        /* Normalized Amplitude Quotient (pico fluxo / periodo) */
    float glottal_h1h2;       /* Diferenca de amplitude entre 1o e 2o harmonicos */
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
