/*
 * feature_wavelet.c - Extracao de features via DWT Daubechies-4
 *
 * Implementa a Discrete Wavelet Transform com filtros db4.
 * Decomposicao em 6 niveis: cada nivel aplica convolucao com
 * filtros passa-baixa (aproximacao) e passa-alta (detalhe),
 * seguida de downsampling por fator 2.
 *
 * Coeficientes db4 (4 taps):
 *   h0 = (1 + sqrt(3)) / (4*sqrt(2))  ≈  0.4829629
 *   h1 = (3 + sqrt(3)) / (4*sqrt(2))  ≈  0.8365163
 *   h2 = (3 - sqrt(3)) / (4*sqrt(2))  ≈  0.2241439
 *   h3 = (1 - sqrt(3)) / (4*sqrt(2))  ≈ -0.1294095
 *
 * Filtro passa-alta (detalhe): g[k] = (-1)^k * h[3-k]
 */

#include "feature_wavelet.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Coeficientes do filtro passa-baixa Daubechies-4 */
#define DB4_LEN 4
static const float db4_lo[DB4_LEN] = {
     0.4829629131445341f,
     0.8365163037378079f,
     0.2241438680420134f,
    -0.1294095225512604f
};

/* Coeficientes do filtro passa-alta Daubechies-4 */
static const float db4_hi[DB4_LEN] = {
    -0.1294095225512604f,
    -0.2241438680420134f,
     0.8365163037378079f,
    -0.4829629131445341f
};

/*
 * Aplica um nivel de decomposicao DWT.
 * input: sinal de entrada (n amostras)
 * approx: saida dos coeficientes de aproximacao (n/2 amostras)
 * detail: saida dos coeficientes de detalhe (n/2 amostras)
 * n: tamanho da entrada (deve ser par)
 */
static void dwt_step(const float *input, int n, float *approx, float *detail)
{
    int half = n / 2;

    for (int i = 0; i < half; i++) {
        float lo = 0.0f;
        float hi = 0.0f;

        for (int k = 0; k < DB4_LEN; k++) {
            /* Indice com borda periodica (extensao circular) */
            int idx = (2 * i + k) % n;
            lo += db4_lo[k] * input[idx];
            hi += db4_hi[k] * input[idx];
        }

        approx[i] = lo;
        detail[i] = hi;
    }
}

/*
 * Calcula estatisticas de um vetor de coeficientes.
 */
static void compute_stats(const float *coeff, int n,
                          float *mean, float *variance, float *energy)
{
    if (n <= 0) {
        *mean = *variance = *energy = 0.0f;
        return;
    }

    float sum = 0.0f;
    float sq_sum = 0.0f;

    for (int i = 0; i < n; i++) {
        sum += coeff[i];
        sq_sum += coeff[i] * coeff[i];
    }

    *mean = sum / n;
    *energy = sq_sum / n;
    *variance = *energy - (*mean) * (*mean);
    if (*variance < 0.0f) *variance = 0.0f;  /* protecao numerica */
}

int wavelet_extract(const float *signal, int n, WaveletFeatures *out)
{
    memset(out, 0, sizeof(WaveletFeatures));

    /* Garantir que n eh par (truncar se necessario) */
    if (n % 2 != 0) n--;
    if (n < (1 << WAVELET_LEVELS)) {
        /* Sinal muito curto para 6 niveis de decomposicao */
        return -1;
    }

    /* Buffer para a aproximacao atual */
    int current_len = n;
    float *current = (float *)safe_malloc(current_len * sizeof(float));
    memcpy(current, signal, current_len * sizeof(float));

    for (int level = 0; level < WAVELET_LEVELS; level++) {
        if (current_len < DB4_LEN || current_len < 2) break;

        /* Garantir tamanho par */
        if (current_len % 2 != 0) current_len--;

        int half = current_len / 2;
        float *approx = (float *)safe_malloc(half * sizeof(float));
        float *detail = (float *)safe_malloc(half * sizeof(float));

        dwt_step(current, current_len, approx, detail);

        /* Extrair estatisticas dos coeficientes de detalhe */
        compute_stats(detail, half,
                      &out->mean[level],
                      &out->variance[level],
                      &out->energy[level]);

        free(detail);
        free(current);

        /* Proximo nivel: trabalhar com a aproximacao */
        current = approx;
        current_len = half;
    }

    free(current);
    return 0;
}
