/*
 * dsp_utils.c - Primitivas de Processamento Digital de Sinais
 *
 * Implementacao de pre-enfase, janela Hamming, autocorrelacao,
 * FFT radix-2 (Cooley-Tukey) e funcoes auxiliares.
 */

#include "dsp_utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========== Pre-enfase ========== */

void dsp_pre_emphasis(float *signal, int n, float alpha)
{
    /* Aplicar de tras para frente para nao sobrescrever valores necessarios */
    for (int i = n - 1; i > 0; i--) {
        signal[i] = signal[i] - alpha * signal[i - 1];
    }
    /* signal[0] fica inalterado (nao ha amostra anterior) */
}

/* ========== Janela Hamming ========== */

void dsp_hamming_window(float *frame, int n)
{
    for (int i = 0; i < n; i++) {
        float w = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (n - 1));
        frame[i] *= w;
    }
}

/* ========== Autocorrelacao ========== */

void dsp_autocorrelation(const float *signal, int n, float *out, int max_lag)
{
    for (int k = 0; k < max_lag; k++) {
        float sum = 0.0f;
        for (int i = 0; i < n - k; i++) {
            sum += signal[i] * signal[i + k];
        }
        out[k] = sum;
    }
}

/* ========== FFT Radix-2 ========== */

/* Bit-reversal permutation */
static void bit_reverse(float *real, float *imag, int n)
{
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float tmp_r = real[i];
            float tmp_i = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = tmp_r;
            imag[j] = tmp_i;
        }
        int m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

void dsp_fft(float *real, float *imag, int n)
{
    bit_reverse(real, imag, n);

    /* Butterfly */
    for (int step = 2; step <= n; step <<= 1) {
        int half = step >> 1;
        float angle = -2.0f * (float)M_PI / step;

        float w_real = cosf(angle);
        float w_imag = sinf(angle);

        for (int k = 0; k < n; k += step) {
            float wr = 1.0f;
            float wi = 0.0f;

            for (int j = 0; j < half; j++) {
                int even = k + j;
                int odd = k + j + half;

                float tr = wr * real[odd] - wi * imag[odd];
                float ti = wr * imag[odd] + wi * real[odd];

                real[odd] = real[even] - tr;
                imag[odd] = imag[even] - ti;
                real[even] = real[even] + tr;
                imag[even] = imag[even] + ti;

                /* Rotacao do twiddle factor */
                float wr_new = wr * w_real - wi * w_imag;
                wi = wr * w_imag + wi * w_real;
                wr = wr_new;
            }
        }
    }
}

/* ========== Espectros ========== */

void dsp_magnitude_spectrum(const float *real, const float *imag, int n, float *mag)
{
    int half = n / 2 + 1;
    for (int i = 0; i < half; i++) {
        mag[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
    }
}

void dsp_power_spectrum(const float *real, const float *imag, int n, float *power)
{
    int half = n / 2 + 1;
    float inv_n = 1.0f / n;
    for (int i = 0; i < half; i++) {
        power[i] = (real[i] * real[i] + imag[i] * imag[i]) * inv_n;
    }
}

/* ========== Utilitarios ========== */

int dsp_next_power_of_2(int n)
{
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

float dsp_energy(const float *signal, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += signal[i] * signal[i];
    }
    return sum / n;
}

/* ========== US-010: Wavelet Denoising ========== */

/* One-level in-place Haar forward DWT on work[0..len-1].
 * Approximation → work[0..len/2-1], Detail → work[len/2..len-1] */
static void haar_forward(float *work, int len, float *scratch)
{
    int half = len / 2;
    static const float SQRT2_INV = 0.70710678118f;
    for (int i = 0; i < half; i++) {
        scratch[i]        = (work[2*i] + work[2*i+1]) * SQRT2_INV;
        scratch[half + i] = (work[2*i] - work[2*i+1]) * SQRT2_INV;
    }
    memcpy(work, scratch, len * sizeof(float));
}

/* One-level in-place Haar inverse DWT on work[0..len-1]. */
static void haar_inverse(float *work, int len, float *scratch)
{
    int half = len / 2;
    static const float SQRT2_INV = 0.70710678118f;
    for (int i = 0; i < half; i++) {
        scratch[2*i]     = (work[i] + work[half + i]) * SQRT2_INV;
        scratch[2*i + 1] = (work[i] - work[half + i]) * SQRT2_INV;
    }
    memcpy(work, scratch, len * sizeof(float));
}

static float soft_threshold(float x, float t)
{
    if (x > t)  return x - t;
    if (x < -t) return x + t;
    return 0.0f;
}

void dsp_wavelet_denoise(float *signal, int n, int levels)
{
    if (n < 4 || levels < 1) return;

    /* Largest power of 2 <= n */
    int n_work = 1;
    while (n_work * 2 <= n) n_work *= 2;
    if (n_work < 4) return;

    /* Limit levels */
    int max_lev = 0;
    for (int tmp = n_work; tmp >= 2; tmp >>= 1) max_lev++;
    if (levels > max_lev) levels = max_lev;

    float *work    = (float *)malloc(n_work * sizeof(float));
    float *scratch = (float *)malloc(n_work * sizeof(float));
    if (!work || !scratch) { free(work); free(scratch); return; }

    memcpy(work, signal, n_work * sizeof(float));

    /* Forward DWT: levels passes */
    int cur_len = n_work;
    for (int l = 0; l < levels; l++) {
        haar_forward(work, cur_len, scratch);
        cur_len >>= 1;
    }

    /* Estimate sigma from finest detail (level-1 detail: work[n_work/2..n_work-1]) */
    int finest_start = n_work / 2;
    int finest_len   = n_work / 2;
    float rms2 = 0.0f;
    for (int i = finest_start; i < finest_start + finest_len; i++)
        rms2 += work[i] * work[i];
    float sigma = (finest_len > 0) ? sqrtf(rms2 / finest_len) : 1e-6f;
    if (sigma < 1e-10f) sigma = 1e-6f;

    /* Universal threshold T = sigma * sqrt(2 * log(n_work)) */
    float T = sigma * sqrtf(2.0f * logf((float)n_work));

    /* Soft-threshold all detail sub-bands */
    cur_len = n_work;
    for (int l = 0; l < levels; l++) {
        int det_start = cur_len / 2;
        for (int i = det_start; i < cur_len; i++)
            work[i] = soft_threshold(work[i], T);
        cur_len >>= 1;
    }

    /* Inverse DWT: levels passes in reverse */
    for (int l = levels - 1; l >= 0; l--) {
        int inv_len = n_work >> l;
        haar_inverse(work, inv_len, scratch);
    }

    memcpy(signal, work, n_work * sizeof(float));
    free(work);
    free(scratch);
}
