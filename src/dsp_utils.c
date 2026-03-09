/*
 * dsp_utils.c - Primitivas de Processamento Digital de Sinais
 *
 * Implementacao de pre-enfase, janela Hamming, autocorrelacao,
 * FFT radix-2 (Cooley-Tukey) e funcoes auxiliares.
 */

#include "dsp_utils.h"
#include <math.h>
#include <string.h>

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
