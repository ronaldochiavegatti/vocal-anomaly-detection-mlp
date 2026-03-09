/*
 * feature_temporal.c - Extracao de features temporais
 *
 * Detecta periodos via autocorrelacao para estimar T0, depois
 * localiza picos positivos do sinal usando busca guiada.
 *
 * Definicoes (padrao Praat / literatura):
 *   Jitter local = media(|T_i - T_{i+1}|) / media(T_i) * 100%
 *   Jitter RAP   = media(|T_i - media(T_{i-1}, T_i, T_{i+1})|) / media(T_i) * 100%
 *   Shimmer local = media(|A_i - A_{i+1}|) / media(A_i) * 100%
 */

#include "feature_temporal.h"
#include "dsp_utils.h"
#include "config.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MIN_PERIODS 5

/*
 * Estima o periodo fundamental T0 via autocorrelacao.
 * Retorna T0 em amostras, ou 0 se nao detectado.
 */
static int estimate_t0_autocorr(const float *signal, int n, int sample_rate)
{
    int min_lag = sample_rate / F0_MAX_HZ;
    int max_lag = sample_rate / F0_MIN_HZ;

    if (max_lag >= n) max_lag = n - 1;
    if (min_lag >= max_lag) return 0;

    /* Usar trecho central (mais estavel) */
    int chunk_len = n;
    if (chunk_len > sample_rate) chunk_len = sample_rate;
    int offset = (n - chunk_len) / 2;

    float *acorr = (float *)safe_malloc((max_lag + 1) * sizeof(float));
    dsp_autocorrelation(signal + offset, chunk_len, acorr, max_lag + 1);

    float r0 = acorr[0];
    if (r0 < 1e-10f) {
        free(acorr);
        return 0;
    }

    float best_val = 0.0f;
    int best_lag = 0;
    for (int k = min_lag; k <= max_lag; k++) {
        float norm = acorr[k] / r0;
        if (norm > best_val) {
            best_val = norm;
            best_lag = k;
        }
    }

    free(acorr);

    if (best_val < 0.2f || best_lag == 0) return 0;
    return best_lag;
}

/*
 * Detecta picos positivos usando T0 como guia.
 * Usa janelas deslizantes de tamanho T0 e busca o maximo em cada uma.
 * Tolerancia de 40% para acomodar sinais patologicos.
 */
static int detect_peaks_guided(const float *signal, int n,
                               int t0, int *peaks, int max_peaks)
{
    int count = 0;
    int tol = (int)(t0 * 0.4f);
    if (tol < 4) tol = 4;

    /* Primeiro pico: buscar o maximo global no primeiro periodo */
    int end = t0 + tol;
    if (end > n) end = n;
    float best = -1e30f;
    int best_idx = 0;
    for (int i = 0; i < end; i++) {
        if (signal[i] > best) {
            best = signal[i];
            best_idx = i;
        }
    }
    peaks[count++] = best_idx;

    /* Picos subsequentes: buscar maximo na janela centrada em last_peak + T0 */
    while (count < max_peaks) {
        int center = peaks[count - 1] + t0;
        int win_start = center - tol;
        int win_end = center + tol;

        if (win_start < peaks[count - 1] + 1)
            win_start = peaks[count - 1] + 1;
        if (win_end > n) win_end = n;
        if (win_start >= n) break;

        best = -1e30f;
        best_idx = win_start;
        for (int i = win_start; i < win_end; i++) {
            if (signal[i] > best) {
                best = signal[i];
                best_idx = i;
            }
        }

        peaks[count++] = best_idx;
    }

    return count;
}

/*
 * Calcula HNR (Harmonics-to-Noise Ratio) em dB.
 * HNR = 10 * log10(R[T0] / (R[0] - R[T0]))
 */
static float calculate_hnr(const float *signal, int n, int t0)
{
    if (t0 <= 0 || t0 >= n) return 0.0f;

    float *acorr = (float *)safe_malloc((t0 + 1) * sizeof(float));
    dsp_autocorrelation(signal, n, acorr, t0 + 1);

    float r0 = acorr[0];
    float rt0 = acorr[t0];
    free(acorr);

    if (r0 <= 0.0f || rt0 <= 0.0f || r0 <= rt0) return 0.0f;

    return 10.0f * log10f(rt0 / (r0 - rt0));
}

/*
 * Calcula Zero Crossing Rate.
 */
static float calculate_zcr(const float *signal, int n)
{
    int crossings = 0;
    for (int i = 1; i < n; i++) {
        if ((signal[i] >= 0.0f && signal[i - 1] < 0.0f) ||
            (signal[i] < 0.0f && signal[i - 1] >= 0.0f)) {
            crossings++;
        }
    }
    return (float)crossings / (n - 1);
}

/*
 * Calcula perturbation quotient generico com janela de tamanho w.
 * Usado para Jitter PPQ5, Shimmer APQ3/5/11.
 */
static float perturbation_quotient(const float *values, int n, int window)
{
    if (n < window) return 0.0f;
    int half = window / 2;
    float sum_diff = 0.0f;
    float sum_val = 0.0f;
    int count = 0;

    for (int i = half; i < n - half; i++) {
        float local_mean = 0.0f;
        for (int j = i - half; j <= i + half; j++) {
            local_mean += values[j];
        }
        local_mean /= window;
        sum_diff += fabsf(values[i] - local_mean);
        sum_val += values[i];
        count++;
    }

    if (count == 0 || sum_val <= 0.0f) return 0.0f;
    float mean_val = sum_val / count;
    return (sum_diff / count) / mean_val * 100.0f;
}

int temporal_extract(const float *signal, int n, int sample_rate,
                     TemporalFeatures *out)
{
    memset(out, 0, sizeof(TemporalFeatures));
    out->energy_mean = dsp_energy(signal, n);
    out->zcr = calculate_zcr(signal, n);

    int t0 = estimate_t0_autocorr(signal, n, sample_rate);
    if (t0 == 0) return 0;

    out->hnr = calculate_hnr(signal, n, t0);

    int max_peaks = n / (t0 / 2) + 1;
    int *peaks = (int *)safe_malloc(max_peaks * sizeof(int));
    int num_peaks = detect_peaks_guided(signal, n, t0, peaks, max_peaks);

    if (num_peaks < MIN_PERIODS) {
        free(peaks);
        return 0;
    }

    int num_periods = num_peaks - 1;
    float *periods = (float *)safe_malloc(num_periods * sizeof(float));
    float *amplitudes = (float *)safe_malloc(num_peaks * sizeof(float));

    float period_sum = 0.0f;
    for (int i = 0; i < num_periods; i++) {
        periods[i] = (float)(peaks[i + 1] - peaks[i]);
        period_sum += periods[i];
    }
    float period_mean = period_sum / num_periods;

    float amp_sum = 0.0f;
    for (int i = 0; i < num_peaks; i++) {
        amplitudes[i] = fabsf(signal[peaks[i]]);
        amp_sum += amplitudes[i];
    }
    float amp_mean = amp_sum / num_peaks;

    /* Jitter local */
    if (num_periods >= 2 && period_mean > 0.0f) {
        float jitter_sum = 0.0f;
        for (int i = 0; i < num_periods - 1; i++) {
            jitter_sum += fabsf(periods[i] - periods[i + 1]);
        }
        out->jitter_local = (jitter_sum / (num_periods - 1)) / period_mean * 100.0f;
    }

    /* Jitter RAP (janela 3) */
    if (num_periods >= 3 && period_mean > 0.0f) {
        out->jitter_rap = perturbation_quotient(periods, num_periods, 3);
    }

    /* Jitter PPQ5 (janela 5) */
    if (num_periods >= 5 && period_mean > 0.0f) {
        out->jitter_ppq5 = perturbation_quotient(periods, num_periods, 5);
    }

    /* Shimmer local */
    if (num_peaks >= 2 && amp_mean > 0.0f) {
        float shimmer_sum = 0.0f;
        for (int i = 0; i < num_peaks - 1; i++) {
            shimmer_sum += fabsf(amplitudes[i] - amplitudes[i + 1]);
        }
        out->shimmer_local = (shimmer_sum / (num_peaks - 1)) / amp_mean * 100.0f;
    }

    /* Shimmer APQ3/5/11 */
    if (num_peaks >= 3) {
        out->shimmer_apq3 = perturbation_quotient(amplitudes, num_peaks, 3);
    }
    if (num_peaks >= 5) {
        out->shimmer_apq5 = perturbation_quotient(amplitudes, num_peaks, 5);
    }
    if (num_peaks >= 11) {
        out->shimmer_apq11 = perturbation_quotient(amplitudes, num_peaks, 11);
    }

    free(peaks);
    free(periods);
    free(amplitudes);
    return 0;
}
