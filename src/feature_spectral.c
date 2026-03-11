/*
 * feature_spectral.c - Extracao de features espectrais
 *
 * F0 via autocorrelacao por frames, formantes via LPC (Levinson-Durbin)
 * com busca de picos no envelope espectral, e entropia espectral.
 */

#include "feature_spectral.h"
#include "dsp_utils.h"
#include "config.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========== F0 via autocorrelacao ========== */

/*
 * Estima f0 de um unico frame via autocorrelacao.
 * Retorna f0 em Hz, ou 0.0 se nao detectado (frame nao-vozeado).
 */
static float estimate_f0_frame(const float *frame, int frame_len, int sample_rate)
{
    int min_lag = sample_rate / F0_MAX_HZ;  /* 88 para 44100/500 */
    int max_lag = sample_rate / F0_MIN_HZ;  /* 551 para 44100/80 */

    if (max_lag >= frame_len) max_lag = frame_len - 1;
    if (min_lag >= max_lag) return 0.0f;

    /* Calcular autocorrelacao */
    float *acorr = (float *)safe_malloc((max_lag + 1) * sizeof(float));
    dsp_autocorrelation(frame, frame_len, acorr, max_lag + 1);

    /* R[0] eh a energia do frame */
    float r0 = acorr[0];
    if (r0 < 1e-10f) {
        free(acorr);
        return 0.0f;  /* silencio */
    }

    /* Buscar pico maximo em [min_lag, max_lag] */
    float best_val = 0.0f;
    int best_lag = 0;

    for (int k = min_lag; k <= max_lag; k++) {
        float normalized = acorr[k] / r0;
        if (normalized > best_val) {
            best_val = normalized;
            best_lag = k;
        }
    }

    free(acorr);

    /* Limiar de vozeamento: pico normalizado deve ser > 0.3 */
    if (best_val < 0.3f || best_lag == 0) {
        return 0.0f;
    }

    return (float)sample_rate / best_lag;
}

/*
 * Estima f0 media e desvio-padrao sobre frames do sinal.
 */
static void estimate_f0_stats(const float *signal, int n, int sample_rate,
                              float *f0_mean, float *f0_std)
{
    int frame_size = FRAME_SIZE;
    int frame_step = FRAME_STEP;
    int num_frames = (n - frame_size) / frame_step + 1;

    if (num_frames <= 0) {
        *f0_mean = 0.0f;
        *f0_std = 0.0f;
        return;
    }

    /* Buffer para frame com janela Hamming */
    float *frame = (float *)safe_malloc(frame_size * sizeof(float));

    float *f0_values = (float *)safe_malloc(num_frames * sizeof(float));
    int voiced_count = 0;

    for (int i = 0; i < num_frames; i++) {
        int offset = i * frame_step;
        memcpy(frame, signal + offset, frame_size * sizeof(float));
        dsp_hamming_window(frame, frame_size);

        float f0 = estimate_f0_frame(frame, frame_size, sample_rate);
        if (f0 > 0.0f) {
            f0_values[voiced_count++] = f0;
        }
    }

    if (voiced_count == 0) {
        *f0_mean = 0.0f;
        *f0_std = 0.0f;
    } else {
        /* Media */
        float sum = 0.0f;
        for (int i = 0; i < voiced_count; i++) sum += f0_values[i];
        *f0_mean = sum / voiced_count;

        /* Desvio-padrao */
        float var_sum = 0.0f;
        for (int i = 0; i < voiced_count; i++) {
            float diff = f0_values[i] - *f0_mean;
            var_sum += diff * diff;
        }
        *f0_std = sqrtf(var_sum / voiced_count);
    }

    free(frame);
    free(f0_values);
}

/* ========== Formantes via LPC (Levinson-Durbin) ========== */

/*
 * Resolve coeficientes LPC via algoritmo de Levinson-Durbin.
 * r: autocorrelacao R[0..order]
 * a: coeficientes LPC de saida (a[0..order], onde a[0]=1.0)
 * order: ordem do LPC
 *
 * Retorna o erro de predicao.
 */
static float levinson_durbin(const float *r, float *a, int order)
{
    float *a_prev = (float *)safe_calloc(order + 1, sizeof(float));

    a[0] = 1.0f;
    float error = r[0];

    for (int i = 1; i <= order; i++) {
        /* Calcular coeficiente de reflexao */
        float sum = 0.0f;
        for (int j = 1; j < i; j++) {
            sum += a[j] * r[i - j];
        }
        float k = -(r[i] + sum) / error;

        /* Atualizar coeficientes */
        memcpy(a_prev, a, (order + 1) * sizeof(float));
        for (int j = 1; j < i; j++) {
            a[j] = a_prev[j] + k * a_prev[i - j];
        }
        a[i] = k;

        error *= (1.0f - k * k);
        if (error <= 0.0f) break;
    }

    free(a_prev);
    return error;
}

/*
 * Subamostra o sinal por fator inteiro (media de fator amostras).
 * Retorna o sinal subamostrado (alocado internamente).
 */
static float *downsample(const float *signal, int n, int factor, int *out_n)
{
    *out_n = n / factor;
    float *out = (float *)safe_malloc(*out_n * sizeof(float));
    for (int i = 0; i < *out_n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < factor; j++) {
            sum += signal[i * factor + j];
        }
        out[i] = sum / factor;
    }
    return out;
}

/*
 * Encontra formantes de um frame via LPC + busca de picos no envelope.
 * O sinal deve estar a ~11025 Hz para LPC ordem 12 funcionar bem.
 * frame_formants[4] recebe as frequencias dos formantes encontrados.
 * Retorna o numero de formantes encontrados (0-4).
 */
static int find_formants_frame(const float *frame, int n, int effective_sr,
                               float frame_formants[4])
{
    int order = LPC_ORDER;
    frame_formants[0] = frame_formants[1] = frame_formants[2] = frame_formants[3] = 0.0f;

    float *r = (float *)safe_calloc(order + 1, sizeof(float));
    dsp_autocorrelation(frame, n, r, order + 1);

    if (r[0] < 1e-10f) {
        free(r);
        return 0;
    }

    float *a = (float *)safe_calloc(order + 1, sizeof(float));
    levinson_durbin(r, a, order);
    free(r);

    /* Avaliar envelope espectral: resolucao de ~5 Hz */
    int num_bins = effective_sr / 10;
    float freq_step = (float)(effective_sr / 2) / num_bins;
    float *envelope = (float *)safe_malloc(num_bins * sizeof(float));

    for (int i = 0; i < num_bins; i++) {
        float freq = i * freq_step;
        float w = 2.0f * (float)M_PI * freq / effective_sr;

        float re = 0.0f, im = 0.0f;
        for (int k = 0; k <= order; k++) {
            re += a[k] * cosf(w * k);
            im -= a[k] * sinf(w * k);
        }

        float mag_sq = re * re + im * im;
        envelope[i] = (mag_sq > 1e-10f) ? 1.0f / mag_sq : 0.0f;
    }

    free(a);

    /* Encontrar picos entre 200 Hz e 5000 Hz */
    int formant_count = 0;
    int start_bin = (int)(200.0f / freq_step);
    int end_bin = (int)(5000.0f / freq_step);
    if (start_bin < 1) start_bin = 1;
    if (end_bin >= num_bins) end_bin = num_bins - 1;

    for (int i = start_bin; i < end_bin && formant_count < 4; i++) {
        if (envelope[i] > envelope[i - 1] && envelope[i] > envelope[i + 1]) {
            frame_formants[formant_count++] = i * freq_step;
        }
    }

    free(envelope);
    return formant_count;
}

/*
 * Calcula formantes F1-F4 usando multiplos frames com media.
 * Subamostra a ~11025 Hz (fator 4 de 44100) para melhor resolucao LPC.
 */
static void find_formants_lpc(const float *signal, int n, int sample_rate,
                              float formants[4])
{
    formants[0] = formants[1] = formants[2] = formants[3] = 0.0f;

    /* Subamostrar a ~11025 Hz */
    int ds_factor = sample_rate / 11025;
    if (ds_factor < 1) ds_factor = 1;
    int ds_n;
    float *ds_signal = downsample(signal, n, ds_factor, &ds_n);
    int effective_sr = sample_rate / ds_factor;

    /* Frame de ~30ms a 11025 Hz = ~330 amostras */
    int frame_size = effective_sr * 30 / 1000;
    int frame_step = effective_sr * 10 / 1000;

    if (frame_size > ds_n) {
        frame_size = ds_n;
        frame_step = ds_n;
    }

    int num_frames = (ds_n - frame_size) / frame_step + 1;
    if (num_frames <= 0) {
        free(ds_signal);
        return;
    }

    /* Acumular formantes de todos os frames */
    float f_sum[4] = {0, 0, 0, 0};
    int f_count[4] = {0, 0, 0, 0};
    float *frame_buf = (float *)safe_malloc(frame_size * sizeof(float));

    for (int i = 0; i < num_frames; i++) {
        int offset = i * frame_step;
        memcpy(frame_buf, ds_signal + offset, frame_size * sizeof(float));
        dsp_hamming_window(frame_buf, frame_size);

        float ff[4];
        int nf = find_formants_frame(frame_buf, frame_size, effective_sr, ff);
        for (int j = 0; j < nf && j < 4; j++) {
            f_sum[j] += ff[j];
            f_count[j]++;
        }
    }

    for (int j = 0; j < 4; j++) {
        if (f_count[j] > 0) {
            formants[j] = f_sum[j] / f_count[j];
        }
    }

    free(frame_buf);
    free(ds_signal);
}

/* ========== Entropia espectral ========== */

/*
 * Calcula a entropia espectral normalizada.
 * H = -sum(p_i * log2(p_i)) / log2(N)
 * onde p_i = |X_i|^2 / sum(|X_k|^2) eh a distribuicao espectral normalizada.
 */
static float compute_spectral_entropy(const float *signal, int n, int sample_rate)
{
    (void)sample_rate;

    int fft_size = dsp_next_power_of_2(n);
    float *real = (float *)safe_calloc(fft_size, sizeof(float));
    float *imag = (float *)safe_calloc(fft_size, sizeof(float));

    /* Copiar sinal e aplicar janela Hamming */
    memcpy(real, signal, n * sizeof(float));
    dsp_hamming_window(real, n);
    /* Zero-padding ja feito pelo calloc */

    dsp_fft(real, imag, fft_size);

    /* Espectro de potencia (metade positiva) */
    int half = fft_size / 2 + 1;
    float *power = (float *)safe_malloc(half * sizeof(float));
    dsp_power_spectrum(real, imag, fft_size, power);

    free(real);
    free(imag);

    /* Normalizar como distribuicao de probabilidade */
    float total_power = 0.0f;
    for (int i = 0; i < half; i++) total_power += power[i];

    if (total_power < 1e-10f) {
        free(power);
        return 0.0f;
    }

    /* Calcular entropia */
    float entropy = 0.0f;
    for (int i = 0; i < half; i++) {
        float p = power[i] / total_power;
        if (p > 1e-10f) {
            entropy -= p * log2f(p);
        }
    }

    /* Normalizar por log2(N) para ficar em [0, 1] */
    float max_entropy = log2f((float)half);
    if (max_entropy > 0.0f) {
        entropy /= max_entropy;
    }

    free(power);
    return entropy;
}

/* ========== Spectral Centroid e Rolloff ========== */

/*
 * Calcula centroide espectral: frequencia media ponderada pela energia.
 * Centroid = sum(f_k * P_k) / sum(P_k)
 */
static float compute_spectral_centroid(const float *signal, int n, int sample_rate)
{
    int fft_size = dsp_next_power_of_2(n);
    float *real = (float *)safe_calloc(fft_size, sizeof(float));
    float *imag = (float *)safe_calloc(fft_size, sizeof(float));

    memcpy(real, signal, n * sizeof(float));
    dsp_hamming_window(real, n);
    dsp_fft(real, imag, fft_size);

    int half = fft_size / 2 + 1;
    float freq_step = (float)sample_rate / fft_size;
    float weighted_sum = 0.0f;
    float total_power = 0.0f;

    for (int i = 0; i < half; i++) {
        float power = real[i] * real[i] + imag[i] * imag[i];
        float freq = i * freq_step;
        weighted_sum += freq * power;
        total_power += power;
    }

    free(real);
    free(imag);

    return (total_power > 1e-10f) ? weighted_sum / total_power : 0.0f;
}

/*
 * Calcula rolloff espectral: frequencia abaixo da qual 85% da energia esta.
 */
static float compute_spectral_rolloff(const float *signal, int n, int sample_rate)
{
    int fft_size = dsp_next_power_of_2(n);
    float *real = (float *)safe_calloc(fft_size, sizeof(float));
    float *imag = (float *)safe_calloc(fft_size, sizeof(float));

    memcpy(real, signal, n * sizeof(float));
    dsp_hamming_window(real, n);
    dsp_fft(real, imag, fft_size);

    int half = fft_size / 2 + 1;
    float freq_step = (float)sample_rate / fft_size;
    float total = 0.0f;

    for (int i = 0; i < half; i++) {
        total += real[i] * real[i] + imag[i] * imag[i];
    }

    float threshold = 0.85f * total;
    float cumsum = 0.0f;
    float rolloff = 0.0f;

    for (int i = 0; i < half; i++) {
        cumsum += real[i] * real[i] + imag[i] * imag[i];
        if (cumsum >= threshold) {
            rolloff = i * freq_step;
            break;
        }
    }

    free(real);
    free(imag);
    return rolloff;
}

/* ========== MFCC (Mel-Frequency Cepstral Coefficients) ========== */

#define NUM_MEL_FILTERS 26
#define NUM_MFCC_COEFFS 13

static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/*
 * US-011: Computa delta de coeficientes frame-a-frame com janela N=2.
 * Formula: Delta[t] = (2*c[t+2] + c[t+1] - c[t-1] - 2*c[t-2]) / 10
 * Com replicacao de bordas.
 */
static void compute_delta_frames(const float *frames, int num_frames, int n_coef,
                                  float *delta)
{
    for (int t = 0; t < num_frames; t++) {
        for (int k = 0; k < n_coef; k++) {
            int t1p = (t + 1 < num_frames) ? t + 1 : num_frames - 1;
            int t1m = (t - 1 >= 0)         ? t - 1 : 0;
            int t2p = (t + 2 < num_frames) ? t + 2 : num_frames - 1;
            int t2m = (t - 2 >= 0)         ? t - 2 : 0;
            delta[t * n_coef + k] =
                (2.0f * frames[t2p * n_coef + k] + frames[t1p * n_coef + k]
                - frames[t1m * n_coef + k] - 2.0f * frames[t2m * n_coef + k]) / 10.0f;
        }
    }
}

/*
 * Extrai MFCCs estaticos, delta e delta-delta do sinal.
 * Usa frames com Hamming, Mel filterbank, log energia, DCT-II.
 * Saida: mfcc_out[13], delta_out[13], delta2_out[13] (medias sobre frames)
 */
static void compute_mfcc(const float *signal, int n, int sample_rate,
                          float *mfcc_out, float *delta_out, float *delta2_out)
{
    memset(mfcc_out, 0, NUM_MFCC_COEFFS * sizeof(float));
    memset(delta_out, 0, NUM_MFCC_COEFFS * sizeof(float));
    memset(delta2_out, 0, NUM_MFCC_COEFFS * sizeof(float));

    int frame_size = FRAME_SIZE;
    int frame_step = FRAME_STEP;
    int num_frames = (n - frame_size) / frame_step + 1;
    if (num_frames <= 0) return;

    int fft_size = dsp_next_power_of_2(frame_size);
    int half = fft_size / 2 + 1;
    float freq_step = (float)sample_rate / fft_size;

    /* Calcular pontos centrais dos filtros Mel */
    float low_mel = hz_to_mel(0.0f);
    float high_mel = hz_to_mel((float)sample_rate / 2.0f);

    float mel_points[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        float mel = low_mel + (high_mel - low_mel) * i / (NUM_MEL_FILTERS + 1);
        mel_points[i] = mel_to_hz(mel);
    }

    /* Converter para bins de FFT */
    int bins[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        bins[i] = (int)(mel_points[i] / freq_step);
        if (bins[i] >= half) bins[i] = half - 1;
    }

    /* Buffer per-frame de MFCCs para calculo de deltas */
    float *mfcc_frames = (float *)safe_calloc(num_frames * NUM_MFCC_COEFFS, sizeof(float));

    float *frame = (float *)safe_malloc(fft_size * sizeof(float));
    float *imag = (float *)safe_calloc(fft_size, sizeof(float));
    float *mel_energies = (float *)safe_malloc(NUM_MEL_FILTERS * sizeof(float));

    for (int fi = 0; fi < num_frames; fi++) {
        int offset = fi * frame_step;

        /* Preparar frame com Hamming e zero-padding */
        memset(frame, 0, fft_size * sizeof(float));
        memset(imag, 0, fft_size * sizeof(float));
        memcpy(frame, signal + offset, frame_size * sizeof(float));
        dsp_hamming_window(frame, frame_size);

        /* FFT */
        dsp_fft(frame, imag, fft_size);

        /* Power spectrum */
        float *power = (float *)safe_malloc(half * sizeof(float));
        for (int i = 0; i < half; i++) {
            power[i] = (frame[i] * frame[i] + imag[i] * imag[i]) / fft_size;
        }

        /* Aplicar Mel filterbank (filtros triangulares) */
        for (int m = 0; m < NUM_MEL_FILTERS; m++) {
            float energy = 0.0f;
            int start = bins[m];
            int center = bins[m + 1];
            int end = bins[m + 2];

            /* Subida */
            for (int k = start; k < center && k < half; k++) {
                float weight = (center > start) ?
                    (float)(k - start) / (center - start) : 0.0f;
                energy += power[k] * weight;
            }
            /* Descida */
            for (int k = center; k < end && k < half; k++) {
                float weight = (end > center) ?
                    (float)(end - k) / (end - center) : 0.0f;
                energy += power[k] * weight;
            }

            mel_energies[m] = (energy > 1e-10f) ? log10f(energy) : -10.0f;
        }

        free(power);

        /* DCT-II para obter MFCCs deste frame */
        for (int k = 0; k < NUM_MFCC_COEFFS; k++) {
            float sum = 0.0f;
            for (int m = 0; m < NUM_MEL_FILTERS; m++) {
                sum += mel_energies[m] *
                       cosf((float)M_PI * k * (m + 0.5f) / NUM_MEL_FILTERS);
            }
            mfcc_frames[fi * NUM_MFCC_COEFFS + k] = sum;
        }
    }

    /* US-011: Calcular delta e delta-delta frame-a-frame */
    float *delta_frames  = (float *)safe_malloc(num_frames * NUM_MFCC_COEFFS * sizeof(float));
    float *delta2_frames = (float *)safe_malloc(num_frames * NUM_MFCC_COEFFS * sizeof(float));
    compute_delta_frames(mfcc_frames, num_frames, NUM_MFCC_COEFFS, delta_frames);
    compute_delta_frames(delta_frames, num_frames, NUM_MFCC_COEFFS, delta2_frames);

    /* Computar media dos MFCCs estaticos e desvio-padrao dos deltas.
     * Nota: mean(delta) tende a zero para vogais sustentadas (pouco informativo).
     * Std(delta) captura a variabilidade da dinamica espectral — mais discriminativo
     * para vozes patologicas que exibem irregularidades periodicas. */
    for (int k = 0; k < NUM_MFCC_COEFFS; k++) {
        float sum = 0.0f, sum_d = 0.0f, sum_d2 = 0.0f;
        for (int fi = 0; fi < num_frames; fi++) {
            sum    += mfcc_frames[fi  * NUM_MFCC_COEFFS + k];
            sum_d  += delta_frames[fi  * NUM_MFCC_COEFFS + k];
            sum_d2 += delta2_frames[fi * NUM_MFCC_COEFFS + k];
        }
        mfcc_out[k] = sum / num_frames;
        float mean_d  = sum_d  / num_frames;
        float mean_d2 = sum_d2 / num_frames;

        /* Desvio-padrao temporal dos deltas */
        float var_d = 0.0f, var_d2 = 0.0f;
        for (int fi = 0; fi < num_frames; fi++) {
            float dd  = delta_frames[fi  * NUM_MFCC_COEFFS + k] - mean_d;
            float dd2 = delta2_frames[fi * NUM_MFCC_COEFFS + k] - mean_d2;
            var_d  += dd  * dd;
            var_d2 += dd2 * dd2;
        }
        delta_out[k]  = sqrtf(var_d  / num_frames);
        delta2_out[k] = sqrtf(var_d2 / num_frames);
    }

    free(mfcc_frames);
    free(delta_frames);
    free(delta2_frames);
    free(frame);
    free(imag);
    free(mel_energies);
}

/* ========== Interface publica ========== */

int spectral_extract(const float *signal, int n, int sample_rate,
                     SpectralFeatures *out)
{
    memset(out, 0, sizeof(SpectralFeatures));

    if (n < FRAME_SIZE) {
        log_warn("Sinal muito curto para analise espectral (%d amostras)", n);
        return -1;
    }

    /* F0 (media e desvio-padrao) */
    estimate_f0_stats(signal, n, sample_rate, &out->f0_mean, &out->f0_std);

    /* Formantes via LPC */
    find_formants_lpc(signal, n, sample_rate, out->formants);

    /* Entropia espectral */
    out->spectral_entropy = compute_spectral_entropy(signal, n, sample_rate);

    /* Centroide e rolloff espectral */
    out->spectral_centroid = compute_spectral_centroid(signal, n, sample_rate);
    out->spectral_rolloff = compute_spectral_rolloff(signal, n, sample_rate);

    /* MFCC + Delta + Delta-Delta (US-011) */
    compute_mfcc(signal, n, sample_rate, out->mfcc, out->delta_mfcc, out->delta2_mfcc);

    return 0;
}
