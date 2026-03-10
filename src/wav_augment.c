#include "wav_augment.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void wav_aug_noise(float *samples, int n, float snr_db)
{
    /* Calcular potencia do sinal */
    float power = 0.0f;
    for (int i = 0; i < n; i++) power += samples[i] * samples[i];
    power /= n;
    if (power < 1e-10f) return;

    /* Desvio padrao do ruido para a SNR desejada */
    float noise_power = power / powf(10.0f, snr_db / 10.0f);
    float noise_std   = sqrtf(noise_power);

    for (int i = 0; i < n; i++) {
        float s = samples[i] + noise_std * rng_normal();
        /* Clamp a [-1, 1] */
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        samples[i] = s;
    }
}

void wav_aug_gain(float *samples, int n, float db)
{
    float scale = powf(10.0f, db / 20.0f);
    for (int i = 0; i < n; i++) {
        float s = samples[i] * scale;
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        samples[i] = s;
    }
}

/*
 * Reamostragem com interpolacao linear.
 * Para cada indice de saida i, a posicao de origem e i * src_step.
 * Resultado copiado de volta para samples[].
 */
static void resample_inplace(float *samples, int n, float src_step)
{
    float *tmp = (float *)safe_malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float pos = i * src_step;
        int   lo  = (int)pos;
        float frac = pos - lo;
        if (lo >= n - 1) {
            tmp[i] = samples[n - 1];
        } else {
            tmp[i] = samples[lo] * (1.0f - frac) + samples[lo + 1] * frac;
        }
    }
    memcpy(samples, tmp, n * sizeof(float));
    free(tmp);
}

void wav_aug_stretch(float *samples, int n, float factor)
{
    /* factor > 1 → mais lento: cada amostra de saida vem de 1/factor na entrada */
    if (factor <= 0.0f) return;
    resample_inplace(samples, n, 1.0f / factor);
}

void wav_aug_pitch(float *samples, int n, float semitones)
{
    /* pitch_factor = 2^(semitones/12): >1 → pitch mais alto */
    float pitch_factor = powf(2.0f, semitones / 12.0f);
    /* Para subir o pitch, comprimir o sinal (src avanca mais rapido) */
    resample_inplace(samples, n, pitch_factor);
}
