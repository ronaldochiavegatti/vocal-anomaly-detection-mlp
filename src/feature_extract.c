/*
 * feature_extract.c - Orquestrador de extracao de features
 *
 * Para cada paciente no dataset:
 *   1. Le os 3 WAVs de vogal (a_n, i_n, u_n)
 *   2. Aplica pre-enfase ao sinal
 *   3. Extrai 4 features temporais (do sinal original)
 *   4. Extrai 7 features espectrais (do sinal com pre-enfase)
 *   5. Extrai 18 features wavelet (do sinal original)
 *   6. Concatena as 29 features de cada vogal -> 87 features por paciente
 */

#include "feature_extract.h"
#include "wav_io.h"
#include "dsp_utils.h"
#include "feature_temporal.h"
#include "feature_spectral.h"
#include "feature_wavelet.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Extrai as 29 features de uma unica vogal.
 * Preenche out[0..28] com os valores.
 * Retorna 0 em sucesso.
 */
static int extract_vowel_features(const char *wav_path, float *out)
{
    WavFile wav;
    if (wav_read(wav_path, &wav) != 0) {
        /* Preencher com zeros em caso de erro */
        memset(out, 0, FEATURES_PER_VOWEL * sizeof(float));
        return -1;
    }

    /* Copiar sinal para pre-enfase (manter original para features temporais) */
    float *pre_emph = (float *)safe_malloc(wav.num_samples * sizeof(float));
    memcpy(pre_emph, wav.samples, wav.num_samples * sizeof(float));
    dsp_pre_emphasis(pre_emph, wav.num_samples, PRE_EMPHASIS_ALPHA);

    int idx = 0;

    /* Features temporais (sinal original, sem pre-enfase) */
    TemporalFeatures tf;
    temporal_extract(wav.samples, wav.num_samples, wav.sample_rate, &tf);
    out[idx++] = tf.jitter_local;
    out[idx++] = tf.jitter_rap;
    out[idx++] = tf.jitter_ppq5;
    out[idx++] = tf.shimmer_local;
    out[idx++] = tf.shimmer_apq3;
    out[idx++] = tf.shimmer_apq5;
    out[idx++] = tf.shimmer_apq11;
    out[idx++] = tf.energy_mean;
    out[idx++] = tf.hnr;
    out[idx++] = tf.zcr;

    /* Features espectrais (sinal com pre-enfase) */
    SpectralFeatures sf;
    spectral_extract(pre_emph, wav.num_samples, wav.sample_rate, &sf);
    out[idx++] = sf.f0_mean;
    out[idx++] = sf.f0_std;
    out[idx++] = sf.formants[0];  /* F1 */
    out[idx++] = sf.formants[1];  /* F2 */
    out[idx++] = sf.formants[2];  /* F3 */
    out[idx++] = sf.formants[3];  /* F4 */
    out[idx++] = sf.spectral_entropy;
    out[idx++] = sf.spectral_centroid;
    out[idx++] = sf.spectral_rolloff;
    for (int m = 0; m < 13; m++) out[idx++] = sf.mfcc[m];

    /* Features wavelet (sinal original) */
    WaveletFeatures wf;
    wavelet_extract(wav.samples, wav.num_samples, &wf);
    for (int l = 0; l < WAVELET_LEVELS; l++) out[idx++] = wf.mean[l];
    for (int l = 0; l < WAVELET_LEVELS; l++) out[idx++] = wf.variance[l];
    for (int l = 0; l < WAVELET_LEVELS; l++) out[idx++] = wf.energy[l];

    free(pre_emph);
    wav_free(&wav);
    return 0;
}

int features_extract_all(const Dataset *ds, FeatureMatrix *fm)
{
    fm->count = ds->count;
    fm->num_features = TOTAL_FEATURES;
    fm->features = (float *)safe_calloc(fm->count * fm->num_features, sizeof(float));
    fm->labels = (int *)safe_malloc(fm->count * sizeof(int));

    double t_start = timer_now();
    int errors = 0;

    for (int i = 0; i < ds->count; i++) {
        const Patient *p = &ds->patients[i];
        fm->labels[i] = p->class_label;

        /* Extrair features de cada vogal */
        for (int v = 0; v < NUM_VOWELS; v++) {
            float *dest = &fm->features[i * fm->num_features + v * FEATURES_PER_VOWEL];
            if (extract_vowel_features(p->vowel_paths[v], dest) != 0) {
                errors++;
            }
        }

        /* Progresso a cada 100 pacientes */
        if ((i + 1) % 100 == 0 || i == ds->count - 1) {
            double elapsed = timer_now() - t_start;
            log_info("Extracao: %d/%d pacientes (%.1fs)", i + 1, ds->count, elapsed);
        }
    }

    double total_time = timer_now() - t_start;
    log_info("Extracao completa: %d pacientes, %d erros, %.1fs",
             fm->count, errors, total_time);

    /* Verificar NaN/Inf e substituir por 0 */
    int nan_count = 0;
    for (int i = 0; i < fm->count * fm->num_features; i++) {
        if (isnan(fm->features[i]) || isinf(fm->features[i])) {
            fm->features[i] = 0.0f;
            nan_count++;
        }
    }
    if (nan_count > 0) {
        log_warn("Substituidos %d valores NaN/Inf por 0", nan_count);
    }

    return 0;
}

int features_export_csv(const FeatureMatrix *fm, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) {
        log_error("Nao foi possivel criar: %s", path);
        return -1;
    }

    /* Cabecalho */
    const char *vowels[] = {"a", "i", "u"};
    const char *feat_names[] = {
        "jitter_local", "jitter_rap", "jitter_ppq5",
        "shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11",
        "energy", "hnr", "zcr",
        "f0_mean", "f0_std", "F1", "F2", "F3", "F4", "spectral_entropy",
        "spectral_centroid", "spectral_rolloff",
        "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
        "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12",
        "wl_mean_1", "wl_mean_2", "wl_mean_3", "wl_mean_4", "wl_mean_5", "wl_mean_6",
        "wl_var_1", "wl_var_2", "wl_var_3", "wl_var_4", "wl_var_5", "wl_var_6",
        "wl_energy_1", "wl_energy_2", "wl_energy_3", "wl_energy_4", "wl_energy_5", "wl_energy_6"
    };

    for (int v = 0; v < NUM_VOWELS; v++) {
        for (int j = 0; j < FEATURES_PER_VOWEL; j++) {
            if (v > 0 || j > 0) fprintf(f, ",");
            fprintf(f, "%s_%s", vowels[v], feat_names[j]);
        }
    }
    fprintf(f, ",label\n");

    /* Dados */
    for (int i = 0; i < fm->count; i++) {
        for (int j = 0; j < fm->num_features; j++) {
            if (j > 0) fprintf(f, ",");
            fprintf(f, "%.6f", fm->features[i * fm->num_features + j]);
        }
        fprintf(f, ",%d\n", fm->labels[i]);
    }

    fclose(f);
    log_info("Features exportadas: %s (%d x %d)", path, fm->count, fm->num_features);
    return 0;
}

void features_free(FeatureMatrix *fm)
{
    if (fm) {
        free(fm->features);
        free(fm->labels);
        fm->features = NULL;
        fm->labels = NULL;
        fm->count = 0;
    }
}
