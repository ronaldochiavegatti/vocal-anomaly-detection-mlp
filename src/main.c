/*
 * main.c - Entry point do classificador de anomalias vocais
 *
 * Modos de operacao:
 *   extract  - Extrai features de todos os pacientes e salva CSV
 *   train    - Treina MLP com k-fold cross-validation
 *   test     - Carrega modelo e avalia num conjunto
 *   full     - Pipeline completa (extract + train com k-fold)
 *
 * Uso: ./vocal_detect <modo>
 */

#include "config.h"
#include "utils.h"
#include "dataset.h"
#include "feature_extract.h"
#include "normalize.h"
#include "mlp.h"
#include "mlp_train.h"
#include "kfold.h"
#include "metrics.h"
#include "feature_select.h"
#include "knn.h"
#include "logreg.h"
#include "wav_augment.h"
#include "wav_io.h"
#include "dsp_utils.h"
#include "feature_temporal.h"
#include "feature_spectral.h"
#include "feature_wavelet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========== Modo: extract ========== */

static int mode_extract(const char *base_dir)
{
    log_info("=== MODO: EXTRACAO DE FEATURES ===");

    Dataset ds;
    char csv_path[1024];
    snprintf(csv_path, sizeof(csv_path), "%s/%s", base_dir, CSV_METADATA);

    if (dataset_load(base_dir, csv_path, &ds) != 0) {
        log_error("Falha ao carregar dataset");
        return -1;
    }

    FeatureMatrix fm;
    if (features_extract_all(&ds, &fm) != 0) {
        log_error("Falha na extracao de features");
        dataset_free(&ds);
        return -1;
    }

    char out_path[1024];
    snprintf(out_path, sizeof(out_path), "%s/features.csv", RESULTS_DIR);
    features_export_csv(&fm, out_path);

    features_free(&fm);
    dataset_free(&ds);
    return 0;
}

/* ========== Feature Selection ========== */

/*
 * Feature selection: remove low-variance and highly correlated features.
 * Works on normalized data. Returns selected feature indices.
 * - var_threshold: minimum variance to keep (after z-score, most have var~1,
 *   but some may be near-constant)
 * - corr_threshold: maximum absolute Pearson correlation between features
 */
static int select_features(const float *x, int n, int nf,
                           float var_threshold, float corr_threshold,
                           int *selected, int max_selected)
{
    /* Step 1: compute variance of each feature */
    float *var = (float *)safe_calloc(nf, sizeof(float));
    float *mean = (float *)safe_calloc(nf, sizeof(float));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < nf; j++)
            mean[j] += x[i * nf + j];
    for (int j = 0; j < nf; j++) mean[j] /= n;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < nf; j++) {
            float d = x[i * nf + j] - mean[j];
            var[j] += d * d;
        }
    for (int j = 0; j < nf; j++) var[j] /= n;

    /* Step 2: mark low-variance features */
    int *keep = (int *)safe_calloc(nf, sizeof(int));
    int n_keep = 0;
    for (int j = 0; j < nf; j++) {
        if (var[j] >= var_threshold) {
            keep[j] = 1;
            n_keep++;
        }
    }

    /* Step 3: remove highly correlated features (greedy) */
    /* Build list of kept feature indices */
    int *kept_idx = (int *)safe_malloc(n_keep * sizeof(int));
    int ki = 0;
    for (int j = 0; j < nf; j++)
        if (keep[j]) kept_idx[ki++] = j;

    /* For each pair, if correlation > threshold, remove the one with lower variance */
    for (int a = 0; a < ki; a++) {
        if (!keep[kept_idx[a]]) continue;
        for (int b = a + 1; b < ki; b++) {
            if (!keep[kept_idx[b]]) continue;

            int fa = kept_idx[a], fb = kept_idx[b];

            /* Compute Pearson correlation */
            float sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
            for (int i = 0; i < n; i++) {
                float da = x[i * nf + fa] - mean[fa];
                float db = x[i * nf + fb] - mean[fb];
                sum_ab += da * db;
                sum_a2 += da * da;
                sum_b2 += db * db;
            }
            float denom = sqrtf(sum_a2 * sum_b2);
            float corr = (denom > 1e-10f) ? fabsf(sum_ab / denom) : 0.0f;

            if (corr > corr_threshold) {
                /* Remove the feature with lower variance */
                if (var[fa] < var[fb]) {
                    keep[fa] = 0;
                } else {
                    keep[fb] = 0;
                }
            }
        }
    }

    /* Build final selected list */
    int n_selected = 0;
    for (int j = 0; j < nf; j++) {
        if (keep[j] && n_selected < max_selected) {
            selected[n_selected++] = j;
        }
    }

    free(var); free(mean); free(keep); free(kept_idx);
    return n_selected;
}

/*
 * Apply feature selection: extract only selected columns.
 */
static void apply_feature_selection(const float *x_in, int n, int nf_in,
                                     const int *selected, int n_selected,
                                     float *x_out)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_selected; j++) {
            x_out[i * n_selected + j] = x_in[i * nf_in + selected[j]];
        }
    }
}

/* ========== Modo: train (k-fold) ========== */

/*
 * Encontra os k vizinhos mais proximos de um ponto dentro da mesma classe.
 * Retorna os indices dos vizinhos em neighbors[].
 */
static void find_knn(const float *x, int base, const int *class_indices,
                     int n_class, int nf, int k, int *neighbors)
{
    float *dists = (float *)safe_malloc(n_class * sizeof(float));
    int *order = (int *)safe_malloc(n_class * sizeof(int));

    for (int i = 0; i < n_class; i++) {
        order[i] = i;
        if (class_indices[i] == base) {
            dists[i] = 1e30f;  /* excluir a si mesmo */
            continue;
        }
        float dist = 0.0f;
        for (int f = 0; f < nf; f++) {
            float diff = x[base * nf + f] - x[class_indices[i] * nf + f];
            dist += diff * diff;
        }
        dists[i] = dist;
    }

    /* Selecao parcial: encontrar os k menores */
    for (int i = 0; i < k && i < n_class; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n_class; j++) {
            if (dists[order[j]] < dists[order[min_idx]]) {
                min_idx = j;
            }
        }
        int tmp = order[i];
        order[i] = order[min_idx];
        order[min_idx] = tmp;
        neighbors[i] = class_indices[order[i]];
    }

    free(dists);
    free(order);
}

/*
 * Borderline-SMOTE: Only generates synthetic samples from minority instances
 * that are near the decision boundary (have neighbors from other classes).
 * This produces higher-quality synthetic samples than regular SMOTE.
 */
static void smote_oversample(const float *x_in, const int *y_in, int n_in, int nf,
                             float **x_out, int **y_out, int *n_out)
{
    int k = 5;

    /* Contar por classe */
    int counts[NUM_CLASSES] = {0};
    for (int i = 0; i < n_in; i++) counts[y_in[i]]++;

    int max_count = 0;
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (counts[c] > max_count) max_count = counts[c];
    }

    *n_out = max_count * NUM_CLASSES;
    *x_out = (float *)safe_malloc(*n_out * nf * sizeof(float));
    *y_out = (int *)safe_malloc(*n_out * sizeof(int));

    /* Separar indices por classe */
    int *class_idx[NUM_CLASSES];
    int class_pos[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) {
        class_idx[c] = (int *)safe_malloc(counts[c] * sizeof(int));
        class_pos[c] = 0;
    }
    for (int i = 0; i < n_in; i++) {
        int c = y_in[i];
        class_idx[c][class_pos[c]++] = i;
    }

    /*
     * Step 1: For each sample, find k nearest neighbors (across ALL classes).
     * Classify as DANGER if half or more of neighbors are from other classes.
     * These are the borderline samples.
     */
    int *is_borderline = (int *)safe_calloc(n_in, sizeof(int));

    for (int i = 0; i < n_in; i++) {
        /* Find k nearest neighbors across all samples */
        float *dists = (float *)safe_malloc(n_in * sizeof(float));
        int *order = (int *)safe_malloc(n_in * sizeof(int));

        for (int j = 0; j < n_in; j++) {
            order[j] = j;
            if (j == i) { dists[j] = 1e30f; continue; }
            float dist = 0.0f;
            for (int f = 0; f < nf; f++) {
                float diff = x_in[i * nf + f] - x_in[j * nf + f];
                dist += diff * diff;
            }
            dists[j] = dist;
        }

        /* Partial sort to find k smallest */
        int knn = (k < n_in - 1) ? k : n_in - 1;
        for (int a = 0; a < knn; a++) {
            int min_idx = a;
            for (int b = a + 1; b < n_in; b++) {
                if (dists[order[b]] < dists[order[min_idx]])
                    min_idx = b;
            }
            int tmp = order[a]; order[a] = order[min_idx]; order[min_idx] = tmp;
        }

        /* Count neighbors from other classes */
        int other_class_count = 0;
        for (int a = 0; a < knn; a++) {
            if (y_in[order[a]] != y_in[i]) other_class_count++;
        }

        /* DANGER zone: at least k/2 neighbors are from other classes */
        if (other_class_count >= (knn + 1) / 2 && other_class_count < knn) {
            is_borderline[i] = 1;
        }

        free(dists);
        free(order);
    }

    int out_idx = 0;

    for (int c = 0; c < NUM_CLASSES; c++) {
        int n_class = counts[c];
        int n_needed = max_count;
        int knn = (k < n_class - 1) ? k : n_class - 1;
        if (knn < 1) knn = 1;

        /* Copiar todas as amostras originais */
        for (int i = 0; i < n_class; i++) {
            int src = class_idx[c][i];
            memcpy(&(*x_out)[out_idx * nf], &x_in[src * nf], nf * sizeof(float));
            (*y_out)[out_idx++] = c;
        }

        /* Count borderline samples for this class */
        int n_borderline = 0;
        int *borderline_idx = (int *)safe_malloc(n_class * sizeof(int));
        for (int i = 0; i < n_class; i++) {
            if (is_borderline[class_idx[c][i]]) {
                borderline_idx[n_borderline++] = class_idx[c][i];
            }
        }

        /* If no borderline samples, fall back to all samples */
        int *synth_pool = borderline_idx;
        int synth_pool_size = n_borderline;
        if (n_borderline == 0) {
            synth_pool = class_idx[c];
            synth_pool_size = n_class;
        }

        /* Gerar amostras sinteticas para completar ate max_count */
        int n_synthetic = n_needed - n_class;
        int *neighbors = (int *)safe_malloc(knn * sizeof(int));

        for (int s = 0; s < n_synthetic; s++) {
            /* Choose from borderline samples */
            int base_local = (int)(rng_uniform() * synth_pool_size);
            if (base_local >= synth_pool_size) base_local = synth_pool_size - 1;
            int base_idx = synth_pool[base_local];

            /* Find k nearest neighbors within same class */
            find_knn(x_in, base_idx, class_idx[c], n_class, nf, knn, neighbors);

            /* Choose random neighbor */
            int nn_local = (int)(rng_uniform() * knn);
            if (nn_local >= knn) nn_local = knn - 1;
            int neighbor_idx = neighbors[nn_local];

            /* Interpolate: x_new = x_base + alpha * (x_neighbor - x_base) */
            float alpha = rng_uniform();
            for (int f = 0; f < nf; f++) {
                (*x_out)[out_idx * nf + f] =
                    x_in[base_idx * nf + f] +
                    alpha * (x_in[neighbor_idx * nf + f] - x_in[base_idx * nf + f]);
            }
            (*y_out)[out_idx++] = c;
        }

        free(neighbors);
        free(borderline_idx);
        free(class_idx[c]);
    }

    free(is_borderline);
}

/*
 * Carrega features de um arquivo CSV previamente exportado.
 * Retorna 0 em sucesso, -1 se o arquivo nao existe ou tem erro.
 */
static int features_load_csv(const char *path, FeatureMatrix *fm)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    /* Contar linhas (excluindo header) */
    int n_lines = 0;
    char buf[65536];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return -1; } /* header */
    while (fgets(buf, sizeof(buf), f)) n_lines++;

    if (n_lines == 0) { fclose(f); return -1; }

    fm->count = n_lines;
    fm->num_features = TOTAL_FEATURES;
    fm->features = (float *)safe_malloc(n_lines * TOTAL_FEATURES * sizeof(float));
    fm->labels = (int *)safe_malloc(n_lines * sizeof(int));

    /* Reler */
    rewind(f);
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return -1; } /* skip header */

    for (int i = 0; i < n_lines; i++) {
        if (!fgets(buf, sizeof(buf), f)) { fclose(f); return -1; }
        char *tok = buf;
        for (int j = 0; j < TOTAL_FEATURES; j++) {
            fm->features[i * TOTAL_FEATURES + j] = strtof(tok, &tok);
            if (*tok == ',') tok++;
        }
        fm->labels[i] = (int)strtol(tok, NULL, 10);
    }

    fclose(f);
    return 0;
}

/*
 * Extrai FEATURES_PER_VOWEL features de um buffer float em memoria.
 * Equivalente a extract_vowel_features() de feature_extract.c mas sem I/O.
 */
static void extract_vowel_from_float(const float *samples, int n, int sr, float *out)
{
    /* US-010: Wavelet denoising on a copy before feature extraction */
    float *denoised = (float *)safe_malloc(n * sizeof(float));
    memcpy(denoised, samples, n * sizeof(float));
    dsp_wavelet_denoise(denoised, n, 3);

    float *pre_emph = (float *)safe_malloc(n * sizeof(float));
    memcpy(pre_emph, denoised, n * sizeof(float));
    dsp_pre_emphasis(pre_emph, n, PRE_EMPHASIS_ALPHA);

    int idx = 0;
    TemporalFeatures tf;
    /* Temporal features: original signal (jitter/shimmer are pathological biomarkers) */
    temporal_extract(samples, n, sr, &tf);
    out[idx++] = tf.jitter_local; out[idx++] = tf.jitter_rap;
    out[idx++] = tf.jitter_ppq5; out[idx++] = tf.shimmer_local;
    out[idx++] = tf.shimmer_apq3; out[idx++] = tf.shimmer_apq5;
    out[idx++] = tf.shimmer_apq11; out[idx++] = tf.energy_mean;
    out[idx++] = tf.hnr; out[idx++] = tf.zcr;

    SpectralFeatures sf;
    spectral_extract(pre_emph, n, sr, &sf);
    out[idx++] = sf.f0_mean; out[idx++] = sf.f0_std;
    out[idx++] = sf.formants[0]; out[idx++] = sf.formants[1];
    out[idx++] = sf.formants[2]; out[idx++] = sf.formants[3];
    out[idx++] = sf.spectral_entropy; out[idx++] = sf.spectral_centroid;
    out[idx++] = sf.spectral_rolloff;
    for (int m = 0; m < 13; m++) out[idx++] = sf.mfcc[m];
    /* US-011: Delta e Delta-Delta MFCCs */
    for (int m = 0; m < 13; m++) out[idx++] = sf.delta_mfcc[m];
    for (int m = 0; m < 13; m++) out[idx++] = sf.delta2_mfcc[m];

    WaveletFeatures wf;
    /* Wavelet features: original signal */
    wavelet_extract(samples, n, &wf);
    for (int l = 0; l < WAVELET_LEVELS; l++) out[idx++] = wf.mean[l];
    for (int l = 0; l < WAVELET_LEVELS; l++) out[idx++] = wf.variance[l];
    for (int l = 0; l < WAVELET_LEVELS; l++) out[idx++] = wf.energy[l];

    free(pre_emph);
    free(denoised);
}

/*
 * Aplica 4 tecnicas de augmentacao nas classes minoritarias (Laryngite, Disfonia)
 * do fold de treino. Expande train_x e train_y com as amostras augmentadas.
 * Modifica *x_ptr, *y_ptr, *n_ptr in-place (realoca).
 */
static void augment_fold_training(float **x_ptr, int **y_ptr, int *n_ptr,
                                   int nf, const Dataset *ds,
                                   const int *indices, int n_orig)
{
    /* Contar quantas amostras minoritarias existem */
    int n_minority = 0;
    for (int i = 0; i < n_orig; i++) {
        int cls = ds->patients[indices[i]].class_label;
        if (cls == CLASS_LARYNGITIS || cls == CLASS_DYSPHONIA) n_minority++;
    }
    if (n_minority == 0) return;

    int n_aug_per_sample = 4;
    int n_new = *n_ptr + n_minority * n_aug_per_sample;
    float *new_x = (float *)safe_malloc((size_t)n_new * nf * sizeof(float));
    int   *new_y = (int   *)safe_malloc((size_t)n_new * sizeof(int));

    /* Copiar dados originais */
    memcpy(new_x, *x_ptr, (size_t)(*n_ptr) * nf * sizeof(float));
    memcpy(new_y, *y_ptr, (size_t)(*n_ptr) * sizeof(int));

    int out_idx = *n_ptr;

    for (int i = 0; i < n_orig; i++) {
        int pat_idx = indices[i];
        const Patient *p = &ds->patients[pat_idx];
        if (p->class_label != CLASS_LARYNGITIS && p->class_label != CLASS_DYSPHONIA)
            continue;

        /* Carregar WAVs das 3 vogais */
        WavFile wavs[NUM_VOWELS];
        int wav_ok[NUM_VOWELS];
        for (int v = 0; v < NUM_VOWELS; v++)
            wav_ok[v] = (wav_read(p->vowel_paths[v], &wavs[v]) == 0);

        /* 4 configuracoes de augmentacao */
        for (int aug = 0; aug < n_aug_per_sample; aug++) {
            float *feat_row = &new_x[out_idx * nf];
            memset(feat_row, 0, nf * sizeof(float));

            for (int v = 0; v < NUM_VOWELS; v++) {
                if (!wav_ok[v]) continue;
                int ns = wavs[v].num_samples;
                int sr = wavs[v].sample_rate;
                float *buf = (float *)safe_malloc(ns * sizeof(float));
                memcpy(buf, wavs[v].samples, ns * sizeof(float));

                switch (aug) {
                    case 0: wav_aug_noise(buf, ns, 25.0f); break;
                    case 1: wav_aug_gain(buf, ns,  3.0f); break;
                    case 2: wav_aug_gain(buf, ns, -3.0f); break;
                    case 3: wav_aug_stretch(buf, ns, 1.05f); break;
                }

                extract_vowel_from_float(buf, ns, sr,
                                         &feat_row[v * FEATURES_PER_VOWEL]);
                free(buf);
            }
            new_y[out_idx] = p->class_label;
            out_idx++;
        }

        for (int v = 0; v < NUM_VOWELS; v++)
            if (wav_ok[v]) wav_free(&wavs[v]);
    }

    free(*x_ptr);
    free(*y_ptr);
    *x_ptr = new_x;
    *y_ptr = new_y;
    *n_ptr = out_idx;
}

/*
 * inner_cv_select_thresholds - Seleciona var/corr thresholds via 3-fold CV interno.
 * x_raw:      features brutas (pre-normalizacao) [n x nf]
 * y:          labels [n]
 * outer_fold: indice do fold externo (0..K_FOLDS-1); usado para variar a seed do
 *             shuffle interno, garantindo splits independentes por fold externo.
 * Avalia 9 combinacoes (var x corr), retorna a que maximiza Macro F1 medio no inner val.
 */
static void inner_cv_select_thresholds(const float *x_raw, const int *y,
                                        int n, int nf, int outer_fold,
                                        float *best_var_out, float *best_corr_out)
{
    static const float var_grid[]  = {0.005f, 0.01f,  0.02f};
    static const float corr_grid[] = {0.90f,  0.95f,  0.98f};
    int n_var = 3, n_corr = 3, n_inner = 3;

    /* Shuffled index array for inner split — seed varia por fold externo */
    int *idx = (int *)safe_malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    rng_seed(RANDOM_SEED + 77 + outer_fold * 13);
    rng_shuffle_int(idx, n);

    *best_var_out  = 0.01f;
    *best_corr_out = 0.95f;
    float best_f1 = -1.0f;

    for (int vi = 0; vi < n_var; vi++) {
        for (int ci = 0; ci < n_corr; ci++) {
            float vt = var_grid[vi];
            float ct = corr_grid[ci];
            float f1_sum = 0.0f;

            for (int fi = 0; fi < n_inner; fi++) {
                int val_start     = (fi * n) / n_inner;
                int val_end       = ((fi + 1) * n) / n_inner;
                int n_ival        = val_end - val_start;
                int n_itrain      = n - n_ival;

                float *ix_tr = (float *)safe_malloc(n_itrain * nf * sizeof(float));
                int   *iy_tr = (int   *)safe_malloc(n_itrain * sizeof(int));
                float *ix_vl = (float *)safe_malloc(n_ival   * nf * sizeof(float));
                int   *iy_vl = (int   *)safe_malloc(n_ival   * sizeof(int));

                int ti = 0, vii = 0;
                for (int i = 0; i < n; i++) {
                    int orig = idx[i];
                    if (i >= val_start && i < val_end) {
                        memcpy(&ix_vl[vii * nf], &x_raw[orig * nf], nf * sizeof(float));
                        iy_vl[vii++] = y[orig];
                    } else {
                        memcpy(&ix_tr[ti * nf], &x_raw[orig * nf], nf * sizeof(float));
                        iy_tr[ti++] = y[orig];
                    }
                }

                /* Normalize within inner fold */
                NormParams inorm;
                norm_fit(ix_tr, n_itrain, nf, &inorm);
                norm_transform(ix_tr, n_itrain, &inorm);
                norm_transform(ix_vl, n_ival, &inorm);

                /* Feature selection */
                int *sel   = (int *)safe_malloc(nf * sizeof(int));
                int n_sel  = select_features(ix_tr, n_itrain, nf, vt, ct, sel, nf);

                float *sx_tr = (float *)safe_malloc(n_itrain * n_sel * sizeof(float));
                float *sx_vl = (float *)safe_malloc(n_ival   * n_sel * sizeof(float));
                apply_feature_selection(ix_tr, n_itrain, nf, sel, n_sel, sx_tr);
                apply_feature_selection(ix_vl, n_ival,   nf, sel, n_sel, sx_vl);

                /* Train LR and evaluate */
                int *y_pred = (int *)safe_malloc(n_ival * sizeof(int));
                LRModel lr_inner;
                rng_seed(RANDOM_SEED + fi * 37 + vi * 11 + ci * 5);
                lr_init(&lr_inner, n_sel, NUM_CLASSES);
                lr_train(&lr_inner, sx_tr, iy_tr, n_itrain,
                         sx_vl,   iy_vl, n_ival, n_sel, y_pred);
                lr_free(&lr_inner);

                MetricsResult mr;
                metrics_compute(iy_vl, y_pred, n_ival, &mr);
                f1_sum += mr.macro_f1;

                free(ix_tr); free(iy_tr); free(ix_vl); free(iy_vl);
                free(sel); free(sx_tr); free(sx_vl); free(y_pred);
                norm_free(&inorm);
            }

            float avg_f1 = f1_sum / n_inner;
            if (avg_f1 > best_f1) {
                best_f1       = avg_f1;
                *best_var_out  = vt;
                *best_corr_out = ct;
            }
        }
    }
    free(idx);
}

static int mode_train(const char *base_dir)
{
    log_info("=== MODO: TREINAMENTO COM K-FOLD ===");

    /* Carregar dataset */
    Dataset ds;
    char csv_path[1024];
    snprintf(csv_path, sizeof(csv_path), "%s/%s", base_dir, CSV_METADATA);

    if (dataset_load(base_dir, csv_path, &ds) != 0) {
        log_error("Falha ao carregar dataset");
        return -1;
    }

    /* Tentar carregar features do cache */
    char feat_path[1024];
    snprintf(feat_path, sizeof(feat_path), "%s/features.csv", RESULTS_DIR);

    FeatureMatrix fm;
    if (features_load_csv(feat_path, &fm) == 0 && fm.count == ds.count) {
        log_info("Features carregadas do cache: %s (%d x %d)",
                 feat_path, fm.count, fm.num_features);
    } else {
        /* Extrair features */
        log_info("Extraindo features de %d pacientes...", ds.count);
        if (features_extract_all(&ds, &fm) != 0) {
            log_error("Falha na extracao de features");
            dataset_free(&ds);
            return -1;
        }
        features_export_csv(&fm, feat_path);
    }

    /* K-fold split */
    KFoldSplits splits;
    kfold_split(fm.labels, fm.count, RANDOM_SEED, &splits);

    /* Acumular metricas de todos os folds */
    float acc_sum = 0.0f, macro_f1_sum = 0.0f, weighted_f1_sum = 0.0f;
    int *all_y_true = (int *)safe_malloc(fm.count * sizeof(int));
    int *all_y_pred = (int *)safe_malloc(fm.count * sizeof(int));
    float *all_y_prob = (float *)safe_malloc(fm.count * MLP_OUTPUT_SIZE * sizeof(float));
    int all_count = 0;

    /* Predicoes acumuladas para baselines (majority class, kNN, logistic regression) */
    int *maj_all_pred = (int *)safe_malloc(fm.count * sizeof(int));
    int *knn_all_pred = (int *)safe_malloc(fm.count * sizeof(int));
    int *lr_all_pred  = (int *)safe_malloc(fm.count * sizeof(int));

    /* Acumular importancia de features por fold */
    float *imp_acc_sum = (float *)safe_calloc(TOTAL_FEATURES, sizeof(float));
    float *imp_f1_sum  = (float *)safe_calloc(TOTAL_FEATURES, sizeof(float));
    int   *imp_count   = (int   *)safe_calloc(TOTAL_FEATURES, sizeof(int));

    float best_val_f1 = -1.0f;

    for (int f = 0; f < K_FOLDS; f++) {
        log_info("\n========== FOLD %d/%d ==========", f + 1, K_FOLDS);

        FoldSplit *fold = &splits.folds[f];

        /* Montar arrays de treino e validacao */
        int nf = fm.num_features;
        float *train_x = (float *)safe_malloc(fold->n_train * nf * sizeof(float));
        int *train_y = (int *)safe_malloc(fold->n_train * sizeof(int));
        float *val_x = (float *)safe_malloc(fold->n_val * nf * sizeof(float));
        int *val_y = (int *)safe_malloc(fold->n_val * sizeof(int));

        for (int i = 0; i < fold->n_train; i++) {
            int idx = fold->train_indices[i];
            memcpy(&train_x[i * nf], &fm.features[idx * nf], nf * sizeof(float));
            train_y[i] = fm.labels[idx];
        }
        for (int i = 0; i < fold->n_val; i++) {
            int idx = fold->val_indices[i];
            memcpy(&val_x[i * nf], &fm.features[idx * nf], nf * sizeof(float));
            val_y[i] = fm.labels[idx];
        }

        /* Augmentacao no dominio do audio (somente classes minoritarias no treino) */
        int n_train_aug = fold->n_train;
        augment_fold_training(&train_x, &train_y, &n_train_aug,
                              nf, &ds, fold->train_indices, fold->n_train);

        /* Nested CV: selecionar thresholds de feature selection via 3-fold interno.
         * Usa apenas amostras originais (pre-augmentacao) para evitar que amostras
         * sinteticas bias the threshold selection e reduzir custo computacional. */
        float best_var_thresh, best_corr_thresh;
        inner_cv_select_thresholds(train_x, train_y, fold->n_train, nf, f,
                                   &best_var_thresh, &best_corr_thresh);
        log_info("Fold %d: inner CV selecionou var=%.3f corr=%.2f",
                 f + 1, best_var_thresh, best_corr_thresh);

        /* Normalizar (fit no treino, transform em ambos) */
        NormParams norm;
        norm_fit(train_x, n_train_aug, nf, &norm);
        norm_transform(train_x, n_train_aug, &norm);
        norm_transform(val_x, fold->n_val, &norm);

        /* Feature selection com thresholds escolhidos pelo inner CV */
        int *selected = (int *)safe_malloc(nf * sizeof(int));
        int n_selected = select_features(train_x, n_train_aug, nf,
                                         best_var_thresh, best_corr_thresh,
                                         selected, nf);
        log_info("Fold %d: inner CV var=%.3f corr=%.2f -> n_features=%d",
                 f + 1, best_var_thresh, best_corr_thresh, n_selected);

        float *sel_train_x = (float *)safe_malloc(n_train_aug * n_selected * sizeof(float));
        float *sel_val_x = (float *)safe_malloc(fold->n_val * n_selected * sizeof(float));
        apply_feature_selection(train_x, n_train_aug, nf, selected, n_selected, sel_train_x);
        apply_feature_selection(val_x, fold->n_val, nf, selected, n_selected, sel_val_x);

        /* Salvar indices de features selecionadas para reproducibilidade */
        char sel_path[1024];
        snprintf(sel_path, sizeof(sel_path), "%s/selected_fold%d.bin", MODELS_DIR, f);
        selected_save(sel_path, selected, n_selected);

        if (f == 0) {
            log_info("Feature selection: %d -> %d features", nf, n_selected);
        }

        /* Baselines: majority class, kNN e logistic regression (pre-SMOTE) */
        {
            int base_offset = all_count; /* ainda nao incrementado */
            /* Majority class: prediz sempre classe 0 (Normal) */
            for (int i = 0; i < fold->n_val; i++)
                maj_all_pred[base_offset + i] = 0;
            /* kNN k=5 */
            knn_predict(sel_train_x, train_y, n_train_aug,
                        sel_val_x, fold->n_val, n_selected,
                        5, &knn_all_pred[base_offset]);
            /* Logistic regression (Adam, cosine LR, early stopping) */
            LRModel lr_model;
            rng_seed(RANDOM_SEED + f * 100 + 1);
            lr_init(&lr_model, n_selected, NUM_CLASSES);
            lr_train(&lr_model,
                     sel_train_x, train_y, n_train_aug,
                     sel_val_x,   val_y,   fold->n_val,
                     n_selected, &lr_all_pred[base_offset]);
            lr_free(&lr_model);
        }

        /* Borderline-SMOTE oversampling */
        float *os_train_x; int *os_train_y; int os_n_train;
        smote_oversample(sel_train_x, train_y, n_train_aug, n_selected,
                         &os_train_x, &os_train_y, &os_n_train);

        log_info("Treino: %d -> %d (Borderline-SMOTE), Validacao: %d, Features: %d",
                 n_train_aug, os_n_train, fold->n_val, n_selected);

        MLP net;
        TrainHistory hist;
        rng_seed(RANDOM_SEED + f * 100);
        mlp_init_dynamic(&net, n_selected);
        mlp_train(&net, os_train_x, os_train_y, os_n_train,
                  sel_val_x, val_y, fold->n_val, n_selected, &hist);

        /* Predictions */
        float output[MLP_OUTPUT_SIZE];
        for (int i = 0; i < fold->n_val; i++) {
            mlp_forward(&net, &sel_val_x[i * n_selected], output, 0);

            int pred = 0;
            for (int c = 1; c < MLP_OUTPUT_SIZE; c++) {
                if (output[c] > output[pred]) pred = c;
            }

            all_y_true[all_count] = val_y[i];
            all_y_pred[all_count] = pred;
            for (int c = 0; c < MLP_OUTPUT_SIZE; c++)
                all_y_prob[all_count * MLP_OUTPUT_SIZE + c] = output[c];
            all_count++;
        }

        /* Permutation importance no fold */
        {
            float *fimp_acc = (float *)safe_calloc(n_selected, sizeof(float));
            float *fimp_f1  = (float *)safe_calloc(n_selected, sizeof(float));
            int *fold_true = &all_y_true[all_count - fold->n_val];
            metrics_permutation_importance(sel_val_x, fold_true,
                                           fold->n_val, n_selected,
                                           &net, selected, n_selected,
                                           fimp_acc, fimp_f1);
            for (int j = 0; j < n_selected; j++) {
                int orig = selected[j];
                imp_acc_sum[orig] += fimp_acc[j];
                imp_f1_sum[orig]  += fimp_f1[j];
                imp_count[orig]++;
            }
            free(fimp_acc); free(fimp_f1);
        }

        /* Metricas do fold */
        MetricsResult fold_metrics;
        int *fold_pred = &all_y_pred[all_count - fold->n_val];
        int *fold_true = &all_y_true[all_count - fold->n_val];
        metrics_compute(fold_true, fold_pred, fold->n_val, &fold_metrics);
        metrics_print(&fold_metrics, stderr);

        acc_sum += fold_metrics.accuracy;
        macro_f1_sum += fold_metrics.macro_f1;
        weighted_f1_sum += fold_metrics.weighted_f1;

        /* Salvar modelo do fold */
        char model_path[1024], norm_path[1024];
        snprintf(model_path, sizeof(model_path), "%s/mlp_fold%d.bin", MODELS_DIR, f);
        mlp_save(&net, model_path);

        /* Salvar normalizacao do fold */
        snprintf(norm_path, sizeof(norm_path), "%s/norm_fold%d.bin", MODELS_DIR, f);
        norm_save(&norm, norm_path);

        /* Salvar melhor modelo (baseado no F1) */
        if (fold_metrics.macro_f1 > best_val_f1) {
            best_val_f1 = fold_metrics.macro_f1;
            
            char b_model[1024], b_norm[1024], b_sel[1024];
            snprintf(b_model, sizeof(b_model), "%s/best_model.bin", MODELS_DIR);
            snprintf(b_norm,  sizeof(b_norm),  "%s/best_norm.bin",  MODELS_DIR);
            snprintf(b_sel,   sizeof(b_sel),   "%s/best_selected.bin", MODELS_DIR);
            
            mlp_save(&net, b_model);
            norm_save(&norm, b_norm);
            selected_save(b_sel, selected, n_selected);
            log_info("Fold %d e o melhor ate agora (F1=%.4f). Salvo como 'best'.", f + 1, best_val_f1);
        }

        /* Liberar recursos */
        mlp_free(&net);
        train_history_free(&hist);
        norm_free(&norm);
        free(train_x); free(train_y);
        free(val_x); free(val_y);
        free(sel_train_x); free(sel_val_x);
        free(selected);
        free(os_train_x); free(os_train_y);
    }

    /* Metricas agregadas (todos os folds) */
    log_info("\n========== RESULTADOS AGREGADOS (%d-FOLD) ==========", K_FOLDS);
    log_info("Acuracia media:     %.4f", acc_sum / K_FOLDS);
    log_info("Macro F1 medio:     %.4f", macro_f1_sum / K_FOLDS);
    log_info("Weighted F1 medio:  %.4f", weighted_f1_sum / K_FOLDS);

    /* Metricas globais (todas as predicoes de validacao) */
    MetricsResult global_metrics;
    metrics_compute(all_y_true, all_y_pred, all_count, &global_metrics);
    log_info("\n=== Metricas Globais (todos os folds combinados) ===");
    metrics_print(&global_metrics, stderr);

    /* Exportar metricas */
    char metrics_path[1024];
    snprintf(metrics_path, sizeof(metrics_path), "%s/metrics_global.csv", RESULTS_DIR);
    metrics_export_csv(&global_metrics, metrics_path);

    /* Baselines: metricas globais e tabela comparativa */
    {
        MetricsResult maj_m, knn_m, lr_m;
        metrics_compute(all_y_true, maj_all_pred, all_count, &maj_m);
        metrics_compute(all_y_true, knn_all_pred, all_count, &knn_m);
        metrics_compute(all_y_true, lr_all_pred,  all_count, &lr_m);

        log_info("\n=== Tabela Comparativa de Baselines ===");
        log_info("%-22s %8s %8s %8s %10s %9s",
                 "Method", "Accuracy", "Macro_F1",
                 "F1_Norm", "F1_Laring", "F1_Disf");
        log_info("%-22s %8.4f %8.4f %8.4f %10.4f %9.4f",
                 "MajorityClass",
                 maj_m.accuracy, maj_m.macro_f1,
                 maj_m.f1[0], maj_m.f1[1], maj_m.f1[2]);
        log_info("%-22s %8.4f %8.4f %8.4f %10.4f %9.4f",
                 "kNN(k=5)",
                 knn_m.accuracy, knn_m.macro_f1,
                 knn_m.f1[0], knn_m.f1[1], knn_m.f1[2]);
        log_info("%-22s %8.4f %8.4f %8.4f %10.4f %9.4f",
                 "LogisticRegression",
                 lr_m.accuracy, lr_m.macro_f1,
                 lr_m.f1[0], lr_m.f1[1], lr_m.f1[2]);
        log_info("%-22s %8.4f %8.4f %8.4f %10.4f %9.4f",
                 "MLP(proposed)",
                 global_metrics.accuracy, global_metrics.macro_f1,
                 global_metrics.f1[0], global_metrics.f1[1], global_metrics.f1[2]);

        char bl_path[1024];
        snprintf(bl_path, sizeof(bl_path), "%s/baselines.csv", RESULTS_DIR);
        FILE *bl_f = fopen(bl_path, "w");
        if (bl_f) {
            fprintf(bl_f, "Method,Accuracy,Macro_F1,F1_Normal,F1_Laryngite,F1_Disfonia\n");
            fprintf(bl_f, "MajorityClass,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    maj_m.accuracy, maj_m.macro_f1,
                    maj_m.f1[0], maj_m.f1[1], maj_m.f1[2]);
            fprintf(bl_f, "kNN_k5,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    knn_m.accuracy, knn_m.macro_f1,
                    knn_m.f1[0], knn_m.f1[1], knn_m.f1[2]);
            fprintf(bl_f, "LogisticRegression,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    lr_m.accuracy, lr_m.macro_f1,
                    lr_m.f1[0], lr_m.f1[1], lr_m.f1[2]);
            fprintf(bl_f, "MLP_proposed,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    global_metrics.accuracy, global_metrics.macro_f1,
                    global_metrics.f1[0], global_metrics.f1[1], global_metrics.f1[2]);
            fclose(bl_f);
            log_info("Baselines salvos em %s", bl_path);
        }
    }
    free(maj_all_pred);
    free(knn_all_pred);
    free(lr_all_pred);

    /* Feature importance: escrever CSV e sumarizar por grupo */
    {
        char fi_path[1024];
        snprintf(fi_path, sizeof(fi_path), "%s/feature_importance.csv", RESULTS_DIR);
        FILE *fi_f = fopen(fi_path, "w");
        if (fi_f) {
            fprintf(fi_f, "feature_idx,importance_acc,importance_f1\n");
            for (int j = 0; j < TOTAL_FEATURES; j++) {
                if (imp_count[j] > 0) {
                    fprintf(fi_f, "%d,%.6f,%.6f\n", j,
                            imp_acc_sum[j] / imp_count[j],
                            imp_f1_sum[j]  / imp_count[j]);
                }
            }
            fclose(fi_f);
            log_info("Feature importance salvo em %s", fi_path);
        }

        /* Sumarizar por grupo */
        float g_acc[3] = {0}, g_f1[3] = {0};
        int   g_cnt[3] = {0};
        for (int j = 0; j < TOTAL_FEATURES; j++) {
            if (imp_count[j] == 0) continue;
            int g = (j < 30) ? 0 : (j < 96) ? 1 : 2;
            g_acc[g] += imp_acc_sum[j] / imp_count[j];
            g_f1[g]  += imp_f1_sum[j]  / imp_count[j];
            g_cnt[g]++;
        }
        static const char *gnames[3] = {"Temporal (0-29)", "Spectral (30-95)", "Wavelet (96-149)"};
        log_info("=== Feature Importance por Grupo ===");
        for (int g = 0; g < 3; g++) {
            if (g_cnt[g] > 0)
                log_info("  %-18s acc_drop=%.4f  f1_drop=%.4f  (n=%d)",
                         gnames[g],
                         g_acc[g] / g_cnt[g],
                         g_f1[g]  / g_cnt[g],
                         g_cnt[g]);
        }
    }
    free(imp_acc_sum); free(imp_f1_sum); free(imp_count);

    /* Bootstrap CI sobre todas as predicoes acumuladas */
    ConfidenceInterval ci[CI_N_METRICS];
    metrics_bootstrap_ci(all_y_true, all_y_pred, all_count, 1000, 42, ci);
    log_info("Bootstrap CI 95%% (N=1000, seed=42):");
    log_info("  Accuracy:    %.4f [%.4f, %.4f]", ci[CI_ACCURACY].mean,     ci[CI_ACCURACY].lower,     ci[CI_ACCURACY].upper);
    log_info("  Macro F1:    %.4f [%.4f, %.4f]", ci[CI_MACRO_F1].mean,     ci[CI_MACRO_F1].lower,     ci[CI_MACRO_F1].upper);
    log_info("  F1 Normal:   %.4f [%.4f, %.4f]", ci[CI_F1_NORMAL].mean,    ci[CI_F1_NORMAL].lower,    ci[CI_F1_NORMAL].upper);
    log_info("  F1 Laringite:%.4f [%.4f, %.4f]", ci[CI_F1_LARYNGITE].mean, ci[CI_F1_LARYNGITE].lower, ci[CI_F1_LARYNGITE].upper);
    log_info("  F1 Disfonia: %.4f [%.4f, %.4f]", ci[CI_F1_DISFONIA].mean,  ci[CI_F1_DISFONIA].lower,  ci[CI_F1_DISFONIA].upper);

    /* Anexar secao de CI ao CSV de metricas */
    {
        FILE *mf = fopen(metrics_path, "a");
        if (mf) {
            static const char *ci_names[CI_N_METRICS] = {
                "accuracy", "macro_f1", "f1_normal", "f1_laryngite", "f1_disfonia"
            };
            fprintf(mf, "# bootstrap_ci\nmetric,mean,ci_lower,ci_upper\n");
            for (int m = 0; m < CI_N_METRICS; m++)
                fprintf(mf, "%s,%.6f,%.6f,%.6f\n",
                        ci_names[m], ci[m].mean, ci[m].lower, ci[m].upper);
            fclose(mf);
        }
    }

    /* ROC/AUC e curvas Precision-Recall por classe */
    char roc_path[1024], pr_path[1024];
    snprintf(roc_path, sizeof(roc_path), "%s/roc_curves.csv", RESULTS_DIR);
    snprintf(pr_path,  sizeof(pr_path),  "%s/pr_curves.csv",  RESULTS_DIR);
    metrics_roc_auc(all_y_true, all_y_prob, all_count, NUM_CLASSES,
                    global_metrics.auc, roc_path);
    metrics_pr_curve(all_y_true, all_y_prob, all_count, NUM_CLASSES, pr_path);
    log_info("AUC Normal=%.3f Laryngite=%.3f Disfonia=%.3f",
             global_metrics.auc[0], global_metrics.auc[1], global_metrics.auc[2]);

    free(all_y_true);
    free(all_y_pred);
    free(all_y_prob);
    kfold_free(&splits);
    features_free(&fm);
    dataset_free(&ds);
    return 0;
}

/* ========== Modo: full ========== */

static int mode_full(const char *base_dir)
{
    return mode_train(base_dir);
}

/* ========== Main ========== */

static void print_usage(const char *prog)
{
    fprintf(stderr, "Uso: %s <modo> [diretorio_base]\n", prog);
    fprintf(stderr, "\nModos:\n");
    fprintf(stderr, "  extract  - Extrai features e salva CSV\n");
    fprintf(stderr, "  train    - Treina MLP com %d-fold cross-validation\n", K_FOLDS);
    fprintf(stderr, "  full     - Pipeline completa (extract + train)\n");
    fprintf(stderr, "\nDiretorio base padrao: .\n");
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    const char *base_dir = (argc >= 3) ? argv[2] : ".";

    log_set_level(LOG_INFO);
    log_info("Classificador de Anomalias Vocais - MLP em C");
    log_info("Modo: %s | Diretorio: %s | Seed: %d", mode, base_dir, RANDOM_SEED);

    double t_start = timer_now();
    int result;

    if (strcmp(mode, "extract") == 0) {
        result = mode_extract(base_dir);
    } else if (strcmp(mode, "train") == 0) {
        result = mode_train(base_dir);
    } else if (strcmp(mode, "full") == 0) {
        result = mode_full(base_dir);
    } else {
        fprintf(stderr, "Modo desconhecido: %s\n", mode);
        print_usage(argv[0]);
        return 1;
    }

    double elapsed = timer_now() - t_start;
    log_info("Tempo total: %.1f segundos", elapsed);

    return (result == 0) ? 0 : 1;
}
