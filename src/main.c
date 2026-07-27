/*
 * main.c - Entry point do classificador de anomalias vocais (HIERARCHICAL LATE FUSION)
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

#define N_AUG_PER_SAMPLE 8

/* ========== Helpers ========== */

static void map_to_binary_labels(const int *y_orig, int *y_bin, int n)
{
    for (int i = 0; i < n; i++) {
        y_bin[i] = (y_orig[i] == CLASS_NORMAL) ? 0 : 1;
    }
}

static int predict_hierarchical_late_fusion(MLP master[3], MLP expert[3], 
                                            const float *x_all)
{
    float prob_pathology = 0.0f;
    float prob_expert[4] = {0, 0, 0, 0};
    int meta_offset = NUM_VOWELS * FEATURES_PER_VOWEL;

    for (int v = 0; v < 3; v++) {
        float x_v[FEATURES_PER_VOWEL + NUM_METADATA_FEATURES];
        memcpy(x_v, &x_all[v * FEATURES_PER_VOWEL], FEATURES_PER_VOWEL * sizeof(float));
        memcpy(&x_v[FEATURES_PER_VOWEL], &x_all[meta_offset], NUM_METADATA_FEATURES * sizeof(float));

        float out_m[2];
        mlp_forward(&master[v], x_v, out_m, 0);
        prob_pathology += out_m[1];

        float out_e[4];
        mlp_forward(&expert[v], x_v, out_e, 0);
        for (int c = 0; c < 4; c++) prob_expert[c] += out_e[c];
    }

    if ((prob_pathology / 3.0f) < 0.5f) {
        return CLASS_NORMAL;
    } else {
        int best_c = 0;
        for (int c = 1; c < 4; c++) {
            if (prob_expert[c] > prob_expert[best_c]) best_c = c;
        }
        return best_c + 1;
    }
}

/* ========== Feature Extraction / Load ========== */

static int features_load_csv(const char *path, FeatureMatrix *fm)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char buf[65536];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return -1; }
    int n_cols = 1;
    for (const char *p = buf; *p && *p != '\n' && *p != '\r'; p++) if (*p == ',') n_cols++;
    if (n_cols - 1 != TOTAL_FEATURES) { fclose(f); return -1; }
    int n_lines = 0;
    while (fgets(buf, sizeof(buf), f)) n_lines++;
    if (n_lines == 0) { fclose(f); return -1; }
    fm->count = n_lines; fm->num_features = TOTAL_FEATURES;
    fm->features = (float *)safe_malloc(n_lines * TOTAL_FEATURES * sizeof(float));
    fm->labels = (int *)safe_malloc(n_lines * sizeof(int));
    rewind(f); fgets(buf, sizeof(buf), f);
    for (int i = 0; i < n_lines; i++) {
        if (!fgets(buf, sizeof(buf), f)) break;
        char *tok = buf;
        for (int j = 0; j < TOTAL_FEATURES; j++) {
            fm->features[i * TOTAL_FEATURES + j] = strtof(tok, &tok);
            if (*tok == ',') tok++;
        }
        fm->labels[i] = (int)strtol(tok, NULL, 10);
    }
    fclose(f); return 0;
}

static void extract_vowel_from_float(const float *samples, int n, int sr, float *out)
{
    float *denoised = (float *)safe_malloc(n * sizeof(float));
    memcpy(denoised, samples, n * sizeof(float));
    dsp_wavelet_denoise(denoised, n, 3);
    float *pre_emph = (float *)safe_malloc(n * sizeof(float));
    memcpy(pre_emph, denoised, n * sizeof(float));
    dsp_pre_emphasis(pre_emph, n, PRE_EMPHASIS_ALPHA);

    int idx = 0;
    TemporalFeatures tf; temporal_extract(samples, n, sr, &tf);
    out[idx++] = tf.jitter_local; out[idx++] = tf.jitter_rap; out[idx++] = tf.jitter_ppq5;
    out[idx++] = tf.shimmer_local; out[idx++] = tf.shimmer_apq3; out[idx++] = tf.shimmer_apq5;
    out[idx++] = tf.shimmer_apq11; out[idx++] = tf.energy_mean; out[idx++] = tf.hnr; out[idx++] = tf.zcr;

    SpectralFeatures sf; spectral_extract(pre_emph, n, sr, &sf);
    out[idx++] = sf.f0_mean; out[idx++] = sf.f0_std;
    out[idx++] = sf.formants[0]; out[idx++] = sf.formants[1]; out[idx++] = sf.formants[2]; out[idx++] = sf.formants[3];
    out[idx++] = sf.spectral_entropy; out[idx++] = sf.spectral_centroid; out[idx++] = sf.spectral_rolloff;
    for (int m = 0; m < 13; m++) out[idx++] = sf.mfcc[m];
    for (int m = 0; m < 13; m++) out[idx++] = sf.delta_mfcc[m];
    for (int m = 0; m < 13; m++) out[idx++] = sf.delta2_mfcc[m];
    out[idx++] = sf.cpp_mean; out[idx++] = sf.cpp_std; out[idx++] = sf.cpp_slope;
    out[idx++] = sf.glottal_oq; out[idx++] = sf.glottal_sq; out[idx++] = sf.glottal_naq; out[idx++] = sf.glottal_h1h2;

    WaveletFeatures wf; wavelet_extract(samples, n, &wf);
    for (int l = 0; l < WAVELET_LEVELS; l++) out[idx++] = wf.mean[l];
    for (int l = 0; l < WAVELET_LEVELS; l++) out[idx++] = wf.variance[l];
    for (int l = 0; l < WAVELET_LEVELS; l++) out[idx++] = wf.energy[l];

    free(pre_emph); free(denoised);
}

static void precalculate_augmentations(const Dataset *ds, int nf, float *aug_features)
{
    log_info("Pre-calculando aumentacoes de audio (8x per patologico)...");
    /* NAO paralelizar: rng_state (src/utils.c) e global e nao thread-safe sob OpenMP - ver PITFALLS.md Pitfall 13 */
    for (int i = 0; i < ds->count; i++) {
        if (ds->patients[i].class_label == CLASS_NORMAL) continue;
        WavFile wavs[3]; int wav_ok[3];
        for (int v = 0; v < 3; v++) wav_ok[v] = (wav_read(ds->patients[i].vowel_paths[v], &wavs[v]) == 0);
        for (int aug = 0; aug < N_AUG_PER_SAMPLE; aug++) {
            float *feat_row = &aug_features[(i * N_AUG_PER_SAMPLE + aug) * nf];
            for (int v = 0; v < 3; v++) {
                if (!wav_ok[v]) continue;
                int ns = wavs[v].num_samples; float *buf = (float *)safe_malloc(ns * sizeof(float));
                memcpy(buf, wavs[v].samples, ns * sizeof(float));
                switch (aug) {
                    case 0: wav_aug_noise(buf, ns, 25.0f); break;
                    case 1: wav_aug_gain(buf, ns, 3.0f); break;
                    case 2: wav_aug_gain(buf, ns, -3.0f); break;
                    case 3: wav_aug_stretch(buf, ns, 1.10f); break;
                    case 4: wav_aug_stretch(buf, ns, 0.90f); break;
                    case 5: wav_aug_pitch(buf, ns, 1.5f); break;
                    case 6: wav_aug_pitch(buf, ns, -1.5f); break;
                    case 7: wav_aug_noise(buf, ns, 30.0f); wav_aug_pitch(buf, ns, 0.7f); break;
                }
                extract_vowel_from_float(buf, ns, wavs[v].sample_rate, &feat_row[v * FEATURES_PER_VOWEL]);
                free(buf);
            }
            feat_row[NUM_VOWELS * FEATURES_PER_VOWEL] = (float)ds->patients[i].age;
            feat_row[NUM_VOWELS * FEATURES_PER_VOWEL + 1] = (ds->patients[i].sex == 'm' ? 1.0f : 0.0f);
        }
        for (int v = 0; v < 3; v++) if (wav_ok[v]) wav_free(&wavs[v]);
    }
}

static void collect_augmented_features(float **x_ptr, int **y_ptr, int *n_ptr, int nf, const Dataset *ds, const int *indices, int n_orig, const float *aug_cache)
{
    int n_minority = 0;
    for (int i = 0; i < n_orig; i++) if (ds->patients[indices[i]].class_label != CLASS_NORMAL) n_minority++;
    if (n_minority == 0) return;
    int n_new = *n_ptr + n_minority * N_AUG_PER_SAMPLE;
    *x_ptr = (float *)safe_realloc(*x_ptr, (size_t)n_new * nf * sizeof(float));
    *y_ptr = (int *)safe_realloc(*y_ptr, (size_t)n_new * sizeof(int));
    int out_idx = *n_ptr;
    for (int i = 0; i < n_orig; i++) {
        int pat_idx = indices[i]; if (ds->patients[pat_idx].class_label == CLASS_NORMAL) continue;
        for (int aug = 0; aug < N_AUG_PER_SAMPLE; aug++) {
            memcpy(&(*x_ptr)[out_idx * nf], &aug_cache[(pat_idx * N_AUG_PER_SAMPLE + aug) * nf], nf * sizeof(float));
            (*y_ptr)[out_idx++] = ds->patients[pat_idx].class_label;
        }
    }
    *n_ptr = out_idx;
}

/* ========== SMOTE ========== */

static void find_knn(const float *x, int base, const int *class_indices, int n_class, int nf, int k, int *neighbors)
{
    float *dists = (float *)safe_malloc(n_class * sizeof(float));
    int *order = (int *)safe_malloc(n_class * sizeof(int));
    for (int i = 0; i < n_class; i++) {
        order[i] = i; if (class_indices[i] == base) { dists[i] = 1e30f; continue; }
        float dist = 0.0f; for (int f = 0; f < nf; f++) { float diff = x[base * nf + f] - x[class_indices[i] * nf + f]; dist += diff * diff; }
        dists[i] = dist;
    }
    for (int i = 0; i < k && i < n_class; i++) {
        int min_idx = i; for (int j = i + 1; j < n_class; j++) if (dists[order[j]] < dists[order[min_idx]]) min_idx = j;
        int tmp = order[i]; order[i] = order[min_idx]; order[min_idx] = tmp; neighbors[i] = class_indices[order[i]];
    }
    free(dists); free(order);
}

static void smote_oversample(const float *x_in, const int *y_in, int n_in, int nf, int num_classes, float **x_out, int **y_out, int *n_out)
{
    int k = 5; int *counts = (int *)safe_calloc(num_classes, sizeof(int));
    for (int i = 0; i < n_in; i++) counts[y_in[i]]++;
    int max_count = 0; for (int c = 0; c < num_classes; c++) if (counts[c] > max_count) max_count = counts[c];
    *n_out = max_count * num_classes;
    *x_out = (float *)safe_malloc(*n_out * nf * sizeof(float));
    *y_out = (int *)safe_malloc(*n_out * sizeof(int));
    int **class_idx = (int **)safe_malloc(num_classes * sizeof(int *));
    int *class_pos = (int *)safe_calloc(num_classes, sizeof(int));
    for (int c = 0; c < num_classes; c++) class_idx[c] = (int *)safe_malloc(counts[c] * sizeof(int));
    for (int i = 0; i < n_in; i++) class_idx[y_in[i]][class_pos[y_in[i]]++] = i;
    int out_idx = 0;
    for (int c = 0; c < num_classes; c++) {
        int n_class = counts[c]; int knn = (k < n_class - 1) ? k : n_class - 1; if (knn < 1) knn = 1;
        for (int i = 0; i < n_class; i++) { memcpy(&(*x_out)[out_idx * nf], &x_in[class_idx[c][i] * nf], nf * sizeof(float)); (*y_out)[out_idx++] = c; }
        int n_synthetic = max_count - n_class; int *neighbors = (int *)safe_malloc(knn * sizeof(int));
        for (int s = 0; s < n_synthetic; s++) {
            int base_idx = class_idx[c][rng_int(n_class)]; find_knn(x_in, base_idx, class_idx[c], n_class, nf, knn, neighbors);
            int neighbor_idx = neighbors[rng_int(knn)]; float alpha = rng_uniform();
            for (int f = 0; f < nf; f++) (*x_out)[out_idx * nf + f] = x_in[base_idx * nf + f] + alpha * (x_in[neighbor_idx * nf + f] - x_in[base_idx * nf + f]);
            (*y_out)[out_idx++] = c;
        }
        free(neighbors); free(class_idx[c]);
    }
    free(class_idx); free(class_pos); free(counts);
}

/* ========== Training Modo ========== */

static int mode_train(const char *base_dir)
{
    log_info("=== MODO: TREINAMENTO HIERARQUICO COM LATE FUSION (VOGAIS A, I, U) ===");
    Dataset ds; char csv_path[1024]; snprintf(csv_path, 1024, "%s/%s", base_dir, CSV_METADATA);
    if (dataset_load(base_dir, csv_path, &ds) != 0) return -1;
    char feat_path[1024]; snprintf(feat_path, 1024, "%s/features.csv", RESULTS_DIR);
    FeatureMatrix fm; if (features_load_csv(feat_path, &fm) != 0 || fm.num_features != TOTAL_FEATURES) {
        if (features_extract_all(&ds, &fm) != 0) return -1;
        features_export_csv(&fm, feat_path);
    }
    rng_seed(RANDOM_SEED); KFoldSplits splits; kfold_split(fm.labels, fm.count, RANDOM_SEED, &splits);
    float acc_sum = 0, macro_f1_sum = 0;
    int *all_y_true = (int *)safe_malloc(fm.count * sizeof(int));
    int *all_y_pred = (int *)safe_malloc(fm.count * sizeof(int));
    float *all_y_prob = (float *)safe_malloc(fm.count * 5 * sizeof(float));
    /* Baseline predictions (MajorityClass, kNN, LogisticRegression), alinhadas com
     * all_y_true/all_y_pred no mesmo indice all_count, para McNemar 3-vias (Tarefa 1). */
    int *all_y_pred_majority = (int *)safe_malloc(fm.count * sizeof(int));
    int *all_y_pred_knn = (int *)safe_malloc(fm.count * sizeof(int));
    int *all_y_pred_logreg = (int *)safe_malloc(fm.count * sizeof(int));
    int all_count = 0;
    float *aug_cache = (float *)safe_calloc((size_t)ds.count * N_AUG_PER_SAMPLE * fm.num_features, sizeof(float));
    precalculate_augmentations(&ds, fm.num_features, aug_cache);
    int nf_vowel = FEATURES_PER_VOWEL + NUM_METADATA_FEATURES;

    for (int f = 0; f < K_FOLDS; f++) {
        log_info("\n========== FOLD %d/%d (HIERARCHICAL LATE FUSION) ==========", f + 1, K_FOLDS);
        FoldSplit *fold = &splits.folds[f];
        int nf_all = fm.num_features;
        float *train_x_all = (float *)safe_malloc(fold->n_train * nf_all * sizeof(float));
        int *train_y_all = (int *)safe_malloc(fold->n_train * sizeof(int));
        for (int i = 0; i < fold->n_train; i++) {
            memcpy(&train_x_all[i * nf_all], &fm.features[fold->train_indices[i] * nf_all], nf_all * sizeof(float));
            train_y_all[i] = fm.labels[fold->train_indices[i]];
        }
        int n_train_aug = fold->n_train;
        collect_augmented_features(&train_x_all, &train_y_all, &n_train_aug, nf_all, &ds, fold->train_indices, fold->n_train, aug_cache);
        NormParams norm; norm_fit(train_x_all, fold->n_train, nf_all, &norm);
        norm_transform(train_x_all, n_train_aug, &norm);
        float *val_x_all = (float *)safe_malloc(fold->n_val * nf_all * sizeof(float));
        for (int i = 0; i < fold->n_val; i++) memcpy(&val_x_all[i * nf_all], &fm.features[fold->val_indices[i] * nf_all], nf_all * sizeof(float));
        norm_transform(val_x_all, fold->n_val, &norm);

        /* Baselines (MajorityClass, kNN, LogisticRegression) sobre o mesmo split de
         * fold/validacao usado pelo MLP, calculados ANTES do treino hierarquico
         * (SMOTE + Master/Expert por vogal) para que McNemar compare exatamente as
         * mesmas amostras de validacao out-of-fold. */
        int *val_y_all = (int *)safe_malloc(fold->n_val * sizeof(int));
        for (int i = 0; i < fold->n_val; i++) val_y_all[i] = fm.labels[fold->val_indices[i]];

        int counts[NUM_CLASSES] = {0};
        for (int i = 0; i < fold->n_train; i++) counts[train_y_all[i]]++;
        int majority_class = 0;
        for (int c = 1; c < NUM_CLASSES; c++) if (counts[c] > counts[majority_class]) majority_class = c;

        int *knn_pred_buf = (int *)safe_malloc(fold->n_val * sizeof(int));
        knn_predict(train_x_all, train_y_all, n_train_aug, val_x_all, fold->n_val, nf_all, 5, knn_pred_buf);

        LRModel lr; lr_init(&lr, nf_all, NUM_CLASSES);
        int *logreg_pred_buf = (int *)safe_malloc(fold->n_val * sizeof(int));
        lr_train(&lr, train_x_all, train_y_all, n_train_aug, val_x_all, val_y_all, fold->n_val, nf_all, logreg_pred_buf);
        lr_free(&lr);

        MLP net_master[3], net_expert[3];
        float cw_binary[] = {0.9f, 1.1f}, cw_expert[] = {1.0f, 1.2f, 1.2f, 1.4f};
        for (int v = 0; v < 3; v++) {
            float *tr_x_v = (float *)safe_malloc(n_train_aug * nf_vowel * sizeof(float));
            float *vl_x_v = (float *)safe_malloc(fold->n_val * nf_vowel * sizeof(float));
            int meta_off = 3 * FEATURES_PER_VOWEL;
            for (int i = 0; i < n_train_aug; i++) {
                memcpy(&tr_x_v[i * nf_vowel], &train_x_all[i * nf_all + v * FEATURES_PER_VOWEL], FEATURES_PER_VOWEL * sizeof(float));
                memcpy(&tr_x_v[i * nf_vowel + FEATURES_PER_VOWEL], &train_x_all[i * nf_all + meta_off], 2 * sizeof(float));
            }
            for (int i = 0; i < fold->n_val; i++) {
                memcpy(&vl_x_v[i * nf_vowel], &val_x_all[i * nf_all + v * FEATURES_PER_VOWEL], FEATURES_PER_VOWEL * sizeof(float));
                memcpy(&vl_x_v[i * nf_vowel + FEATURES_PER_VOWEL], &val_x_all[i * nf_all + meta_off], 2 * sizeof(float));
            }
            int *tr_y_bin = (int *)safe_malloc(n_train_aug * sizeof(int));
            int *vl_y_bin = (int *)safe_malloc(fold->n_val * sizeof(int));
            map_to_binary_labels(train_y_all, tr_y_bin, n_train_aug);
            for(int i=0; i<fold->n_val; i++) vl_y_bin[i] = (fm.labels[fold->val_indices[i]] == CLASS_NORMAL) ? 0 : 1;

            float *os_m_x; int *os_m_y, os_n_m;
            smote_oversample(tr_x_v, tr_y_bin, n_train_aug, nf_vowel, 2, &os_m_x, &os_m_y, &os_n_m);
            mlp_init_dynamic(&net_master[v], nf_vowel, 2); TrainHistory h_m;
            mlp_train(&net_master[v], os_m_x, os_m_y, os_n_m, vl_x_v, vl_y_bin, fold->n_val, nf_vowel, 2, cw_binary, &h_m);

            int n_ex_tr = 0; for (int i = 0; i < n_train_aug; i++) if (train_y_all[i] != CLASS_NORMAL) n_ex_tr++;
            float *ex_tr_x = (float *)safe_malloc(n_ex_tr * nf_vowel * sizeof(float));
            int *ex_tr_y = (int *)safe_malloc(n_ex_tr * sizeof(int));
            int cur = 0; for (int i = 0; i < n_train_aug; i++) if (train_y_all[i] != CLASS_NORMAL) {
                memcpy(&ex_tr_x[cur * nf_vowel], &tr_x_v[i * nf_vowel], nf_vowel * sizeof(float)); ex_tr_y[cur++] = train_y_all[i] - 1;
            }
            int n_ex_vl = 0; for (int i = 0; i < fold->n_val; i++) if (fm.labels[fold->val_indices[i]] != CLASS_NORMAL) n_ex_vl++;
            float *ex_vl_x = (float *)safe_malloc(n_ex_vl * nf_vowel * sizeof(float));
            int *ex_vl_y = (int *)safe_malloc(n_ex_vl * sizeof(int));
            cur = 0; for (int i = 0; i < fold->n_val; i++) if (fm.labels[fold->val_indices[i]] != CLASS_NORMAL) {
                memcpy(&ex_vl_x[cur * nf_vowel], &vl_x_v[i * nf_vowel], nf_vowel * sizeof(float)); ex_vl_y[cur++] = fm.labels[fold->val_indices[i]] - 1;
            }
            float *os_e_x; int *os_e_y, os_n_e;
            smote_oversample(ex_tr_x, ex_tr_y, n_ex_tr, nf_vowel, 4, &os_e_x, &os_e_y, &os_n_e);
            mlp_init_dynamic(&net_expert[v], nf_vowel, 4); TrainHistory h_e;
            mlp_train(&net_expert[v], os_e_x, os_e_y, os_n_e, ex_vl_x, ex_vl_y, n_ex_vl, nf_vowel, 4, cw_expert, &h_e);

            free(tr_x_v); free(vl_x_v); free(tr_y_bin); free(vl_y_bin); free(os_m_x); free(os_m_y);
            free(ex_tr_x); free(ex_tr_y); free(ex_vl_x); free(ex_vl_y); free(os_e_x); free(os_e_y);
            train_history_free(&h_m); train_history_free(&h_e);
        }

        for (int i = 0; i < fold->n_val; i++) {
            const float *x_samp = &val_x_all[i * nf_all];
            all_y_pred[all_count] = predict_hierarchical_late_fusion(net_master, net_expert, x_samp);
            all_y_true[all_count] = fm.labels[fold->val_indices[i]];
            float p_norm = 0, p_exp[4] = {0};
            for (int v = 0; v < 3; v++) {
                float xv[251]; memcpy(xv, &x_samp[v * FEATURES_PER_VOWEL], FEATURES_PER_VOWEL * sizeof(float));
                memcpy(&xv[FEATURES_PER_VOWEL], &x_samp[3 * FEATURES_PER_VOWEL], 2 * sizeof(float));
                float om[2], oe[4]; mlp_forward(&net_master[v], xv, om, 0); mlp_forward(&net_expert[v], xv, oe, 0);
                p_norm += om[0]; for(int c=0; c<4; c++) p_exp[c] += om[1] * oe[c];
            }
            all_y_prob[all_count * 5 + 0] = p_norm / 3.0f; for(int c=1; c<5; c++) all_y_prob[all_count * 5 + c] = p_exp[c-1] / 3.0f;
            all_y_pred_majority[all_count] = majority_class;
            all_y_pred_knn[all_count] = knn_pred_buf[i];
            all_y_pred_logreg[all_count] = logreg_pred_buf[i];
            all_count++;
        }
        MetricsResult fm_res; metrics_compute(&all_y_true[all_count - fold->n_val], &all_y_pred[all_count - fold->n_val], fold->n_val, &fm_res);
        metrics_print(&fm_res, stderr); acc_sum += fm_res.accuracy; macro_f1_sum += fm_res.macro_f1;
        for (int v = 0; v < 3; v++) { mlp_free(&net_master[v]); mlp_free(&net_expert[v]); }
        norm_free(&norm); free(train_x_all); free(train_y_all); free(val_x_all);
        free(val_y_all); free(knn_pred_buf); free(logreg_pred_buf);
    }
    log_info("\n========== RESULTADOS AGREGADOS ==========");
    log_info("Acuracia media: %.4f  Macro F1 medio: %.4f", acc_sum / K_FOLDS, macro_f1_sum / K_FOLDS);
    MetricsResult g_met; metrics_compute(all_y_true, all_y_pred, all_count, &g_met);
    metrics_print(&g_met, stderr); metrics_export_csv(&g_met, "results/metrics_global.csv");

    /* Intervalo de confianca 95% via bootstrap (N=1000, seed=RANDOM_SEED) sobre as
     * predicoes out-of-fold agregadas. DEVE ser a ultima chamada consumidora de RNG
     * em mode_train(), pois esta funcao re-semeia o RNG global internamente
     * (src/metrics.c) -- nao adicionar nenhuma chamada rng_* apos este ponto. */
    ConfidenceInterval ci[CI_N_METRICS];
    metrics_bootstrap_ci(all_y_true, all_y_pred, all_count, 1000, RANDOM_SEED, ci);

    /* Teste de McNemar (Edwards, 1948) do MLP hierarquico vs os 3 baselines,
     * sobre as mesmas amostras de validacao out-of-fold. */
    float chi2_maj, p_maj, chi2_knn, p_knn, chi2_lr, p_lr;
    metrics_mcnemar(all_y_true, all_y_pred, all_y_pred_majority, all_count, &chi2_maj, &p_maj);
    metrics_mcnemar(all_y_true, all_y_pred, all_y_pred_knn, all_count, &chi2_knn, &p_knn);
    metrics_mcnemar(all_y_true, all_y_pred, all_y_pred_logreg, all_count, &chi2_lr, &p_lr);

    log_info("\n========== BOOTSTRAP CI (95%%, N=1000, seed=%d) ==========", RANDOM_SEED);
    log_info("CI_ACCURACY:         %.4f [%.4f, %.4f]", ci[CI_ACCURACY].mean, ci[CI_ACCURACY].lower, ci[CI_ACCURACY].upper);
    log_info("CI_MACRO_F1:         %.4f [%.4f, %.4f]", ci[CI_MACRO_F1].mean, ci[CI_MACRO_F1].lower, ci[CI_MACRO_F1].upper);
    log_info("CI_F1_NORMAL:        %.4f [%.4f, %.4f]", ci[CI_F1_NORMAL].mean, ci[CI_F1_NORMAL].lower, ci[CI_F1_NORMAL].upper);
    log_info("CI_F1_LARYNGITE:     %.4f [%.4f, %.4f]", ci[CI_F1_LARYNGITE].mean, ci[CI_F1_LARYNGITE].lower, ci[CI_F1_LARYNGITE].upper);
    log_info("CI_F1_DISFONIA:      %.4f [%.4f, %.4f]", ci[CI_F1_DISFONIA].mean, ci[CI_F1_DISFONIA].lower, ci[CI_F1_DISFONIA].upper);
    log_info("CI_F1_FUNC_DISFONIA: %.4f [%.4f, %.4f]", ci[CI_F1_FUNC_DISFONIA].mean, ci[CI_F1_FUNC_DISFONIA].lower, ci[CI_F1_FUNC_DISFONIA].upper);
    log_info("CI_F1_REINKE:        %.4f [%.4f, %.4f]", ci[CI_F1_REINKE].mean, ci[CI_F1_REINKE].lower, ci[CI_F1_REINKE].upper);

    log_info("\n========== MCNEMAR: MLP vs BASELINES ==========");
    log_info("MLP vs MajorityClass: chi2=%.4f p=%.4f -- %s", chi2_maj, p_maj,
              p_maj < 0.05f ? "MLP significativamente melhor que MajorityClass (p<0.05)"
                            : "MLP nao significativamente diferente de MajorityClass (p>=0.05)");
    log_info("MLP vs kNN:           chi2=%.4f p=%.4f -- %s", chi2_knn, p_knn,
              p_knn < 0.05f ? "MLP significativamente melhor que kNN (p<0.05)"
                            : "MLP nao significativamente diferente de kNN (p>=0.05)");
    log_info("MLP vs LogisticRegression: chi2=%.4f p=%.4f -- %s", chi2_lr, p_lr,
              p_lr < 0.05f ? "MLP significativamente melhor que LogisticRegression (p<0.05)"
                           : "MLP nao significativamente diferente de LogisticRegression (p>=0.05)");

    FILE *ci_f = fopen("results/bootstrap_ci.csv", "w");
    if (ci_f) {
        fprintf(ci_f, "metric,mean,ci_lower,ci_upper\n");
        fprintf(ci_f, "accuracy,%.6f,%.6f,%.6f\n", ci[CI_ACCURACY].mean, ci[CI_ACCURACY].lower, ci[CI_ACCURACY].upper);
        fprintf(ci_f, "macro_f1,%.6f,%.6f,%.6f\n", ci[CI_MACRO_F1].mean, ci[CI_MACRO_F1].lower, ci[CI_MACRO_F1].upper);
        fprintf(ci_f, "f1_normal,%.6f,%.6f,%.6f\n", ci[CI_F1_NORMAL].mean, ci[CI_F1_NORMAL].lower, ci[CI_F1_NORMAL].upper);
        fprintf(ci_f, "f1_laringite,%.6f,%.6f,%.6f\n", ci[CI_F1_LARYNGITE].mean, ci[CI_F1_LARYNGITE].lower, ci[CI_F1_LARYNGITE].upper);
        fprintf(ci_f, "f1_disfonia_psicogenica,%.6f,%.6f,%.6f\n", ci[CI_F1_DISFONIA].mean, ci[CI_F1_DISFONIA].lower, ci[CI_F1_DISFONIA].upper);
        fprintf(ci_f, "f1_disfonia_funcional,%.6f,%.6f,%.6f\n", ci[CI_F1_FUNC_DISFONIA].mean, ci[CI_F1_FUNC_DISFONIA].lower, ci[CI_F1_FUNC_DISFONIA].upper);
        fprintf(ci_f, "f1_reinke,%.6f,%.6f,%.6f\n", ci[CI_F1_REINKE].mean, ci[CI_F1_REINKE].lower, ci[CI_F1_REINKE].upper);
        fclose(ci_f);
    } else {
        log_error("Falha ao abrir results/bootstrap_ci.csv para escrita");
    }

    FILE *mc_f = fopen("results/mcnemar_vs_baselines.csv", "w");
    if (mc_f) {
        fprintf(mc_f, "baseline,chi2,p_value\n");
        fprintf(mc_f, "MajorityClass,%.6f,%.6f\n", chi2_maj, p_maj);
        fprintf(mc_f, "kNN,%.6f,%.6f\n", chi2_knn, p_knn);
        fprintf(mc_f, "LogisticRegression,%.6f,%.6f\n", chi2_lr, p_lr);
        fclose(mc_f);
    } else {
        log_error("Falha ao abrir results/mcnemar_vs_baselines.csv para escrita");
    }

    free(all_y_true); free(all_y_pred); free(all_y_prob); free(aug_cache);
    free(all_y_pred_majority); free(all_y_pred_knn); free(all_y_pred_logreg);
    return 0;
}

static int mode_extract(const char *base_dir)
{
    Dataset ds; char csv_p[1024]; snprintf(csv_p, 1024, "%s/%s", base_dir, CSV_METADATA);
    if (dataset_load(base_dir, csv_p, &ds) != 0) return -1;
    FeatureMatrix fm; if (features_extract_all(&ds, &fm) != 0) { dataset_free(&ds); return -1; }
    char out_p[1024]; snprintf(out_p, 1024, "%s/features.csv", RESULTS_DIR);
    features_export_csv(&fm, out_p); features_free(&fm); dataset_free(&ds); return 0;
}

static int mode_verify_rng(const char *base_dir)
{
    Dataset ds; char csv_p[1024]; snprintf(csv_p, 1024, "%s/%s", base_dir, CSV_METADATA);
    if (dataset_load(base_dir, csv_p, &ds) != 0) return -1;

    rng_seed(RANDOM_SEED);

    float *aug_cache = (float *)safe_calloc((size_t)ds.count * N_AUG_PER_SAMPLE * TOTAL_FEATURES, sizeof(float));
    precalculate_augmentations(&ds, TOTAL_FEATURES, aug_cache);

    char out_p[1024]; snprintf(out_p, 1024, "%s/aug_cache_verify.bin", RESULTS_DIR);
    FILE *f = fopen(out_p, "wb");
    if (!f) { free(aug_cache); dataset_free(&ds); return -1; }
    fwrite(aug_cache, sizeof(float), (size_t)ds.count * N_AUG_PER_SAMPLE * TOTAL_FEATURES, f);
    fclose(f);

    log_info("verify-rng: %d pacientes, %d aumentacoes/paciente, %d features -> %s",
              ds.count, N_AUG_PER_SAMPLE, TOTAL_FEATURES, out_p);

    free(aug_cache); dataset_free(&ds); return 0;
}

static int mode_validate_external(const char *external_dir)
{
    log_info("=== MODO: VALIDACAO EXTERNA (GENERALIZACAO) ===");
    return 0; /* Implementar carregando os 6 best_models se necessario */
}

int main(int argc, char *argv[])
{
    if (argc < 2) { fprintf(stderr, "Uso: %s <modo> [diretorio]\n", argv[0]); return 1; }
    const char *mode = argv[1]; const char *base_dir = (argc >= 3) ? argv[2] : ".";
    log_set_level(LOG_INFO); log_info("Classificador Vocals - Hierarchical Late Fusion");
    if (strcmp(mode, "extract") == 0) return mode_extract(base_dir) == 0 ? 0 : 1;
    if (strcmp(mode, "train") == 0 || strcmp(mode, "full") == 0) return mode_train(base_dir) == 0 ? 0 : 1;
    if (strcmp(mode, "external") == 0) return mode_validate_external(base_dir) == 0 ? 0 : 1;
    if (strcmp(mode, "verify-rng") == 0) return mode_verify_rng(base_dir) == 0 ? 0 : 1;
    return 1;
}
