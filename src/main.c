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
    int all_count = 0;

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

        /* Normalizar (fit no treino, transform em ambos) */
        NormParams norm;
        norm_fit(train_x, fold->n_train, nf, &norm);
        norm_transform(train_x, fold->n_train, &norm);
        norm_transform(val_x, fold->n_val, &norm);

        /* Feature selection */
        int *selected = (int *)safe_malloc(nf * sizeof(int));
        int n_selected = select_features(train_x, fold->n_train, nf,
                                         0.01f, 0.95f, selected, nf);

        float *sel_train_x = (float *)safe_malloc(fold->n_train * n_selected * sizeof(float));
        float *sel_val_x = (float *)safe_malloc(fold->n_val * n_selected * sizeof(float));
        apply_feature_selection(train_x, fold->n_train, nf, selected, n_selected, sel_train_x);
        apply_feature_selection(val_x, fold->n_val, nf, selected, n_selected, sel_val_x);

        if (f == 0) {
            log_info("Feature selection: %d -> %d features", nf, n_selected);
        }

        /* Borderline-SMOTE oversampling */
        float *os_train_x; int *os_train_y; int os_n_train;
        smote_oversample(sel_train_x, train_y, fold->n_train, n_selected,
                         &os_train_x, &os_train_y, &os_n_train);

        log_info("Treino: %d -> %d (Borderline-SMOTE), Validacao: %d, Features: %d",
                 fold->n_train, os_n_train, fold->n_val, n_selected);

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
            all_count++;
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
        char model_path[1024];
        snprintf(model_path, sizeof(model_path), "%s/mlp_fold%d.bin", MODELS_DIR, f);
        mlp_save(&net, model_path);

        /* Salvar normalizacao do fold */
        char norm_path[1024];
        snprintf(norm_path, sizeof(norm_path), "%s/norm_fold%d.bin", MODELS_DIR, f);
        norm_save(&norm, norm_path);

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

    free(all_y_true);
    free(all_y_pred);
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
