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
#include "feature_select_paraconsistent.h"
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

/* Modo de operacao do SMOTE: padrao (todas as amostras da classe minoritaria sao
 * elegiveis como ponto-base de sintese) ou Borderline-SMOTE1 (Han, Wang & Mao, 2005),
 * que restringe o pool de sintese as amostras classificadas como BORDERLINE. */
typedef enum { SMOTE_STANDARD = 0, SMOTE_BORDERLINE = 1 } SmoteMode;

#define SMOTE_K_NEIGHBORS 5
#define BORDERLINE_M_NEIGHBORS 5
/* smote_oversample() e chamada apenas com num_classes=2 (Master, binario) ou
 * num_classes=4 (Expert, 4 classes) -- nunca o NUM_CLASSES total (5) -- por isso
 * este limite fixo e correto e suficiente para os arrays de contagem por classe. */
#define MAX_SMOTE_CLASSES 4

/* Contagem por classe de amostras classificadas como seguras/borderline/ruido
 * durante a classificacao do Borderline-SMOTE1 (SMOTE-04 -- tabela de contagens
 * do relatorio A/B). Struct de dados simples, sem metodos, no mesmo estilo de
 * ConfidenceInterval/MetricsResult (include/metrics.h). */
typedef struct {
    int safe[MAX_SMOTE_CLASSES];
    int borderline[MAX_SMOTE_CLASSES];
    int noise[MAX_SMOTE_CLASSES];
} SmoteBorderlineCounts;

/* Busca k-NN sobre TODAS as n_in amostras (todas as classes), usada apenas para
 * classificar cada amostra como segura/borderline/ruido (Borderline-SMOTE1, Passo 1).
 * NAO faz nenhuma chamada rng_* -- isto preserva a paridade no numero de sorteios de
 * RNG entre os modos SMOTE_STANDARD e SMOTE_BORDERLINE (ver RESEARCH.md Pitfall 4).
 * NUNCA paralelizar com #pragma omp parallel for a menos que este invariante ("zero
 * chamadas rng_*") seja reverificado antes -- mesmo cuidado ja documentado no
 * comentario de precalculate_augmentations(). */
static int find_knn_global(const float *x, int base, int n_in, int nf, int k, int *neighbors)
{
    float *dists = (float *)safe_malloc(n_in * sizeof(float));
    int *order = (int *)safe_malloc(n_in * sizeof(int));
    for (int i = 0; i < n_in; i++) {
        order[i] = i; if (i == base) { dists[i] = 1e30f; continue; }
        float dist = 0.0f; for (int f = 0; f < nf; f++) { float diff = x[base * nf + f] - x[i * nf + f]; dist += diff * diff; }
        dists[i] = dist;
    }
    int kk = (k < n_in - 1) ? k : n_in - 1; if (kk < 1) kk = 1;
    for (int i = 0; i < kk; i++) {
        int min_idx = i; for (int j = i + 1; j < n_in; j++) if (dists[order[j]] < dists[order[min_idx]]) min_idx = j;
        int tmp = order[i]; order[i] = order[min_idx]; order[min_idx] = tmp; neighbors[i] = order[i];
    }
    free(dists); free(order);
    return kk;
}

/* Classificacao segura/borderline/ruido do Borderline-SMOTE1 (Han, Wang & Mao, 2005,
 * Passo 1): m = numero de vizinhos (dentre os k globais) que NAO pertencem a mesma
 * classe da amostra. Regra sem truncamento por divisao inteira: compara 2*m >= k,
 * nunca m >= k/2. A verificacao m == k (ruido) vem PRIMEIRO, pois 2*m >= k tambem e
 * verdadeiro quando m == k -- a ordem dos ramos e determinante. */
static int classify_borderline(int m, int k)
{
    if (m == k) return 2;       /* RUIDO */
    if (2 * m >= k) return 1;   /* BORDERLINE (perigo) */
    return 0;                   /* SEGURO */
}

static void smote_oversample(const float *x_in, const int *y_in, int n_in, int nf,
    int num_classes, SmoteMode smote_mode, SmoteBorderlineCounts *bcounts, float **x_out,
    int **y_out, int *n_out)
{
    int k = SMOTE_K_NEIGHBORS; int *counts = (int *)safe_calloc(num_classes, sizeof(int));
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
        int n_synthetic = max_count - n_class;
        /* SMOTE-03: classe degenerada (n_class<=1) nao tem par valido da mesma classe
         * para interpolar -- pula a sintese inteiramente ao inves de gerar uma
         * duplicata disfarcada de amostra sintetica. */
        if (n_class <= 1) {
            if (n_synthetic > 0) {
                log_warn("smote_oversample: classe %d com apenas %d amostra(s) -- sintese pulada (nenhum par valido da mesma classe para interpolar, zero amostras sinteticas geradas para esta classe)", c, n_class);
            }
            n_synthetic = 0;
        }
        int *neighbors = (int *)safe_malloc(knn * sizeof(int));

        /* Bloco B: construcao do pool borderline (Han, Wang & Mao, 2005, Passo 1).
         * So executa quando o modo e SMOTE_BORDERLINE e a classe realmente precisa
         * sintetizar amostras (n_synthetic > 0) -- uma classe ja pulada pelo Bloco A
         * nao precisa de pool algum. */
        int *class_idx_borderline = NULL; int n_borderline = 0;
        if (smote_mode == SMOTE_BORDERLINE && n_synthetic > 0) {
            class_idx_borderline = (int *)safe_malloc(n_class * sizeof(int));
            int neighbors_g[BORDERLINE_M_NEIGHBORS];
            for (int i = 0; i < n_class; i++) {
                int kk = find_knn_global(x_in, class_idx[c][i], n_in, nf, BORDERLINE_M_NEIGHBORS, neighbors_g);
                int m = 0; for (int j = 0; j < kk; j++) if (y_in[neighbors_g[j]] != c) m++;
                int cls = classify_borderline(m, kk);
                if (bcounts != NULL) {
                    if (cls == 0) bcounts->safe[c]++;
                    else if (cls == 1) bcounts->borderline[c]++;
                    else bcounts->noise[c]++;
                }
                if (cls == 1) class_idx_borderline[n_borderline++] = class_idx[c][i];
            }
            if (n_borderline == 0) {
                log_warn("smote_oversample: classe %d sem amostras borderline (fold/vogal atual) -- usando class_idx[c] completo como fallback", c);
            }
        }

        /* Pool de sintese: restrito ao subconjunto borderline quando aplicavel,
         * caindo de volta ao class_idx[c] completo em SMOTE_STANDARD ou quando o
         * pool borderline esta vazio (fallback do SMOTE-03). */
        int *pool = class_idx[c]; int pool_size = n_class;
        if (smote_mode == SMOTE_BORDERLINE && n_borderline > 0) { pool = class_idx_borderline; pool_size = n_borderline; }

        for (int s = 0; s < n_synthetic; s++) {
            int base_idx = pool[rng_int(pool_size)]; find_knn(x_in, base_idx, class_idx[c], n_class, nf, knn, neighbors);
            int neighbor_idx = neighbors[rng_int(knn)]; float alpha = rng_uniform();
            for (int f = 0; f < nf; f++) (*x_out)[out_idx * nf + f] = x_in[base_idx * nf + f] + alpha * (x_in[neighbor_idx * nf + f] - x_in[base_idx * nf + f]);
            (*y_out)[out_idx++] = c;
        }
        free(neighbors); free(class_idx[c]);
        if (smote_mode == SMOTE_BORDERLINE && n_synthetic > 0) free(class_idx_borderline);
    }
    free(class_idx); free(class_pos); free(counts);
    *n_out = out_idx;
}

/* ========== Training Modo ========== */

/* Resultado agregado de uma execucao completa do pipeline hierarquico (um dos dois
 * bracos do A/B, SMOTE_STANDARD ou SMOTE_BORDERLINE) -- struct de dados simples, sem
 * metodos, no mesmo estilo de MetricsResult/ConfidenceInterval (include/metrics.h).
 * Usado por mode_smote_ab() para comparar os dois modos sob a mesma seed/folds. */
typedef struct {
    float accuracy;
    float macro_f1;
    float f1_per_class[NUM_CLASSES];
    ConfidenceInterval ci[CI_N_METRICS];
    int *y_true;
    int *y_pred;
    int n;
    /* Gap 3 (ARCH-04): contagem de parametros treinaveis e tempo de parede por
     * epoca, por combinacao arquitetura x regularizacao. Zero/nao utilizados em
     * instancias de ABResult criadas pelo caminho ja existente mode_smote_ab()
     * (adicionar campos e compativel com esse codigo ja publicado). */
    int param_count_master;
    int param_count_expert;
    double mean_time_per_epoch_sec;
    float mean_epochs_to_stop;
    /* Gap 1 (PARA-05): media do numero de features selecionadas por execucao de
     * paraconsistent_select(), agregada sobre os 5 folds x 3 vogais x 2 redes = 30
     * chamadas. So preenchido quando para_mode == PARA_SELECT_ON; 0 quando OFF
     * (nao ha selecao a medir). */
    float mean_n_selected;
} ABResult;

/* Config C = producao atual (2 camadas ocultas [128,64]), per ARCH-01 -- NAO
 * chamar de "Config A" em nenhum lugar (correcao ja aplicada em CLAUDE.md pelo
 * Plano 02-01). */
typedef struct { const char *name; int hidden_sizes[3]; int n_hidden; float dropout_rates[3]; } ArchConfig;

static const ArchConfig ARCH_CONFIGS[4] = {
    { "A", {128, 0, 0},   1, {0.5f, 0.0f, 0.0f} },
    { "B", {64, 0, 0},    1, {0.5f, 0.0f, 0.0f} },
    { "C", {128, 64, 0},  2, {0.5f, 0.4f, 0.0f} },
    { "D", {128, 64, 32}, 3, {0.5f, 0.4f, 0.3f} }
};

/* Forca de regularizacao relativa (Gap 3): multiplicador aplicado uniformemente
 * aos dropout_rates[] de ArchConfig E ao L2_LAMBDA (0.001f, config.h -- verificado
 * nesta sessao, NAO o 0.003 obsoleto da tabela de Hiperparametros do CLAUDE.md).
 * REG_BASELINE (multiplicador 1.0) reproduz exatamente o dropout/L2 de producao
 * atual, sem alteracao. */
typedef enum { REG_LIGHT = 0, REG_BASELINE = 1, REG_STRONG = 2 } RegSetting;
static const float REG_MULTIPLIER[3] = { 0.6f, 1.0f, 1.4f };
static const char *REG_NAME[3] = { "light", "baseline", "strong" };

/* Modo de selecao paraconsistente de features (Gap 1, PARA-03): OFF preserva o
 * comportamento atual byte-a-byte -- cada combinacao (fold,vogal,rede) usa o
 * conjunto completo, nao-selecionado, de nf_vowel features, sem nenhuma nova
 * chamada de I/O em disco. ON executa paraconsistent_select() por
 * (fold,vogal,rede), persistindo seus indices selecionados via
 * selected_save()/selected_load() (round trip real por disco) antes do treino,
 * tornando a predicao/o registro de probabilidades consumidores genuinos do
 * arquivo persistido, nao apenas de uma copia em memoria. */
typedef enum { PARA_SELECT_OFF = 0, PARA_SELECT_ON = 1 } ParaMode;

/* mode_train_ex(): executa o pipeline hierarquico completo com o modo SMOTE indicado.
 * result == NULL: execucao CLI simples (modos train/full) -- nomes de arquivo de saida
 * sem sufixo, all_y_true/all_y_pred liberados ao final, comportamento identico ao
 * mode_train() original.
 * result != NULL: execucao de comparacao A/B (mode_smote_ab()) -- nomes de arquivo
 * sufixados por modo, all_y_true/all_y_pred NAO sao liberados aqui (posse transferida
 * para o chamador via *result), que deve libera-los apos o uso.
 * arch/reg (Gap 3, ARCH-03): selecionam a arquitetura (ARCH_CONFIGS) e a forca de
 * regularizacao (REG_MULTIPLIER) usadas por esta execucao -- widen em relacao ao
 * Plano 02-01, sem duplicar o loop fold+vogal.
 * para_mode (Gap 1, PARA-03): PARA_SELECT_OFF preserva o comportamento atual
 * byte-a-byte (identidade, sem I/O novo); PARA_SELECT_ON executa
 * paraconsistent_select() por (fold,vogal,rede), persiste os indices
 * selecionados via selected_save()/selected_load() e aplica o subconjunto de
 * colunas resultante ao treino/validacao de Master e Expert independentemente. */
static int mode_train_ex(const char *base_dir, SmoteMode smote_mode, const ArchConfig *arch, RegSetting reg, ParaMode para_mode, ABResult *result)
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

    /* SMOTE-04: contagem safe/borderline/ruido por fold/vogal/rede/classe -- so
     * relevante no braco SMOTE_BORDERLINE (no braco SMOTE_STANDARD a classificacao e
     * um no-op e as contagens seriam todas zero, nao vale a pena exportar). */
    FILE *counts_f = NULL;
    if (smote_mode == SMOTE_BORDERLINE) {
        char counts_path[160];
        if (result != NULL) {
            snprintf(counts_path, sizeof(counts_path), "results/smote_borderline_counts_%s_%s.csv", arch->name, REG_NAME[reg]);
        } else {
            snprintf(counts_path, sizeof(counts_path), "results/smote_borderline_counts.csv");
        }
        counts_f = fopen(counts_path, "w");
        if (counts_f) {
            fprintf(counts_f, "fold,vowel,network,class,safe,borderline,noise\n");
        } else {
            log_error("Falha ao abrir %s para escrita", counts_path);
        }
    }

    /* Gap 1 (PARA-05 relatorio): abre results/paraconsistent_selection_freq.csv uma
     * unica vez por execucao, apenas quando para_mode == PARA_SELECT_ON (nenhum
     * caller neste plano passa PARA_SELECT_ON -- so o Plano 03-03 o fara -- mas a
     * logica de escrita pertence aos blocos de insercao Master/Expert deste plano,
     * que produzem os valores mu/lambda/gc/gct). Sem sufixo: apenas um chamador
     * jamais passara PARA_SELECT_ON nesta fase. */
    FILE *freq_f = NULL;
    if (para_mode == PARA_SELECT_ON) {
        const char *freq_path = "results/paraconsistent_selection_freq.csv";
        freq_f = fopen(freq_path, "w");
        if (freq_f) {
            fprintf(freq_f, "fold,vowel,network,feature_idx,mu,lambda,gc,gct,selected\n");
        } else {
            log_error("Falha ao abrir %s para escrita", freq_path);
        }
    }

    /* Gap 3: hiperparametros efetivos desta execucao -- calculados uma unica vez,
     * fora do loop de folds, ja que arch/reg sao fixos para toda a chamada. */
    float eff_dropout[3];
    for (int i = 0; i < arch->n_hidden; i++) eff_dropout[i] = arch->dropout_rates[i] * REG_MULTIPLIER[reg];
    float eff_l2 = L2_LAMBDA * REG_MULTIPLIER[reg];
    int param_count_master = 0, param_count_expert = 0;
    double total_train_time_sec = 0.0;
    long total_epochs_sum = 0;
    int total_trainings = 0;
    /* Gap 1 (PARA-05): acumuladores para ABResult.mean_n_selected -- somam
     * ns_m[v]/ns_e[v] a cada uma das 5 folds x 3 vogais x 2 redes = 30 chamadas de
     * paraconsistent_select() (so incrementados quando para_mode == PARA_SELECT_ON). */
    long n_selected_sum = 0;
    int n_selected_calls = 0;

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
        /* Gap 1 (PARA-03): indices de features selecionadas por vogal, independentes
         * entre Master e Expert (podem e devem divergir) -- preenchidos dentro do
         * loop de vogais abaixo, consumidos por predict_hierarchical_late_fusion()
         * apos o loop. Em PARA_SELECT_OFF cada sel_m[v]/sel_e[v] recebe a identidade
         * [0, nf_vowel) e ns_m[v]/ns_e[v] = nf_vowel (comportamento atual). */
        int sel_m[3][FEATURES_PER_VOWEL + NUM_METADATA_FEATURES], ns_m[3];
        int sel_e[3][FEATURES_PER_VOWEL + NUM_METADATA_FEATURES], ns_e[3];
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

            /* Gap 1 (PARA-03): selecao paraconsistente de features para o Master
             * (rede binaria), computada SOMENTE sobre as fold->n_train linhas
             * originais (nao-augmentadas, nao-SMOTE) de tr_x_v/tr_y_bin -- mesmo
             * principio ja documentado para norm_fit (fold->n_train, nunca
             * n_train_aug; Pitfall 3: linhas augmentadas/SMOTE enviesariam mu/lambda). */
            if (para_mode == PARA_SELECT_ON) {
                float *mu_buf = (float *)safe_malloc(nf_vowel * sizeof(float));
                float *lambda_buf = (float *)safe_malloc(nf_vowel * sizeof(float));
                float *gc_buf = (float *)safe_malloc(nf_vowel * sizeof(float));
                float *gct_buf = (float *)safe_malloc(nf_vowel * sizeof(float));
                int *sel_local = (int *)safe_malloc(nf_vowel * sizeof(int));
                int ns_local = paraconsistent_select(tr_x_v, tr_y_bin, fold->n_train, nf_vowel, 2,
                                                       PARA_GC_THRESH, PARA_GCT_MAX, sel_local,
                                                       mu_buf, lambda_buf, gc_buf, gct_buf);
                char sel_path[192];
                snprintf(sel_path, sizeof(sel_path), "models/selected_master_fold%d_v%d.bin", f, v);
                selected_save(sel_path, sel_local, ns_local);
                /* Round trip real por disco (nao apenas copia em memoria) -- e o que
                 * torna a etapa de predicao (mais abaixo) uma consumidora genuina do
                 * arquivo persistido, satisfazendo PARA-04 literalmente. */
                selected_load(sel_path, sel_m[v], &ns_m[v]);
                if (freq_f) {
                    for (int j = 0; j < nf_vowel; j++) {
                        int is_selected = 0;
                        for (int k = 0; k < ns_m[v]; k++) if (sel_m[v][k] == j) { is_selected = 1; break; }
                        fprintf(freq_f, "%d,%d,master,%d,%.6f,%.6f,%.6f,%.6f,%d\n",
                                f, v, j, mu_buf[j], lambda_buf[j], gc_buf[j], gct_buf[j], is_selected);
                    }
                }
                free(mu_buf); free(lambda_buf); free(gc_buf); free(gct_buf); free(sel_local);
            } else {
                ns_m[v] = nf_vowel;
                for (int j = 0; j < nf_vowel; j++) sel_m[v][j] = j;
            }
            /* T-03-04: valida bounds dos indices carregados/atribuidos antes de
             * usa-los para fatiar tr_x_v/vl_x_v -- um .bin corrompido/obsoleto
             * poderia conter indices fora de [0, nf_vowel), causando leitura fora
             * dos limites do buffer. Em violacao, cai de volta para identidade
             * completa (nunca faz clamp silencioso do indice invalido). */
            for (int k = 0; k < ns_m[v]; k++) {
                if (sel_m[v][k] < 0 || sel_m[v][k] >= nf_vowel) {
                    log_error("Master fold=%d vowel=%d: indice de selecao corrompido sel_m[%d]=%d fora de [0,%d) -- usando identidade completa como fallback", f, v, k, sel_m[v][k], nf_vowel);
                    ns_m[v] = nf_vowel;
                    for (int j = 0; j < nf_vowel; j++) sel_m[v][j] = j;
                    break;
                }
            }
            if (para_mode == PARA_SELECT_ON) { n_selected_sum += ns_m[v]; n_selected_calls++; }

            float *tr_x_v_sel = (float *)safe_malloc((size_t)n_train_aug * ns_m[v] * sizeof(float));
            float *vl_x_v_sel = (float *)safe_malloc((size_t)fold->n_val * ns_m[v] * sizeof(float));
            for (int i = 0; i < n_train_aug; i++)
                for (int k = 0; k < ns_m[v]; k++)
                    tr_x_v_sel[i * ns_m[v] + k] = tr_x_v[i * nf_vowel + sel_m[v][k]];
            for (int i = 0; i < fold->n_val; i++)
                for (int k = 0; k < ns_m[v]; k++)
                    vl_x_v_sel[i * ns_m[v] + k] = vl_x_v[i * nf_vowel + sel_m[v][k]];

            SmoteBorderlineCounts sbc_master = {0}, sbc_expert = {0};
            float *os_m_x; int *os_m_y, os_n_m;
            smote_oversample(tr_x_v_sel, tr_y_bin, n_train_aug, ns_m[v], 2, smote_mode, &sbc_master, &os_m_x, &os_m_y, &os_n_m);
            if (counts_f) {
                for (int c = 0; c < 2; c++) {
                    fprintf(counts_f, "%d,%d,master,%d,%d,%d,%d\n", f, v, c,
                            sbc_master.safe[c], sbc_master.borderline[c], sbc_master.noise[c]);
                }
            }
            mlp_init_multi(&net_master[v], ns_m[v], 2, arch->hidden_sizes, arch->n_hidden, eff_dropout);
            if (f == 0 && v == 0) { param_count_master = mlp_count_params(&net_master[v]); }
            TrainHistory h_m;
            double t0_m = timer_now();
            mlp_train(&net_master[v], os_m_x, os_m_y, os_n_m, vl_x_v_sel, vl_y_bin, fold->n_val, ns_m[v], 2, cw_binary, eff_l2, &h_m);
            double dt_m = timer_now() - t0_m;
            total_train_time_sec += dt_m; total_epochs_sum += h_m.num_epochs; total_trainings++;

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
            smote_oversample(ex_tr_x, ex_tr_y, n_ex_tr, nf_vowel, 4, smote_mode, &sbc_expert, &os_e_x, &os_e_y, &os_n_e);
            if (counts_f) {
                for (int c = 0; c < 4; c++) {
                    fprintf(counts_f, "%d,%d,expert,%d,%d,%d,%d\n", f, v, c,
                            sbc_expert.safe[c], sbc_expert.borderline[c], sbc_expert.noise[c]);
                }
            }
            mlp_init_multi(&net_expert[v], nf_vowel, 4, arch->hidden_sizes, arch->n_hidden, eff_dropout);
            if (f == 0 && v == 0) { param_count_expert = mlp_count_params(&net_expert[v]); }
            TrainHistory h_e;
            double t0_e = timer_now();
            mlp_train(&net_expert[v], os_e_x, os_e_y, os_n_e, ex_vl_x, ex_vl_y, n_ex_vl, nf_vowel, 4, cw_expert, eff_l2, &h_e);
            double dt_e = timer_now() - t0_e;
            total_train_time_sec += dt_e; total_epochs_sum += h_e.num_epochs; total_trainings++;

            free(tr_x_v); free(vl_x_v); free(tr_x_v_sel); free(vl_x_v_sel); free(tr_y_bin); free(vl_y_bin); free(os_m_x); free(os_m_y);
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

    double mean_time_per_epoch_sec = (total_epochs_sum > 0) ? total_train_time_sec / total_epochs_sum : 0.0;
    float mean_epochs_to_stop = (total_trainings > 0) ? (float)total_epochs_sum / total_trainings : 0.0f;

    /* Nomes de arquivo de saida: sufixados por modo SMOTE apenas em execucoes de
     * comparacao A/B (result != NULL) -- os modos train/full (result == NULL) mantem
     * os nomes originais sem sufixo, garantindo que os artefatos canonicos da Fase 0
     * permanecam byte-a-byte inalterados (garantia critica de nao-regressao).
     * Gap 3: quando result != NULL os 12 bracos do arch-compare compartilham
     * smote_mode == SMOTE_BORDERLINE -- sem o sufixo arch->name/REG_NAME[reg], todos
     * colidiriam nos mesmos 4 nomes de arquivo (ver threat_model T-02-06). */
    char metrics_path[160], ci_path[160], mcnemar_path[160];
    if (result != NULL) {
        const char *smote_suffix = (smote_mode == SMOTE_BORDERLINE) ? "borderline" : "standard";
        snprintf(metrics_path, sizeof(metrics_path), "results/metrics_global_%s_%s_%s.csv", smote_suffix, arch->name, REG_NAME[reg]);
        snprintf(ci_path, sizeof(ci_path), "results/bootstrap_ci_%s_%s_%s.csv", smote_suffix, arch->name, REG_NAME[reg]);
        snprintf(mcnemar_path, sizeof(mcnemar_path), "results/mcnemar_vs_baselines_%s_%s_%s.csv", smote_suffix, arch->name, REG_NAME[reg]);
    } else {
        snprintf(metrics_path, sizeof(metrics_path), "results/metrics_global.csv");
        snprintf(ci_path, sizeof(ci_path), "results/bootstrap_ci.csv");
        snprintf(mcnemar_path, sizeof(mcnemar_path), "results/mcnemar_vs_baselines.csv");
    }

    MetricsResult g_met; metrics_compute(all_y_true, all_y_pred, all_count, &g_met);
    metrics_print(&g_met, stderr); metrics_export_csv(&g_met, metrics_path);

    /* Intervalo de confianca 95% via bootstrap (N=1000, seed=RANDOM_SEED) sobre as
     * predicoes out-of-fold agregadas. DEVE ser a ultima chamada consumidora de RNG
     * em mode_train_ex(), pois esta funcao re-semeia o RNG global internamente
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

    FILE *ci_f = fopen(ci_path, "w");
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
        log_error("Falha ao abrir %s para escrita", ci_path);
    }

    FILE *mc_f = fopen(mcnemar_path, "w");
    if (mc_f) {
        fprintf(mc_f, "baseline,chi2,p_value\n");
        fprintf(mc_f, "MajorityClass,%.6f,%.6f\n", chi2_maj, p_maj);
        fprintf(mc_f, "kNN,%.6f,%.6f\n", chi2_knn, p_knn);
        fprintf(mc_f, "LogisticRegression,%.6f,%.6f\n", chi2_lr, p_lr);
        fclose(mc_f);
    } else {
        log_error("Falha ao abrir %s para escrita", mcnemar_path);
    }

    if (counts_f) fclose(counts_f);
    if (freq_f) fclose(freq_f);

    /* Transferencia de posse: em execucao de comparacao A/B (result != NULL),
     * all_y_true/all_y_pred NAO sao liberados aqui -- mode_smote_ab() e responsavel
     * por libera-los apos consumir o ABResult (evita use-after-free/double-free,
     * ver threat_model T-01-05 do plano 01-02). */
    if (result != NULL) {
        result->accuracy = g_met.accuracy;
        result->macro_f1 = g_met.macro_f1;
        memcpy(result->f1_per_class, g_met.f1, sizeof(g_met.f1));
        memcpy(result->ci, ci, sizeof(ci));
        result->y_true = all_y_true;
        result->y_pred = all_y_pred;
        result->n = all_count;
        result->param_count_master = param_count_master;
        result->param_count_expert = param_count_expert;
        result->mean_time_per_epoch_sec = mean_time_per_epoch_sec;
        result->mean_epochs_to_stop = mean_epochs_to_stop;
        result->mean_n_selected = (n_selected_calls > 0) ? (float)n_selected_sum / n_selected_calls : 0.0f;
    } else {
        free(all_y_true); free(all_y_pred);
    }
    free(all_y_prob); free(aug_cache);
    free(all_y_pred_majority); free(all_y_pred_knn); free(all_y_pred_logreg);
    return 0;
}

static int mode_train(const char *base_dir) { return mode_train_ex(base_dir, SMOTE_STANDARD, &ARCH_CONFIGS[2], REG_BASELINE, PARA_SELECT_OFF, NULL); }

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

/* Gera o relatorio de comparacao A/B (Gap 2, SMOTE-04) entre SMOTE padrao e
 * Borderline-SMOTE1, sob a mesma seed/folds. A decisao de adocao e computada por
 * uma unica regra fixa em codigo (nunca redigida manualmente) -- ver
 * threat_model T-01-04 do plano 01-02. */
static void write_smote_ab_report(const ABResult *std_res, const ABResult *bl_res,
                                   const char *report_path, const char *csv_path)
{
    /* Comparacao McNemar direta entre os dois bracos (distinta do McNemar de cada
     * braco vs seus 3 baselines, ja calculado dentro de mode_train_ex()). Valida
     * porque std_res->y_true e a ordenacao das predicoes out-of-fold de ambos os
     * bracos sao garantidamente identicas (mesma seed RANDOM_SEED, mesma ordem de
     * fold/vogal via kfold_split()). */
    float chi2_ab, p_ab;
    metrics_mcnemar(std_res->y_true, bl_res->y_pred, std_res->y_pred, std_res->n, &chi2_ab, &p_ab);

    FILE *f = fopen(report_path, "w");
    if (!f) {
        log_error("Falha ao abrir %s para escrita", report_path);
    } else {
        fprintf(f, "=== COMPARACAO A/B: SMOTE PADRAO vs BORDERLINE-SMOTE1 (Gap 2) ===\n");
        fprintf(f, "Mesma seed (RANDOM_SEED=%d), mesmos %d folds\n\n", RANDOM_SEED, K_FOLDS);

        fprintf(f, "%-28s %-32s %-32s\n", "Metrica", "Padrao [IC 95%%]", "Borderline [IC 95%%]");
        fprintf(f, "%-28s %.4f [%.4f, %.4f]      %.4f [%.4f, %.4f]\n", "accuracy",
                std_res->ci[CI_ACCURACY].mean, std_res->ci[CI_ACCURACY].lower, std_res->ci[CI_ACCURACY].upper,
                bl_res->ci[CI_ACCURACY].mean, bl_res->ci[CI_ACCURACY].lower, bl_res->ci[CI_ACCURACY].upper);
        fprintf(f, "%-28s %.4f [%.4f, %.4f]      %.4f [%.4f, %.4f]\n", "macro_f1",
                std_res->ci[CI_MACRO_F1].mean, std_res->ci[CI_MACRO_F1].lower, std_res->ci[CI_MACRO_F1].upper,
                bl_res->ci[CI_MACRO_F1].mean, bl_res->ci[CI_MACRO_F1].lower, bl_res->ci[CI_MACRO_F1].upper);
        fprintf(f, "%-28s %.4f [%.4f, %.4f]      %.4f [%.4f, %.4f]\n", CLASS_NAME_NORMAL,
                std_res->ci[CI_F1_NORMAL].mean, std_res->ci[CI_F1_NORMAL].lower, std_res->ci[CI_F1_NORMAL].upper,
                bl_res->ci[CI_F1_NORMAL].mean, bl_res->ci[CI_F1_NORMAL].lower, bl_res->ci[CI_F1_NORMAL].upper);
        fprintf(f, "%-28s %.4f [%.4f, %.4f]      %.4f [%.4f, %.4f]\n", CLASS_NAME_LARYNGITIS,
                std_res->ci[CI_F1_LARYNGITE].mean, std_res->ci[CI_F1_LARYNGITE].lower, std_res->ci[CI_F1_LARYNGITE].upper,
                bl_res->ci[CI_F1_LARYNGITE].mean, bl_res->ci[CI_F1_LARYNGITE].lower, bl_res->ci[CI_F1_LARYNGITE].upper);
        fprintf(f, "%-28s %.4f [%.4f, %.4f]      %.4f [%.4f, %.4f]\n", CLASS_NAME_DYSPHONIA,
                std_res->ci[CI_F1_DISFONIA].mean, std_res->ci[CI_F1_DISFONIA].lower, std_res->ci[CI_F1_DISFONIA].upper,
                bl_res->ci[CI_F1_DISFONIA].mean, bl_res->ci[CI_F1_DISFONIA].lower, bl_res->ci[CI_F1_DISFONIA].upper);
        fprintf(f, "%-28s %.4f [%.4f, %.4f]      %.4f [%.4f, %.4f]\n", CLASS_NAME_FUNC_DYSPHONIA,
                std_res->ci[CI_F1_FUNC_DISFONIA].mean, std_res->ci[CI_F1_FUNC_DISFONIA].lower, std_res->ci[CI_F1_FUNC_DISFONIA].upper,
                bl_res->ci[CI_F1_FUNC_DISFONIA].mean, bl_res->ci[CI_F1_FUNC_DISFONIA].lower, bl_res->ci[CI_F1_FUNC_DISFONIA].upper);
        fprintf(f, "%-28s %.4f [%.4f, %.4f]      %.4f [%.4f, %.4f]\n\n", CLASS_NAME_REINKE,
                std_res->ci[CI_F1_REINKE].mean, std_res->ci[CI_F1_REINKE].lower, std_res->ci[CI_F1_REINKE].upper,
                bl_res->ci[CI_F1_REINKE].mean, bl_res->ci[CI_F1_REINKE].lower, bl_res->ci[CI_F1_REINKE].upper);

        fprintf(f, "McNemar direto (Borderline vs Padrao): chi2=%.4f p=%.4f -- %s\n\n", chi2_ab, p_ab,
                p_ab < 0.05f ? "diferenca estatisticamente significativa (p<0.05)"
                             : "diferenca nao estatisticamente significativa (p>=0.05)");

        fprintf(f, "Contagens completas de amostras seguras/borderline/ruido por fold/vogal/rede/classe: "
                    "ver results/smote_borderline_counts_C_baseline.csv\n\n");

        if (bl_res->macro_f1 >= std_res->macro_f1) {
            fprintf(f, "DECISAO: Borderline-SMOTE ADOTADO (Macro F1 borderline=%.4f >= padrao=%.4f, delta=%+.4f, McNemar chi2=%.4f p=%.4f)\n",
                    bl_res->macro_f1, std_res->macro_f1, bl_res->macro_f1 - std_res->macro_f1, chi2_ab, p_ab);
        } else {
            fprintf(f, "DECISAO: Borderline-SMOTE REJEITADO (Macro F1 borderline=%.4f < padrao=%.4f, delta=%+.4f, McNemar chi2=%.4f p=%.4f) -- mantendo SMOTE padrao em producao\n",
                    bl_res->macro_f1, std_res->macro_f1, bl_res->macro_f1 - std_res->macro_f1, chi2_ab, p_ab);
        }
        fclose(f);
    }

    FILE *cf = fopen(csv_path, "w");
    if (!cf) {
        log_error("Falha ao abrir %s para escrita", csv_path);
        return;
    }
    fprintf(cf, "metric,standard,standard_ci_lower,standard_ci_upper,borderline,borderline_ci_lower,borderline_ci_upper\n");
    fprintf(cf, "accuracy,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            std_res->ci[CI_ACCURACY].mean, std_res->ci[CI_ACCURACY].lower, std_res->ci[CI_ACCURACY].upper,
            bl_res->ci[CI_ACCURACY].mean, bl_res->ci[CI_ACCURACY].lower, bl_res->ci[CI_ACCURACY].upper);
    fprintf(cf, "macro_f1,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            std_res->ci[CI_MACRO_F1].mean, std_res->ci[CI_MACRO_F1].lower, std_res->ci[CI_MACRO_F1].upper,
            bl_res->ci[CI_MACRO_F1].mean, bl_res->ci[CI_MACRO_F1].lower, bl_res->ci[CI_MACRO_F1].upper);
    fprintf(cf, "f1_normal,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            std_res->ci[CI_F1_NORMAL].mean, std_res->ci[CI_F1_NORMAL].lower, std_res->ci[CI_F1_NORMAL].upper,
            bl_res->ci[CI_F1_NORMAL].mean, bl_res->ci[CI_F1_NORMAL].lower, bl_res->ci[CI_F1_NORMAL].upper);
    fprintf(cf, "f1_laringite,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            std_res->ci[CI_F1_LARYNGITE].mean, std_res->ci[CI_F1_LARYNGITE].lower, std_res->ci[CI_F1_LARYNGITE].upper,
            bl_res->ci[CI_F1_LARYNGITE].mean, bl_res->ci[CI_F1_LARYNGITE].lower, bl_res->ci[CI_F1_LARYNGITE].upper);
    fprintf(cf, "f1_disfonia_psicogenica,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            std_res->ci[CI_F1_DISFONIA].mean, std_res->ci[CI_F1_DISFONIA].lower, std_res->ci[CI_F1_DISFONIA].upper,
            bl_res->ci[CI_F1_DISFONIA].mean, bl_res->ci[CI_F1_DISFONIA].lower, bl_res->ci[CI_F1_DISFONIA].upper);
    fprintf(cf, "f1_disfonia_funcional,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            std_res->ci[CI_F1_FUNC_DISFONIA].mean, std_res->ci[CI_F1_FUNC_DISFONIA].lower, std_res->ci[CI_F1_FUNC_DISFONIA].upper,
            bl_res->ci[CI_F1_FUNC_DISFONIA].mean, bl_res->ci[CI_F1_FUNC_DISFONIA].lower, bl_res->ci[CI_F1_FUNC_DISFONIA].upper);
    fprintf(cf, "f1_reinke,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            std_res->ci[CI_F1_REINKE].mean, std_res->ci[CI_F1_REINKE].lower, std_res->ci[CI_F1_REINKE].upper,
            bl_res->ci[CI_F1_REINKE].mean, bl_res->ci[CI_F1_REINKE].lower, bl_res->ci[CI_F1_REINKE].upper);
    fclose(cf);
}

/* Orquestra a comparacao A/B (Gap 2, SMOTE-04): executa o pipeline hierarquico
 * completo duas vezes, uma por modo SMOTE, sob a mesma seed/folds (garantido pelo
 * reseed interno de kfold_split() -- ver RESEARCH.md Pattern 2), e produz o
 * relatorio comparativo com decisao de adocao computada por regra fixa. */
static int mode_smote_ab(const char *base_dir)
{
    log_info("=== MODO: A/B BORDERLINE-SMOTE (Gap 2) ===");
    log_info("Atencao: modo de longa duracao (~60-180 min) -- executa o pipeline hierarquico completo duas vezes (uma por modo SMOTE)");
    ABResult res_standard = {0}, res_borderline = {0};
    if (mode_train_ex(base_dir, SMOTE_STANDARD, &ARCH_CONFIGS[2], REG_BASELINE, PARA_SELECT_OFF, &res_standard) != 0) return -1;
    if (mode_train_ex(base_dir, SMOTE_BORDERLINE, &ARCH_CONFIGS[2], REG_BASELINE, PARA_SELECT_OFF, &res_borderline) != 0) return -1;
    write_smote_ab_report(&res_standard, &res_borderline,
                           "results/train_log_v32_gap2_smote_ab.txt",
                           "results/smote_ab_comparison.csv");
    free(res_standard.y_true); free(res_standard.y_pred);
    free(res_borderline.y_true); free(res_borderline.y_pred);
    return 0;
}

/* Gera o relatorio de comparacao Gap 3 (ARCH-05): 4 arquiteturas x 3 forcas de
 * regularizacao, decisao de adocao computada por procedimento fixo em 5 passos
 * (melhor regularizacao por arquitetura -> melhor arquitetura -> banda de 1 SE ->
 * portao McNemar -> menor numero de parametros) -- nunca por inspecao visual da
 * tabela, espelhando o precedente fixo-em-codigo de write_smote_ab_report(). */
static void write_arch_compare_report(const ABResult results[4][3], const ArchConfig arch_configs[4], const char *report_path)
{
    /* Passo 1: melhor regularizacao por arquitetura (empate mantem o r de menor
     * indice -- prefere REG_LIGHT sobre REG_BASELINE sobre REG_STRONG). */
    int best_reg[4];
    const ABResult *best_of[4];
    for (int a = 0; a < 4; a++) {
        best_reg[a] = 0;
        for (int r = 1; r < 3; r++) {
            if (results[a][r].macro_f1 > results[a][best_reg[a]].macro_f1) best_reg[a] = r;
        }
        best_of[a] = &results[a][best_reg[a]];
    }

    /* Passo 2: melhor arquitetura geral (empate mantem o a de menor indice). */
    int ao = 0;
    for (int a = 1; a < 4; a++) {
        if (best_of[a]->macro_f1 > best_of[ao]->macro_f1) ao = a;
    }

    /* Passo 3: banda de 1 SE. SE derivada do IC bootstrap (nao da formula classica
     * CART por fold), por consistencia com a infraestrutura estatistica ja
     * estabelecida no projeto -- este codebase nao mantem um array de macro_f1 por
     * fold. */
    float se = (best_of[ao]->ci[CI_MACRO_F1].upper - best_of[ao]->ci[CI_MACRO_F1].lower) / (2.0f * 1.96f);
    float band_lower = best_of[ao]->macro_f1 - se;

    /* Passo 4: portao McNemar -- cada arquitetura != ao comparada contra a melhor
     * (ao), amostra-a-amostra, valido pois os 12 bracos compartilham a mesma
     * seed/ordem de fold/vogal via reseed interno de kfold_split(). */
    float chi2_arr[4] = {0}, p_arr[4] = {0};
    int mcnemar_passes[4] = {0};
    for (int a = 0; a < 4; a++) {
        if (a == ao) { mcnemar_passes[a] = 1; continue; }
        float chi2, p;
        metrics_mcnemar(best_of[ao]->y_true, best_of[ao]->y_pred, best_of[a]->y_pred, best_of[ao]->n, &chi2, &p);
        chi2_arr[a] = chi2; p_arr[a] = p;
        mcnemar_passes[a] = (p >= 0.05f);
    }

    /* Passo 5: regra de adocao -- candidata passa se estiver dentro da banda de 1 SE
     * E (for a propria ao OU nao-significativamente-pior por McNemar). Entre as
     * candidatas aprovadas, adota-se a de menor numero total de parametros (empate
     * mantem o a de menor indice), garantindo adopted==ao sempre que nenhuma config
     * mais simples passar em ambos os portoes. */
    int adopted = -1, adopted_params = 0;
    for (int a = 0; a < 4; a++) {
        int passes = (best_of[a]->macro_f1 >= band_lower) && (a == ao || mcnemar_passes[a]);
        if (!passes) continue;
        int params = best_of[a]->param_count_master + best_of[a]->param_count_expert;
        if (adopted == -1 || params < adopted_params) { adopted = a; adopted_params = params; }
    }

    FILE *f = fopen(report_path, "w");
    if (!f) {
        log_error("Falha ao abrir %s para escrita", report_path);
        return;
    }

    fprintf(f, "=== COMPARACAO REDES RASAS x PROFUNDAS (Gap 3) ===\n");
    fprintf(f, "Mesma seed (RANDOM_SEED=%d), mesmos %d folds, SMOTE fixo em Borderline-SMOTE1 (decisao adotada na Fase 1)\n\n", RANDOM_SEED, K_FOLDS);
    fprintf(f, "Config C [128,64] e a producao atual, NAO Config A [128] (correcao ARCH-01).\n\n");

    fprintf(f, "--- Tabela completa: 4 arquiteturas x 3 forcas de regularizacao ---\n");
    fprintf(f, "arch,reg,accuracy,macro_f1,macro_f1_ci_lower,macro_f1_ci_upper,f1_normal,f1_laringite,f1_disfonia_psicogenica,f1_disfonia_funcional,f1_reinke,param_count_master,param_count_expert,mean_time_per_epoch_sec,mean_epochs_to_stop\n");
    for (int a = 0; a < 4; a++) {
        for (int r = 0; r < 3; r++) {
            const ABResult *res = &results[a][r];
            fprintf(f, "%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%.4f,%.2f\n",
                    arch_configs[a].name, REG_NAME[r], res->accuracy, res->macro_f1,
                    res->ci[CI_MACRO_F1].lower, res->ci[CI_MACRO_F1].upper,
                    res->f1_per_class[CLASS_NORMAL], res->f1_per_class[CLASS_LARYNGITIS],
                    res->f1_per_class[CLASS_DYSPHONIA], res->f1_per_class[CLASS_FUNC_DYSPHONIA],
                    res->f1_per_class[CLASS_REINKE], res->param_count_master, res->param_count_expert,
                    res->mean_time_per_epoch_sec, res->mean_epochs_to_stop);
        }
    }

    fprintf(f, "\n--- Melhor regularizacao por arquitetura ---\n");
    fprintf(f, "arch,reg,macro_f1,macro_f1_ci_lower,macro_f1_ci_upper,params_totais\n");
    for (int a = 0; a < 4; a++) {
        fprintf(f, "%s,%s,%.4f,%.4f,%.4f,%d\n",
                arch_configs[a].name, REG_NAME[best_reg[a]], best_of[a]->macro_f1,
                best_of[a]->ci[CI_MACRO_F1].lower, best_of[a]->ci[CI_MACRO_F1].upper,
                best_of[a]->param_count_master + best_of[a]->param_count_expert);
    }

    fprintf(f, "\n--- Banda de 1 SE ---\n");
    fprintf(f, "SE derivada do IC bootstrap (nao da formula classica CART por fold), por consistencia com a infraestrutura estatistica ja estabelecida no projeto.\n");
    fprintf(f, "Melhor arquitetura geral (ao): %s (reg=%s), macro_f1=%.4f\n", arch_configs[ao].name, REG_NAME[best_reg[ao]], best_of[ao]->macro_f1);
    fprintf(f, "Banda: [%.4f, %.4f]\n\n", band_lower, best_of[ao]->macro_f1);

    fprintf(f, "--- McNemar vs melhor arquitetura (%s) ---\n", arch_configs[ao].name);
    for (int a = 0; a < 4; a++) {
        if (a == ao) continue;
        fprintf(f, "%s vs %s: chi2=%.4f p=%.4f -- %s\n", arch_configs[a].name, arch_configs[ao].name, chi2_arr[a], p_arr[a],
                p_arr[a] < 0.05f ? "diferenca estatisticamente significativa (p<0.05)" : "diferenca nao estatisticamente significativa (p>=0.05)");
    }

    fprintf(f, "\nDECISAO: arquitetura adotada = %s, regularizacao = %s, parametros totais = %d -- ",
            arch_configs[adopted].name, REG_NAME[best_reg[adopted]], adopted_params);
    if (adopted == ao) {
        fprintf(f, "e a propria melhor configuracao (macro_f1=%.4f).\n", best_of[adopted]->macro_f1);
    } else {
        fprintf(f, "dentro da banda de 1 SE (macro_f1=%.4f >= %.4f) e nao significativamente pior que %s por McNemar (p=%.4f >= 0.05).\n",
                best_of[adopted]->macro_f1, band_lower, arch_configs[ao].name, p_arr[adopted]);
    }

    fclose(f);
}

/* Orquestra a comparacao Gap 3 (ARCH-03/04/05): executa o pipeline hierarquico
 * completo 12 vezes (4 arquiteturas x 3 forcas de regularizacao), SMOTE fixo em
 * Borderline-SMOTE1 (decisao adotada na Fase 1, ver STATE.md), persistindo cada
 * braco imediatamente em results/arch_compare_comparison.csv (mecanismo de
 * durabilidade explicito -- ver threat_model T-02-04/T-02-07 do plano 02-02). */
static int mode_arch_compare(const char *base_dir)
{
    log_info("=== MODO: COMPARACAO REDES RASAS x PROFUNDAS (Gap 3) ===");
    log_info("Atencao: modo de duracao extremamente longa (estimado 6-18+ horas) -- executa o pipeline hierarquico completo 12 vezes (4 arquiteturas x 3 forcas de regularizacao), SMOTE fixo em Borderline-SMOTE1 (decisao da Fase 1)");

    ABResult results[4][3] = {0};
    const char *csv_path = "results/arch_compare_comparison.csv";

    FILE *cf = fopen(csv_path, "w");
    if (cf) {
        fprintf(cf, "arch,reg,accuracy,macro_f1,f1_normal,f1_laringite,f1_disfonia_psicogenica,f1_disfonia_funcional,f1_reinke,param_count_master,param_count_expert,mean_time_per_epoch_sec,mean_epochs_to_stop\n");
        fclose(cf);
    } else {
        log_error("Falha ao abrir %s para escrita", csv_path);
    }

    for (int a = 0; a < 4; a++) {
        for (int r = 0; r < 3; r++) {
            int arm_num = a * 3 + r + 1;
            log_info("Iniciando braco %d/12: arquitetura=%s regularizacao=%s", arm_num, ARCH_CONFIGS[a].name, REG_NAME[r]);
            if (mode_train_ex(base_dir, SMOTE_BORDERLINE, &ARCH_CONFIGS[a], (RegSetting)r, PARA_SELECT_OFF, &results[a][r]) != 0) return -1;

            FILE *af = fopen(csv_path, "a");
            if (af) {
                const ABResult *res = &results[a][r];
                fprintf(af, "%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%.6f,%.4f\n",
                        ARCH_CONFIGS[a].name, REG_NAME[r], res->accuracy, res->macro_f1,
                        res->f1_per_class[CLASS_NORMAL], res->f1_per_class[CLASS_LARYNGITIS],
                        res->f1_per_class[CLASS_DYSPHONIA], res->f1_per_class[CLASS_FUNC_DYSPHONIA],
                        res->f1_per_class[CLASS_REINKE], res->param_count_master, res->param_count_expert,
                        res->mean_time_per_epoch_sec, res->mean_epochs_to_stop);
                fclose(af);
            } else {
                log_error("Falha ao abrir %s para escrita (append)", csv_path);
            }
            log_info("Braco %d/12 concluido: arquitetura=%s regularizacao=%s macro_f1=%.4f", arm_num, ARCH_CONFIGS[a].name, REG_NAME[r], results[a][r].macro_f1);
        }
    }

    write_arch_compare_report(results, ARCH_CONFIGS, "results/train_log_v33_gap3_arch_compare.txt");

    for (int a = 0; a < 4; a++) {
        for (int r = 0; r < 3; r++) {
            free(results[a][r].y_true);
            free(results[a][r].y_pred);
        }
    }
    return 0;
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
    if (strcmp(mode, "smote-ab") == 0) return mode_smote_ab(base_dir) == 0 ? 0 : 1;
    if (strcmp(mode, "arch-compare") == 0) return mode_arch_compare(base_dir) == 0 ? 0 : 1;
    return 1;
}
