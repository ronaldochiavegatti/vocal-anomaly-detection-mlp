/*
 * config.h - Constantes globais do projeto
 *
 * Deteccao de Anomalias Vocais via MLP/CNN em C
 * Iniciacao Cientifica - 2026
 */

#ifndef CONFIG_H
#define CONFIG_H

/* ========== Caminhos ========== */
#define DATA_DIR_NORMAL       "saudavel"
#define DATA_DIR_LARYNGITIS   "laringite"
#define DATA_DIR_DYSPHONIA    "disfonia_psicog\u00eanica"
#define CSV_METADATA          "overview_merged.csv"
#define MODELS_DIR            "models"
#define RESULTS_DIR           "results"

/* ========== Audio ========== */
#define SAMPLE_RATE           44100
#define BITS_PER_SAMPLE       16
#define NUM_CHANNELS          1
#define PRE_EMPHASIS_ALPHA    0.97f

/* ========== Classes ========== */
#define NUM_CLASSES           3
#define CLASS_NORMAL          0
#define CLASS_LARYNGITIS      1
#define CLASS_DYSPHONIA       2

/* Nomes das classes (para exibicao) */
#define CLASS_NAME_NORMAL     "Normal"
#define CLASS_NAME_LARYNGITIS "Laringite"
#define CLASS_NAME_DYSPHONIA  "Disfonia Psicogenica"

/* ========== Vogais para extracao de features ========== */
#define NUM_VOWELS            3
#define VOWEL_A               "a_n"
#define VOWEL_I               "i_n"
#define VOWEL_U               "u_n"

/* ========== Extracao de Features ========== */

/* Features temporais por vogal */
#define NUM_TEMPORAL_FEATURES 10  /* jitter_local, jitter_rap, jitter_ppq5, shimmer_local, shimmer_apq3/5/11, energia, hnr, zcr */

/* Features espectrais por vogal */
#define NUM_SPECTRAL_FEATURES 48  /* f0_mean, f0_std, F1-F4, entropia, centroid, rolloff, mfcc[13], delta_mfcc[13], delta2_mfcc[13] */

/* Features wavelet por vogal */
#define WAVELET_LEVELS        6
#define WAVELET_STATS         3   /* media, variancia, energia */
#define NUM_WAVELET_FEATURES  (WAVELET_LEVELS * WAVELET_STATS)  /* 18 */

/* Total de features por vogal e por paciente */
#define FEATURES_PER_VOWEL    (NUM_TEMPORAL_FEATURES + NUM_SPECTRAL_FEATURES + NUM_WAVELET_FEATURES)  /* 50 */
#define TOTAL_FEATURES        (NUM_VOWELS * FEATURES_PER_VOWEL)  /* 150 */

/* Parametros DSP */
#define FRAME_SIZE_MS         30      /* tamanho do frame em ms */
#define FRAME_STEP_MS         10      /* passo entre frames em ms */
#define FRAME_SIZE            (SAMPLE_RATE * FRAME_SIZE_MS / 1000)   /* 1323 amostras */
#define FRAME_STEP            (SAMPLE_RATE * FRAME_STEP_MS / 1000)   /* 441 amostras */
#define F0_MIN_HZ             80
#define F0_MAX_HZ             500
#define LPC_ORDER             12

/* ========== Arquitetura MLP ========== */
#define MLP_INPUT_SIZE        TOTAL_FEATURES  /* 150 */
#define MLP_HIDDEN1_SIZE      128
#define MLP_HIDDEN2_SIZE      64
#define MLP_HIDDEN3_SIZE      32   /* unused when MLP_NUM_LAYERS=3 */
#define MLP_OUTPUT_SIZE       NUM_CLASSES      /* 3 */
#define MLP_NUM_LAYERS        3                /* 2 hidden + 1 output */

/* ========== Dropout ========== */
#define DROPOUT_RATE_HIDDEN1  0.5f
#define DROPOUT_RATE_HIDDEN2  0.4f
#define DROPOUT_RATE_HIDDEN3  0.0f

/* ========== Batch Normalization ========== */
#define BN_MOMENTUM           0.1f
#define BN_EPSILON            1e-5f

/* ========== Treinamento ========== */
#define LEARNING_RATE         0.001f
#define LR_MIN                0.00001f
#define BATCH_SIZE            32
#define MAX_EPOCHS            500
#define EARLY_STOP_PATIENCE   30
#define L2_LAMBDA             0.003f
#define N_ENSEMBLE            1
#define ADAM_BETA1            0.9f
#define ADAM_BETA2            0.999f
#define ADAM_EPSILON          1e-8f
#define GRAD_CLIP_NORM        5.0f
#define LABEL_SMOOTHING       0.05f
#define LEAKY_RELU_ALPHA      0.01f
#define NOISE_STDDEV          0.05f
#define FOCAL_LOSS_GAMMA      2.0f

/* ========== Validacao ========== */
#define K_FOLDS               5
#define RANDOM_SEED           42

/* ========== Pesos de classe (US-012: Weighted Cross-Entropy, ajustado para Macro F1 early stop) ========== */
/* Moderate boost for minority classes; paired with Macro F1 early stopping (US-013).
 * Disfonia weight increased to 1.7 to boost recall without over-correction
 * (Macro F1 early stopping prevents model from sacrificing Normal recall). */
#define CLASS_WEIGHT_NORMAL     0.80f
#define CLASS_WEIGHT_LARYNGITIS 1.25f
#define CLASS_WEIGHT_DYSPHONIA  1.70f

#endif /* CONFIG_H */
