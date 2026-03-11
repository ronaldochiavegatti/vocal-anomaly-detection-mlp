# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vocal anomaly detection pipeline implemented in pure C. Classifies patients into 3 classes (Normal, Laryngite, Disfonia) from voice recordings using a Multi-Layer Perceptron. Dataset: 918 patients (687/140/91), severe imbalance.

## Prerequisites

- Audio files (SVD database) placed in `saudavel/`, `laringite/`, `disfonia_psicogênica/` at the working directory
- `overview_merged.csv` metadata file at the working directory
- `results/` and `models/` directories must be created manually before running (not auto-created):
  ```bash
  mkdir -p results models
  ```

## Build & Run Commands

```bash
make                          # Compile → build/vocal_detect
make clean                    # Remove build/

make extract                  # Extract features from WAV files (~257s), caches to results/features.csv
make train                    # Train 5-fold CV (loads cached CSV if present, ~45-53s)
make full                     # Alias for train (auto-loads cache if results/features.csv exists)

./build/vocal_detect extract [base_dir]
./build/vocal_detect train   [base_dir]
./build/vocal_detect full    [base_dir]
```

**Note**: `make test` target exists in Makefile but `test` mode is **not implemented** in `main.c`.

**Compiler flags**: `gcc -O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -lm`

## Architecture

### Data Flow
```
WAV files (saudavel/, laringite/, disfonia_psicogênica/)
  → dataset.c: enumerate patients + load metadata (overview_merged.csv)
  → feature_extract.c: orchestrate per-patient extraction
      ├─ wav_io.c: read PCM 16-bit WAV (handles non-standard RIFF chunks)
      ├─ dsp_utils.c: pre-emphasis, Hamming window, FFT, autocorr
      ├─ feature_temporal.c: Jitter, Shimmer, HNR, ZCR (10 features)
      ├─ feature_spectral.c: F0, Formants, MFCC×13, entropy (22 features)
      └─ feature_wavelet.c: DWT Daubechies-4, 6 levels × 3 stats (18 features)
  → results/features.csv: 918 × 150 cached feature matrix
  → kfold.c: stratified 5-fold splits
  → Per fold in main.c:
      ├─ normalize.c: Z-score (fit on train, apply to val)
      ├─ select_features() [in main.c]: remove low-variance (threshold=0.01) + correlated (threshold=0.95) features (150 → ~114)
      ├─ smote_oversample() [in main.c]: Borderline-SMOTE (k=5) to balance minority classes
      ├─ mlp_train.c: mini-batch Adam, cosine LR decay, early stopping
      │     ├─ mlp.c: forward (LeakyReLU + Dropout + Softmax), backward, Adam update
      │     └─ saves best checkpoint to models/mlp_fold{k}.bin
      └─ metrics.c: confusion matrix, per-class P/R/F1, macro/weighted F1
  → results/metrics_global.csv
```

### MLP Architecture (config.h)
- **Layers**: Input(~114) → Dense(128) + LeakyReLU + Dropout(0.5) → Dense(64) + LeakyReLU + Dropout(0.4) → Dense(3) + Softmax
- `MLP_NUM_LAYERS=3` is a preprocessor constant used with `#if` guards in `mlp.c`
- For variable input sizes use `mlp_init_dynamic(net, input_size)` (not `mlp_init()`)

### Key Hyperparameters (config.h)
| Parameter | Value |
|---|---|
| Learning rate | 0.001 → 0.00001 (cosine annealing) |
| Batch size | 32 |
| Max epochs | 500 |
| Early stopping patience | 30 (val_acc) |
| L2 lambda | 0.003 |
| Label smoothing | 0.05 |
| Gaussian noise | 0.05 |
| Gradient clip norm | 5.0 |
| Class weights | Normal=0.9, Laryngite=1.1, Disfonia=1.2 |

### Key Structures
- `MLP` / `Layer` / `BatchNorm` — in `mlp.h`
- `Dataset` / `Patient` — in `dataset.h`
- `FeatureMatrix` — in `feature_extract.h`
- `KFoldSplit` — in `kfold.h`
- `MetricsResult` — in `metrics.h`
- `NormParams` — in `normalize.h`

## Current Best Results (v15/v20 config)
- Accuracy: **80.6%**, Macro F1: 0.612, Weighted F1: 0.789
- Normal: P=0.85, R=0.94, F1=0.89
- Laryngite: P=0.64, R=0.49, F1=0.55
- Disfonia: P=0.55, R=0.31, F1=0.39

## Important Constraints

**What works**: Wider layers (128+), LeakyReLU, Dropout, gradient clipping, Borderline-SMOTE, feature selection, mild class weights, feature caching.

**What doesn't work** (do not re-attempt): SWA, ensemble averaging, focal loss, strong class weights + SMOTE, inter-vowel difference features, Mixup augmentation, Batch Normalization (hurts on small datasets), post-hoc probability boosting.

**Fundamental bottleneck**: 91 Disfonia samples caps generalization. Train ~99% vs val ~80% is persistent overfitting. Breaking 81% likely requires more real data or different features.

## Output Files
- `results/features.csv` — cached feature matrix (skip ~5min extraction on re-runs)
- `results/metrics_global.csv` — final evaluation metrics
- `models/mlp_fold{0-4}.bin` — trained network weights per fold
- `models/norm_fold{0-4}.bin` — Z-score normalization parameters per fold
- `results/train_log_v*.txt` — training logs per version
