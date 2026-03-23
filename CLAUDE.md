# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vocal anomaly detection pipeline implemented in pure C. Classifies patients into **5 classes** (Normal, Laringite, Disfonia Psicogênica, Disfonia Funcional, Edema de Reinke) from voice recordings using a Multi-Layer Perceptron. Dataset: ~1098 patients (687/140/91/112/68), severe imbalance.

**Branch**: `ralph/academic-improvements` — all active development happens here.

## Prerequisites

- Audio files (SVD database) placed at the working directory:
  - `saudavel/` — Normal (687 patients)
  - `laringite/` — Laringite (140 patients)
  - `disfonia_psicogênica/` — Disfonia Psicogênica (91 patients)
  - `disfonia_funcional/` — Disfonia Funcional (112 patients)
  - `edema_de_reinke/` — Edema de Reinke (68 patients)
- `overview_merged.csv` metadata file at the working directory
- `results/` and `models/` directories must be created manually before running:
  ```bash
  mkdir -p results models
  ```

## Build & Run Commands

```bash
make                          # Compile → build/vocal_detect
make clean                    # Remove build/ AND results/*.csv (invalidates feature cache)

make extract                  # Extract features from WAV files (~42s with OpenMP), caches to results/features.csv
make train                    # Train 5-fold CV (loads cached CSV if present, ~60-90 min with 237 features)
make full                     # Alias for train (auto-loads cache if results/features.csv exists)

./build/vocal_detect extract [base_dir]
./build/vocal_detect train   [base_dir]
./build/vocal_detect full    [base_dir]
```

**Note**: `make test` target exists in Makefile but `test` mode is **not implemented** in `main.c`.

**Compiler flags**: `gcc -O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -fopenmp -lm`

**Cache validation**: `results/features.csv` column count is validated at load time. If `TOTAL_FEATURES` changes, the cache is automatically rejected and re-extracted.

## Architecture

### Data Flow
```
WAV files (5 class directories)
  → dataset.c: enumerate patients + load metadata (overview_merged.csv)
  → feature_extract.c: orchestrate per-patient extraction [OpenMP parallelized]
      ├─ wav_io.c: read PCM 16-bit WAV (handles non-standard RIFF chunks)
      ├─ dsp_utils.c: pre-emphasis, Hamming window, FFT, autocorr, wavelet denoise
      ├─ feature_temporal.c: Jitter, Shimmer, HNR, ZCR (10 features/vowel)
      ├─ feature_spectral.c: F0, Formants, MFCC×13, δMFCC×13, δδMFCC×13, CPP×3 (51 features/vowel)
      └─ feature_wavelet.c: DWT Daubechies-4, 6 levels × 3 stats (18 features/vowel)
  → results/features.csv: ~1098 × 237 cached feature matrix
  → kfold.c: stratified 5-fold splits
  → Per fold in main.c:
      ├─ Audio-domain augmentation (noise/gain/stretch on minority classes)
      ├─ inner_cv_select_thresholds(): nested 3-fold CV selects variance+correlation thresholds
      ├─ normalize.c: Z-score fit on ORIGINAL training samples only (not augmented)
      ├─ select_features(): remove low-variance + correlated features (237 → ~150-190)
      ├─ Baselines: majority class, kNN(k=5), logistic regression
      ├─ smote_oversample(): Borderline-SMOTE (k=5) to balance minority classes
      ├─ mlp_train.c: mini-batch Adam, cosine LR decay, early stopping on Macro F1
      │     ├─ mlp.c: forward (LeakyReLU + Dropout + Softmax), backward, Adam update
      │     ├─ saves best checkpoint to models/mlp_fold{k}.bin
      │     └─ exports learning curves to results/learning_curves.csv
      └─ metrics.c: confusion matrix, per-class P/R/F1, macro/weighted F1, AUC, McNemar
  → results/metrics_global.csv + bootstrap CI section
  → results/learning_curves.csv (fold, epoch, train/val loss/acc/F1)
  → results/roc_curves.csv + results/pr_curves.csv
  → results/baselines.csv + results/feature_importance.csv
```

### MLP Architecture (config.h)
- **Layers**: Input(~160) → Dense(128) + LeakyReLU + Dropout(0.5) → Dense(64) + LeakyReLU + Dropout(0.4) → Dense(5) + Softmax
- `MLP_NUM_LAYERS=3` is a preprocessor constant used with `#if` guards in `mlp.c`
- For variable input sizes use `mlp_init_dynamic(net, input_size)` (not `mlp_init()`)

### Feature Count (config.h)
| Level | Count | Components |
|-------|-------|------------|
| Temporal/vowel | 10 | jitter×3, shimmer×4, energy, HNR, ZCR |
| Spectral/vowel | 51 | f0×2, F1-F4, entropy, centroid, rolloff, MFCC×13, δMFCC×13, δδMFCC×13, CPP×3 |
| Wavelet/vowel | 18 | 6 levels × (mean, variance, energy) |
| **Per vowel** | **79** | |
| **Total (3 vowels)** | **237** | |
| After selection | ~150-190 | Varies per fold |

### Key Hyperparameters (config.h)
| Parameter | Value |
|---|---|
| Learning rate | 0.001 → 0.00001 (cosine annealing) |
| Batch size | 32 |
| Max epochs | 500 |
| Early stopping patience | 30 (val Macro F1) |
| L2 lambda | 0.003 |
| Label smoothing | 0.05 |
| Gaussian noise | 0.05 |
| Gradient clip norm | 5.0 |
| Class weights | Normal=0.75, Laringite=1.10, Disfonia Psicog.=1.50, Disfonia Func.=1.40, Reinke=1.80 |
| Random seed | 42 (controls splits, SMOTE, noise, bootstrap) |

### Key Structures
- `MLP` / `Layer` / `BatchNorm` — in `mlp.h`
- `Dataset` / `Patient` — in `dataset.h`
- `FeatureMatrix` — in `feature_extract.h`
- `SpectralFeatures` — in `feature_spectral.h` (includes cpp_mean, cpp_std, cpp_slope)
- `KFoldSplit` — in `kfold.h`
- `MetricsResult` — in `metrics.h`
- `NormParams` — in `normalize.h`
- `TrainHistory` — in `mlp_train.h`
- `ConfidenceInterval` — in `metrics.h`

## Current Best Results

**3-class baseline** (v20 config, pre-expansion):
- Accuracy: **80.6%**, Macro F1: 0.612, Disfonia F1: 0.39

**5-class v26 (CPP + all fixes, 2026-03-23)**:
- Accuracy: **65.4%** [CI: 62.6–68.1%], Macro F1: **0.4115** [0.374–0.449]
- Normal: F1=0.822, Laringite: F1=0.360, DisfPsicog: F1=0.273, DisfFunc: F1=0.240, Reinke: F1=0.362
- McNemar: MLP vs LogReg p=0.009 ✓, MLP vs MajorityClass p=0.031 ✓, MLP vs kNN p=0.714 ✗
- AUC: Normal=0.808, Laringite=0.764, DisfPsicog=0.682, DisfFunc=0.657, Reinke=0.797

**5-class v27 (weight tuning US-026, in progress)**:
- Weights: Normal=0.65, Laringite=1.35, DisfPsicog=1.70, DisfFunc=1.70, Reinke=2.10

## Important Constraints

**What works**: Wider layers (128+), LeakyReLU, Dropout, gradient clipping, Borderline-SMOTE, feature selection, mild class weights, std of delta MFCCs, Macro F1 early stopping, feature caching, OpenMP extraction.

**What doesn't work** (do not re-attempt): SWA, ensemble averaging, focal loss, strong class weights + SMOTE, inter-vowel difference features, Mixup augmentation, Batch Normalization (hurts on small datasets), post-hoc probability boosting, wavelet denoising on initial features (removes pathological markers), mean delta MFCCs (near-zero for sustained vowels).

**Fundamental bottleneck**: Acoustic ceiling for Disfonia Psicogênica vs Disfonia Funcional — AUC one-vs-rest ≈ 0.62-0.64 for these two classes. They are acoustically nearly indistinguishable (both functional dysphonias without structural lesion). No architecture/hyperparameter change can break this ceiling with acoustic features alone.

**5-class reality**: Accuracy targets >75% and Macro F1 >0.55 are optimistic given the ceiling. Realistic expectations: Accuracy 62-68%, Macro F1 0.40-0.50. Edema de Reinke (structural lesion) should be separable; the two functional dysphonias are the hard problem.

## Output Files
- `results/features.csv` — cached 1098×237 feature matrix (re-extracted if TOTAL_FEATURES changes)
- `results/metrics_global.csv` — per-class metrics + bootstrap CI (7 metrics, 5 classes)
- `results/learning_curves.csv` — loss/F1 per epoch per fold (for overfitting diagnosis)
- `results/roc_curves.csv` — one-vs-rest ROC points (5 classes)
- `results/pr_curves.csv` — Precision-Recall points (5 classes)
- `results/baselines.csv` — MajorityClass, kNN, LogReg comparison
- `results/feature_importance.csv` — permutation importance per feature
- `models/mlp_fold{0-4}.bin` — trained network weights per fold
- `models/norm_fold{0-4}.bin` — Z-score normalization parameters per fold
- `models/selected_fold{0-4}.bin` — selected feature indices per fold
- `models/best_model.bin` / `models/best_norm.bin` / `models/best_selected.bin` — best fold
- `results/train_log_v*.txt` — training logs per version

## Methodological Notes (for academic rigor)

- **norm_fit**: Must be called with `fold->n_train` (original samples), NOT `n_train_aug` (augmented)
- **rng_seed**: Called globally before `kfold_split()` with `RANDOM_SEED=42` — ensures all randomness is reproducible
- **Cache validation**: `features_load_csv()` counts columns in header; rejects cache if count ≠ `TOTAL_FEATURES`
- **McNemar test**: Compares MLP vs all 3 baselines using Edwards continuity correction; p-value via `erfc()`
- **CPP**: Computed via cepstrum (FFT → log|spec| → FFT → peak in F0 quefrency range); see `src/feature_spectral.c`
- **Bootstrap CI**: N=1000, seed=42, on concatenated out-of-fold predictions
