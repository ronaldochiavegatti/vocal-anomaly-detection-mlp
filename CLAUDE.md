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

**What works**: Wider layers (128+), LeakyReLU, Dropout, gradient clipping, Borderline-SMOTE1 (Han/Wang/Mao 2005, validated by reproducible A/B in v32 — see `results/train_log_v32_gap2_smote_ab.txt`; Macro F1 +0.0235 point-estimate delta, McNemar p=0.66 not statistically significant), feature selection, mild class weights, std of delta MFCCs, Macro F1 early stopping, feature caching, OpenMP extraction.

**What doesn't work** (do not re-attempt): SWA, ensemble averaging, focal loss, strong class weights + SMOTE, inter-vowel difference features, Mixup augmentation, Batch Normalization (hurts on small datasets), post-hoc probability boosting, wavelet denoising on initial features (removes pathological markers), mean delta MFCCs (near-zero for sustained vowels).

**Fundamental bottleneck**: Acoustic ceiling for Disfonia Psicogênica vs Disfonia Funcional — AUC one-vs-rest ≈ 0.62-0.64 for these two classes. They are acoustically nearly indistinguishable (both functional dysphonias without structural lesion). No architecture/hyperparameter change can break this ceiling with acoustic features alone.

**5-class reality**: Accuracy targets >75% and Macro F1 >0.55 are optimistic given the ceiling. Realistic expectations: Accuracy 62-68%, Macro F1 0.40-0.50. Edema de Reinke (structural lesion) should be separable; the two functional dysphonias are the hard problem.

## Gap 2 Outcome — Borderline-SMOTE A/B (v32)

**DECISAO** (verbatim from `results/train_log_v32_gap2_smote_ab.txt`):

> DECISAO: Borderline-SMOTE ADOTADO (Macro F1 borderline=0.4587 >= padrao=0.4351, delta=+0.0235, McNemar chi2=0.1928 p=0.6606)

**Comparison** (bootstrap mean [95% CI], N=1000, seed=42, same 5-folds):

| Metric | Standard (Padrão) | Borderline-SMOTE1 |
|---|---|---|
| Accuracy | 0.6915 [0.6648, 0.7177] | 0.6958 [0.6694, 0.7231] |
| **Macro F1** | **0.4338 [0.3976, 0.4713]** | **0.4565 [0.4194, 0.4949]** |
| Normal F1 | 0.8725 | 0.8664 |
| Laringite F1 | 0.3884 | 0.3974 |
| Disfonia Psicogênica F1 | 0.2648 | 0.3273 |
| Disfonia Funcional F1 | 0.1892 | 0.2386 |
| Edema de Reinke F1 | 0.4540 | 0.4531 |

**Significance caveat**: Direct McNemar test between the two arms' out-of-fold predictions: chi2=0.1928, **p=0.6606 — NOT statistically significant (p >= 0.05)**. The Macro F1 delta (point-estimate +0.0235) is directionally positive under both the raw point estimate and the bootstrap-CI mean readings, but this is a fixed adopt/reject rule (`borderline_macro_f1 >= standard_macro_f1`) with no significance gate — do not present this as a proven/statistically significant improvement, only as the outcome of that rule. Gains concentrate in the two hardest/smallest classes (Disfonia Psicogênica +0.063, Disfonia Funcional +0.049), with a small give-back on Normal (-0.006) and Reinke (-0.001).

**Empty-borderline-pool fallback**: Fired in 30 of 90 rows in `results/smote_borderline_counts.csv`, but all 30 are structural placeholder rows for class slots each network doesn't use (Master's unused class=1 slot, Expert's unused class=0 slot), always `(0,0,0)` by construction — not real fallback events. Among the 60 real classification rows (Master's healthy class + Expert's 4 pathology classes, across 5 folds × 3 vowels × 2 networks), the fallback fired **zero times** — it did not concentrate in the smallest classes as anticipated.

**Files**: `results/train_log_v32_gap2_smote_ab.txt` (full A/B report + DECISAO sentence), `results/smote_ab_comparison.csv` (machine-readable 7-metric × 2-arm comparison with bootstrap CI), `results/smote_borderline_counts.csv` (90-row safe/borderline/noise counts per fold/vowel/network/class).

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

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Detecção de Anomalias Vocais (MLP em C)**

Pipeline de aprendizado de máquina em C99 puro (sem frameworks externos) que classifica
pacientes em 5 classes (Normal, Laringite, Disfonia Psicogênica, Disfonia Funcional,
Edema de Reinke) a partir de gravações de voz (base SVD, ~1098 pacientes). É o produto
de uma Iniciação Científica (PIBIC) e a arquitetura HEAD atual (v29, `Hierarchical Late
Fusion`, commit `e63483a`) usa um ensemble Master binário + Expert 4 classes, replicado
por vogal (/a/, /i/, /u/), com fusão tardia por média de probabilidades.

**Core Value:** Fechar, com rigor metodológico comprovável por comparação A/B (mesma seed, mesmos
5-folds), os 3 gaps entre a implementação atual e a proposta PIBIC original — sem
piorar o baseline de referência (Macro F1 0,4423 / Acurácia 69,4%, `results/metrics_global.csv`).

### Constraints

- **Metodológico**: Nenhuma mudança é incorporada sem comparação A/B reprodutível (mesma seed `RANDOM_SEED=42`, mesmos 5-folds), log salvo em `results/train_log_vXX_<nome-do-gap>.txt` — exigência do SPEC.md para rigor acadêmico perante a banca PIBIC
- **Regressão**: Reverter ou manter apenas como experimento documentado qualquer mudança que derrube o Macro F1 global abaixo de 0,42
- **Tech stack**: C99 puro, sem dependências externas de ML — `gcc -O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -fopenmp -lm`
- **Ordem de implementação**: Gap 2 → Gap 3 → Gap 1 (do SPEC.md — Gap 2 valida o fluxo de comparação A/B com baixo risco; Gap 1 é o mais complexo e deve incorporar as melhores config/modo já validados nos passos anteriores)
- **Documentação**: Atualizar `CLAUDE.md` (Optimization History / What Worked / What Didn't Work) ao final de cada gap, independentemente do resultado
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C (C99 standard, `-std=c99`) - 100% of the codebase: `src/*.c` (19 files, ~5,666 total lines across `src/` and `include/`), `include/*.h` (17 headers)
- None. No Python, shell scripts, or other languages are part of the build/runtime pipeline. (Non-code artifacts in the repo root — `.docx`/`.pptx` files, `overview_merged.csv` — are academic/documentation/data assets, not part of the toolchain.)
## Runtime
- Native compiled binary (ELF executable), no VM/interpreter. Built and run directly on Linux (developed/tested on Ubuntu 24.04, gcc 13.3.0, kernel 6.17).
- No containerization (no `Dockerfile`, no `docker-compose.yml` in repo).
- None. There is no language-level package manager (no `pip`, `npm`, `cargo`, `conan`, `vcpkg`). All dependencies are system libraries linked directly by the compiler (`libm`, OpenMP runtime `libgomp`).
- Lockfile: not applicable — no dependency manifest exists.
## Frameworks
- None. No ML/DL framework (no TensorFlow, PyTorch, ONNX, scikit-learn). The Multi-Layer Perceptron is hand-implemented from scratch in `src/mlp.c` / `src/mlp_train.c` (forward pass with LeakyReLU + Dropout + Softmax, manual backprop, Adam optimizer, cosine LR annealing, gradient clipping).
- Baseline classifiers (kNN in `src/knn.c`, logistic regression in `src/logreg.c`) are also hand-implemented, not from a library.
- None detected. No unit test framework (no CUnit, Check, Unity). The `Makefile` defines a `test` target (`./build/vocal_detect test`) but this mode is explicitly **not implemented** in `src/main.c` (per `CLAUDE.md`). There is no `tests/` directory and no automated test suite.
- GNU Make (`Makefile`, GNU Make syntax with `$(wildcard ...)`, pattern rules) - drives the entire build.
- GCC 13.3.0 (`gcc`) - sole compiler; flags: `-O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -fopenmp`.
- OpenMP 4.5 (`_OPENMP 201511`, via `-fopenmp`) - used for parallel feature extraction (`#pragma omp parallel for reduction(+:errors)` in `src/feature_extract.c`), giving ~6.5× speedup (257s → ~39-42s for 1098 patients).
## Key Dependencies
- `libm` (math library, linked via `-lm`) - all DSP math (FFT, autocorrelation, log, trig, `erfc` for McNemar test p-values).
- `libgomp` (OpenMP runtime, linked via `-fopenmp`) - parallelizes the per-patient feature-extraction loop across CPU cores.
- Standard C library (`libc`) headers used throughout: `stdio.h`, `stdlib.h`, `string.h`, `math.h`, `time.h`, `stdint.h`, `stddef.h`, `stdarg.h`, `dirent.h`, `sys/stat.h`. No third-party C libraries (no libsndfile, no FFTW, no BLAS/LAPACK) — WAV parsing, FFT, and CSV parsing are all custom implementations.
- None. No message queues, caches, or service dependencies. The only "infrastructure" is the local filesystem (input WAV directories, `overview_merged.csv`, and generated `results/`/`models/` directories).
## Configuration
- No environment variables are read anywhere in `src/` (no `getenv` calls found). All tunables are compile-time constants.
- All configuration lives in a single header: `include/config.h` — paths, audio parameters, class definitions, feature counts, MLP architecture, hyperparameters, class weights, random seed. Changing any of these requires recompilation (`make clean && make`).
- Command-line arguments select pipeline mode: `./build/vocal_detect {extract|train|full} [base_dir]` (parsed in `src/main.c`); `base_dir` defaults to the current working directory if omitted.
- `Makefile` (root) — single build config file; no CMake, no Meson, no Autotools.
- No `tsconfig.json`/`eslint.config`/`package.json`-equivalent exists; this is a pure C project with no auxiliary tool configs beyond the Makefile and `include/config.h`.
## Platform Requirements
- Linux (developed on Ubuntu 24.04 LTS, kernel 6.17).
- `gcc` supporting C99 and OpenMP (tested with gcc 13.3.0).
- `make` (GNU Make).
- Multi-core CPU recommended (OpenMP feature extraction scales with core count).
- Local copies of the SVD (Saarbrücken Voice Database) WAV files, organized into 5 class directories (`saudavel/`, `laringite/`, `disfonia_psicogênica/`, `disfonia_funcional/`, `edema_de_reinke/`) plus `overview_merged.csv` metadata, placed in the working directory (not checked into git — see `.gitignore`).
- `results/` and `models/` directories must be created manually (`mkdir -p results models`) before running.
- No deployment target — this is a research/academic pipeline run locally via CLI (`./build/vocal_detect train`), not a deployed service. Output artifacts (`results/*.csv`, `models/*.bin`) are consumed manually/offline for analysis, not served.
- Portable to any POSIX-like system with a C99 compiler and OpenMP support (Linux primarily; not verified on macOS/Windows). Uses POSIX-specific APIs (`dirent.h`, `sys/stat.h`, `_POSIX_C_SOURCE 200809L` in `src/dataset.c`), so Windows would require WSL/MinGW/Cygwin.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Language & Compiler
- **Standard:** C99 (`-std=c99`)
- **Compiler:** gcc, flags `-O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -fopenmp`
- **Linker:** `-lm -fopenmp`
- Code must compile cleanly under `-Wall -Wextra` (format-truncation warnings from `snprintf` path building are the only suppressed class). Treat any new warning as a bug to fix, not to suppress.
- `_POSIX_C_SOURCE` is defined at the top of files that need POSIX APIs (e.g. `#define _POSIX_C_SOURCE 199309L` in `src/utils.c` for `clock_gettime`, `#define _POSIX_C_SOURCE 200809L` in `src/dataset.c` for `dirent.h`/`sys/stat.h`). Add this **before any `#include`** when a new file needs POSIX-only functions.
- There is a repo hook (`.claude/hooks/compile-check.sh`) that automatically runs `make` after any edit to a `.c`/`.h` file — code is expected to build after every change, not just at the end of a task.
## File Organization
## Naming Patterns
- `mlp_forward`, `mlp_backward`, `mlp_init_dynamic`, `mlp_save_checkpoint` (module `mlp`)
- `norm_fit`, `norm_transform`, `norm_save`, `norm_load`, `norm_free` (module `norm`)
- `kfold_split`, `kfold_free` (module `kfold`)
- `metrics_compute`, `metrics_print`, `metrics_bootstrap_ci`, `metrics_mcnemar` (module `metrics`)
- `dataset_load`, `dataset_free` (module `dataset`)
- `rng_seed`, `rng_uniform`, `rng_int`, `rng_normal`, `rng_shuffle_int` (module `rng`, lives in `utils.c`)
- `log_debug`, `log_info`, `log_warn`, `log_error` (module `log`, lives in `utils.c`)
## Comments
- All prose comments are in **Portuguese** (matching the academic/PIBIC context of the project) — code identifiers themselves are in English/technical terms (`weights`, `dropout_mask`, `forward`).
- Comment style: `/* ... */` block comments for section headers and function documentation; `//` is essentially absent from the code (grep found none used as a marker) — do not introduce `//` line comments, stay consistent with `/* */`.
- Section dividers inside header files use a consistent banner: `/* ========== Nome da Secao ========== */` (see `config.h`, `utils.h`, `mlp_train.c`'s "Checkpoint buffers" style groupings are inline, but constants files always use this banner).
- Every non-trivial public function in a `.h` file has a `/* ... */` doc comment above its declaration describing parameters and return value semantics (see `metrics.h` for the most complete example — every function documents input array shapes with `[n]` / `[n x m]` notation and cites the algorithm's academic source, e.g. "Teste de McNemar ... (Edwards, 1948)", "Breiman, 2001" for permutation importance).
- No TODO/FIXME/HACK/XXX markers exist anywhere in `src/` or `include/` — the project's own convention is to resolve or document issues in `SPEC.md`/`CLAUDE.md` rather than leave inline markers.
## Error Handling
## Logging
## Function Design
## Module Design
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## System Overview
```text
```
## Component Responsibilities
| Component | Responsibility | File |
|-----------|----------------|------|
| Dataset enumeration | Walk 5 class dirs, validate 3 vowel WAVs exist, join CSV metadata | `src/dataset.c`, `src/csv_parser.c` |
| WAV I/O | Parse non-standard RIFF/PCM16 WAV into float samples | `src/wav_io.c` |
| DSP primitives | Pre-emphasis, Hamming window, autocorrelation, radix-2 FFT, Haar wavelet denoise | `src/dsp_utils.c` |
| Temporal features | Jitter (local/RAP/PPQ5), Shimmer (local/APQ3/5/11), energy, HNR, ZCR | `src/feature_temporal.c` |
| Spectral features | F0, formants (LPC), spectral entropy/centroid/rolloff, MFCC+Δ+ΔΔ (std-dev), CPP, glottal source (Oq/Sq/NAQ/H1-H2) | `src/feature_spectral.c` |
| Wavelet features | 6-level Daubechies-4 DWT, mean/variance/energy per level | `src/feature_wavelet.c` |
| Feature orchestration | Per-patient extraction loop (OpenMP parallel), CSV export/cache | `src/feature_extract.c` |
| Audio augmentation | Noise/gain/stretch/pitch-shift on raw audio (minority classes only) | `src/wav_augment.c` |
| Fold splitting | Stratified, patient-level 5-fold assignment | `src/kfold.c` |
| Normalization | Z-score fit/transform, fit only on original (non-augmented) training rows | `src/normalize.c` |
| SMOTE oversampling | Standard SMOTE (interpolation between same-class k-NN) — implemented inline | `src/main.c` (`smote_oversample`, `find_knn`) |
| MLP core | Layer struct, forward/backward, Adam optimizer, LeakyReLU+Dropout, unused BatchNorm plumbing | `src/mlp.c`, `include/mlp.h` |
| Training loop | Mini-batch SGD, cosine-annealed LR, Gaussian noise injection, Macro-F1 early stopping, unused SWA accumulation | `src/mlp_train.c` |
| Hierarchical fusion orchestration | Builds per-vowel Master/Expert pairs, fold loop, late-fusion prediction, aggregate metrics | `src/main.c` |
| Metrics | Confusion matrix, P/R/F1, macro/weighted F1; also unused-in-current-flow bootstrap CI/ROC/PR/McNemar/permutation importance | `src/metrics.c`, `include/metrics.h` |
| Legacy baselines (dormant) | kNN and multinomial LogReg baselines from earlier PRD iterations; headers included but never invoked in current `mode_train` | `src/knn.c`, `src/logreg.c` |
| Legacy feature selection (dormant) | Binary persistence of selected feature indices from earlier variance/correlation-based selection; header included but selection logic never called in current flow | `src/feature_select.c` |
| Config constants | All architectural/hyperparameter constants (paths, feature counts, network sizes, training hyperparameters, class weights) | `include/config.h` |
## Pattern Overview
- No dynamic dispatch/polymorphism: C structs + free functions, one `.c`/`.h` pair per concern (SRP at file granularity).
- Hierarchical decomposition of the classification problem itself: a binary "sick vs. healthy" gate (Master) followed by a 4-way pathology classifier (Expert), rather than one flat 5-class softmax.
- Per-vowel model replication: the Master/Expert pair is trained independently for each of 3 sustained vowels (`/a/`, `/i/`, `/u/`), then fused by probability averaging at inference time (late fusion, not feature concatenation).
- Aggressive feature caching: `results/features.csv` is a materialized view of the entire (expensive, ~1min) extraction step, column-count-validated against `TOTAL_FEATURES` at load time so stale caches self-invalidate.
- OpenMP data parallelism at the patient level only (`#pragma omp parallel for` in `feature_extract.c` and in `precalculate_augmentations`); no other multithreading (training is single-threaded).
- Everything is parameterized through `include/config.h` preprocessor constants — no runtime config file, no CLI flags beyond `<mode> [base_dir]`.
- The codebase carries substantial **dormant code** from prior PRD iterations (feature selection, kNN/LogReg baselines, bootstrap CI, ROC/AUC, McNemar test, model checkpoint persistence, BatchNorm, SWA) that still compiles and links but is not exercised by the current `mode_train` path — see Anti-Patterns and dedicated section below.
## Layers
- Purpose: turn on-disk WAV files + CSV metadata into an in-memory `Dataset`
- Location: `src/dataset.c`, `src/csv_parser.c`, `src/wav_io.c`
- Contains: directory walking, RIFF/PCM16 parsing, CSV field parsing
- Depends on: POSIX `dirent.h`/`sys/stat.h`, nothing else in the pipeline
- Used by: feature extraction orchestrator (`feature_extract.c`), augmentation precompute (`main.c`)
- Purpose: convert raw audio samples into fixed-length numeric feature vectors
- Location: `src/dsp_utils.c`, `src/feature_temporal.c`, `src/feature_spectral.c`, `src/feature_wavelet.c`, `src/feature_extract.c`
- Contains: pure numeric DSP functions (no I/O except in the orchestrator)
- Depends on: `dataset.h`/`wav_io.h` types, `config.h` constants (frame sizes, F0 range, feature counts)
- Used by: `main.c` (both the cached-CSV path and the on-the-fly augmentation path via `extract_vowel_from_float`)
- Purpose: fold splitting, normalization, class balancing
- Location: `src/kfold.c`, `src/normalize.c`, SMOTE code inline in `src/main.c`
- Contains: stratification logic, Z-score stats, k-NN-based synthetic sample generation
- Depends on: `FeatureMatrix`/`Dataset` types
- Used by: `main.c` fold loop
- Purpose: neural network definition, training, and inference
- Location: `src/mlp.c`, `src/mlp_train.c`, `include/mlp.h`, `include/mlp_train.h`
- Contains: `Layer`/`MLP` structs, forward/backward pass, Adam optimizer, loss functions, checkpoint save/load (checkpoint here means best-epoch-in-RAM, not disk — see Anti-Patterns)
- Depends on: `config.h` (`MLP_NUM_LAYERS`, hidden sizes, dropout rates, Adam betas, etc.) and `utils.c` RNG/log helpers
- Used by: `main.c` (instantiates `net_master[3]` and `net_expert[3]` per fold)
- Purpose: wire every other layer together into the two CLI modes
- Location: `src/main.c`
- Contains: `mode_extract()`, `mode_train()` (the actual pipeline), `mode_validate_external()` (stub, returns 0 without doing anything), CSV feature-cache load/store, hierarchical-fusion prediction, SMOTE, augmentation caching
- Depends on: every other layer
- Used by: nothing (entry point)
- Purpose: turn predictions into reportable numbers
- Location: `src/metrics.c`, `include/metrics.h`
- Contains: confusion matrix, precision/recall/F1, plus a larger surface (bootstrap CI, ROC/AUC, PR curves, McNemar test, permutation importance) inherited from earlier single-model PRD iterations
- Depends on: `MLP` type (for permutation importance only)
- Used by: `main.c` — but only `metrics_compute`/`metrics_print`/`metrics_export_csv` are actually called in the current `mode_train`; the rest is dead from the orchestrator's perspective (still unit-testable/linkable)
## Data Flow
### Primary Request Path (training run: `./build/vocal_detect train .`)
### Feature Extraction Path (`./build/vocal_detect extract .`)
- No persistent application state between runs beyond the two on-disk caches: `results/features.csv` (feature cache, self-invalidating) and stale `models/*.bin` files (see Anti-Patterns — no longer written by the current `mode_train`).
- Within a run, all state is stack/heap-allocated C structs passed by pointer; RNG state is a single global seeded once via `rng_seed(RANDOM_SEED)` in `src/utils.c`.
## Key Abstractions
- Purpose: in-memory catalogue of every patient's class label, demographic metadata, and the 3 WAV file paths needed for feature extraction
- Examples: `src/dataset.c`, `include/dataset.h`
- Pattern: flat array of fixed-size structs (`Patient patients[]`), no per-record heap pointers except the struct itself
- Purpose: dense row-major `count × TOTAL_FEATURES` matrix + parallel `labels[]` array — the single source of truth once extraction is done
- Examples: `include/feature_extract.h`
- Pattern: plain `float*`/`int*` buffers with manual `count`/`num_features` bookkeeping; consumed directly by fold-splitting, normalization, and per-vowel slicing code
- Purpose: a small feedforward network (this codebase currently always instantiates 2 hidden layers via `MLP_NUM_LAYERS=3`, i.e. Input→Hidden1→Hidden2→Output)
- Examples: `include/mlp.h`, `src/mlp.c`
- Pattern: `Layer[MLP_NUM_LAYERS]` fixed-size array inside `MLP`; `mlp_init_dynamic(net, input_size, output_size)` is the variant actually used because per-vowel-per-network input/output sizes vary (85→2 for Master, 85→4 for Expert); `mlp_init()`/`MLP_OUTPUT_SIZE` (5-class legacy path) is unused by `main.c`.
- Purpose: precomputed train/val index arrays per fold, stratified by class at the patient level
- Examples: `include/kfold.h`, `src/kfold.c`
- Pattern: index-array based (no data copying at split time — copying happens later per-fold in `main.c`)
- Purpose: decompose 5-class classification into an easier binary decision (Normal vs. Pathological) followed by a 4-way decision only among pathological classes, then average that decomposition's confidence across 3 independent per-vowel views of the same patient
- Examples: `net_master[3]`, `net_expert[3]` arrays in `src/main.c:273`, fusion logic in `predict_hierarchical_late_fusion()` (`src/main.c:39-69`)
- Pattern: 2 (Master/Expert) × 3 (vowels) = 6 independently trained MLPs per fold, 30 MLPs total across 5-fold CV; probabilities are averaged (not stacked/concatenated) at inference — this is what "late fusion" means here, as opposed to feeding all 3 vowels' features into one wider network
## Entry Points
- Location: `src/main.c`
- Triggers: CLI invocation `./build/vocal_detect <mode> [base_dir]`
- Responsibilities: parse mode string (`extract`|`train`|`full`|`external`), set log level, dispatch
- Location: `src/main.c`
- Triggers: `mode == "extract"`
- Responsibilities: dataset load → feature extraction → CSV export only (no training)
- Location: `src/main.c`
- Triggers: `mode == "train"` or `mode == "full"` (both are aliases — there is no separate "load cache vs. force re-extract" distinction beyond the cache validity check already inside `mode_train`)
- Responsibilities: the entire pipeline described in Data Flow above
- Location: `src/main.c`
- Triggers: `mode == "external"`
- Responsibilities: **stub only** — logs a message and returns 0 immediately; the docstring-style comment says "Implementar carregando os 6 best_models se necessario" (not implemented). Do not assume external validation works.
- Triggers `./build/vocal_detect test`, but no `mode == "test"` branch exists in `main()` — running this returns exit code 1 with the usage message. Treat as non-functional.
## Architectural Constraints
- **Threading:** OpenMP `parallel for` only around (1) the per-patient feature extraction loop in `features_extract_all()` and (2) the per-patient augmented-feature precomputation loop in `precalculate_augmentations()`. Training itself (`mlp_train`) is single-threaded; do not assume thread-safety of `MLP`/`Layer` structs — each fold/vowel/network combination must own its own `MLP` instance (the code already does this via `net_master[3]`/`net_expert[3]` arrays, never sharing one `MLP` across threads).
- **Global state:** RNG is process-global (`rng_seed()`/`rng_uniform()`/`rng_normal()` in `src/utils.c`) and reseeded exactly once at `RANDOM_SEED=42` before `kfold_split()`. Any code path that calls `rng_*` after that point (SMOTE, dropout masks, noise injection, augmentation choice) consumes from the same shared stream — call order matters for exact reproducibility.
- **Fixed-size stack buffers:** several hot paths use fixed-size stack arrays sized from `config.h` constants (e.g. `float xv[251]` in `src/main.c:325`, `char vowel_paths[NUM_VOWELS][4096]` in `dataset.h`). Changing `TOTAL_FEATURES`/`FEATURES_PER_VOWEL` requires auditing these literals, not just `config.h`.
- **Cache/constant coupling:** `results/features.csv` encodes `TOTAL_FEATURES` in its column count; `features_load_csv()` rejects the cache silently (returns -1, triggers re-extraction) if `config.h` feature counts change. There is no versioning beyond this column-count check.
- **No inter-module circular dependencies observed:** the dependency graph is a DAG rooted at `main.c`; lower layers (`dsp_utils`, `wav_io`, `utils`) have no upward includes.
## Anti-Patterns
### Dormant/dead modules still compiled and linked
### In-RAM "checkpoint" naming conflated with disk persistence
### Disabled features left fully wired (BatchNorm, SWA)
### Fixed hard-coded per-vowel feature width duplicated in three places
## Error Handling
- File-open failures in loaders (`features_load_csv`, `wav_read`, `csv_parse`) return `-1` and are checked by the caller, which falls back to re-extraction/re-computation rather than crashing.
- NaN/Inf produced by degenerate audio (e.g., silence, all-zero signal) is swept to 0.0 in a post-extraction pass (`src/feature_extract.c:136-146`) rather than rejecting the patient.
- No error propagation beyond `int` return codes (0 success / -1 failure); no `errno`-style detail, just `log_error()` messages via `src/utils.c`.
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
