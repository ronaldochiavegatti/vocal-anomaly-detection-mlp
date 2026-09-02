<!-- refreshed: 2026-07-27 -->
# Architecture

**Analysis Date:** 2026-07-27

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                     WAV CORPUS (5 class directories)                    │
│  `saudavel/` `laringite/` `disfonia_psicogênica/` `disfonia_funcional/` │
│  `edema_de_reinke/`  +  `overview_merged.csv` (age/sex metadata)        │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATASET ENUMERATION            `src/dataset.c` (+ `src/csv_parser.c`)  │
│  Walks class dirs, builds Patient[] (id, class, sex, age, 3 WAV paths)  │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION (OpenMP, one thread/patient)  `src/feature_extract.c`│
│  ├─ `src/wav_io.c`         raw PCM16 WAV → float[-1,1] samples          │
│  ├─ `src/dsp_utils.c`      pre-emphasis, FFT, autocorr, wavelet denoise │
│  ├─ `src/feature_temporal.c`   10 feat/vowel (jitter/shimmer/HNR/ZCR)   │
│  ├─ `src/feature_spectral.c`   55 feat/vowel (F0,formants,MFCC+Δ+ΔΔ,   │
│  │                                CPP, glottal source Oq/Sq/NAQ/H1-H2)  │
│  └─ `src/feature_wavelet.c`    18 feat/vowel (DWT Daubechies-4 stats)   │
│  Output: `results/features.csv` — 1098 × 251 cached matrix              │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STRATIFIED 5-FOLD SPLIT                    `src/kfold.c`               │
│  Per-class round-robin shuffle, RANDOM_SEED=42, patient-level split     │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼   (repeated per fold, src/main.c:mode_train)
┌─────────────────────────────────────────────────────────────────────────┐
│  PER-FOLD PREPROCESSING                                                  │
│  ├─ Audio-domain augmentation on non-Normal classes (8×/sample)         │
│  │    `src/wav_augment.c` (noise/gain/stretch/pitch) — precomputed once │
│  │    for the whole dataset in `precalculate_augmentations()`,         │
│  │    then re-attached per fold in `collect_augmented_features()`      │
│  ├─ Z-score fit on ORIGINAL (non-augmented) train rows `src/normalize.c`│
│  └─ Per-vowel slicing: 251-dim vector → 3× (83 vowel feat + 2 meta)     │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         HIERARCHICAL LATE-FUSION ENSEMBLE  (9 MLPs trained per fold)    │
│                                                                           │
│   For each vowel v ∈ {a, i, u}:                                         │
│     ┌───────────────────────────┐   ┌────────────────────────────────┐ │
│     │ MASTER MLP (binary)       │   │ EXPERT MLP (4-class)            │ │
│     │ Normal(0) vs Pathol.(1)   │   │ Laring./Psicog./Func./Reinke    │ │
│     │ SMOTE-balanced, cw=[.9,1.1]│   │ SMOTE-balanced, cw=[1,1.2,1.2,1.4]│ │
│     │ `mlp_init_dynamic(...,2)` │   │ `mlp_init_dynamic(...,4)`        │ │
│     └───────────────────────────┘   └────────────────────────────────┘ │
│   Both trained via `src/mlp_train.c` (Adam, cosine LR, Macro-F1 early   │
│   stop) on network defined in `src/mlp.c`/`include/mlp.h`               │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LATE FUSION / PREDICTION   `predict_hierarchical_late_fusion()`        │
│  Average P(pathology) and P(expert class) across the 3 vowel-experts;  │
│  if avg P(pathology) < 0.5 → Normal, else argmax of averaged expert     │
│  probabilities (+1 to map back to 5-class label space)                 │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  METRICS                                    `src/metrics.c`             │
│  Confusion matrix, per-class P/R/F1, macro/weighted F1 → aggregated     │
│  over all 5 fold validation sets → `results/metrics_global.csv`         │
└─────────────────────────────────────────────────────────────────────────┘
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

**Overall:** Batch offline ML pipeline in pure C99, structured as a linear feature-engineering pipeline feeding a **hierarchical late-fusion ensemble** of small feedforward networks — not a single end-to-end model. There is no shared runtime/server; the program is invoked once per mode (`extract`, `train`/`full`) and exits.

**Key Characteristics:**
- No dynamic dispatch/polymorphism: C structs + free functions, one `.c`/`.h` pair per concern (SRP at file granularity).
- Hierarchical decomposition of the classification problem itself: a binary "sick vs. healthy" gate (Master) followed by a 4-way pathology classifier (Expert), rather than one flat 5-class softmax.
- Per-vowel model replication: the Master/Expert pair is trained independently for each of 3 sustained vowels (`/a/`, `/i/`, `/u/`), then fused by probability averaging at inference time (late fusion, not feature concatenation).
- Aggressive feature caching: `results/features.csv` is a materialized view of the entire (expensive, ~1min) extraction step, column-count-validated against `TOTAL_FEATURES` at load time so stale caches self-invalidate.
- OpenMP data parallelism at the patient level only (`#pragma omp parallel for` in `feature_extract.c` and in `precalculate_augmentations`); no other multithreading (training is single-threaded).
- Everything is parameterized through `include/config.h` preprocessor constants — no runtime config file, no CLI flags beyond `<mode> [base_dir]`.
- The codebase carries substantial **dormant code** from prior PRD iterations (feature selection, kNN/LogReg baselines, bootstrap CI, ROC/AUC, McNemar test, model checkpoint persistence, BatchNorm, SWA) that still compiles and links but is not exercised by the current `mode_train` path — see Anti-Patterns and dedicated section below.

## Layers

**Data Access Layer:**
- Purpose: turn on-disk WAV files + CSV metadata into an in-memory `Dataset`
- Location: `src/dataset.c`, `src/csv_parser.c`, `src/wav_io.c`
- Contains: directory walking, RIFF/PCM16 parsing, CSV field parsing
- Depends on: POSIX `dirent.h`/`sys/stat.h`, nothing else in the pipeline
- Used by: feature extraction orchestrator (`feature_extract.c`), augmentation precompute (`main.c`)

**Signal Processing / Feature Layer:**
- Purpose: convert raw audio samples into fixed-length numeric feature vectors
- Location: `src/dsp_utils.c`, `src/feature_temporal.c`, `src/feature_spectral.c`, `src/feature_wavelet.c`, `src/feature_extract.c`
- Contains: pure numeric DSP functions (no I/O except in the orchestrator)
- Depends on: `dataset.h`/`wav_io.h` types, `config.h` constants (frame sizes, F0 range, feature counts)
- Used by: `main.c` (both the cached-CSV path and the on-the-fly augmentation path via `extract_vowel_from_float`)

**Data Preparation Layer:**
- Purpose: fold splitting, normalization, class balancing
- Location: `src/kfold.c`, `src/normalize.c`, SMOTE code inline in `src/main.c`
- Contains: stratification logic, Z-score stats, k-NN-based synthetic sample generation
- Depends on: `FeatureMatrix`/`Dataset` types
- Used by: `main.c` fold loop

**Model Layer:**
- Purpose: neural network definition, training, and inference
- Location: `src/mlp.c`, `src/mlp_train.c`, `include/mlp.h`, `include/mlp_train.h`
- Contains: `Layer`/`MLP` structs, forward/backward pass, Adam optimizer, loss functions, checkpoint save/load (checkpoint here means best-epoch-in-RAM, not disk — see Anti-Patterns)
- Depends on: `config.h` (`MLP_NUM_LAYERS`, hidden sizes, dropout rates, Adam betas, etc.) and `utils.c` RNG/log helpers
- Used by: `main.c` (instantiates `net_master[3]` and `net_expert[3]` per fold)

**Orchestration / Application Layer:**
- Purpose: wire every other layer together into the two CLI modes
- Location: `src/main.c`
- Contains: `mode_extract()`, `mode_train()` (the actual pipeline), `mode_validate_external()` (stub, returns 0 without doing anything), CSV feature-cache load/store, hierarchical-fusion prediction, SMOTE, augmentation caching
- Depends on: every other layer
- Used by: nothing (entry point)

**Metrics Layer:**
- Purpose: turn predictions into reportable numbers
- Location: `src/metrics.c`, `include/metrics.h`
- Contains: confusion matrix, precision/recall/F1, plus a larger surface (bootstrap CI, ROC/AUC, PR curves, McNemar test, permutation importance) inherited from earlier single-model PRD iterations
- Depends on: `MLP` type (for permutation importance only)
- Used by: `main.c` — but only `metrics_compute`/`metrics_print`/`metrics_export_csv` are actually called in the current `mode_train`; the rest is dead from the orchestrator's perspective (still unit-testable/linkable)

## Data Flow

### Primary Request Path (training run: `./build/vocal_detect train .`)

1. `main()` dispatches to `mode_train(base_dir)` (`src/main.c:360-368`)
2. `dataset_load()` enumerates the 5 class directories into `Dataset` (`src/main.c:239`, `src/dataset.c:134`)
3. `features_load_csv()` attempts to reuse `results/features.csv`; on miss or column-count mismatch, `features_extract_all()` re-extracts (OpenMP-parallel over patients) and `features_export_csv()` re-caches (`src/main.c:241-244`, `src/feature_extract.c:102`)
4. `rng_seed(RANDOM_SEED)` then `kfold_split()` produce 5 stratified `FoldSplit`s (`src/main.c:245`, `src/kfold.c:13`)
5. `precalculate_augmentations()` generates 8 audio-domain augmentations per non-Normal patient ONCE for the whole dataset, feature-extracting each into `aug_cache` (`src/main.c:134-166`) — this is done before the fold loop to avoid recomputing per fold
6. For each of the 5 folds (`src/main.c:255-337`):
   a. Build `train_x_all`/`train_y_all` from the fold's train indices, then append augmented minority rows via `collect_augmented_features()` (`src/main.c:259-266`)
   b. `norm_fit()` on the ORIGINAL (non-augmented) `fold->n_train` rows, `norm_transform()` applied to both augmented-train and validation rows (`src/main.c:267-271`)
   c. For each vowel v ∈ {0,1,2}: slice the 251-dim vector down to the 85-dim per-vowel-plus-metadata vector (`src/main.c:276-286`)
   d. Map 5-class labels to binary (Normal=0/Pathological=1), SMOTE-balance, train Master MLP (`mlp_init_dynamic(...,2)` + `mlp_train`) (`src/main.c:287-295`)
   e. Filter to pathological-only rows, remap to 0-3, SMOTE-balance, train Expert MLP (`mlp_init_dynamic(...,4)` + `mlp_train`) (`src/main.c:297-312`)
   f. On the fold's validation set, run `predict_hierarchical_late_fusion()` per sample, averaging Master/Expert softmax outputs across the 3 vowel models (`src/main.c:319-332`)
   g. `metrics_compute()`/`metrics_print()` on this fold's predictions (`src/main.c:333-334`)
7. After all folds: aggregate accuracy/Macro-F1 across folds, and recompute global metrics on the concatenated out-of-fold predictions → `results/metrics_global.csv` (`src/main.c:338-341`)

### Feature Extraction Path (`./build/vocal_detect extract .`)

1. `mode_extract()` → `dataset_load()` → `features_extract_all()` → `features_export_csv()` (`src/main.c:345-352`)
2. `features_extract_all()` parallelizes over patients with `#pragma omp parallel for reduction(+:errors) schedule(dynamic, 4)`, each thread calling `extract_vowel_features()` for all 3 vowels + writing 2 metadata features (`src/feature_extract.c:102-149`)
3. Post-loop NaN/Inf sanitization replaces invalid values with 0 (`src/feature_extract.c:136-146`)

**State Management:**
- No persistent application state between runs beyond the two on-disk caches: `results/features.csv` (feature cache, self-invalidating) and stale `models/*.bin` files (see Anti-Patterns — no longer written by the current `mode_train`).
- Within a run, all state is stack/heap-allocated C structs passed by pointer; RNG state is a single global seeded once via `rng_seed(RANDOM_SEED)` in `src/utils.c`.

## Key Abstractions

**`Dataset` / `Patient`:**
- Purpose: in-memory catalogue of every patient's class label, demographic metadata, and the 3 WAV file paths needed for feature extraction
- Examples: `src/dataset.c`, `include/dataset.h`
- Pattern: flat array of fixed-size structs (`Patient patients[]`), no per-record heap pointers except the struct itself

**`FeatureMatrix`:**
- Purpose: dense row-major `count × TOTAL_FEATURES` matrix + parallel `labels[]` array — the single source of truth once extraction is done
- Examples: `include/feature_extract.h`
- Pattern: plain `float*`/`int*` buffers with manual `count`/`num_features` bookkeeping; consumed directly by fold-splitting, normalization, and per-vowel slicing code

**`MLP` / `Layer`:**
- Purpose: a small feedforward network (this codebase currently always instantiates 2 hidden layers via `MLP_NUM_LAYERS=3`, i.e. Input→Hidden1→Hidden2→Output)
- Examples: `include/mlp.h`, `src/mlp.c`
- Pattern: `Layer[MLP_NUM_LAYERS]` fixed-size array inside `MLP`; `mlp_init_dynamic(net, input_size, output_size)` is the variant actually used because per-vowel-per-network input/output sizes vary (85→2 for Master, 85→4 for Expert); `mlp_init()`/`MLP_OUTPUT_SIZE` (5-class legacy path) is unused by `main.c`.

**`FoldSplit` / `KFoldSplits`:**
- Purpose: precomputed train/val index arrays per fold, stratified by class at the patient level
- Examples: `include/kfold.h`, `src/kfold.c`
- Pattern: index-array based (no data copying at split time — copying happens later per-fold in `main.c`)

**Hierarchical Late-Fusion Ensemble (design pattern, not a struct):**
- Purpose: decompose 5-class classification into an easier binary decision (Normal vs. Pathological) followed by a 4-way decision only among pathological classes, then average that decomposition's confidence across 3 independent per-vowel views of the same patient
- Examples: `net_master[3]`, `net_expert[3]` arrays in `src/main.c:273`, fusion logic in `predict_hierarchical_late_fusion()` (`src/main.c:39-69`)
- Pattern: 2 (Master/Expert) × 3 (vowels) = 6 independently trained MLPs per fold, 30 MLPs total across 5-fold CV; probabilities are averaged (not stacked/concatenated) at inference — this is what "late fusion" means here, as opposed to feeding all 3 vowels' features into one wider network

## Entry Points

**`main()` (`src/main.c:360`):**
- Location: `src/main.c`
- Triggers: CLI invocation `./build/vocal_detect <mode> [base_dir]`
- Responsibilities: parse mode string (`extract`|`train`|`full`|`external`), set log level, dispatch

**`mode_extract()` (`src/main.c:345`):**
- Location: `src/main.c`
- Triggers: `mode == "extract"`
- Responsibilities: dataset load → feature extraction → CSV export only (no training)

**`mode_train()` (`src/main.c:235`):**
- Location: `src/main.c`
- Triggers: `mode == "train"` or `mode == "full"` (both are aliases — there is no separate "load cache vs. force re-extract" distinction beyond the cache validity check already inside `mode_train`)
- Responsibilities: the entire pipeline described in Data Flow above

**`mode_validate_external()` (`src/main.c:354`):**
- Location: `src/main.c`
- Triggers: `mode == "external"`
- Responsibilities: **stub only** — logs a message and returns 0 immediately; the docstring-style comment says "Implementar carregando os 6 best_models se necessario" (not implemented). Do not assume external validation works.

**`make test` target (`Makefile:27`):**
- Triggers `./build/vocal_detect test`, but no `mode == "test"` branch exists in `main()` — running this returns exit code 1 with the usage message. Treat as non-functional.

## Architectural Constraints

- **Threading:** OpenMP `parallel for` only around (1) the per-patient feature extraction loop in `features_extract_all()` and (2) the per-patient augmented-feature precomputation loop in `precalculate_augmentations()`. Training itself (`mlp_train`) is single-threaded; do not assume thread-safety of `MLP`/`Layer` structs — each fold/vowel/network combination must own its own `MLP` instance (the code already does this via `net_master[3]`/`net_expert[3]` arrays, never sharing one `MLP` across threads).
- **Global state:** RNG is process-global (`rng_seed()`/`rng_uniform()`/`rng_normal()` in `src/utils.c`) and reseeded exactly once at `RANDOM_SEED=42` before `kfold_split()`. Any code path that calls `rng_*` after that point (SMOTE, dropout masks, noise injection, augmentation choice) consumes from the same shared stream — call order matters for exact reproducibility.
- **Fixed-size stack buffers:** several hot paths use fixed-size stack arrays sized from `config.h` constants (e.g. `float xv[251]` in `src/main.c:325`, `char vowel_paths[NUM_VOWELS][4096]` in `dataset.h`). Changing `TOTAL_FEATURES`/`FEATURES_PER_VOWEL` requires auditing these literals, not just `config.h`.
- **Cache/constant coupling:** `results/features.csv` encodes `TOTAL_FEATURES` in its column count; `features_load_csv()` rejects the cache silently (returns -1, triggers re-extraction) if `config.h` feature counts change. There is no versioning beyond this column-count check.
- **No inter-module circular dependencies observed:** the dependency graph is a DAG rooted at `main.c`; lower layers (`dsp_utils`, `wav_io`, `utils`) have no upward includes.

## Anti-Patterns

### Dormant/dead modules still compiled and linked

**What happens:** `src/knn.c`, `src/logreg.c`, and `src/feature_select.c` (plus large portions of `src/metrics.c`: `metrics_bootstrap_ci`, `metrics_roc_auc`, `metrics_pr_curve`, `metrics_mcnemar`, `metrics_permutation_importance`) are compiled by the wildcard `Makefile` rule and their headers are `#include`d in `src/main.c`, but none of their functions are actually called from `mode_train()` in the current (v29, Hierarchical Late Fusion) architecture.

**Why it's wrong:** Anyone reading `CLAUDE.md`'s "Methodological Notes" (McNemar test, bootstrap CI, baselines) or `include/metrics.h`'s rich API will assume these run every training pass; they don't. `results/baselines.csv`, `results/roc_curves.csv`, `results/pr_curves.csv`, `results/feature_importance.csv` described in `CLAUDE.md`'s Output Files section are **not produced** by the current `mode_train`.

**Do this instead:** Before trusting any doc that references these outputs, grep `src/main.c` for the function name first. If reviving one of these (e.g., McNemar to compare Master+Expert fusion vs. a flat baseline), wire the call explicitly into the fold loop in `src/main.c` and confirm the corresponding CSV appears in `results/` after a run.

### In-RAM "checkpoint" naming conflated with disk persistence

**What happens:** `mlp_save_checkpoint()`/`mlp_load_checkpoint()` in `src/mlp.c` copy weights into RAM buffers for early-stopping rollback within `mlp_train()` — no file I/O involved. Separately, `mlp_save()`/`mlp_load()` (disk-based, binary format) exist in `mlp.h` but are **never called** from `src/main.c`. The `models/*.bin` files present on disk (`models/best_master.bin`, `models/mlp_fold*.bin`, etc.) are stale artifacts from an earlier PRD iteration (April) and are not regenerated by the current training run.

**Why it's wrong:** A reader expecting `make full` to leave usable trained models in `models/` (as `CLAUDE.md`'s Output Files section states) will find only outdated files that don't correspond to the current 251-feature/late-fusion architecture — there is currently no way to reload a trained fold's Master/Expert networks for inference outside of the training run itself.

**Do this instead:** If persistence is needed, add explicit `mlp_save(&net_master[v], path)` / `mlp_save(&net_expert[v], path)` calls at the end of each fold's vowel loop in `mode_train()`, using a naming scheme that encodes fold and vowel (e.g. `models/master_fold{k}_v{vowel}.bin`).

### Disabled features left fully wired (BatchNorm, SWA)

**What happens:** `mlp_init_dynamic()` hard-codes `int use_bn[] = { 0, 0, 0 }` (`src/mlp.c`), so `BatchNorm` is always disabled — yet the full BN forward/backward/running-stats/checkpoint machinery remains in `mlp.c` and `mlp_train.c` (including a whole `update_bn_stats_from_data()` helper that iterates the network doing nothing useful when BN is off). Similarly, `mlp_train()` allocates and accumulates `swa_weights`/`swa_biases` (Stochastic Weight Averaging) every `swa_freq` epochs after `swa_start`, but the averaged weights are **never written back into `net`** — they are computed, then freed.

**Why it's wrong:** This is exactly the "what doesn't work" list from `CLAUDE.md` (BN hurts small datasets; SWA doesn't help) implemented as live, executing-but-inert code, which wastes CPU cycles per epoch and confuses future readers into thinking these techniques are active.

**Do this instead:** Treat `use_bn[]` and the SWA accumulation loop as intentionally disabled experiments. If re-enabling either, do so by flipping `use_bn[i]` to 1 (retest per `CLAUDE.md`'s A/B protocol) or by adding the missing `mlp_load_checkpoint`-equivalent step that copies `swa_weights`/`swa_biases` (divided by `swa_count`) into `net->layers[i].weights/biases` at the end of training. If not re-enabling, consider removing the dead computation to reduce per-epoch cost and reader confusion (this is a candidate CONCERNS.md item, not addressed here).

### Fixed hard-coded per-vowel feature width duplicated in three places

**What happens:** The per-vowel feature width (`FEATURES_PER_VOWEL + NUM_METADATA_FEATURES` = 85) is recomputed inline in at least three separate spots in `src/main.c` (`extract_vowel_from_float`'s caller, the fold-loop vowel-slicing block, and the inference-time `float xv[251]` buffer at `src/main.c:325` — note this one is actually sized to `TOTAL_FEATURES`, not the per-vowel width, oversized but functionally safe since only the first 85 bytes are used).

**Why it's wrong:** Any future change to `NUM_METADATA_FEATURES` or `FEATURES_PER_VOWEL` requires updating all these call sites consistently; a mismatch would silently read/write out of bounds or wrong offsets since there is no single named constant like `NF_VOWEL_MODEL` used everywhere.

**Do this instead:** When modifying feature widths, grep for `FEATURES_PER_VOWEL` and `NUM_METADATA_FEATURES` across `src/main.c` and `src/feature_extract.c` and update every occurrence together; consider introducing a single `NF_VOWEL_MODEL` constant in `config.h` if this file is touched again.

## Error Handling

**Strategy:** Fail-fast for unrecoverable allocation errors (`safe_malloc`/`safe_calloc`/`safe_realloc` in `src/utils.c` abort the process on OOM rather than returning NULL), but tolerant/silent for per-record data errors (a missing or corrupt WAV file zero-fills that vowel's feature slice and increments an error counter rather than aborting the whole extraction — `src/feature_extract.c:36-40`).

**Patterns:**
- File-open failures in loaders (`features_load_csv`, `wav_read`, `csv_parse`) return `-1` and are checked by the caller, which falls back to re-extraction/re-computation rather than crashing.
- NaN/Inf produced by degenerate audio (e.g., silence, all-zero signal) is swept to 0.0 in a post-extraction pass (`src/feature_extract.c:136-146`) rather than rejecting the patient.
- No error propagation beyond `int` return codes (0 success / -1 failure); no `errno`-style detail, just `log_error()` messages via `src/utils.c`.

## Cross-Cutting Concerns

**Logging:** Custom leveled logger in `src/utils.c` (`LOG_DEBUG`/`LOG_INFO`/`LOG_WARN`/`LOG_ERROR`), output to `stderr` by default, level set once in `main()` via `log_set_level(LOG_INFO)`. Training progress is logged every 10 epochs plus epoch 1 (`src/mlp_train.c:270-273`).

**Validation:** Feature-cache schema validation is the only formal validation gate (`features_load_csv` column-count check against `TOTAL_FEATURES`). There is no schema validation on `overview_merged.csv` beyond best-effort field parsing in `csv_parser.c`; unmatched patient IDs simply keep default `sex='?'`/`age=0`.

**Determinism/Reproducibility:** All stochastic behavior (fold assignment, SMOTE neighbor/interpolation choice, dropout masks, Gaussian noise injection, He-initialization weights) flows through the single global RNG seeded once with `RANDOM_SEED=42`; call-order sensitivity means restructuring the pipeline (e.g., reordering fold loop vs. augmentation precompute) will change results even with the same seed.

---

*Architecture analysis: 2026-07-27*
