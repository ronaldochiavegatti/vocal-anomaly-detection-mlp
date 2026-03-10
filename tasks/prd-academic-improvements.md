# PRD: Academic Rigor Improvements for Vocal Anomaly Detection Pipeline

## Introduction

The current vocal anomaly detection pipeline achieves 80.6% accuracy and Macro F1=0.612 on a 5-fold stratified CV, but lacks the evaluation rigor and comparative context required for academic publication. This PRD specifies 7 full-featured improvements — targeting biomedical journal standards — to be implemented in a specific order to minimize integration risk.

**Context:** Pure C pipeline, 918 patients (687 Normal / 140 Laryngite / 91 Disfonia), MLP with 150→~114 features, 5-fold CV.

---

## Goals

- Enable reproducible inference by persisting feature selection state per fold
- Provide AUC/ROC and Precision-Recall curves required by biomedical reviewers
- Report 95% confidence intervals on all metrics (not just point estimates)
- Quantify per-feature and per-group clinical interpretability via permutation importance
- Contextualize MLP against 3 standard baselines (majority class, logistic regression, kNN)
- Eliminate indirect data leakage in hyperparameter selection via nested CV
- Generate physically-grounded augmented samples via audio-domain augmentation

---

## User Stories

### US-001: Save Selected Feature Indices Per Fold
**Description:** As a researcher, I want to persist which features were selected in each fold so that I can run inference on new patients without re-running the full pipeline.

**Acceptance Criteria:**
- [ ] After `select_features()` in each fold, save `models/selected_fold{k}.bin` with format `[int n_selected][int idx_0]...[int idx_{n-1}]`
- [ ] Functions `selected_save(path, indices, n)` and `selected_load(path, indices, n)` implemented (in `src/feature_select.c` or `src/normalize.c`)
- [ ] Running `make train` produces 5 files: `models/selected_fold0.bin` … `models/selected_fold4.bin`
- [ ] `make` compiles with 0 errors and 0 relevant warnings

---

### US-002: ROC Curves and AUC Per Class
**Description:** As a researcher, I want per-class ROC and Precision-Recall curves so that I can report AUC in publications and assess model discrimination on imbalanced classes.

**Acceptance Criteria:**
- [ ] Softmax probabilities `all_y_prob[N][3]` accumulated across all folds (parallel to `all_y_pred`)
- [ ] `metrics_roc_auc(y_true, y_prob, n_samples, auc_out)` implemented in `src/metrics.c` using one-vs-rest + trapezoidal rule
- [ ] `metrics_pr_curve(y_true, y_prob, n_samples)` implemented in `src/metrics.c`
- [ ] `results/roc_curves.csv` written with columns: `class,threshold,tpr,fpr`
- [ ] `results/pr_curves.csv` written with columns: `class,threshold,precision,recall`
- [ ] AUC per class printed to stdout after final evaluation
- [ ] `make` compiles with 0 errors and 0 relevant warnings

---

### US-003: Bootstrap Confidence Intervals (N=1000)
**Description:** As a researcher, I want 95% confidence intervals on accuracy, Macro F1, and per-class F1 so that I can report statistically valid results.

**Acceptance Criteria:**
- [ ] `struct ConfidenceInterval { float mean; float lower; float upper; }` added to `include/metrics.h`
- [ ] `metrics_bootstrap_ci(y_true, y_pred, n_samples, n_bootstrap, results)` implemented in `src/metrics.c`; resamples with replacement N=1000 times, computes percentiles 2.5 and 97.5
- [ ] CI section appended to `results/metrics_global.csv` with columns: `metric,mean,ci_lower,ci_upper`
- [ ] Metrics covered: accuracy, macro_f1, f1_normal, f1_laryngite, f1_disfonia
- [ ] Runtime overhead < 500ms (no re-training)
- [ ] `make` compiles with 0 errors and 0 relevant warnings

---

### US-004: Feature Importance via Permutation
**Description:** As a researcher, I want to know which acoustic features most affect diagnosis so I can provide clinical interpretability in publications.

**Acceptance Criteria:**
- [ ] `metrics_permutation_importance(X_val, y_true, net, norm, selected, n_selected, importance_out)` implemented in `src/metrics.c`; for each feature j, shuffles column j, re-evaluates accuracy and Macro F1, records `importance[j] = accuracy_original - accuracy_permuted`
- [ ] Group-level permutation computed for 3 groups: Temporal (features 0–29), Spectral (30–95), Wavelet (96–149)
- [ ] `results/feature_importance.csv` written with columns: `feature_idx,importance_acc,importance_f1`
- [ ] Group summary printed to stdout after each fold
- [ ] `make` compiles with 0 errors and 0 relevant warnings

---

### US-005: Baselines for Comparison
**Description:** As a researcher, I want to compare the MLP against simple baselines so that I can contextualize the 80.6% result in a publication.

**Acceptance Criteria:**
- [ ] **Majority class baseline**: always predicts class 0 (Normal); accuracy, Macro F1, per-class F1 computed in `metrics.c`
- [ ] **Logistic Regression baseline**: MLP with 0 hidden layers (Input → Dense(3) + Softmax); use `mlp_init_dynamic` with `n_hidden=0`; trained with same Adam + early stopping as main model
- [ ] **kNN (k=5) baseline**: implemented in `src/knn.c` + `include/knn.h`; Euclidean distance on normalized features; majority vote among 5 nearest neighbors
- [ ] Comparison table printed to stdout: `Method | Accuracy | Macro F1 | F1_Normal | F1_Laryngite | F1_Disfonia`
- [ ] Results saved to `results/baselines.csv` with same columns
- [ ] All baselines use same train/val splits, same normalization, same feature selection as main model
- [ ] `make` compiles with 0 errors and 0 relevant warnings

---

### US-006: Nested Cross-Validation for Threshold Selection
**Description:** As a researcher, I want to select feature selection thresholds via inner CV so that hyperparameter choices are unbiased and defensible to reviewers.

**Acceptance Criteria:**
- [ ] `inner_cv_select_thresholds(train_data, n_train, best_var_thresh, best_corr_thresh)` implemented in `src/main.c`
- [ ] Inner grid: `var_threshold ∈ {0.005, 0.01, 0.02}`, `corr_threshold ∈ {0.90, 0.95, 0.98}` (9 combinations)
- [ ] Inner loop: 3-fold CV on the outer fold's training set; selects combination maximizing inner val Macro F1
- [ ] Selected thresholds logged per outer fold: `Fold k: var=X corr=Y → n_features=Z`
- [ ] `select_features()` already accepts thresholds as parameters — use those, do not hardcode
- [ ] `make` compiles with 0 errors and 0 relevant warnings

---

### US-007: Audio-Domain Augmentation
**Description:** As a researcher, I want to generate augmented WAV samples from minority classes using physical audio transformations so that my augmentation methodology is more scientifically sound than feature-space SMOTE.

**Acceptance Criteria:**
- [ ] `src/wav_augment.c` and `include/wav_augment.h` created with 4 functions:
  - `wav_add_noise(samples, n, snr_db)` — additive white Gaussian noise at specified SNR
  - `wav_pitch_shift(samples, n, sample_rate, semitones)` — ±1–2 semitone shift
  - `wav_time_stretch(samples, n, factor)` — ±5–10% duration change preserving spectrum
  - `wav_gain_perturb(samples, n, db)` — ±3 dB amplitude scaling
- [ ] Augmentation applied **only** to Laryngite and Disfonia patients in each fold's training set
- [ ] Augmented samples fed into feature extraction pipeline, not stored as WAV files (to preserve feature caching)
- [ ] `main.c` calls augmentation before feature extraction in training folds
- [ ] No data leakage: augmented samples never appear in validation sets
- [ ] `make` compiles with 0 errors and 0 relevant warnings

---

## Functional Requirements

- FR-1: `make train` must produce `models/selected_fold{0-4}.bin` in addition to existing model and norm files
- FR-2: `results/roc_curves.csv` and `results/pr_curves.csv` must be written after every full training run
- FR-3: `results/metrics_global.csv` must include a bootstrap CI section with 5 metric rows
- FR-4: `results/feature_importance.csv` must be written after every full training run
- FR-5: `results/baselines.csv` must be written and comparison table printed to stdout after every full training run
- FR-6: Nested CV thresholds must be logged per fold during training
- FR-7: Audio augmentation must integrate with the existing feature caching mechanism without breaking `results/features.csv` compatibility
- FR-8: All new code must compile with `gcc -O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -lm` with 0 errors
- FR-9: No existing hyperparameters in `config.h` may be changed without explicit justification
- FR-10: `mlp_init_dynamic(net, input_size)` must be used wherever input size varies; `MLP_NUM_LAYERS=3` must not be altered

---

## Non-Goals

- No GUI or visualization (CSV output only; plots are external)
- No repeated K-Fold (10×5-fold) — too costly for iterative development
- No new feature engineering (inter-vowel differences, etc. — known to hurt)
- No changes to existing hyperparameters (lr, dropout, L2, class weights)
- No SWA, ensemble averaging, focal loss, BatchNorm, Mixup (all known to hurt)
- No inference mode (`predict` command) — only training/evaluation pipeline
- No automatic CSV invalidation when augmentation is toggled

---

## Technical Considerations

- `MLP_NUM_LAYERS=3` is a preprocessor constant used with `#if` guards in `mlp.c` — do not change without auditing all `#if MLP_NUM_LAYERS` guards
- `mlp_init_dynamic(net, input_size)` must be used for any model with variable input (baselines, nested CV variants)
- Feature indices in `selected_fold{k}.bin` are indices into the post-extraction 150-feature space, not the original WAV features
- kNN must operate on Z-score normalized features (same `NormParams` used for MLP)
- Bootstrap must use a fixed seed (e.g., seed=42) for reproducibility; document the seed in output CSV header
- Permutation importance must be computed on held-out validation data only (not training data)
- Audio augmentation functions operate on `int16_t` PCM samples (same format as `wav_io.c` output)

---

## Implementation Order

Per TASK.md (ordered to minimize integration risk):

1. **US-001** — Feature index persistence (prerequisite for reproducibility)
2. **US-002** — ROC/AUC (requires saving softmax probs; low risk)
3. **US-003** — Bootstrap CI (operates on existing predictions; lowest risk)
4. **US-004** — Feature importance (requires model + validation data)
5. **US-005** — Baselines (new files, minimal touch to existing code)
6. **US-006** — Nested CV (refactors training loop; highest integration risk)
7. **US-007** — Audio augmentation (new module; affects feature extraction path)

---

## Success Metrics

- All 7 improvements compile cleanly and produce their expected output files
- Bootstrap CI width for Macro F1 < 0.05 (indicates stable estimate)
- MLP outperforms all 3 baselines on both Accuracy and Macro F1
- Feature importance identifies at least one dominant feature group (Temporal/Spectral/Wavelet)
- Nested CV thresholds vary across folds (confirming the search is non-trivial)
- No regression in main model accuracy (must remain ≥ 80.0% after integration)

---

## Open Questions

- Should `wav_time_stretch` use linear interpolation or sinc resampling? (sinc is higher quality but ~3× slower)
- For the kNN baseline, should distance be computed on all ~114 selected features or on all 150 pre-selection features?
- Should `results/baselines.csv` include per-fold breakdown or only aggregate metrics?
- Should augmented samples be counted in the SMOTE step or replace SMOTE for minority classes?
