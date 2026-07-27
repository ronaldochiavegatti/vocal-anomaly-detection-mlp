---
phase: 00-statistical-infrastructure-rng-reproducibility-prerequisite
plan: 02
subsystem: infra
tags: [c99, metrics, bootstrap-ci, mcnemar, baselines, statistical-rigor]

# Dependency graph
requires:
  - "00-01: RNG race fix (precalculate_augmentations is deterministic under RANDOM_SEED=42)"
provides:
  - "mode_train() computes per-fold MajorityClass/kNN(k=5)/LogisticRegression baseline predictions aligned index-for-index with the MLP's out-of-fold all_y_true/all_y_pred arrays"
  - "mode_train() calls metrics_bootstrap_ci() (N=1000, seed=RANDOM_SEED, 7 metrics) and metrics_mcnemar() x3 (vs MajorityClass/kNN/LogisticRegression) on the aggregated out-of-fold predictions, as the last RNG-consuming step in the function"
  - "results/bootstrap_ci.csv and results/mcnemar_vs_baselines.csv exported on every training run"
affects: [01-gap2-borderline-smote, 02-gap3-shallow-vs-deep, 03-gap1-paraconsistent-selection, 00-03-baseline-reconfirmation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Baseline-classifier reconnection pattern: compute dormant kNN/LogReg baselines on the exact same pre-SMOTE, already-normalized train_x_all/val_x_all fold buffers the MLP consumes, writing into aggregate arrays at the identical all_count index used by the MLP's own prediction loop -- eliminates a whole class of index-misalignment bugs in statistical comparisons"
    - "RNG-reseeding call ordering: any function that internally re-seeds the global RNG (metrics_bootstrap_ci) must be the last RNG consumer in the enclosing function, documented inline at the call site, not just in planning docs"

key-files:
  created: []
  modified:
    - src/main.c

key-decisions:
  - "Baseline computation placed BEFORE the per-vowel SMOTE+MLP training loop within each fold (not after), so LogisticRegression's internal RNG draws (rng_normal/rng_shuffle_int in lr_train's Adam training loop) happen first and shift the sequence consumed afterward by smote_oversample/mlp_init_dynamic/dropout -- this is the plan's explicitly accepted ordering, not a bug"
  - "Majority-class baseline counts ONLY the original (non-augmented) fold->n_train rows of train_y_all, matching the existing norm_fit precedent in CLAUDE.md ('fit on ORIGINAL training samples only, not augmented')"
  - "kNN(k=5) and LogisticRegression baselines are trained on the FULL n_train_aug/nf_all-dimensioned, already-normalized fold data (same buffers as the MLP), not a reduced/original-only subset -- this maximizes apples-to-apples comparability with the MLP's own training distribution"

patterns-established:
  - "Any future statistical-comparison addition to mode_train() should reuse the all_count index (not a separately tracked offset) when writing into a new aggregate array, per the scientific-integrity threat (T-00-04) this plan mitigated"

requirements-completed: [INFRA-01]

# Metrics
duration: 25min
completed: 2026-07-27
---

# Phase 0 Plan 2: Bootstrap CI + 3-Way McNemar Statistical Infrastructure Summary

**Reconnected the dormant kNN/LogisticRegression baseline classifiers and wired the already-implemented `metrics_bootstrap_ci()`/`metrics_mcnemar()` into `mode_train()`, so every training run now reports a 95% bootstrap CI (7 metrics) and 3-way McNemar significance test (vs MajorityClass, kNN, LogisticRegression) on the aggregated out-of-fold predictions instead of a bare point-estimate Macro F1.**

## Performance

- **Duration:** ~25 min (pure code-edit + `make` build verification; no `make full` pipeline run required for this plan)
- **Completed:** 2026-07-27
- **Tasks:** 2
- **Files modified:** 1 (`src/main.c`)

## Accomplishments

- **Task 1:** Added `all_y_pred_majority`/`all_y_pred_knn`/`all_y_pred_logreg` aggregate arrays (sized `fm.count`), computed per-fold on the same `train_x_all`/`val_x_all`/`nf_all` fold buffers the MLP uses:
  - MajorityClass: argmax over class counts of `train_y_all[0, fold->n_train)` (original samples only, matching `norm_fit`'s established precedent)
  - kNN(k=5) via `knn_predict()` on the full `n_train_aug` (post-augmentation) training buffer
  - LogisticRegression via `lr_init()`/`lr_train()`/`lr_free()` on the same full training buffer
  - All three written into the aggregate arrays at the identical `all_count` index used by `all_y_true`/`all_y_pred` in the existing per-sample validation loop, eliminating any risk of index drift between the MLP's predictions and the baselines'
- **Task 2:** Added, immediately after the existing point-estimate `metrics_compute`/`metrics_print`/`metrics_export_csv` block:
  - `metrics_bootstrap_ci(all_y_true, all_y_pred, all_count, 1000, RANDOM_SEED, ci)` — placed as the LAST RNG-consuming call in `mode_train()`, per the function's internal `rng_seed()` reseed documented in `src/metrics.c`
  - Three `metrics_mcnemar()` calls: MLP vs MajorityClass, vs kNN, vs LogisticRegression
  - `log_info()` output for all 7 CI metrics (`mean [lower, upper]`) and all 3 McNemar results (`chi2`/`p_value` plus an explicit Portuguese significance verdict sentence)
  - CSV export: `results/bootstrap_ci.csv` (8 lines: header + 7 metric rows) and `results/mcnemar_vs_baselines.csv` (4 lines: header + 3 baseline rows)
  - Freed the three new baseline aggregate arrays alongside the existing `free(all_y_true)`/etc. at the end of `mode_train()`

## Task Commits

Each task was committed atomically:

1. **Task 1: Compute per-fold MajorityClass/kNN/LogReg baseline predictions** - `0edc3a0` (feat)
2. **Task 2: Wire metrics_bootstrap_ci()/metrics_mcnemar() into end-of-function reporting** - `88c05c0` (feat)

## Files Created/Modified

- `src/main.c` — `mode_train()`:
  - New allocations: `all_y_pred_majority`, `all_y_pred_knn`, `all_y_pred_logreg` (before the fold loop)
  - New per-fold block (after `norm_transform(val_x_all, ...)`, before the vowel loop): `val_y_all`, majority-class `counts[NUM_CLASSES]`/`majority_class`, `knn_pred_buf` via `knn_predict()`, `logreg_pred_buf` via `lr_init`/`lr_train`/`lr_free`
  - Three new lines in the existing per-sample validation loop writing into the baseline aggregate arrays at `all_count`
  - New frees at end of fold iteration: `val_y_all`, `knn_pred_buf`, `logreg_pred_buf`
  - New end-of-function block: `ConfidenceInterval ci[CI_N_METRICS]` + `metrics_bootstrap_ci()`; `chi2_maj/p_maj`, `chi2_knn/p_knn`, `chi2_lr/p_lr` + 3× `metrics_mcnemar()`; `log_info()` reporting; `fopen`/`fprintf`/`fclose` export of both CSVs; frees of the three baseline aggregate arrays

## Decisions Made

- Baseline computation placed before the per-vowel SMOTE+MLP training loop (per the plan's explicit RNG-ordering note): `lr_train()`'s internal Adam-training RNG draws happen first in each fold, shifting the subsequent `rng_*` sequence consumed by SMOTE/dropout/noise injection relative to a hypothetical baseline-free run. This is documented as an accepted, deliberate ordering rather than a bug — matching plan 00-02's objective section and plan 00-03's divergence-hypothesis text.
- Majority-class counts restricted to `train_y_all[0, fold->n_train)` (original, non-augmented rows), consistent with the existing `norm_fit(train_x_all, fold->n_train, ...)` precedent already documented in `CLAUDE.md`.
- kNN and LogisticRegression baselines trained on the full `n_train_aug`-dimensioned buffer (post-augmentation), matching exactly what the MLP itself trains on downstream, for maximum comparability.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reworded an inline comment to avoid a false-positive grep match**
- **Found during:** Task 2 acceptance-criteria verification
- **Issue:** The plan's acceptance criterion `grep -c 'metrics_bootstrap_ci(' src/main.c` equals `1` initially returned `2`, because an inline comment explaining the RNG-reseed ordering constraint mentioned `metrics_bootstrap_ci()` by name (with parentheses), which the grep pattern also matched.
- **Fix:** Reworded the comment to say "esta funcao re-semeia o RNG global internamente" instead of repeating the function name with parentheses, preserving the same documentation intent without tripping the literal grep count.
- **Files modified:** `src/main.c` (comment only, no functional change)
- **Verification:** `grep -c 'metrics_bootstrap_ci(' src/main.c` now returns `1`; `make` rebuild confirmed no new warnings.
- **Committed in:** `88c05c0` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (grep false-positive from a self-referential comment, no functional impact)
**Impact on plan:** None — purely a comment wording adjustment to satisfy the plan's own literal acceptance-criteria check; the underlying code behavior is unchanged from the plan's specification.

## Verification Results

- `make` rebuilds with zero errors and exactly the same 2 pre-existing warnings seen at HEAD before this plan (`external_dir` unused parameter in `mode_validate_external`, `fgets` ignored return value in `features_load_csv`). The third pre-existing warning mentioned in the plan's acceptance criteria (`best_val_acc` unused variable in `mlp_train.c`) was not observed in this build because `mlp_train.c` was not recompiled (unchanged object file reused by `make`); it is unrelated to this plan's `src/main.c`-only changes.
- `grep -c 'knn_predict(' src/main.c` → `1`
- `grep -c 'lr_train(' src/main.c` → `1`
- `grep -c 'all_y_pred_majority\[all_count\]' src/main.c` → `1`
- `grep -c 'all_y_pred_knn\[all_count\]' src/main.c` → `1`
- `grep -c 'all_y_pred_logreg\[all_count\]' src/main.c` → `1`
- `grep -c 'metrics_bootstrap_ci(' src/main.c` → `1`
- `grep -c 'metrics_mcnemar(' src/main.c` → `3`
- Full runtime confirmation (an actual `make full`/`make train` execution producing well-formed `results/bootstrap_ci.csv` with exactly 8 lines and `results/mcnemar_vs_baselines.csv` with exactly 4 lines) is explicitly deferred to plan 00-03's full pipeline run, per this plan's own `<verification>` section (a 60-90 minute `make full` execution should not be duplicated across plans in Phase 0).

## Issues Encountered

None beyond the grep false-positive documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- INFRA-01 is satisfied at the code level: `mode_train()` now computes 3-way baseline predictions and calls both `metrics_bootstrap_ci()` and `metrics_mcnemar()` (x3) on the aggregated out-of-fold predictions, with CSV export logic in place.
- Plan 00-03 (baseline reconfirmation / full pipeline run) will provide the first actual runtime evidence that `results/bootstrap_ci.csv` (8 lines) and `results/mcnemar_vs_baselines.csv` (4 lines) are produced correctly, and will reconcile any Macro F1 delta against the SPEC.md-cited reference baseline (0.4423 / 69.4%) against the two RNG-sequence-shifting causes already documented in this plan's objective (00-01's RNG-race fix + this plan's newly-inserted baseline RNG consumption before SMOTE/MLP training).
- No blockers identified for plan 00-03.

---
*Phase: 00-statistical-infrastructure-rng-reproducibility-prerequisite*
*Completed: 2026-07-27*

## Self-Check: PASSED

- FOUND: src/main.c
- FOUND: .planning/phases/00-statistical-infrastructure-rng-reproducibility-prerequisite/00-02-SUMMARY.md
- FOUND: commit 0edc3a0
- FOUND: commit 88c05c0
