---
phase: 03-gap-1-paraconsistent-feature-selection-final-reporting
plan: 02
subsystem: ml-feature-selection
tags: [c99, paraconsistent-logic, LPA2v, feature-selection, hierarchical-late-fusion]

# Dependency graph
requires:
  - phase: 03-gap-1-paraconsistent-feature-selection-final-reporting
    plan: 01
    provides: "paraconsistent_select() — feature_select_paraconsistent.c/.h module (mu/lambda/Gc/Gct per feature) plus PARA_* hyperparameter constants in config.h"
provides:
  - "ParaMode enum (PARA_SELECT_OFF/PARA_SELECT_ON) threaded through mode_train_ex(), all 4 existing call sites explicitly passing PARA_SELECT_OFF (behavior-preserving default)"
  - "Master and Expert paraconsistent feature selection wired into the fold/vowel loop, computed on fold->n_train-prefix rows only, persisted via selected_save()/selected_load() round trip"
  - "Refactored predict_hierarchical_late_fusion() as the single owner of per-vowel slicing+selection+forward-pass for both networks, returning discrete class + probabilities via out-parameters (PARA-04 fix)"
  - "models/selected_master_fold*_v*.bin / models/selected_expert_fold*_v*.bin persistence, ready for Plan 03-03's comparison-mode CLI entry"
affects: [03-03, 03-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fit-narrow/apply-wide asymmetry for feature selection (mirrors norm_fit's fold->n_train-only precedent): compute paraconsistent_select() on fold->n_train-prefix rows, apply the resulting column subset to the full n_train_aug/n_val (or n_ex_tr/n_ex_vl) buffers"
    - "Genuine save-then-load round trip through disk (not an in-memory copy) between selection computation and prediction consumption, satisfying PARA-04's literal wording"
    - "Single-function ownership of per-vowel slicing+forward-pass for both discrete prediction and recorded probabilities, eliminating dual-code-path divergence by construction rather than by convention"

key-files:
  created: []
  modified:
    - src/main.c

key-decisions:
  - "ParaMode declared as a 2-value enum (OFF/ON) matching SmoteMode/RegSetting's existing style, positioned after RegSetting in the same enum/struct region"
  - "Expert-path selection computation uses a separate, narrower fold->n_train-prefix + non-NORMAL-filtered subset (ex_tr_x_orig/ex_tr_y_orig), distinct from the existing ex_tr_x/ex_tr_y (built from all n_train_aug rows, used unchanged for the real SMOTE/training step)"
  - "Bounds-validation on loaded indices falls back to full identity selection (never silently clamps) on corruption, logging the exact corrupt index -- mitigates T-03-04"
  - "predict_hierarchical_late_fusion() widened to take sel_m/ns_m/sel_e/ns_e and out-parameters for probabilities, deleting the validation loop's duplicated inline slicing block and its float xv[251] magic-number buffer entirely"

patterns-established:
  - "Pattern: paraconsistent selection insertion mirrors smote_oversample's existing per-vowel loop structure -- compute-persist-load-bounds_check-slice, repeated identically for Master (binary) and Expert (4-class), each with its own independent selection"

requirements-completed: [PARA-03, PARA-04]

# Metrics
duration: ~10min
completed: 2026-07-29
---

# Phase 3 Plan 02: Paraconsistent Feature Selection Integration + predict_hierarchical_late_fusion Refactor Summary

**Wired `paraconsistent_select()` (Plan 03-01) into `mode_train_ex()`'s Master/Expert fold/vowel loop behind a new `ParaMode` toggle (default OFF, byte-for-byte behavior-preserving), and collapsed the discrete-prediction/recorded-probability duplicate code paths into a single refactored `predict_hierarchical_late_fusion()`, deleting the `float xv[251]` magic-number buffer.**

## Performance

- **Duration:** ~10 min
- **Tasks:** 2 completed (2 committed)
- **Files modified:** 1 (src/main.c)

## Accomplishments

- Added `#include "feature_select_paraconsistent.h"` and a new `ParaMode` enum (`PARA_SELECT_OFF = 0, PARA_SELECT_ON = 1`) in the same enum/struct region as `SmoteMode`/`RegSetting`.
- Extended `ABResult` with `float mean_n_selected;` for Plan 03-03's future reporting layer.
- Widened `mode_train_ex()`'s signature with a new `ParaMode para_mode` parameter (inserted before the final `ABResult *result` parameter) and updated all 4 existing call sites (`mode_train()`, both `mode_smote_ab()` calls, `mode_arch_compare()`'s loop) to explicitly pass `PARA_SELECT_OFF`, preserving current CLI behavior exactly with zero new file I/O on that path.
- Declared fold-level `sel_m[3][nf_vowel]`/`ns_m[3]`/`sel_e[3][nf_vowel]`/`ns_e[3]` arrays and opened `results/paraconsistent_selection_freq.csv` (guarded by `para_mode == PARA_SELECT_ON`) mirroring the existing `counts_f` open/append/close idiom.
- Inserted Master-path paraconsistent selection: computed via `paraconsistent_select(tr_x_v, tr_y_bin, fold->n_train, nf_vowel, 2, PARA_GC_THRESH, PARA_GCT_MAX, ...)` — critically `fold->n_train`, not `n_train_aug` — persisted to `models/selected_master_fold%d_v%d.bin` via `selected_save()` then immediately reloaded via `selected_load()` (a genuine disk round trip), bounds-validated (falls back to full identity selection on any out-of-range index, never silently clamps), then applied to column-sliced `tr_x_v_sel`/`vl_x_v_sel` buffers feeding `smote_oversample`/`mlp_init_multi`/`mlp_train`.
- Inserted Expert-path paraconsistent selection mirroring Master's pattern, with the required distinction: the selection computation runs on a separate, narrower subset (`ex_tr_x_orig`/`ex_tr_y_orig`, filtered from `tr_x_v`/`train_y_all`'s `fold->n_train`-prefix for `!= CLASS_NORMAL`), distinct from the existing `ex_tr_x`/`ex_tr_y` (built from all `n_train_aug` rows, left unchanged for the real SMOTE/training step). Persisted to `models/selected_expert_fold%d_v%d.bin`, same bounds-validation and column-slicing pattern.
- Refactored `predict_hierarchical_late_fusion()`: new signature takes `sel_m`/`ns_m`/`sel_e`/`ns_e` plus `p_norm_out`/`p_exp_out[4]` out-parameters. Each vowel's per-network slicing (`x_v_m` via `sel_m[v]`, `x_v_e` via `sel_e[v]`) and single `mlp_forward` call now feeds both the discrete-class accumulation (`prob_pathology`/`prob_expert[4]`) and the probability-recording accumulation (`*p_norm_out`/`p_exp_out[c]`) in the same loop iteration — eliminating the possibility of the two consumers ever diverging.
- Deleted the validation loop's duplicated inline slicing block entirely, including its `float xv[251]` magic-number buffer (DEBT-01's flagged literal) — replaced with a single call to the refactored function.

## Task Commits

Each task was committed atomically:

1. **Task 1: ParaMode + ABResult widening + all call-site updates + Master-path selection insertion** - `8426f4c` (feat)
2. **Task 2: Expert-path selection insertion + predict_hierarchical_late_fusion() refactor (PARA-04)** - `1db04c1` (feat)

## Files Created/Modified

- `src/main.c` - Added `ParaMode` enum, `ABResult.mean_n_selected` field, widened `mode_train_ex()` signature (4 call sites updated), Master/Expert paraconsistent selection insertion with save/load persistence and bounds validation, refactored `predict_hierarchical_late_fusion()` as sole slicing+forward-pass owner, deleted duplicate inline probability block and its `251` magic-number literal.

## Decisions Made

- ParaMode positioned after `RegSetting` in the enum/struct region (same style, same block) rather than near `SmoteMode` — both are valid per the plan's "same region" instruction; chosen for proximity to `mode_train_ex()`'s signature which is the primary consumer.
- Expert selection computation intentionally uses a separate, narrower loop (`ex_tr_x_orig`/`ex_tr_y_orig`) rather than reusing the existing `ex_tr_x`/`ex_tr_y` buffers, per the plan's explicit requirement that selection must be computed on `fold->n_train`-prefix rows only while `ex_tr_x`/`ex_tr_y` (built from all `n_train_aug` rows) remain unchanged for the real SMOTE/training step.
- Bounds-validation fallback (T-03-04 mitigation) sets `ns_m[v]`/`ns_e[v] = nf_vowel` with full identity mapping on any corrupted index, rather than clamping — clamping would silently substitute a different feature with no log trace.

## Deviations from Plan

None - plan executed exactly as written. Both tasks' acceptance criteria (grep-based checks on `PARA_SELECT_OFF` count, `ParaMode` signature/typedef, `mean_n_selected` field, `fold->n_train` usage in the selection call, bounds-check presence, `xv[251]` removal, `mlp_forward` call-count-of-one per network, zero remaining `251` literals) all passed, and `make` compiled with 0 errors after both tasks. The `paraconsistent_select()` call site wraps across two lines for readability (vs. the plan's single-line grep pattern) but is semantically identical and uses the exact required identifiers (`tr_x_v`, `tr_y_bin`, `fold->n_train`, `nf_vowel`, `PARA_GC_THRESH`) — the plan's own acceptance criterion explicitly allows "or equivalent exact call using these identifiers."

## Issues Encountered

None. `make` compiled cleanly with 0 errors after each task; warning count returned to the pre-existing baseline of 3 (`mode_validate_external` unused parameter, `fgets` unused-result, `mlp_train.c`'s unused `best_val_acc`) after Task 2 — Task 1 alone temporarily introduced 2 additional "unused variable" warnings for `sel_e`/`ns_e` (declared per the plan's explicit instruction in Task 1, consumed by Task 2 in the same plan), which is an expected artifact of splitting this single logical change across two atomic task commits, not a defect.

## User Setup Required

None - no external service configuration required. Pure C99, no new dependencies.

## Next Phase Readiness

- `mode_train()`'s default CLI path (train/full) is structurally verified to be behavior-preserving: `PARA_SELECT_OFF` at all 4 call sites, zero new file I/O on that path, identity selection (`sel_m[v][j] = j`, `ns_m[v] = nf_vowel`) applied when OFF. A real run to numerically confirm byte-for-byte parity is Plan 03-04's job, as specified.
- `ParaMode`, the `models/selected_{master,expert}_fold*_v*.bin` persistence, and `results/paraconsistent_selection_freq.csv` (header-only until a `PARA_SELECT_ON` caller exists) are ready for Plan 03-03's comparison-mode CLI entry point.
- No blockers identified for Plan 03-03.

---
*Phase: 03-gap-1-paraconsistent-feature-selection-final-reporting*
*Completed: 2026-07-29*
