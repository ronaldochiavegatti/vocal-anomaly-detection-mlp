---
phase: 01-gap-2-borderline-smote
plan: 01
subsystem: ml-pipeline
tags: [c99, smote, borderline-smote, imbalanced-data, oversampling]

# Dependency graph
requires:
  - phase: 00-statistical-infrastructure-rng-reproducibility-prerequisite
    provides: "Deterministic, RNG-race-free precalculate_augmentations() and the verify-rng fast-check precedent this phase's algorithm-correctness-by-construction approach mirrors"
provides:
  - "SmoteMode enum (SMOTE_STANDARD / SMOTE_BORDERLINE) parameterizing smote_oversample()"
  - "find_knn_global(): RNG-free, all-class k-NN search used only for Borderline-SMOTE1 classification"
  - "classify_borderline(): untruncated 2*m>=k safe/borderline/noise classifier, m==k checked first"
  - "smote_oversample() extended with smote_mode + bcounts params; borderline mode restricts the synthesis base pool to samples classified BORDERLINE, with a logged fallback to the full class pool when empty"
  - "n_class<=1 degeneration now skips synthesis entirely (zero synthetic rows), *n_out corrected to actual out_idx rows written"
affects: [01-gap-2-borderline-smote-plan-02, 01-gap-2-borderline-smote-plan-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-phase k-NN search: find_knn_global() (all-class, classification-only, RNG-free) kept textually and structurally distinct from find_knn() (same-class, interpolation-only) -- never share a candidate set or a call site (SMOTE-02 invariant)"
    - "RNG-draw-count parity across SMOTE modes: only the array rng_int() indexes into changes (class_idx[c] vs class_idx_borderline), never the number of rng_int()/rng_uniform() calls per synthesis iteration"

key-files:
  created: []
  modified:
    - src/main.c

key-decisions:
  - "n_class<=1 (SMOTE-03) skips synthesis entirely (n_synthetic forced to 0) rather than logging-and-continuing with a duplicate -- per the plan's explicit, already-resolved user decision, since a logged duplicate still literally violates SMOTE-03's 'no degenerate synthetic samples' wording"
  - "New smote_oversample() parameter named bcounts (not counts) to avoid a hard C redeclaration conflict with the pre-existing local int *counts array"
  - "SMOTE_K_NEIGHBORS/BORDERLINE_M_NEIGHBORS/MAX_SMOTE_CLASSES kept as local #defines in main.c, not promoted to config.h, matching the pre-existing local int k=5 convention"
  - "Both existing smote_oversample() call sites updated to pass SMOTE_STANDARD, NULL explicitly -- zero behavior change in this plan; mode threading through mode_train() itself is deferred to plan 01-02"

patterns-established:
  - "Borderline-SMOTE1 (Han, Wang & Mao, 2005) classification/generation core: find_knn_global() + classify_borderline() gate which points enter the synthesis pool; find_knn() (unchanged) still performs same-class-only interpolation"

requirements-completed: [SMOTE-01, SMOTE-02, SMOTE-03]

# Metrics
duration: 25min
completed: 2026-07-27
---

# Phase 1 Plan 1: Borderline-SMOTE1 Classification/Generation Core Summary

**Added the Borderline-SMOTE1 (Han, Wang & Mao, 2005) classification/generation core (`SmoteMode`, `find_knn_global()`, `classify_borderline()`) as an in-place, behavior-preserving extension of `smote_oversample()` in `src/main.c`, and fixed the pre-existing `n_class<=1` degeneration bug to skip synthesis entirely instead of producing a disguised duplicate synthetic sample.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-07-27T22:58:48Z
- **Completed:** 2026-07-27T23:01:06Z
- **Tasks:** 2
- **Files modified:** 1 (`src/main.c`)

## Accomplishments
- Added `SmoteMode` enum, `SmoteBorderlineCounts` struct, `find_knn_global()` (RNG-free all-class k-NN search), and `classify_borderline()` (untruncated `2*m >= k` rule, `m == k` checked first) as new, standalone functions with fully Portuguese inline comments
- Widened `smote_oversample()`'s signature in place to accept `SmoteMode smote_mode` and a nullable `SmoteBorderlineCounts *bcounts`, wiring in Borderline-SMOTE1's classification step (Block B) without ever letting the interpolation call site (`find_knn()`) reference the borderline pool or the global k-NN result (SMOTE-02 invariant, verified by exact-string grep)
- Fixed the pre-existing `n_class<=1` degeneration (SMOTE-03): synthesis is now skipped entirely for that class (`n_synthetic` forced to `0`), and `*n_out` is corrected to the actual `out_idx` row count so downstream `mlp_train()` calls never read uninitialized memory for skipped rows
- Both call sites inside `mode_train()` updated to pass `SMOTE_STANDARD, NULL` explicitly, preserving today's behavior byte-for-byte (mode threading through `mode_train()` itself is plan 01-02's scope)
- `make` compiles with zero errors and exactly the 2 pre-existing warnings (unused `external_dir` parameter, ignored `fgets` return value) -- no new warnings introduced

## Task Commits

Each task was committed atomically:

1. **Task 1: Add SmoteMode, SmoteBorderlineCounts, find_knn_global(), classify_borderline()** - `40f3efe` (feat)
2. **Task 2: Wire SmoteMode + borderline classification into smote_oversample(), add SMOTE-03 fallback warnings** - `6b371a8` (feat)

**Plan metadata:** (this commit) `docs(01-01): complete Borderline-SMOTE1 core plan`

## Files Created/Modified
- `src/main.c` - Added `SmoteMode`/`SmoteBorderlineCounts`/`find_knn_global()`/`classify_borderline()` (Task 1); extended `smote_oversample()`'s signature and body with borderline-pool construction, the `n_class<=1` skip fix, and the `*n_out` row-count correction, and updated both call sites in `mode_train()` (Task 2)

## Decisions Made
- `n_class<=1` (SMOTE-03) skips synthesis entirely rather than logging-and-continuing with a duplicate -- this decision was already made and recorded in `01-RESEARCH.md`'s Open Question 1 (RESOLVED) prior to this execution; this plan implemented it exactly as specified, no new decision required during execution
- No other deviations from the plan's exact wording were needed -- the plan's insertion points, variable names (`bcounts` vs `counts`), branch order (`m == k` before `2 * m >= k`), and Portuguese-comment requirements were all followed as written

## Deviations from Plan

None - plan executed exactly as written. All acceptance criteria (grep-based checks for both tasks) passed on the first implementation attempt; no auto-fixes were required.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `smote_oversample()` now compiles against the new `SmoteMode`-aware signature with `SMOTE_STANDARD` behavior unchanged from pre-phase -- plan 01-02 can safely thread `SmoteMode` through `mode_train()`'s own parameter list and add the `mode_smote_ab()` CLI orchestration without needing to revisit this plan's code
- `SmoteBorderlineCounts` is defined and its per-class `safe`/`borderline`/`noise` fields are already populated by `smote_oversample()` when a non-NULL `bcounts` pointer is passed -- plan 01-02/01-03 can wire this into the SMOTE-04 A/B report's count table without further struct changes
- No blockers identified. Full runtime proof (an actual training run exercising both SMOTE modes) is deliberately deferred to plan 01-03's dedicated A/B execution, per this plan's own scope boundary (algorithm correctness by construction + clean compilation only)

---
*Phase: 01-gap-2-borderline-smote*
*Completed: 2026-07-27*

## Self-Check: PASSED

- FOUND: src/main.c
- FOUND: .planning/phases/01-gap-2-borderline-smote/01-01-SUMMARY.md
- FOUND: commit 40f3efe
- FOUND: commit 6b371a8
