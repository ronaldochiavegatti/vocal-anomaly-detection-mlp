---
phase: 02-gap-3-shallow-vs-deep-mlp-comparison-with-structural-refacto
plan: 01
subsystem: infra
tags: [c99, mlp, structural-refactor, config.h, mlp.c, mlp_train.c]

# Dependency graph
requires:
  - phase: 01-gap-2-borderline-smote
    provides: Borderline-SMOTE adopted as the fixed SMOTE mode; ABResult/mode_train_ex()/mode_smote_ab() reusable orchestration pattern this phase's Plan 02-02 will extend
provides:
  - "MLP_MAX_LAYERS=5 headroom constant in config.h, MLP_NUM_LAYERS=3 unchanged"
  - "MLP.layers[] widened to MLP_MAX_LAYERS (mlp.h) -- can safely represent Config D (4 layers) with zero out-of-bounds writes"
  - "mlp_init_multi(net, input_size, output_size, hidden_sizes, n_hidden, dropout_rates) -- runtime-configurable-depth network constructor"
  - "mlp_init_dynamic() rewritten as a thin, behavior-identical wrapper around mlp_init_multi() (removes the dead #if MLP_NUM_LAYERS==3 compile-time branch)"
  - "mlp_backward()'s delta buffer sizing computed dynamically from net->layers[i].output_size, not the stale MLP_HIDDEN1_SIZE constant"
  - "mlp_count_params(net) -- sums trainable weights+biases across net->num_layers, for Plan 02-02's parameter-count comparison column"
  - "mlp_train() accepts l2_lambda as a runtime parameter (inserted before TrainHistory*), unblocking Plan 02-02's regularization sweep"
  - "CLAUDE.md ARCH-01 correction: current production network explicitly labeled Config C [128,64], not SPEC.md's assumed Config A [128]"
  - "Verified via ad hoc ASan run: Config D's exact widest dropout/L2 combination trains end-to-end with zero memory-safety errors"
affects: [02-02, 02-03, 02-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fixed-headroom-array generalization: MLP_MAX_LAYERS sizes the struct array; net->num_layers (runtime field) is what every loop actually iterates by -- verified true for every loop in mlp.c/mlp_train.c before making the change"
    - "Thin-wrapper compatibility: mlp_init_dynamic() calls mlp_init_multi() with today's exact [128,64]/[0.5,0.4] arguments, so every existing caller keeps working with zero behavior change"
    - "Ad hoc pre-sweep ASan verification (not a permanent Makefile target) for array-sizing changes before a real multi-hour training run consumes them"

key-files:
  created: []
  modified:
    - CLAUDE.md
    - include/config.h
    - include/mlp.h
    - src/mlp.c
    - include/mlp_train.h
    - src/mlp_train.c
    - src/main.c

key-decisions:
  - "MLP_MAX_LAYERS=5 (one slot of headroom over Config D's exact 4 layers), per RESEARCH.md Assumption A3"
  - "mlp_init_dynamic() kept (not removed) as a compatibility wrapper -- zero call-site changes needed elsewhere in the codebase"
  - "l2_lambda inserted as a new mlp_train() parameter immediately before TrainHistory*, both existing call sites in main.c updated to pass the unchanged L2_LAMBDA constant explicitly"
  - "ASan Config D smoke test built/run/deleted entirely in the scratchpad directory outside src/include -- never committed, never a Makefile target, per REQUIREMENTS.md's Out of Scope table"

requirements-completed: [ARCH-01, ARCH-02]

# Metrics
duration: 15min
completed: 2026-07-28
---

# Phase 2 Plan 1: Structural Refactor (MLP_MAX_LAYERS, mlp_init_multi, runtime l2_lambda) Summary

**Generalized the compile-time-fixed 2-hidden-layer MLP struct into a runtime-configurable-depth network (verified safe for Config D's 3 hidden layers via ad hoc ASan run) and corrected CLAUDE.md's Config A/C mislabeling before any Gap 3 comparison code exists.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-07-28T15:05Z (approx, per first commit)
- **Completed:** 2026-07-28T15:09Z
- **Tasks:** 3 completed
- **Files modified:** 7 (CLAUDE.md, include/config.h, include/mlp.h, src/mlp.c, include/mlp_train.h, src/mlp_train.c, src/main.c)

## Accomplishments
- CLAUDE.md now explicitly states the current production network (2 hidden layers, [128,64]) is Gap 3's "Config C", not SPEC.md's assumed "Config A [128]" -- corrected before any comparison code exists (ARCH-01)
- `MLP.layers[]` widened from `MLP_NUM_LAYERS` (fixed, 3) to `MLP_MAX_LAYERS` (headroom, 5); `mlp_init_multi()` added as the new runtime-configurable-depth constructor; `mlp_init_dynamic()` becomes a thin, behavior-identical wrapper; `mlp_backward()`'s delta buffer sizing is now computed dynamically from the actual network instance, not a stale `MLP_HIDDEN1_SIZE` constant (ARCH-02)
- `mlp_count_params()` added for Plan 02-02's parameter-count comparison column
- `mlp_train()` accepts `l2_lambda` as a runtime parameter, unblocking Plan 02-02's regularization sweep, with zero behavior change for both existing call sites
- Verified the array-sizing fix under `-fsanitize=address` on Config D's exact widest dropout/L2 combination (hidden=[128,64,32], dropout=[0.70,0.56,0.42], L2=0.0014) before any real training run depends on it: exit code 0, zero ASan errors, `num_layers=4`, `param_count=21476` (matches the manual formula exactly)

## Task Commits

Each task was committed atomically:

1. **Task 1: ARCH-01 -- correct the Config A/C mislabeling in CLAUDE.md** - `9d5611c` (docs)
2. **Task 2: ARCH-02 core -- MLP_MAX_LAYERS, mlp_init_multi(), mlp_count_params(), dynamic mlp_backward() sizing** - `d8211e7` (feat)
3. **Task 3: mlp_train.c/mlp_train.h widening + l2_lambda runtime parameter + ASan Config D smoke test** - `ecb4c05` (feat)

**Plan metadata:** (this commit, docs: complete plan)

## Files Created/Modified
- `CLAUDE.md` - Added ARCH-01 correction bullet in the MLP Architecture section (Config C, not Config A)
- `include/config.h` - Added `MLP_MAX_LAYERS 5` immediately after `MLP_NUM_LAYERS 3` (unchanged)
- `include/mlp.h` - `Layer layers[MLP_MAX_LAYERS]`; new `mlp_init_multi()`/`mlp_count_params()` declarations with Portuguese doc comments
- `src/mlp.c` - New `mlp_init_multi()` implementation; `mlp_init_dynamic()` rewritten as a thin wrapper; `mlp_backward()`'s `max_size` computed dynamically; new `mlp_count_params()`
- `include/mlp_train.h` - `mlp_train()` gains `float l2_lambda` parameter (before `TrainHistory *history`), doc comment updated
- `src/mlp_train.c` - `mlp_train()` definition widened to match; internal `mlp_l2_regularization(net, L2_LAMBDA)` call now uses the `l2_lambda` parameter; 6 checkpoint/SWA pointer-array declarations widened from `MLP_NUM_LAYERS` to `MLP_MAX_LAYERS`
- `src/main.c` - Both existing `mlp_train()` call sites (Master/Expert, inside `mode_train_ex()`'s per-vowel loop) updated to pass `L2_LAMBDA` explicitly as the new argument

## Decisions Made
- `MLP_MAX_LAYERS=5` (one slot of headroom over Config D's exact 4 layers) rather than an exact-fit 4, per RESEARCH.md Assumption A3 -- negligible memory cost, avoids an off-by-one risk if a future config needs exactly 4 hidden layers
- `mlp_init_dynamic()` retained as a compatibility wrapper rather than removed/renamed everywhere -- zero call-site churn elsewhere in the codebase (only `main.c`'s 2 `mlp_train()` calls needed a signature-widening edit, and that was for the unrelated `l2_lambda` parameter, not `mlp_init_dynamic`)
- ASan smoke test built and run entirely in the session scratchpad directory (outside `src/`/`include/`), deleted immediately after passing -- confirmed absent from `git status --porcelain` before proceeding, per the plan's explicit "not committed, not a Makefile target" scoping

## Deviations from Plan

None functionally significant - plan executed as written. One documentation note:

### Auto-fixed / Documented Issues

**1. [Scope boundary - documented, not fixed] Pre-existing `best_val_acc` unused-variable warning surfaced during Task 3's forced rebuild**
- **Found during:** Task 3's `make` verification (after `touch`-forcing recompilation of `mlp_train.c`, which hadn't been recompiled in a while)
- **Issue:** `src/mlp_train.c:171` (`float best_val_acc = -1.0f;`) is unused dead code that predates this plan (confirmed present unchanged at `HEAD~2`, i.e. before Phase 2 started) -- it produces a 3rd compiler warning alongside the 2 previously-known pre-existing warnings (`mode_validate_external` unused parameter, `fgets` ignoring return value in `features_load_csv`), pushing the observed warning count to 3 instead of the plan's expected "at most 2"
- **Fix:** None applied -- per the deviation rules' scope boundary ("only auto-fix issues directly caused by the current task's changes"), this variable was not touched by Task 3's `l2_lambda` parameterization and is unrelated dead code from the original v29 HEAD. Documented here rather than silently fixed or silently ignored.
- **Files affected:** `src/mlp_train.c` (not modified for this issue)
- **Verification:** `git show HEAD~2:src/mlp_train.c | grep -n best_val_acc` confirms the line existed unchanged before this plan's commits
- **Committed in:** N/A - not fixed, only documented

---

**Total deviations:** 0 code changes; 1 documentation-only note (pre-existing warning surfaced, out of scope per task boundary)
**Impact on plan:** None on correctness or scope. All 3 tasks' acceptance criteria pass; `make` compiles with 0 errors and 3 warnings, all 3 pre-existing and unrelated to this plan's changes (2 already known, 1 newly surfaced by recompiling a file that hadn't changed in a while).

## Issues Encountered
- `grep -c "#define MLP_MAX_LAYERS 5" include/config.h` (single-space literal match) returned 0 because the actual line uses multi-space column alignment (`#define MLP_MAX_LAYERS        5  /* ... */`), matching this file's existing house style (`config.h`'s other `MLP_*` defines are all multi-space-aligned). Verified functionally correct via `grep -n "MLP_MAX_LAYERS" include/config.h`. Not a deviation -- house-style alignment takes precedence over the plan's illustrative grep spacing.
- `grep -c "MLP_MAX_LAYERS\]" src/mlp_train.c` (line-counting `-c`) returned 4, not the plan's expected 8, because each of the 4 declaration lines contains 2 occurrences on the same line (`grep -c` counts matching *lines*, not matching *instances*). Verified the true occurrence count is 8 via `grep -o "MLP_MAX_LAYERS\]" src/mlp_train.c | wc -l`. Not a deviation -- all 4 declaration lines were correctly widened.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `mlp_init_multi()`, `mlp_count_params()`, and `mlp_train()`'s runtime `l2_lambda` parameter are all in place and verified -- Plan 02-02 can now build the `ArchConfig`/`RegSetting` orchestration layer (`mode_arch_compare()`, `ARCH_CONFIGS[4]` table) directly on top of this session's changes with no further structural work needed in `mlp.c`/`mlp.h`/`mlp_train.c`.
- Config D's array-sizing safety is proven under ASan for its exact widest dropout/L2 combination -- Plan 02-03's real multi-hour 4-config sweep can proceed without re-verifying memory safety.
- No blockers identified for Plan 02-02.

## Self-Check: PASSED

All 7 modified files verified present on disk; all 3 task commits (`9d5611c`, `d8211e7`, `ecb4c05`) verified present in `git log --oneline --all`. No missing items.

---
*Phase: 02-gap-3-shallow-vs-deep-mlp-comparison-with-structural-refacto*
*Completed: 2026-07-28*
