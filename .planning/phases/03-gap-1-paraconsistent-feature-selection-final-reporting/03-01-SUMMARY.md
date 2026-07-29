---
phase: 03-gap-1-paraconsistent-feature-selection-final-reporting
plan: 01
subsystem: ml-feature-selection
tags: [c99, anova, eta-squared, paraconsistent-logic, LPA2v, feature-selection]

# Dependency graph
requires:
  - phase: 02-gap-3-shallow-vs-deep-mlp
    provides: mlp_init_multi()/config-comparison infrastructure (unrelated to this plan but the last completed phase before this one)
provides:
  - "paraconsistent_select() — new, independent public entry point computing (mu, lambda, Gc, Gct) per feature"
  - "4 new PARA_* hyperparameter constants in config.h"
  - "Numerically-verified eta-squared and per-class-sigma/global-sigma lambda formulas, replacing SPEC.md's non-standard Fisher-ratio/CV formulas"
affects: [03-02, 03-03, 03-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Capped relaxation loop with guaranteed all-features fallback (mirrors smote_oversample's empty-borderline-pool precedent)"
    - "File-local MIN_VAR/MIN_STD division-by-near-zero guards (mirrors normalize.c's MIN_STD precedent)"

key-files:
  created:
    - include/feature_select_paraconsistent.h
    - src/feature_select_paraconsistent.c
  modified:
    - include/config.h

key-decisions:
  - "mu computed via one-way ANOVA eta-squared (SSB/SST), not SPEC.md's non-standard Fisher-ratio"
  - "lambda computed via unweighted mean of per-class sigma/global-sigma ratio, not SPEC.md's CV=std/mean (which explodes for near-zero-mean delta-MFCC features)"
  - "Relaxation loop capped at PARA_MAX_RELAX_ITERS=10 iterations with guaranteed non-empty fallback (all nf features), never returns 0 selected features"

patterns-established:
  - "Pattern: new independent feature-selection module kept to a single public function (paraconsistent_select), internal helpers (feature_eta_squared/feature_lambda) stay static, not exposed in the header"

requirements-completed: [PARA-01, PARA-02]

# Metrics
duration: 20min
completed: 2026-07-29
---

# Phase 3 Plan 01: Paraconsistent Feature Selection Module Summary

**New `feature_select_paraconsistent.c`/`.h` module implementing `paraconsistent_select()` — ANOVA eta-squared for mu, unweighted per-class sigma/global-sigma ratio for lambda, capped Gc/Gct relaxation loop with guaranteed non-empty fallback — verified against 3 synthetic-data correctness scenarios.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 3 completed (2 committed, 1 verification-only)
- **Files modified:** 3 (1 modified, 2 created)

## Accomplishments

- Added 4 new `PARA_*` hyperparameter constants to `include/config.h` (`PARA_GC_THRESH=0.35f`, `PARA_GCT_MAX=0.3f`, `PARA_MAX_RELAX_ITERS=10`, `PARA_GC_RELAX_STEP=0.05f`), matching SPEC.md's suggested default ranges.
- Created `include/feature_select_paraconsistent.h` declaring the single public `paraconsistent_select()` entry point, with a full doc comment covering the `fold->n_train`-only contract (mirrors `norm_fit`'s established precedent) and the guaranteed-`>=1`-return contract.
- Implemented `src/feature_select_paraconsistent.c`:
  - `feature_eta_squared()` — one-way ANOVA decomposition (SSB/SST), naturally bounded [0,1], `MIN_VAR=1e-16f`-guarded against division by near-zero total sum of squares (constant features return exactly `0.0f`).
  - `feature_lambda()` — unweighted mean, across classes with `>=2` samples, of `(per-class sigma / global sigma)` clipped to `[0,1]`, `MIN_STD=1e-8f`-guarded (mirrors `normalize.c`'s precedent). Deliberately unweighted per RESEARCH.md's degeneracy proof (a sample-weighted version collapses lambda to `sqrt(1-mu)`).
  - `paraconsistent_select()` — computes `Gc = mu - lambda`, `Gct = mu + lambda - 1` per feature, then a relaxation loop (`gc_thresh` decremented by `PARA_GC_RELAX_STEP` each of up to `PARA_MAX_RELAX_ITERS` iterations) that is mathematically guaranteed to terminate with `n_selected >= 1` via an all-features fallback if the cap is exhausted with zero features selected. No `exit()`/`abort()` anywhere — every degenerate-input path degrades to a safe, logged fallback.
- Verified numerically via a throwaway scratch harness (built and run outside the repo, then deleted) against the real compiled implementation:
  - A perfectly-separated 2-class feature: `mu_out[0] > 0.95f` (PASS).
  - A constant feature: `mu_out[0] == 0.0f` exactly (PASS).
  - Impossible thresholds (`gc_thresh=0.99f, gct_max=0.01f`) on pure-noise data: relaxation loop exhausted its cap (10 `log_warn` messages observed on stderr) and fell back to `n_selected == nf` without crashing (PASS).
- `make clean && make` succeeds afterward with zero errors and the same warning baseline (3 pre-existing warnings) as before this plan — the new module introduces zero new warnings.

## Task Commits

Each task was committed atomically:

1. **Task 1: PARA_* hyperparameters in config.h + feature_select_paraconsistent.h contract** - `02fc36d` (feat)
2. **Task 2: Implement feature_eta_squared(), feature_lambda(), paraconsistent_select()** - `010986e` (feat)
3. **Task 3: Ad hoc numeric correctness verification** - no commit (scratch harness lived only in the scratchpad directory, explicitly not committed per plan instructions; deleted after verification passed)

## Files Created/Modified

- `include/config.h` - Added `/* ========== Selecao Paraconsistente (Gap 1) ========== */` banner section with 4 new PARA_* constants
- `include/feature_select_paraconsistent.h` - New header, single public `paraconsistent_select()` declaration
- `src/feature_select_paraconsistent.c` - New module: `feature_eta_squared()`, `feature_lambda()` (both `static`), `paraconsistent_select()` (public)

## Decisions Made

- mu = one-way ANOVA eta-squared (SSB/SST), not SPEC.md's non-standard per-class Fisher ratio — naturally bounded [0,1] without a separate cross-feature normalization pass.
- lambda = unweighted mean of per-class sigma/global-sigma ratio, not SPEC.md's CV=std/mean — avoids exploding for near-zero-mean delta-MFCC features and avoids collapsing to a fixed function of mu (per RESEARCH.md's degeneracy proof).
- Relaxation loop capped at `PARA_MAX_RELAX_ITERS=10` with guaranteed all-features fallback, mirroring `smote_oversample()`'s empty-borderline-pool precedent from Phase 1 (Gap 2).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing `<stdlib.h>` include caused implicit-declaration warnings on `free()`**
- **Found during:** Task 2 build verification
- **Issue:** `src/feature_select_paraconsistent.c` called `free()` (via `safe_calloc`/`safe_malloc`-allocated buffers) without including `<stdlib.h>`, producing `implicit-function-declaration` and `builtin-declaration-mismatch` warnings under `-Wall -Wextra` — these would have counted as new warnings against the plan's "zero new warnings" acceptance criterion.
- **Fix:** Added `#include <stdlib.h>` to the file's include block.
- **Files modified:** `src/feature_select_paraconsistent.c`
- **Verification:** Rebuilt; warning count returned to the pre-existing baseline of 3 (matching `/tmp/build_03_01_t1.log`), zero new warnings introduced.
- **Committed in:** `010986e` (part of Task 2 commit)

**2. [Rule 1 - Bug] Accidentally deleted tracked `results/*.csv` files during verification, twice**
- **Found during:** Task 2 build verification and Task 3's mandated `make clean && make` step
- **Issue:** `make clean` (run as part of the plan's own verification commands) deletes `results/*.csv` per its documented `Makefile` target, which removed 60 tracked result CSV files (`results/metrics_global*.csv`, `bootstrap_ci*.csv`, `mcnemar_vs_baselines*.csv`, `smote_borderline_counts*.csv`, `arch_compare_comparison.csv`) from the working tree as an unintended side effect, not a task-scoped edit.
- **Fix:** Restored the deleted files with `git checkout -- results/` (a targeted restore of a specific tracked path, not a blanket `reset`/`clean`) each time this occurred, confirmed via `git status --short --porcelain -uall` showing a fully clean tree before proceeding.
- **Files modified:** none (restoration only, no net change)
- **Verification:** `git status --porcelain` empty after each restore; final working tree clean with only the intended Task 3 scratch-harness absence (never committed) as expected.
- **Committed in:** not applicable (no commit needed — files were restored to their already-committed state)

**3. [Documentation-only] Task 1's header-guard grep acceptance criterion is inaccurate relative to its own instructed style**
- **Found during:** Task 1 acceptance-criteria verification
- **Issue:** The plan's acceptance criteria state `grep -c "FEATURE_SELECT_PARACONSISTENT_H" include/feature_select_paraconsistent.h` should output `2` ("guard define + endif comment"), but the plan's own action text instructs mirroring `feature_select.h`'s exact `#ifndef`/`#define`/`#endif /* ... */` guard style, which by construction produces `3` matching lines (verified: `grep -c "FEATURE_SELECT_H" include/feature_select.h` also outputs `3`, not `2`).
- **Fix:** No code change — kept the correct, established project convention (3-line guard) rather than deviating from it to force a miscounted grep target. This is a plan-authoring inconsistency, not an implementation defect.
- **Files modified:** none
- **Verification:** Header follows the exact same structure as `feature_select.h`, confirmed by direct comparison.
- **Committed in:** `02fc36d` (part of Task 1 commit, no separate fix needed)

---

**Total deviations:** 3 (1 blocking auto-fix, 1 unintended-deletion recovery, 1 documented plan-acceptance-criterion inaccuracy — no code impact)
**Impact on plan:** All auto-fixes necessary for correctness (zero-new-warnings requirement) or recovery of accidental tool-caused state (unrelated tracked files). No scope creep; the paraconsistent-selection implementation itself matches the plan's specification exactly.

## Issues Encountered

None beyond the deviations documented above.

## User Setup Required

None - no external service configuration required. Pure C99, no new dependencies.

## Next Phase Readiness

- `paraconsistent_select()` compiles cleanly, is numerically verified against 3 synthetic-data scenarios, and is ready for Plan 03-02 to wire into `mode_train_ex()`'s fold/vowel loop.
- No `main.c` changes were made in this plan, as specified — integration begins in Plan 03-02.
- `PARA_GC_THRESH`/`PARA_GCT_MAX`/`PARA_MAX_RELAX_ITERS`/`PARA_GC_RELAX_STEP` are available in `config.h` for Plan 03-02's call sites to consume as defaults.
- No blockers identified for Plan 03-02.

---
*Phase: 03-gap-1-paraconsistent-feature-selection-final-reporting*
*Completed: 2026-07-29*
