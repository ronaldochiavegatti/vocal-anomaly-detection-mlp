---
phase: 01-gap-2-borderline-smote
plan: 02
subsystem: ml-pipeline
tags: [c99, smote, borderline-smote, cli-orchestration, ab-comparison, mcnemar]

# Dependency graph
requires:
  - phase: 01-gap-2-borderline-smote (plan 01)
    provides: "SmoteMode enum, SmoteBorderlineCounts struct, find_knn_global()/classify_borderline(), smote_oversample() extended with smote_mode+bcounts params"
provides:
  - "mode_train_ex(base_dir, smote_mode, result): pipeline entry point that can run either SMOTE mode and optionally return structured aggregate results (ABResult) without changing train/full CLI output"
  - "mode_train() thin wrapper preserving today's exact train/full CLI behavior"
  - "ABResult struct: accuracy/macro_f1/per-class F1/bootstrap CI/out-of-fold y_true+y_pred+n for one pipeline run"
  - "results/smote_borderline_counts.csv schema (fold,vowel,network,class,safe,borderline,noise), populated whenever mode_train_ex() runs with SMOTE_BORDERLINE"
  - "mode_smote_ab() + write_smote_ab_report(): single CLI mode (smote-ab) running both SMOTE arms under identical seed/folds and producing a code-computed adopt/reject A/B report"
affects: [01-gap-2-borderline-smote-plan-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "mode_train_ex()'s result parameter as a nullable ownership-transfer switch: NULL preserves today's byte-for-byte CLI behavior (frees all_y_true/all_y_pred internally, unsuffixed artifact filenames); non-NULL transfers y_true/y_pred ownership to the caller and suffixes artifact filenames by SMOTE mode"
    - "RNG-stream parity via kfold_split()'s internal reseed reused across A/B arms -- mode_smote_ab() calls mode_train_ex() twice with no manual RNG snapshot/restore, per RESEARCH.md Pattern 2"
    - "Adopt/reject decision computed by a single fixed if/else rule (bl_res->macro_f1 >= std_res->macro_f1) embedded in write_smote_ab_report(), never asserted manually"

key-files:
  created: []
  modified:
    - src/main.c

key-decisions:
  - "Output artifact filenames (metrics_global/bootstrap_ci/mcnemar_vs_baselines) are suffixed with literal _standard/_borderline substrings written via explicit if/else branches (not string-concatenation of a shared mode_suffix variable) -- this keeps the literal suffixed filenames greppable/auditable directly in source, matching the plan's acceptance-criteria wording"
  - "results/smote_borderline_counts.csv path is referenced both in the fopen() call and in the paired log_error() failure message -- 2 occurrences instead of the plan's literal '1', but this exactly matches the codebase's own pre-existing fopen/log_error convention (see bootstrap_ci.csv/mcnemar_vs_baselines.csv writers), so it was kept as the more consistent choice rather than trimmed to match the acceptance-criteria grep count literally"

patterns-established:
  - "ABResult as the project's first 'structured pipeline run result' data holder (no prior analog existed -- MetricsResult/ConfidenceInterval were the closest style reference per PATTERNS.md), enabling programmatic multi-run comparison instead of print-only output"

requirements-completed: [SMOTE-01, SMOTE-04]

# Metrics
duration: 20min
completed: 2026-07-27
---

# Phase 1 Plan 2: mode_train_ex() Refactor + smote-ab CLI Orchestration Summary

**Widened `mode_train()` into `mode_train_ex(base_dir, smote_mode, result)` (threading `SmoteMode`/`SmoteBorderlineCounts` into both `smote_oversample()` call sites and exporting `results/smote_borderline_counts.csv`), preserved `mode_train()` as a behavior-identical thin wrapper, and added a single new `smote-ab` CLI mode that runs both SMOTE arms under an identical seed/fold split and writes a code-computed adopt/reject A/B report.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-07-27T23:03:00Z (approx, continuing directly from 01-01)
- **Completed:** 2026-07-27T23:23:00Z (approx)
- **Tasks:** 2
- **Files modified:** 1 (`src/main.c`)

## Accomplishments
- Added `ABResult` struct (accuracy, macro_f1, per-class F1, bootstrap CI array, out-of-fold `y_true`/`y_pred`/`n`) as a flat plain-data struct, matching `MetricsResult`/`ConfidenceInterval`'s style
- Renamed `mode_train()` to `mode_train_ex(base_dir, smote_mode, result)`; both `smote_oversample()` call sites (Master 2-class, Expert 4-class) now receive the real per-vowel `smote_mode`/`SmoteBorderlineCounts` arguments instead of the `SMOTE_STANDARD, NULL` placeholders left by plan 01-01
- `results/smote_borderline_counts.csv` (schema: `fold,vowel,network,class,safe,borderline,noise`) is opened and populated only when `smote_mode == SMOTE_BORDERLINE` (the classification step is a no-op, all-zero-count operation under `SMOTE_STANDARD`, so it is not exported for that arm)
- The three existing output artifacts (`metrics_global.csv`, `bootstrap_ci.csv`, `mcnemar_vs_baselines.csv`) are suffixed with `_standard`/`_borderline` only when `result != NULL`; when `result == NULL` (today's `train`/`full` CLI modes), the exact original unsuffixed filenames are used, preserving Phase 0's canonical baseline artifacts byte-for-byte
- `mode_train()` is preserved as `static int mode_train(const char *base_dir) { return mode_train_ex(base_dir, SMOTE_STANDARD, NULL); }` -- `main()`'s existing `train`/`full` dispatch line required no change
- Added `write_smote_ab_report()`: computes a direct McNemar comparison between the two arms' out-of-fold predictions (distinct from each arm's own vs-baseline McNemar), writes a Portuguese-language text report (`results/train_log_v32_gap2_smote_ab.txt`) with a Macro F1/per-class F1 + bootstrap CI comparison table, the McNemar significance sentence, a pointer to the safe/borderline/noise count CSV, and the adopt/reject decision sentence -- plus a machine-readable `results/smote_ab_comparison.csv`
- Added `mode_smote_ab()`: runs `mode_train_ex(base_dir, SMOTE_STANDARD, &res_standard)` then `mode_train_ex(base_dir, SMOTE_BORDERLINE, &res_borderline)` sequentially in one process (relying on `kfold_split()`'s internal RNG reseed for identical fold assignments across both arms, per `RESEARCH.md` Pattern 2), calls `write_smote_ab_report()`, then frees both arms' transferred `y_true`/`y_pred` buffers
- Wired the new `smote-ab` CLI mode into `main()`'s dispatch chain, mirroring the existing `verify-rng` ternary-to-exit-code idiom exactly
- `make` compiles with zero errors and exactly the 2 pre-existing warnings (unused `external_dir` parameter, ignored `fgets` return value) -- no new warnings introduced
- CLI argv-parsing regression checks passed: no-args invocation still prints the usage message and exits 1; an unrecognized mode string still falls through to `return 1`

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor mode_train() into mode_train_ex() with ABResult capture, mode-suffixed artifact paths, and per-vowel SMOTE-mode threading** - `a3aef7b` (feat)
2. **Task 2: Add mode_smote_ab() orchestration, write_smote_ab_report(), and the smote-ab CLI dispatch line** - `60d5a10` (feat)

**Plan metadata:** (this commit) `docs(01-02): complete mode_train_ex + smote-ab orchestration plan`

## Files Created/Modified
- `src/main.c` - Added `ABResult` struct and renamed `mode_train()` to `mode_train_ex()` with `SmoteMode`/`ABResult*` parameters, per-vowel `SmoteBorderlineCounts` threading into both `smote_oversample()` call sites, mode-suffixed artifact filenames, and ownership-transfer of `all_y_true`/`all_y_pred` when `result != NULL` (Task 1); added `write_smote_ab_report()`, `mode_smote_ab()`, and the `smote-ab` CLI dispatch line (Task 2)

## Decisions Made
- Literal `_standard`/`_borderline` filename suffixes written via explicit `if (smote_mode == SMOTE_BORDERLINE) {...} else {...}` branches (six literal occurrences total) rather than a single shared `mode_suffix` string variable -- keeps the suffixed filenames directly greppable/auditable in source, matching the plan's acceptance-criteria wording ("results/metrics_global_standard.csv / results/metrics_global_borderline.csv")
- `results/smote_borderline_counts.csv` path string appears twice in source (the `fopen()` call and its paired `log_error()` failure message) rather than once -- this is a direct match to the codebase's own pre-existing `fopen`/`log_error` convention already used for `bootstrap_ci.csv`/`mcnemar_vs_baselines.csv`; kept as the internally-consistent choice over literally minimizing to 1 occurrence

## Deviations from Plan

### Auto-fixed Issues
None requiring code changes beyond what's captured in "Decisions Made" above -- both deviations are cosmetic (grep-count) differences from the plan's exact wording, not behavioral changes, and both were resolved in favor of the more consistent/auditable option.

## Issues Encountered
None. All acceptance criteria (grep-based checks for both tasks, `make` clean-build check, CLI argv-parsing regression checks) passed on the first implementation attempt.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `mode_train_ex()`, `mode_smote_ab()`, `write_smote_ab_report()`, and the `smote-ab` CLI mode all compile cleanly and are ready for plan 01-03's actual execution: `./build/vocal_detect smote-ab .` (a 60-180 minute full pipeline run, deliberately NOT executed in this plan)
- `results/smote_borderline_counts.csv`'s schema is defined and wired to populate on the next `SMOTE_BORDERLINE` arm of a `smote-ab` run
- The adopt/reject decision in `results/train_log_v32_gap2_smote_ab.txt` will be computed entirely by the fixed rule in `write_smote_ab_report()` once plan 01-03 runs the actual comparison -- no manual judgment call is embedded anywhere in the decision logic
- No blockers identified. This plan's own scope boundary (threading + orchestration correctness by construction + clean compilation + CLI regression checks only, no full pipeline run) was respected throughout

---
*Phase: 01-gap-2-borderline-smote*
*Completed: 2026-07-27*

## Self-Check: PASSED

- FOUND: src/main.c
- FOUND: .planning/phases/01-gap-2-borderline-smote/01-02-SUMMARY.md
- FOUND: commit a3aef7b
- FOUND: commit 60d5a10
