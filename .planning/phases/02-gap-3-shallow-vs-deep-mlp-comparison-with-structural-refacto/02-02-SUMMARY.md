---
phase: 02-gap-3-shallow-vs-deep-mlp-comparison-with-structural-refacto
plan: 02
subsystem: infra
tags: [c99, mlp, orchestration, main.c, arch-compare, mcnemar, bootstrap-ci]

# Dependency graph
requires:
  - phase: 02-01
    provides: "mlp_init_multi()/mlp_count_params()/runtime l2_lambda parameter on mlp_train() -- the structural primitives this plan's orchestration layer calls"
provides:
  - "ArchConfig struct (4 configs A[128]/B[64]/C[128,64]/D[128,64,32]) + RegSetting enum (light/baseline/strong) + REG_MULTIPLIER{0.6,1.0,1.4} applied jointly to dropout and L2_LAMBDA"
  - "mode_train_ex() widened to (base_dir, smote_mode, arch, reg, result) -- single reusable function trains any of the 4 architectures at any of the 3 regularization strengths against identical 5-fold partitions"
  - "ABResult extended with param_count_master/param_count_expert/mean_time_per_epoch_sec/mean_epochs_to_stop (ARCH-04 data)"
  - "mode_arch_compare() CLI mode -- runs all 12 (architecture x regularization) combinations sequentially, SMOTE fixed at SMOTE_BORDERLINE, appends each arm to results/arch_compare_comparison.csv incrementally"
  - "write_arch_compare_report() -- fixed 5-step deterministic decision procedure (best reg per arch -> best arch overall -> 1-SE bootstrap-CI band -> McNemar gate -> fewest-parameters adoption), never a manual/visual pick"
  - "Per-arm filename collision fix: metrics_global/bootstrap_ci/mcnemar_vs_baselines/smote_borderline_counts paths now suffix by arch->name + REG_NAME[reg] in addition to SMOTE mode"
affects: [02-03, 02-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Widen-mode_train_ex()-in-place (again): the same reusable-function pattern from Phase 1's SMOTE-04 A/B was extended a second time with 2 new parameters (arch, reg), rather than inventing a parallel orchestration path -- exactly per RESEARCH.md's precedent"
    - "Incremental per-arm CSV append as the explicit durability mechanism for a 12-arm, multi-hour sweep: if mode_arch_compare() is interrupted partway, completed arms' rows survive on disk even though the full .txt report (which needs all 12 in memory) does not get produced"
    - "Fixed-rule-in-code decision procedure (1-SE band + McNemar + fewest-params), mirroring write_smote_ab_report()'s precedent -- the adopted architecture is never asserted by inspecting the table"

key-files:
  created: []
  modified:
    - src/main.c

key-decisions:
  - "ArchConfig/RegSetting/ARCH_CONFIGS/REG_MULTIPLIER/REG_NAME placed immediately after ABResult and before mode_train_ex(), per plan's exact ordering instruction"
  - "Config D's 3rd hidden layer dropout is the hardcoded literal 0.3f in ARCH_CONFIGS, NOT config.h's dead DROPOUT_RATE_HIDDEN3=0.0f -- per the user's explicit Phase-2-planning decision recorded in STATE.md"
  - "REG_MULTIPLIER anchors L2 on config.h's actual L2_LAMBDA=0.001f (verified this session), not CLAUDE.md's stale documented 0.003"
  - "Per-arm filename collision fix applied to metrics_global/bootstrap_ci/mcnemar_vs_baselines AND smote_borderline_counts.csv (the latter was gated only on smote_mode==SMOTE_BORDERLINE, unconditional on result) -- all suffixed by arch->name/REG_NAME[reg] when result != NULL, unsuffixed names preserved unchanged when result == NULL"
  - "mode_smote_ab()'s per-arm diagnostic filenames for its 2 arms change from _standard/_borderline to _standard_C_baseline/_borderline_C_baseline as a deliberate, disclosed side effect of the collision fix -- SMOTE-04's actual required deliverables (results/smote_ab_comparison.csv, results/train_log_v32_gap2_smote_ab.txt, the already-committed original results/smote_borderline_counts.csv) are written by write_smote_ab_report()/mode_smote_ab() directly and are unaffected"
  - "write_smote_ab_report()'s hardcoded file-pointer string updated to results/smote_borderline_counts_C_baseline.csv to stay accurate for any future smote-ab re-run, without touching the already-committed Phase 1 historical file of the unsuffixed name"
  - "1-SE band's SE derived from the bootstrap-CI half-width (ci.upper - ci.lower)/(2*1.96), not the classic per-fold-array CART formula, since this codebase does not track per-fold macro_f1 in an array -- documented verbatim in the report text, per plan instruction"
  - "McNemar tie-break and adoption tie-breaks both keep the lower-index candidate on an exact tie (simplest deterministic rule, documented in code comments)"
  - "mode_arch_compare() is NOT invoked in this plan -- the 12-arm sweep is a 6-18+ hour operation explicitly reserved for Plan 02-03"

requirements-completed: [ARCH-03, ARCH-04, ARCH-05]

# Metrics
duration: 6min
completed: 2026-07-28
---

# Phase 2 Plan 2: Arch-Compare Orchestration (ArchConfig/RegSetting, mode_arch_compare, 1-SE+McNemar decision) Summary

**Widened `mode_train_ex()` a second time (arch, reg params) so one reusable function trains any of 4 architectures at any of 3 regularization strengths against identical 5-fold partitions, and added `mode_arch_compare()` + `write_arch_compare_report()` implementing a fully deterministic 1-SE-band + McNemar + fewest-parameters adoption rule for the 12-arm Gap 3 sweep.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-07-28T18:12Z (approx, per prior commit)
- **Completed:** 2026-07-28T18:18Z
- **Tasks:** 2 completed
- **Files modified:** 1 (src/main.c)

## Accomplishments
- `ArchConfig`/`RegSetting`/`ARCH_CONFIGS[4]`/`REG_MULTIPLIER[3]`/`REG_NAME[3]` added; `mode_train_ex()` now trains any of the 4 architectures (A[128], B[64], C[128,64], D[128,64,32]) at any of 3 regularization strengths (light 0.6x, baseline 1.0x, strong 1.4x, applied jointly to dropout and L2) via one reusable function body -- no duplicated fold+vowel loop anywhere in `main.c` (ARCH-03)
- `ABResult` extended with `param_count_master`/`param_count_expert`/`mean_time_per_epoch_sec`/`mean_epochs_to_stop`, captured via `mlp_count_params()` and `timer_now()` wraps around each `mlp_train()` call
- `mode_arch_compare()` runs all 12 (architecture x regularization) combinations sequentially, SMOTE mode fixed at `SMOTE_BORDERLINE` (Phase 1's adopted decision), appending each arm's result to `results/arch_compare_comparison.csv` immediately after that arm completes -- not only at the very end (ARCH-04)
- `write_arch_compare_report()` implements the exact 5-step deterministic decision procedure (best regularization per architecture -> best architecture overall -> 1-SE bootstrap-CI band -> McNemar significance gate vs the best architecture -> fewest-total-parameters adoption among passing candidates) -- the adopted configuration is never chosen by visual inspection of the comparison table (ARCH-05)
- A real, previously-latent bug was fixed proactively per the plan: all 12 future `arch-compare` arms share `smote_mode == SMOTE_BORDERLINE`, so the existing filename-suffix logic (branching only on `smote_mode`) would have made every arm silently overwrite the same 4 output files. Fixed by suffixing `metrics_global`/`bootstrap_ci`/`mcnemar_vs_baselines`/`smote_borderline_counts` paths with `arch->name`/`REG_NAME[reg]` as well
- `mode_train()`/`mode_smote_ab()` updated to pass `&ARCH_CONFIGS[2]` (Config C) + `REG_BASELINE`, confirmed logically byte-identical to pre-Phase-2 production behavior by inspecting `mlp_init_multi()`'s implementation (Plan 02-01): with `hidden_sizes={128,64,...}`, `n_hidden=2`, `dropout_rates={0.5,0.4}` it reproduces `mlp_init_dynamic()`'s exact original call, and `eff_l2 = L2_LAMBDA * 1.0` matches the explicit `L2_LAMBDA` argument Plan 02-01 already wired in

## Task Commits

Each task was committed atomically:

1. **Task 1: ArchConfig/RegSetting definitions, widen mode_train_ex(), extend ABResult, fix per-arm filename collisions** - `fe1cd3b` (feat)
2. **Task 2: write_arch_compare_report() (1-SE + McNemar decision), mode_arch_compare() orchestration, CLI dispatch** - `55660e1` (feat)

**Plan metadata:** (this commit, docs: complete plan)

## Files Created/Modified
- `src/main.c` -- `ArchConfig`/`RegSetting`/`ARCH_CONFIGS`/`REG_MULTIPLIER`/`REG_NAME` added after `ABResult`; `ABResult` extended with 4 new fields; `mode_train_ex()` widened to `(base_dir, smote_mode, arch, reg, result)`, computing `eff_dropout[]`/`eff_l2` once per call, replacing both `mlp_init_dynamic()` call sites with `mlp_init_multi()`, capturing param counts via `mlp_count_params()`, wrapping both `mlp_train()` calls with `timer_now()`; per-arm filename-suffix fix applied to 4 output paths; `mode_train()`/`mode_smote_ab()` updated to pass Config C + baseline; `write_smote_ab_report()`'s hardcoded file-pointer string updated; new `write_arch_compare_report()` and `mode_arch_compare()` functions added; new `"arch-compare"` CLI dispatch line in `main()`

## Decisions Made
See `key-decisions` in frontmatter above for the full list. The most consequential: the filename-collision fix required for the 12-arm sweep to work correctly was implemented in Task 1 exactly as the plan specified (a proactive bug fix disclosed in the plan itself, not a deviation), and it has a visible, disclosed side effect on `mode_smote_ab()`'s per-arm diagnostic filenames (unaffecting SMOTE-04's actual required deliverables).

## Deviations from Plan

None functionally significant -- plan executed as written. One grep-mechanics-only note (not a deviation in substance):

### Auto-fixed / Documented Issues

**1. [Scope boundary - documented, not fixed] Illustrative single-line grep patterns in the plan's acceptance criteria did not match this file's multi-line/wrapped house style verbatim**
- **Found during:** Task 1's acceptance-criteria verification (`ABResult`'s 4 new fields declared on 4 separate lines, not the plan's single-line illustrative grep string) and Task 1's `smote_borderline_counts_C_baseline.csv` string (split across two `fprintf` argument lines via C string-literal concatenation, matching the surrounding code's existing wrapping style)
- **Issue:** `grep -c` on the plan's exact illustrative single-line pattern returned `0` in both cases because the actual, correct code is formatted across 2-4 lines
- **Fix:** None applied -- verified functionally correct via `grep -n` (all 4 `ABResult` fields present) and `awk`-joined adjacent-line grep (the `smote_borderline_counts_C_baseline.csv` string is present, just line-wrapped). This mirrors the identical precedent documented in Plan 02-01's own SUMMARY (`MLP_MAX_LAYERS` multi-space alignment vs single-space grep) -- house-style line wrapping takes precedence over the plan's illustrative grep spacing/line-breaks
- **Files affected:** `src/main.c` (not modified for this issue -- verification-only)
- **Verification:** `grep -n "param_count_master\|param_count_expert\|mean_time_per_epoch_sec\|mean_epochs_to_stop" src/main.c` shows all 4 fields declared and used correctly; `awk 'NR==772{a=$0} NR==773{print a $0}' src/main.c | grep -c "ver results/smote_borderline_counts_C_baseline.csv"` returns `1`
- **Committed in:** N/A -- not a code change, verification-only

---

**Total deviations:** 0 code changes; 1 verification-mechanics note (illustrative grep line-breaks, semantically confirmed correct)
**Impact on plan:** None on correctness or scope. All Task 1 and Task 2 acceptance criteria pass (all `grep -c` checks return the expected counts once line-wrapping is accounted for); `make` compiles with 0 errors and exactly the 2 previously-known pre-existing warnings (`mode_validate_external` unused parameter, `fgets` ignoring return value) -- no new warnings introduced. (Plan 02-01's separately-documented 3rd pre-existing warning in `mlp_train.c:171` did not appear in this plan's build output because `mlp_train.c` was not recompiled this session; it remains present and unrelated to this plan's changes.)

## Issues Encountered

None beyond the grep-mechanics note above.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

- `mode_arch_compare()`, `write_arch_compare_report()`, and the `arch-compare` CLI mode are fully implemented, compile cleanly, and are ready for Plan 02-03 to actually invoke (`./build/vocal_detect arch-compare .`) as the real multi-hour 12-arm sweep.
- The plain `train`/`full` CLI path was confirmed logically unchanged (not re-run end-to-end in this plan, per the plan's own scope -- a full pipeline run is Plan 02-03's job) by direct inspection of `mlp_init_multi()`'s Config-C-equivalent call and `eff_l2`'s baseline-multiplier arithmetic.
- No blockers identified for Plan 02-03.

## Self-Check: PASSED

Both task commits (`fe1cd3b`, `55660e1`) verified present in `git log --oneline --all`. `src/main.c` verified present and containing all new symbols (`ArchConfig`, `ARCH_CONFIGS`, `RegSetting`, `REG_MULTIPLIER`, `write_arch_compare_report`, `mode_arch_compare`). `make` verified to produce 0 errors. No missing items.

---
*Phase: 02-gap-3-shallow-vs-deep-mlp-comparison-with-structural-refacto*
*Completed: 2026-07-28*
