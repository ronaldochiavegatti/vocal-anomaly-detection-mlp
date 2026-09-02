---
phase: 03-gap-1-paraconsistent-feature-selection-final-reporting
plan: 03
subsystem: ml-reporting-cli
tags: [c99, paraconsistent-logic, LPA2v, gap1-report, cli-dispatch, cross-01, cross-02]

# Dependency graph
requires:
  - phase: 03-gap-1-paraconsistent-feature-selection-final-reporting
    plan: 02
    provides: "ParaMode (PARA_SELECT_OFF/PARA_SELECT_ON), ABResult.mean_n_selected, paraconsistent selection wired into mode_train_ex()'s Master/Expert fold/vowel loop, refactored predict_hierarchical_late_fusion()"
provides:
  - "write_gap1_report() -- SPEC.md's exact 3-branch Gap 1 acceptance rule (equal-or-better Macro F1, OR up to -0.01 worse with >=30% feature reduction, else reject) as fixed code logic, never asserted by inspection (PARA-05)"
  - "write_gap_adoption_status() -- 3-row consolidated Gap Adoption Status table (results/gap_adoption_status.csv), Gap 1's citation field branching on the live decision_gap1 outcome (CROSS-01, CROSS-02)"
  - "mode_paraconsistent_ab() -- dispatchable './build/vocal_detect paraconsistent-ab .' CLI mode running both arms fixed at Borderline-SMOTE1 + Config C + regularizacao baseline"
affects: [03-04, 03-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single decision computation, dual consumption: write_gap1_report() computes SPEC.md's 3-branch rule exactly once and exposes it via out-parameters (decision_out/chi2_out/p_out) so write_gap_adoption_status() never recomputes the rule a second time -- structurally prevents the citation-vs-reality drift CROSS-02 exists to catch"
    - "Fixed-configuration A/B isolation: mode_paraconsistent_ab() passes identical SMOTE_BORDERLINE/&ARCH_CONFIGS[2]/REG_BASELINE to both mode_train_ex() calls, varying only para_mode, guaranteeing the comparison isolates exactly one variable (same discipline as mode_smote_ab()/mode_arch_compare())"

key-files:
  created: []
  modified:
    - src/main.c

key-decisions:
  - "write_gap1_report() signature widened with out-parameters (char *decision_out, size_t decision_out_size, float *chi2_out, float *p_out), all individually NULL-checked/optional, rather than returning a struct or a static buffer -- keeps the function self-contained per the plan's explicit instruction while letting mode_paraconsistent_ab() pass the single computed decision through to write_gap_adoption_status() without duplicating the 3-branch if/else"
  - "chi2_out passed as NULL from mode_paraconsistent_ab() (only decision string + p-value are needed downstream) to avoid an unused-but-set local variable warning under -Wextra"
  - "Task 1/Task 2 boundary split exactly along the plan's own function grouping: write_gap1_report() alone in commit 1 (transiently unused, expected artifact of splitting one logical change across two atomic commits -- same pattern Plan 03-02 documented), write_gap_adoption_status() + mode_paraconsistent_ab() + CLI dispatch registration together in commit 2 (the commit that supplies write_gap1_report()'s first caller)"

patterns-established:
  - "Pattern: reporting/CLI layer for a Gap A/B comparison always follows write_<gap>_report() (per-metric CI table + direct McNemar + fixed if/else DECISAO sentence + parallel CSV) -> mode_<gap>_ab() (log warning, zero-init 2 ABResults, 2 fixed-configuration mode_train_ex() calls varying exactly one parameter, free y_true/y_pred) -> CLI dispatch registration, mirrored byte-for-byte in structure across Gap 2 (write_smote_ab_report/mode_smote_ab), Gap 3 (write_arch_compare_report/mode_arch_compare), and now Gap 1 (write_gap1_report/mode_paraconsistent_ab)"

requirements-completed: [PARA-05, CROSS-01, CROSS-02]

# Metrics
duration: ~15min
completed: 2026-07-29
---

# Phase 3 Plan 03: Gap 1 Reporting/CLI Layer (write_gap1_report, write_gap_adoption_status, mode_paraconsistent_ab) Summary

**Implemented `write_gap1_report()` as SPEC.md's exact 3-branch Gap 1 acceptance rule in fixed code (PARA-05), `write_gap_adoption_status()` as the 3-row consolidated Gap Adoption Status table with a citation field that mechanically enforces CROSS-02 by branching on the live decision, and `mode_paraconsistent_ab()` as a new dispatchable CLI mode (`./build/vocal_detect paraconsistent-ab .`) that runs both selection arms fixed at the already-adopted Gap 2/Gap 3 configuration.**

## Performance

- **Duration:** ~15 min
- **Tasks:** 2 completed (2 committed)
- **Files modified:** 1 (src/main.c)

## Accomplishments

- Implemented `write_gap1_report()` immediately after `write_smote_ab_report()`, mirroring its exact CI-table/McNemar/CSV structure for `off_res`/`on_res` (accuracy, macro_f1, 5 per-class F1 rows), with a direct `metrics_mcnemar(off_res->y_true, on_res->y_pred, off_res->y_pred, off_res->n, ...)` call between the two arms.
- Computed `feature_reduction = 1.0f - (on_res->mean_n_selected / (float)(FEATURES_PER_VOWEL + NUM_METADATA_FEATURES))` and implemented SPEC.md's exact 3-branch trade-off rule (lines 123-131) as fixed `if`/`else if`/`else` code: ADOTADA when `on_res->macro_f1 >= off_res->macro_f1`, ADOTADA via trade-off when the Macro F1 delta is within `-0.01` AND `feature_reduction >= 0.30f`, REJEITADA otherwise.
- Exported a parallel `results/paraconsistent_ab_comparison.csv` with the standard 7-metric comparison columns plus `mean_n_selected`/`feature_reduction` trailing summary rows.
- Added out-parameters (`decision_out`, `decision_out_size`, `chi2_out`, `p_out`, all optional/NULL-checked) so the single decision computation is the sole source of truth, consumed by the caller without recomputation.
- Implemented `write_gap_adoption_status()` producing `results/gap_adoption_status.csv`: 3 rows (`gap,decision,macro_f1_delta,mcnemar_p,citation_status`) -- Gap 2 and Gap 3 rows hardcoded as settled historical facts from CLAUDE.md's existing "Gap 2 Outcome"/"Gap 3 Outcome" sections, Gap 1's row built live from `write_gap1_report()`'s outcome. The citation field is computed by `if (strncmp(decision_gap1, "ADOTAD", 6) == 0)` -- only citing PAL2v (Da Costa 1990 / Abe & Nakamatsu 2009) when the live decision actually starts with "ADOTAD", otherwise "N/A - tecnica nao ativa no modelo final (CROSS-02)".
- Implemented `mode_paraconsistent_ab()` mirroring `mode_smote_ab()`'s exact structure: logs the long-duration warning, zero-initializes `ABResult res_off = {0}, res_on = {0}`, calls `mode_train_ex(base_dir, SMOTE_BORDERLINE, &ARCH_CONFIGS[2], REG_BASELINE, PARA_SELECT_OFF, &res_off)` then the `PARA_SELECT_ON` variant (both fixed at the already-adopted Gap 2/Gap 3 configuration, never `mode_train()`'s unmodified `SMOTE_STANDARD` default), calls `write_gap1_report()` then `write_gap_adoption_status()` with the returned decision/delta/p-value, and frees both arms' `y_true`/`y_pred`.
- Registered `"paraconsistent-ab"` in `main()`'s CLI dispatch chain, immediately after `"arch-compare"`, same `strcmp`/ternary style as every other mode.

## Task Commits

Each task was committed atomically:

1. **Task 1: write_gap1_report() -- SPEC.md's Gap 1 acceptance rule as fixed code logic (PARA-05)** - `5af23ca` (feat)
2. **Task 2: write_gap_adoption_status() (CROSS-01/CROSS-02) + mode_paraconsistent_ab() CLI + dispatch registration** - `aa22ae7` (feat)

## Files Created/Modified

- `src/main.c` - Added `write_gap1_report()` (SPEC.md's exact 3-branch Gap 1 trade-off rule as fixed code, `results/train_log_v34_gap1_paraconsistent_ab.txt` + `results/paraconsistent_ab_comparison.csv` outputs), `write_gap_adoption_status()` (`results/gap_adoption_status.csv`, 3-row consolidated table with CROSS-02-enforcing citation branch), `mode_paraconsistent_ab()` (new CLI comparison mode), and the `"paraconsistent-ab"` dispatch registration in `main()`.

## Decisions Made

- `write_gap1_report()` widened with out-parameters rather than a return value/struct, per the plan's explicit "pick whichever" guidance -- keeps the function self-contained while giving `mode_paraconsistent_ab()` the single computed decision without duplicating the 3-branch rule.
- `chi2_out` passed as `NULL` from the call site (only the decision string and p-value are consumed downstream by `write_gap_adoption_status()`), avoiding an unused-but-set-variable warning that would otherwise appear under `-Wextra`.
- Task boundary split exactly along the plan's own function grouping (write_gap1_report alone in commit 1, write_gap_adoption_status + mode_paraconsistent_ab + dispatch registration together in commit 2) -- required deliberate commit-splitting since both functions were drafted together for design coherence (out-parameter shape had to be finalized before the caller could be written) but needed separate atomic commits matching the plan's task structure; verified via `diff` that the final two-commit result is byte-for-byte identical to the originally drafted combined version.

## Deviations from Plan

None - plan executed exactly as written. All of Task 1's and Task 2's grep-based acceptance criteria passed verbatim (all 3 DECISAO branch strings present, `feature_reduction` referenced >=2 times, the exact `0.01f && feature_reduction >= 0.30f` threshold literal present once, `"paraconsistent-ab"` found in `main()`'s dispatch chain, both exact `mode_train_ex(...)` call strings present, `strncmp(decision_gap1` present), and `make` compiled with 0 errors after each task.

## Issues Encountered

Task 1's intermediate build (before Task 2 supplies the first caller) produced one transient `-Wunused-function` warning for `write_gap1_report` -- an expected artifact of splitting a single logical reporting-layer change across two atomic task commits (same pattern Plan 03-02 documented for `sel_e`/`ns_e`), not a defect; it disappeared once Task 2's commit added `mode_paraconsistent_ab()` as the caller. Final warning count after both tasks returned to the pre-existing baseline (`mode_validate_external` unused parameter, `fgets` unused-result -- the only two warnings visible in an incremental `main.o`-only rebuild; `mlp_train.c`'s unused `best_val_acc` warning is only emitted when `mlp_train.c` itself is recompiled, unaffected by this plan's changes).

## User Setup Required

None - no external service configuration required. Pure C99, no new dependencies, no new CLI flags beyond the new `paraconsistent-ab` mode string.

## Next Phase Readiness

- `./build/vocal_detect paraconsistent-ab .` is dispatchable and structurally verified (build succeeds, dispatch chain intact, `argc<2` usage-line regression check passes). No pipeline run happens in this plan -- executing the mode (which runs the full hierarchical pipeline twice, ~60-180 min) is Plan 03-04's job.
- `write_gap1_report()`'s decision computation is the single source of truth consumed by `write_gap_adoption_status()`'s Gap 1 citation field -- the CROSS-02 mechanical safeguard (T-03-07) is in place and ready to be exercised by a real run in Plan 03-04.
- `results/gap_adoption_status.csv`'s Gap 2/Gap 3 rows are hardcoded historical facts (not recomputed) -- if those sections of CLAUDE.md are ever revised, this function's hardcoded strings would need a manual update in a future plan; out of scope for this plan.
- No blockers identified for Plan 03-04.

---
*Phase: 03-gap-1-paraconsistent-feature-selection-final-reporting*
*Completed: 2026-07-29*

## Self-Check: PASSED

- FOUND: src/main.c
- FOUND: .planning/phases/03-gap-1-paraconsistent-feature-selection-final-reporting/03-03-SUMMARY.md
- FOUND commit: 5af23ca
- FOUND commit: aa22ae7
- FOUND commit: 44f9a16
