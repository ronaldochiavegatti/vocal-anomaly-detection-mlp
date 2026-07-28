---
phase: 01-gap-2-borderline-smote
plan: 03
subsystem: ml-pipeline
tags: [smote, borderline-smote, ab-comparison, mcnemar, bootstrap-ci, reproducibility]

# Dependency graph
requires:
  - phase: 01-gap-2-borderline-smote (plan 02)
    provides: "mode_train_ex()/mode_train()/mode_smote_ab()/write_smote_ab_report() CLI orchestration, ready to actually run"
provides:
  - "results/train_log_v32_gap2_smote_ab.txt: freshly executed A/B report -- Macro F1 borderline=0.4587 >= padrao=0.4351 (point estimate), DECISAO: Borderline-SMOTE ADOTADO, direct McNemar chi2=0.1928 p=0.6606 (NOT statistically significant)"
  - "results/smote_ab_comparison.csv + results/smote_borderline_counts.csv: machine-readable evidence backing the decision"
  - "results/metrics_global.csv/bootstrap_ci.csv/mcnemar_vs_baselines.csv (plain train/full CLI path) reconfirmed byte-identical to Phase 0's baseline -- zero regression from the 01-02 mode_train_ex() refactor"
  - "Final verdict (below) for plan 01-04 to consume directly when documenting the Gap 2 decision in CLAUDE.md"
affects: [01-gap-2-borderline-smote-plan-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Background-run-and-poll pattern (nohup + tee + pgrep-based wait-loop, re-armed after each 10-minute foreground-Bash-tool timeout) reused successfully for both multi-hour runs in this plan, following Phase 0's 00-01/00-03 precedent"

key-files:
  created:
    - .planning/phases/01-gap-2-borderline-smote/01-03-SUMMARY.md
  modified:
    - results/train_log_v32_gap2_regression_check_console.txt
    - results/metrics_global.csv (regenerated, byte-identical to prior commit)
    - results/bootstrap_ci.csv (regenerated, byte-identical to prior commit)
    - results/mcnemar_vs_baselines.csv (regenerated, byte-identical to prior commit)
    - results/train_log_v32_gap2_smote_ab.txt
    - results/train_log_v32_gap2_smote_ab_console.txt
    - results/smote_ab_comparison.csv
    - results/smote_borderline_counts.csv
    - results/metrics_global_standard.csv
    - results/metrics_global_borderline.csv
    - results/bootstrap_ci_standard.csv
    - results/bootstrap_ci_borderline.csv
    - results/mcnemar_vs_baselines_standard.csv
    - results/mcnemar_vs_baselines_borderline.csv

key-decisions:
  - "DECISAO: Borderline-SMOTE ADOTADO is mechanically confirmed consistent with its own underlying numbers under both value sources available (point estimate: borderline=0.4587 >= padrao=0.4351; bootstrap-CI mean: borderline=0.456541 >= standard=0.433753) -- sign of (borderline - standard) is positive either way, so the ADOTADO conclusion does not depend on which of the two number sources is used"
  - "The direct McNemar test between the two arms' out-of-fold predictions (chi2=0.1928, p=0.6606) shows the Macro F1 improvement is NOT statistically significant at p<0.05 -- this is a known, accepted limitation of the fixed adopt/reject rule established in plan 01-02 (rule is 'macro_f1 borderline >= standard', with no significance gate), not a bug in this plan's execution. Plan 01-04 should carry this caveat forward into CLAUDE.md documentation rather than presenting ADOTADO as a strong/significant win."
  - "The standard arm's macro_f1 measured inside the smote-ab process (0.4351 point-estimate / 0.4338 bootstrap-mean) is close to but not identical to Phase 0's baseline (~0.4514) and to this same plan's own Task 1 regression-check result (0.4514, byte-identical to Phase 0) -- delta ~0.016-0.018, well within the ~0.04-0.05 RNG-sequence-variance tolerance already documented as expected/accepted in Phase 0's STATE.md (baseline-classifier RNG draws shifting subsequent SMOTE/dropout sequences). Not an anomaly."
  - "Empty-borderline-pool fallback (borderline count == 0) fires in exactly 30 of 90 data rows in results/smote_borderline_counts.csv, but ALL 30 are structural placeholder rows for class slots each network doesn't use (Master's unused class=1 slot -- Master is binary; Expert's unused class=0 slot -- Expert covers classes 1-4 only), which are always (0,0,0) by construction, not real fallback events. Among the 60 real classification rows (Master class=0, Expert classes 1-4), the empty-pool fallback fired ZERO times in this run -- it did not concentrate in the smallest classes as RESEARCH.md's Pitfall 3 anticipated; it simply never fired."

patterns-established: []

requirements-completed: [SMOTE-04]

# Metrics
duration: 136min
completed: 2026-07-27
---

# Phase 1 Plan 3: Real A/B Execution + Statistical Verification Summary

**Ran the actual, end-to-end Borderline-SMOTE A/B comparison (`./build/vocal_detect smote-ab .`) plus a plain `train`/`full` CLI regression check, both to completion under `RANDOM_SEED=42`/same 5-folds -- result: Borderline-SMOTE improves point-estimate Macro F1 by +0.0235 (ADOTADO per the fixed rule), but the direct McNemar test between arms is NOT statistically significant (p=0.66), and the plain CLI path is confirmed byte-identical/non-regressed against Phase 0's baseline.**

## Performance

- **Duration:** ~136 min (Task 1 pipeline run ~32 min wall clock [20:12-20:44]; Task 2 pipeline run ~61 min wall clock [20:49-21:50]; remainder was setup, background-poll overhead across 10-minute Bash-tool foreground timeouts, verification, and Task 3 analysis)
- **Completed:** 2026-07-27
- **Tasks:** 3
- **Files modified:** 14 (1 regression-check log + 3 regenerated-but-byte-identical baseline CSVs from Task 1; 9 new A/B-specific artifacts from Task 2; this SUMMARY.md)

## Accomplishments

### Task 1: Plain train/full CLI regression check
- Ran `./build/vocal_detect train .` (the `mode_train_ex(base_dir, SMOTE_STANDARD, NULL)` branch -- the one code path `smote-ab` never exercises) to completion, exit clean
- Regenerated `results/metrics_global.csv`, `results/bootstrap_ci.csv`, `results/mcnemar_vs_baselines.csv` with the correct unsuffixed filenames (confirming the `result == NULL` branch's filename behavior survives the 01-02 refactor)
- Result: Macro F1 **0.4514**, Accuracy **0.6976** -- `git diff --stat` against the already-committed Phase 0 baseline showed **zero diff** (byte-identical CSVs), i.e. an exact match, not merely "within tolerance." This is the strongest possible confirmation that plan 01-02's `mode_train()` -> `mode_train_ex()` refactor is behavior-preserving for real users/CI.

### Task 2: smote-ab A/B pipeline run
- Ran `./build/vocal_detect smote-ab .` to completion (both SMOTE arms, same seed/folds, sequential in one process), exit clean
- Produced all 9 A/B-specific artifacts: `results/train_log_v32_gap2_smote_ab.txt` (contains `DECISAO: Borderline-SMOTE ADOTADO ...`), `results/smote_ab_comparison.csv` (8 lines: header + 7 metrics x both arms + 95% bootstrap CI), `results/smote_borderline_counts.csv` (91 lines: header + 90 data rows, `fold,vowel,network,class,safe,borderline,noise`), plus the 6 mode-suffixed `metrics_global`/`bootstrap_ci`/`mcnemar_vs_baselines` CSVs (`_standard`/`_borderline`)

### Task 3: Statistical rigor cross-check
- Mechanically confirmed the `ADOTADO` decision string is consistent with the sign of `(borderline_macro_f1 - standard_macro_f1)` under both the point-estimate numbers embedded in the DECISAO line (0.4587 >= 0.4351) and the bootstrap-CI-mean numbers in `smote_ab_comparison.csv` (0.456541 >= 0.433753) -- both agree, decision is unambiguous
- Flagged that the direct McNemar test between the two arms (chi2=0.1928, p=0.6606) is **not statistically significant** -- the fixed adopt/reject rule from plan 01-02 has no significance gate, so `ADOTADO` here reflects a positive but statistically inconclusive point-estimate difference, not a proven improvement
- Sanity-checked the standard arm's macro_f1 (0.4351 point-estimate / 0.4338 bootstrap-mean) against Phase 0's baseline (~0.4514) and against this plan's own Task 1 result (0.4514, exact match) -- delta ~0.016-0.018, within the documented ~0.04-0.05 RNG-sequence-variance tolerance, not an anomaly
- Counted empty-borderline-pool fallback rows (`borderline == 0`) in `results/smote_borderline_counts.csv`: 30 of 90 rows, but all 30 are structural placeholder rows for unused class slots (Master's class=1, Expert's class=0), not genuine fallback events. Among the 60 real classification rows, the fallback fired **zero times**.

## Final Verdict (for plan 01-04)

| Metric | Standard (bootstrap mean [95% CI]) | Borderline (bootstrap mean [95% CI]) |
|---|---|---|
| accuracy | 0.6915 [0.6648, 0.7177] | 0.6958 [0.6694, 0.7231] |
| **macro_f1** | **0.4338 [0.3976, 0.4713]** | **0.4565 [0.4194, 0.4949]** |
| f1_normal | 0.8725 [0.8531, 0.8904] | 0.8664 [0.8475, 0.8850] |
| f1_laringite | 0.3884 [0.3046, 0.4690] | 0.3974 [0.3131, 0.4891] |
| f1_disfonia_psicogenica | 0.2648 [0.1818, 0.3439] | 0.3273 [0.2375, 0.4149] |
| f1_disfonia_funcional | 0.1892 [0.1118, 0.2703] | 0.2386 [0.1507, 0.3294] |
| f1_reinke | 0.4540 [0.3657, 0.5463] | 0.4531 [0.3636, 0.5444] |

- **Point-estimate Macro F1** (raw, non-bootstrapped, as printed in the DECISAO line): standard=0.4351, borderline=0.4587, **delta=+0.0235**
- **Direct McNemar (Borderline vs Padrao out-of-fold predictions):** chi2=0.1928, **p=0.6606 -- NOT statistically significant (p>=0.05)**
- **Decision:** `DECISAO: Borderline-SMOTE ADOTADO` (per the fixed rule `borderline_macro_f1 >= standard_macro_f1`, mechanically verified consistent with the raw numbers above)
- **Caveat plan 01-04 must carry forward:** the improvement is directionally positive and consistent under both point-estimate and bootstrap-mean readings, but is NOT statistically significant per the direct McNemar test between arms. Per-class gains are concentrated in the two hardest/smallest classes (Disfonia Psicogenica +0.063, Disfonia Funcional +0.049), with a small give-back on Normal (-0.006) and Reinke (-0.001) -- consistent with Borderline-SMOTE's intended effect of focusing synthetic samples near the decision boundary for minority/hard classes.
- **Empty-borderline-pool fallback:** fired 0 times among the 60 real classification rows (Master's healthy class + Expert's 4 pathology classes, across 5 folds x 3 vowels x 2 networks) -- did not concentrate in the smallest classes as anticipated in RESEARCH.md's Pitfall 3; the fallback path exists in code but was not exercised by this run's data.
- **Task 1 regression outcome:** the plain, unsuffixed `train`/`full` CLI path (the code path real users/CI actually invoke, never touched by `smote-ab`) is confirmed byte-identical to Phase 0's reconfirmed baseline (Macro F1 0.4514, Accuracy 0.6976) -- zero regression from plan 01-02's `mode_train_ex()` refactor.

## Task Commits

Each task was committed atomically:

1. **Task 1: Regression-check the plain train/full CLI path** - `09d95f0` (feat)
2. **Task 2: Run the full A/B pipeline (smote-ab CLI mode) to completion** - `830fbcd` (feat)
3. **Task 3: Verify statistical rigor and internal consistency** - inspection-only, no code/data changes beyond what Task 2 already produced; findings captured in this SUMMARY

**Plan metadata:** (this commit) `docs(01-03): complete smote-ab A/B execution and verification plan`

## Files Created/Modified

- `results/train_log_v32_gap2_regression_check_console.txt` - New: full console transcript of the plain `train` regression-check run
- `results/metrics_global.csv`, `results/bootstrap_ci.csv`, `results/mcnemar_vs_baselines.csv` - Regenerated by Task 1's run; byte-identical to the prior committed Phase 0 baseline (no diff)
- `results/train_log_v32_gap2_smote_ab.txt` - New: full A/B report with per-metric comparison table, direct McNemar result, and the `DECISAO:` sentence
- `results/train_log_v32_gap2_smote_ab_console.txt` - New: full console transcript of the `smote-ab` run
- `results/smote_ab_comparison.csv` - New: machine-readable 7-metric x 2-arm comparison table with 95% bootstrap CI bounds
- `results/smote_borderline_counts.csv` - New: 90-row safe/borderline/noise counts per fold/vowel/network/class from the borderline arm
- `results/metrics_global_standard.csv` / `results/metrics_global_borderline.csv` - New: per-arm point-estimate metrics
- `results/bootstrap_ci_standard.csv` / `results/bootstrap_ci_borderline.csv` - New: per-arm 95% bootstrap CI
- `results/mcnemar_vs_baselines_standard.csv` / `results/mcnemar_vs_baselines_borderline.csv` - New: per-arm 3-way McNemar vs MajorityClass/kNN/LogReg

## Decisions Made

- See `key-decisions` in frontmatter above (decision-string consistency, McNemar significance caveat, standard-arm RNG-variance sanity check, empty-pool-fallback finding)

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written. Both multi-hour runs completed successfully on the first attempt with no code changes required.

### Notable observation (not a deviation, documented per Task 3's own instructions)

The `DECISAO:` line's embedded macro_f1 values (0.4587/0.4351, the raw `ABResult->macro_f1` point estimates) differ slightly from the values shown in the comparison table and `smote_ab_comparison.csv` (0.4565/0.4338, the `ci[CI_MACRO_F1].mean` bootstrap-mean estimates) -- this is expected: `write_smote_ab_report()` (from plan 01-02) intentionally uses the direct point estimate for the decision rule but the bootstrap-resampling mean for the reported CI table, a pattern already seen in Phase 0 (bootstrap mean 0.4499 vs. point estimate 0.451440 for the same baseline run). Both number pairs agree on sign (borderline >= standard), so the decision is unambiguous either way; this is a source-of-estimate difference, not a bug.

## Issues Encountered

- The Bash tool's 10-minute foreground timeout was hit while polling both long-running processes (as anticipated by the plan's `critical_process_note`); recovered each time by re-arming a `run_in_background` monitor loop (`pgrep`-based) rather than ending the turn unattended, consistent with Phase 0's established pattern. No process was abandoned.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SMOTE-04 is satisfied: a real, freshly-executed, reproducible A/B comparison now exists in `results/`, with a code-computed adopt/reject decision, full statistical backing (bootstrap CI, direct McNemar), and a quantified empty-pool-fallback rate.
- Plan 01-04 can consume this SUMMARY's "Final Verdict" section directly to document the Gap 2 decision in `CLAUDE.md` (Optimization History / What Worked), including the significant caveat that the McNemar test between arms is not statistically significant despite the positive point-estimate delta.
- No blockers identified.

---
*Phase: 01-gap-2-borderline-smote*
*Completed: 2026-07-27*

## Self-Check: PASSED

- FOUND: results/train_log_v32_gap2_regression_check_console.txt
- FOUND: results/metrics_global.csv
- FOUND: results/bootstrap_ci.csv
- FOUND: results/mcnemar_vs_baselines.csv
- FOUND: results/train_log_v32_gap2_smote_ab.txt
- FOUND: results/train_log_v32_gap2_smote_ab_console.txt
- FOUND: results/smote_ab_comparison.csv
- FOUND: results/smote_borderline_counts.csv
- FOUND: results/metrics_global_standard.csv
- FOUND: results/metrics_global_borderline.csv
- FOUND: results/bootstrap_ci_standard.csv
- FOUND: results/bootstrap_ci_borderline.csv
- FOUND: results/mcnemar_vs_baselines_standard.csv
- FOUND: results/mcnemar_vs_baselines_borderline.csv
- FOUND: commit 09d95f0
- FOUND: commit 830fbcd
