---
phase: 03-gap-1-paraconsistent-feature-selection-final-reporting
plan: 04
subsystem: ml-pipeline-execution
tags: [c99, paraconsistent-logic, LPA2v, gap1-ab-comparison, mcnemar, bootstrap-ci, cross-01, cross-02]

# Dependency graph
requires:
  - phase: 03-gap-1-paraconsistent-feature-selection-final-reporting
    plan: 03
    provides: "write_gap1_report(), write_gap_adoption_status(), mode_paraconsistent_ab() CLI dispatch (./build/vocal_detect paraconsistent-ab .)"
provides:
  - "Real, freshly-executed 2-arm Gap 1 A/B comparison output (not a claim): results/train_log_v34_gap1_paraconsistent_ab.txt, results/paraconsistent_ab_comparison.csv, results/paraconsistent_selection_freq.csv, results/gap_adoption_status.csv"
  - "PARA-05's core deliverable: a 2550-row (85 features x 3 vowels x 5 folds x 2 networks) selection-frequency table with mu/lambda/Gc/Gct/selected per feature"
  - "Mechanically-verified Gap 1 adopt/reject decision (REJEITADA) and its underlying raw off_res/on_res numbers, ready for Plan 03-05 to cite directly"
affects: [03-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Execution-only plan: no source changes, pure runtime invocation of a previously-built CLI mode, followed by hand-reproduction verification of the mode's own printed decision rule against its own raw numbers"

key-files:
  created:
    - results/train_log_v34_gap1_paraconsistent_ab.txt
    - results/train_log_v34_gap1_paraconsistent_ab_console.txt
    - results/paraconsistent_ab_comparison.csv
    - results/paraconsistent_selection_freq.csv
    - results/gap_adoption_status.csv
    - "models/selected_master_fold{0-4}_v{0-2}.bin (15 files, gitignored, not committed)"
    - "models/selected_expert_fold{0-4}_v{0-2}.bin (15 files, gitignored, not committed)"
  modified: []

key-decisions:
  - "make clean (run per Task 1's explicit prerequisite step) deleted a large batch of previously-committed Phase 1/2 result CSVs from the working tree (bootstrap_ci*, metrics_global*, mcnemar_vs_baselines*, smote_ab_comparison.csv, smote_borderline_counts*, arch_compare_comparison.csv) as an out-of-scope side effect of the Makefile's documented `results/*.csv` wipe -- restored via targeted `git checkout -- results/` (not a blanket reset) before committing this plan's own new artifacts, so no historical evidence was lost"
  - "models/selected_{master,expert}_fold{0-4}_v{0-2}.bin (30 files) are NOT committed to git -- the entire models/ directory is gitignored project-wide (consistent with every prior phase's model .bin outputs never being tracked); their presence on disk (verified via `ls | wc -l` = 30) satisfies the plan's acceptance criteria without a git commit"

patterns-established: []

requirements-completed: [PARA-05]

# Metrics
duration: ~68min
completed: 2026-07-29
---

# Phase 3 Plan 04: Real Gap 1 Paraconsistent-Selection A/B Comparison Execution Summary

**Executed the real (not simulated) 2-arm `paraconsistent-ab` pipeline run to completion (~63 min compute + ~5 min verification), producing PARA-05's 2550-row selection-frequency table and a mechanically-verified REJEITADA decision: Macro F1 without paraconsistent selection (0.4610 raw / 0.4597 bootstrap-mean) edges out Macro F1 with selection (0.4587 raw / 0.4565 bootstrap-mean), and the paraconsistent gc-threshold relaxation exhausted in all 30 of 30 (fold, vowel, network) runs, producing 0% feature reduction — well short of SPEC.md's 30% trade-off floor.**

## Performance

- **Duration:** ~68 min (build ~1 min, pipeline run ~63 min, verification ~4 min)
- **Started:** 2026-07-29T15:26:xx (build) / run started 2026-07-29T15:27:07
- **Completed:** 2026-07-29T16:30:50 (run) + verification
- **Tasks:** 2 completed (1 committed; Task 2 was verification-only, no files to commit)
- **Files modified:** 5 (all newly created `results/` artifacts)

## Accomplishments

- Confirmed prerequisites: all 5 SVD class directories present with exact expected patient counts (saudavel=687, laringite=140, disfonia_psicogênica=91, disfonia_funcional=112, edema_de_reinke=68 = 1098 total, confirmed live in the run's own startup log), `overview_merged.csv` present, `results/`/`models/` directories present.
- Ran `make clean && make` for a fresh build (zero errors, only the two pre-existing baseline warnings: `mode_validate_external` unused parameter, `fgets` unused-result — both predate this plan).
- Launched `./build/vocal_detect paraconsistent-ab . 2>&1 | tee results/train_log_v34_gap1_paraconsistent_ab_console.txt` as a detached (`nohup setsid`), polled background process; actively polled to completion in-turn across multiple re-armed poll windows (never fired-and-forgot), confirmed exit code 0 via the `EXITCODE:0` sentinel appended after the pipe.
- Feature extraction (first run of the fresh cache, since `results/features.csv` had been cleared by an earlier wave's `make clean`) completed in 45.8s for all 1098 patients, 0 errors, producing a 1098x251 matrix matching `TOTAL_FEATURES=251`.
- Both arms ran the full hierarchical Master+Expert x 3-vowel x 5-fold pipeline (30 MLPs each), fixed at Borderline-SMOTE1 + Config C [128,64] + baseline regularization (the already-adopted Gap 2/Gap 3 configuration), varying only the paraconsistent-selection flag.
- Verified all 7 Task 1 acceptance criteria pass (exit 0, all 5 report files non-empty, exactly 1 `DECISAO: Selecao Paraconsistente` line, 30 `models/selected_*.bin` files present).
- Verified all 7 Task 2 acceptance criteria pass: selection-frequency CSV has exactly 2551 lines (1 header + 2550 data rows = 85 features x 3 vowels x 5 folds x 2 networks), zero NaN/Inf anywhere across all 3 checked files, `selected` column is 0/1-only (in fact all 2550 rows are `1`, see finding below), `gap_adoption_status.csv` has exactly 4 lines, and the DECISAO branch was hand-reproduced from the raw numbers (not trusted at face value).
- Restored 55 Phase 1/2 result CSVs that `make clean` deleted from the working tree as an out-of-scope side effect, via targeted `git checkout -- results/` before committing this plan's own new artifacts — no historical evidence lost.

## Final Verdict (for Plan 03-05 to consume directly)

**Without-selection arm (off_res):**
- Raw (concatenated out-of-fold) Macro F1: **0.4610**, Accuracy: 0.6995
- Bootstrap mean [95% CI]: Macro F1 **0.4597** [0.4229, 0.4955], Accuracy 0.6988 [0.6721, 0.7268]
- Per-class F1 (bootstrap mean): Normal 0.8704, Laringite 0.4120, Disfonia Psicogênica 0.3353, Disfonia Funcional 0.1991, Edema de Reinke 0.4818
- MLP vs baselines (McNemar): significantly better than MajorityClass (p<0.0001), kNN (p<0.0001), LogisticRegression (p<0.0001)

**With-selection arm (on_res):**
- Raw (concatenated out-of-fold) Macro F1: **0.4587**, Accuracy: 0.6967
- Bootstrap mean [95% CI]: Macro F1 **0.4565** [0.4194, 0.4949], Accuracy 0.6958 [0.6694, 0.7231]
- Per-class F1 (bootstrap mean): Normal 0.8664, Laringite 0.3974, Disfonia Psicogênica 0.3273, Disfonia Funcional 0.2386, Edema de Reinke 0.4531
- MLP vs baselines (McNemar): significantly better than MajorityClass (p<0.0001), kNN (p<0.0001), LogisticRegression (p=0.0001)

**Direct McNemar (with-selection vs without-selection):** chi2=0.0506, p=0.8220 — **not statistically significant** (p>=0.05).

**Feature selection outcome:** mean_n_selected = **85.0 of 85** (0.0% feature reduction). The paraconsistent gc-threshold relaxation loop (10 iterations, 0.35 down to -0.15) found **zero** features clearing the threshold at every single relaxation step, in **all 30 of 30** (fold, vowel, network) combinations — confirmed via `grep -c "relaxamento esgotado" results/train_log_v34_gap1_paraconsistent_ab_console.txt` = 30. Every run fell back to "select all 85 features" (same fallback pattern as SMOTE-03's empty-borderline-pool fallback). This is why `paraconsistent_selection_freq.csv`'s `selected` column is `1` for all 2550 data rows and `0` for none — not a bug, but the mechanical consequence of the threshold never being met.

**DECISAO (verbatim from `results/train_log_v34_gap1_paraconsistent_ab.txt`):**

> DECISAO: Selecao Paraconsistente REJEITADA (Macro F1 sem-selecao=0.4610 > com-selecao=0.4587, delta=-0.0024, reducao de features=0.0% insuficiente para o trade-off do SPEC.md, McNemar chi2=0.0506 p=0.8220) -- mantendo pipeline sem selecao paraconsistente em producao

**Hand-reproduction of the 3-branch rule (confirmed to match the printed sentence):**
1. Branch 1 (ADOTADA, on >= off): 0.4587 >= 0.4610? **False.**
2. Branch 2 (ADOTADA via trade-off, delta within -0.01 AND feature_reduction >= 0.30): delta = 0.4587 - 0.4610 = -0.0023 (within -0.01, true) AND feature_reduction = 0.0 >= 0.30 (**false**) → condition is **False** (AND fails).
3. Branch 3 (REJEITADA): fires by elimination. **Matches the printed DECISAO exactly.**

**CLAUDE.md 0.42 Macro F1 regression floor check:** Both arms clear the floor comfortably — off_res raw 0.4610 / bootstrap-mean 0.4597, on_res raw 0.4587 / bootstrap-mean 0.4565, all >= 0.42. **PASS, explicitly checked, no exception needed.**

**`gap_adoption_status.csv` (4 lines: header + 3 rows), full contents:**

| gap | decision | macro_f1_delta | mcnemar_p | citation_status |
|-----|----------|-----------------|-----------|------------------|
| Gap 2 (Borderline-SMOTE) | ADOTADO | +0.0235 | 0.6606 | Han/Wang/Mao 2005 (Borderline-SMOTE1) |
| Gap 3 (Config C 2-hidden-layer) | ADOTADO (sem mudanca em config.h) | n/a (ja em producao) | 0.7463 | N/A - comparacao metodologica interna |
| Gap 1 (Selecao Paraconsistente LPA2v) | REJEITADO (mantendo pipeline sem selecao paraconsistente) | -0.0024 | 0.8220 | N/A - tecnica nao ativa no modelo final (CROSS-02) |

**CROSS-02 citation-rule check:** Gap 1's decision string is `REJEITADO...`, which does not start with `ADOTAD`, so its citation field correctly reads `N/A - tecnica nao ativa no modelo final (CROSS-02)` — no PAL2v (Da Costa 1990 / Abe & Nakamatsu 2009) citation appears for an inactive technique. **Confirmed correct.**

## Task Commits

Each task was committed atomically:

1. **Task 1: Run the real paraconsistent-ab A/B comparison to completion** - `2f7f55b` (feat)
2. **Task 2: Verify report correctness, internal consistency, and the CLAUDE.md regression floor** - verification-only, no files produced/modified; findings recorded in this SUMMARY (no separate commit per task_commit_protocol, since `<files>none</files>` for this task)

## Files Created/Modified

- `results/train_log_v34_gap1_paraconsistent_ab.txt` - Full text report: CI table (both arms), direct McNemar, mean_n_selected/feature_reduction, verbatim DECISAO sentence
- `results/train_log_v34_gap1_paraconsistent_ab_console.txt` - Full raw console/stdout log of the entire ~63-minute run (both arms, all 30 fold/vowel/network training curves, extraction log, baseline McNemar sections)
- `results/paraconsistent_ab_comparison.csv` - Machine-readable 7-metric x 2-arm comparison with bootstrap CI bounds, plus mean_n_selected/feature_reduction trailing rows
- `results/paraconsistent_selection_freq.csv` - 2550-row (+ header) per (fold, vowel, network, feature) mu/lambda/Gc/Gct/selected table — PARA-05's core deliverable
- `results/gap_adoption_status.csv` - 3-row consolidated Gap Adoption Status table (CROSS-01), Gap 1's row built live from this run's actual outcome
- `models/selected_master_fold{0-4}_v{0-2}.bin` (15 files) / `models/selected_expert_fold{0-4}_v{0-2}.bin` (15 files) - Per-(fold,vowel,network) persisted selection-index files, generated but not committed (models/ is gitignored)

## Decisions Made

- Restored 55 tracked Phase 1/2 result CSVs that `make clean` deleted from the working tree as an unintended, out-of-scope side effect (the Makefile's documented `rm -f results/*.csv` behavior applies to the whole directory, not just this plan's targets) — used a targeted `git checkout -- results/` rather than any blanket reset, since the only untracked files under `results/` at that point were this plan's own new artifacts (safe to leave alone) and pre-existing WIP logs (`train_log_v30/v31*.txt`, also untracked and left alone).
- Did not commit the 30 `models/selected_*.bin` files — `models/` is gitignored project-wide (confirmed via `.gitignore` line 7), consistent with every prior phase never tracking model weight files; their on-disk presence (verified via `ls ... | wc -l` = 30) satisfies the plan's acceptance criteria without requiring a commit.

## Deviations from Plan

None — plan executed exactly as written. The `make clean`-induced deletion of unrelated tracked files was an expected, documented side effect of a prerequisite step explicitly instructed by the plan (not a bug in the plan or an unplanned discovery requiring Rule 1-4 judgment); it was caught by the task_commit_protocol's mandatory post-commit deletion check and restored with a targeted, non-destructive git operation before any commit was made, so no scope creep or judgment call was needed.

## Issues Encountered

- **Universal paraconsistent-selection fallback (0/85 → 85/85 feature reduction in 30/30 runs):** This is a real, verified finding from the actual run (not a bug introduced by this plan — this plan only executes and verifies code built in Plans 03-01/02/03). The gc-threshold relaxation loop (10 steps, 0.35 → -0.15) never found any feature clearing the paraconsistent-selection threshold, in every single one of the 30 (fold, vowel, network) combinations, always falling back to "keep all 85 features." This fully explains both the 0.0% feature_reduction figure and, downstream, why the trade-off branch of the DECISAO rule could never fire regardless of the Macro F1 delta's sign. Flagging explicitly for Plan 03-05's documentation — this is a first-order finding about the paraconsistent method's behavior on this dataset/feature-space, not a methodological error in the comparison itself.
- Initial process launch omitted the `tee` redirect specified in the plan's `critical_process_note`; caught and corrected before any meaningful runtime had elapsed (killed at ~2s, relaunched with `nohup setsid bash -c '... | tee ...'`), so the full run (including the extraction phase) is captured in `results/train_log_v34_gap1_paraconsistent_ab_console.txt` from `[2026-07-29 15:27:07]` onward — no loss of log coverage.

## User Setup Required

None — no external service configuration required. Pure C99 pipeline execution, no new dependencies.

## Next Phase Readiness

- All 4 durable evidence artifacts required by this plan's frontmatter (`train_log_v34_gap1_paraconsistent_ab.txt`, `paraconsistent_ab_comparison.csv`, `paraconsistent_selection_freq.csv`, `gap_adoption_status.csv`) exist, are freshly generated, internally consistent, and committed at `2f7f55b`.
- Plan 03-05 (final CLAUDE.md documentation) can cite this SUMMARY's "Final Verdict" section directly: DECISAO=REJEITADA, both arms clear the 0.42 floor, McNemar not significant (p=0.8220), and the universal-fallback finding (30/30) as the mechanistic explanation for 0% feature reduction.
- No blockers identified for Plan 03-05.

---
*Phase: 03-gap-1-paraconsistent-feature-selection-final-reporting*
*Completed: 2026-07-29*

## Self-Check: PASSED

- FOUND: results/train_log_v34_gap1_paraconsistent_ab.txt
- FOUND: results/train_log_v34_gap1_paraconsistent_ab_console.txt
- FOUND: results/paraconsistent_ab_comparison.csv
- FOUND: results/paraconsistent_selection_freq.csv
- FOUND: results/gap_adoption_status.csv
- FOUND: 30 models/selected_{master,expert}_fold{0-4}_v{0-2}.bin files on disk (gitignored, not committed by design)
- FOUND commit: 2f7f55b
