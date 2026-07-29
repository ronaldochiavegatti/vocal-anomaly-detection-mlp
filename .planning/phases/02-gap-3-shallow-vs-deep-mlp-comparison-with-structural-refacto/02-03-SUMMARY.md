---
phase: 02-gap-3-shallow-vs-deep-mlp-comparison-with-structural-refacto
plan: 03
subsystem: infra
tags: [c99, mlp, arch-compare, mcnemar, bootstrap-ci, gap-3-evidence]

# Dependency graph
requires:
  - phase: 02-02
    provides: "mode_arch_compare()/write_arch_compare_report()/arch-compare CLI dispatch -- the orchestration code this plan actually invoked"
provides:
  - "Real, freshly-executed 12-arm (4 architectures x 3 regularization strengths) comparison in results/, produced by a single ~5h30min ./build/vocal_detect arch-compare . run"
  - "Verified adoption decision: architecture=C [128,64] (today's production), regularization=baseline (dropout/L2 unchanged), 38918 total params, mechanically confirmed via the fixed 1-SE-band + McNemar procedure"
  - "Confirmed zero regression of Plan 02-02's mode_train_ex() widening against the currently committed production baseline (byte-identical metrics_global.csv/bootstrap_ci.csv/mcnemar_vs_baselines.csv)"
  - "Pitfall 3 (fixed-hyperparameter 'deeper is undertuned' artifact) explicitly confirmed present: Config D at strong regularization collapses (mean_epochs_to_stop=46.4, macro_f1=0.2777, statistically indistinguishable from MajorityClass)"
affects: [02-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "nohup + tee + active poll-to-completion (pgrep-based, ~10min intervals) for both the single-run regression check (~33min) and the 12-arm sweep (~5h30min) -- same pattern as Phase 0/Phase 1's long-running-process precedent, scaled to this plan's much longer duration"
    - "Byte-identical-file regression check: rather than a numeric-tolerance comparison, `git diff --stat` on the regenerated results/metrics_global.csv/bootstrap_ci.csv/mcnemar_vs_baselines.csv against the committed baseline showed literally zero changes -- the strongest possible confirmation of RNG-sequence-preserving, behavior-preserving refactoring"

key-files:
  created:
    - results/train_log_v33_gap3_regression_check_console.txt
    - results/arch_compare_comparison.csv
    - results/train_log_v33_gap3_arch_compare.txt
    - results/train_log_v33_gap3_arch_compare_console.txt
    - "results/metrics_global_borderline_{A,B,C,D}_{light,baseline,strong}.csv (12 files)"
    - "results/bootstrap_ci_borderline_{A,B,C,D}_{light,baseline,strong}.csv (12 files)"
    - "results/mcnemar_vs_baselines_borderline_{A,B,C,D}_{light,baseline,strong}.csv (12 files)"
    - "results/smote_borderline_counts_{A,B,C,D}_{light,baseline,strong}.csv (12 files)"
  modified: []

key-decisions:
  - "Task 1's regression-check run reproduced results/metrics_global.csv, results/bootstrap_ci.csv, and results/mcnemar_vs_baselines.csv byte-for-byte identical to the currently committed baseline (git diff --stat = zero changes) -- confirming Plan 02-02's mode_train_ex(base_dir, SMOTE_STANDARD, &ARCH_CONFIGS[2], REG_BASELINE, NULL) widening is fully behavior-preserving for the plain train/full CLI path before the much longer 12-arm sweep began"
  - "12-arm arch-compare sweep completed in ~5h30min wall clock (15:59-21:29), within the plan's estimated 6-18+ hour range but toward the lower end -- Config A/B/C are relatively fast (0.06-0.26 sec/epoch); the augmentation precompute step (~18-25 min per arm, recomputed independently and redundantly for all 12 arms since it depends only on raw audio + RNG stream, not architecture) is the single largest fixed per-arm cost"
  - "DECISAO (mechanically reproduced by hand in Task 3): adopted architecture = C [128,64] (today's production), regularization = baseline (today's production dropout/L2, unchanged) -- best-overall (ao) was D/light (macro_f1=0.4741), but C/baseline (macro_f1=0.4587) falls inside the 1-SE band [0.4554, 0.4741] AND is not significantly worse than D by McNemar (p=0.7463 >= 0.05), and has fewer total parameters (38918) than D (42886). A and B are excluded because their best-reg macro_f1 (0.4365, 0.4452) falls below the 1-SE band's lower bound (0.4554), even though their point-estimate McNemar vs D also shows p>=0.05 (A: p=0.0733, B: p=0.4996) -- the band gate, not McNemar alone, is what excludes them, exactly as the fixed procedure specifies."
  - "Pitfall 3 (RESEARCH.md's fixed-hyperparameter 'deeper is undertuned' caveat) is confirmed present and explicitly documented, not silently absorbed: Config D at strong regularization shows mean_epochs_to_stop=46.4 -- roughly half of D's own light (76.8) and baseline (87.3) arms, and far below every other architecture's strong-regularization arm (A=77.7, B=83.9, C=74.1). D/strong's macro_f1 (0.2777) collapses to the point of being statistically indistinguishable from the MajorityClass baseline (McNemar p=0.1176, the only 'not significantly better than MajorityClass' MLP result across all 12 arms). This does not change the adoption decision (D's own best regularization is light, not strong, so D/strong was never a decision-procedure candidate) but is a caveat Plan 02-04's CLAUDE.md update must carry forward: this run cannot distinguish 'depth doesn't help' from 'depth needs a lighter regularization schedule than tested here.'"

requirements-completed: [ARCH-04, ARCH-05]

# Metrics
duration: 375min
completed: 2026-07-28
---

# Phase 2 Plan 3: Real 12-Arm Arch-Compare Execution + Regression Check + Decision Verification Summary

**Ran the actual 12-arm (4 architectures x 3 regularization strengths) Gap 3 sweep to completion (~5h30min) and the plain train/full CLI regression check (~33min), then mechanically re-derived the 1-SE+McNemar adoption decision by hand from the raw CSV: today's production architecture (Config C [128,64]) at its current baseline regularization is confirmed statistically adopted, with an explicit Pitfall 3 caveat (Config D collapses at strong regularization) documented for Plan 02-04.**

## Performance

- **Duration:** ~6h15min total (Task 1 regression check ~33min: 15:23-15:56; Task 2 arch-compare sweep ~5h30min: 15:59-21:29; Task 3 verification ~10min)
- **Started:** 2026-07-28T15:23 (approx, per console log timestamps)
- **Completed:** 2026-07-28T21:38
- **Tasks:** 3 completed
- **Files modified/created:** 51 (2 commits: 1 regression-check console log; 1 comparison CSV + report + console log + 48 per-arm diagnostic files)

## Accomplishments

### Task 1: Plain train/full CLI regression check
- Ran `./build/vocal_detect train .` to completion (exit 0, ~33min wall clock including ~45s feature (re-)extraction since no `results/features.csv` cache existed at plan start)
- Regenerated `results/metrics_global.csv` (Accuracy 0.6976, Macro F1 0.4514), `results/bootstrap_ci.csv`, `results/mcnemar_vs_baselines.csv` -- all three files are **byte-for-byte identical** to the currently committed baseline (`git diff --stat` showed zero changes), the strongest possible confirmation that Plan 02-02's `mode_train_ex()` widening (adding `arch`/`reg` parameters) introduced zero behavioral regression to the exact code path real users/CI invoke
- This gave high confidence to proceed into the much longer Task 2 sweep

### Task 2: Full 12-arm architecture x regularization sweep
- Ran `./build/vocal_detect arch-compare .` to completion (exit 0, ~5h30min wall clock: 15:59-21:29), detached via `nohup`+`tee`, actively polled to completion across ~35 consecutive Bash poll cycles at ~10-minute intervals -- never ended the turn while the process ran, per the plan's critical_process_note
- Produced `results/arch_compare_comparison.csv` (13 lines: 1 header + 12 arm rows, appended incrementally arm-by-arm as designed in Plan 02-02), `results/train_log_v33_gap3_arch_compare.txt` (full Portuguese report, exactly one `DECISAO:` line), and all 48 per-arm diagnostic files (`metrics_global`/`bootstrap_ci`/`mcnemar_vs_baselines`/`smote_borderline_counts` x 4 architectures x 3 regularization settings)
- Confirmed C/baseline's macro_f1 (0.4587) in this run matches Phase 1's v32 Borderline-SMOTE A/B result (0.4587 point estimate) exactly, cross-validating this run's correctness against prior evidence
- Full 12-row result table (accuracy/macro_f1 per arm):

| Arch | Reg | Accuracy | Macro F1 | Params (M+E) | mean_epochs_to_stop |
|---|---|---|---|---|---|
| A | light | 0.6831 | 0.4365 | 11266+11524 | 72.47 |
| A | baseline | 0.6922 | 0.4239 | 11266+11524 | 78.73 |
| A | strong | 0.6794 | 0.3911 | 11266+11524 | 77.73 |
| B | light | 0.6940 | 0.4452 | 5634+5764 | 82.60 |
| B | baseline | 0.6858 | 0.4118 | 5634+5764 | 84.50 |
| B | strong | 0.6667 | 0.3691 | 5634+5764 | 83.93 |
| C | light | 0.6876 | 0.4431 | 19394+19524 | 76.67 |
| **C** | **baseline** | **0.6967** | **0.4587** | **19394+19524** | 80.53 |
| C | strong | 0.6566 | 0.3723 | 19394+19524 | 74.10 |
| D | light | 0.7004 | 0.4741 | 21410+21476 | 76.80 |
| D | baseline | 0.7040 | 0.4718 | 21410+21476 | 87.27 |
| D | strong | 0.6093 | 0.2777 | 21410+21476 | **46.40** |

(Bold: adopted arm. Bold+italic epoch count: Pitfall 3 collapse.)

### Task 3: Mechanical decision verification + Pitfall 3 check
- **Best-per-architecture cross-check** (reproduced by hand from `arch_compare_comparison.csv`, confirmed matching the report's "melhor regularizacao por arquitetura" table):
  - A: light (0.4365) — B: light (0.4452) — C: baseline (0.4587) — D: light (0.4741)
- **Best-overall (`ao`)**: D/light, macro_f1=0.4741 (highest of the 4 best-per-architecture values) — matches the report's stated `ao`
- **1-SE band**: SE derived from D/light's bootstrap CI half-width: `(0.5097 - 0.4365) / (2*1.96) = 0.0187`. Band = `[0.4741 - 0.0187, 0.4741] = [0.4554, 0.4741]` — matches the report exactly
- **Band + McNemar gate applied by hand**:
  - A (0.4365) and B (0.4452) fall **below** the band's lower bound (0.4554) → excluded, regardless of their McNemar p-values vs D (A: p=0.0733, B: p=0.4996 — both technically "not significantly worse," but the band gate excludes them first, exactly as the fixed procedure specifies: band membership is required, McNemar non-significance is an additional filter, not a substitute)
  - C (0.4587) falls **inside** the band (0.4554 ≤ 0.4587 ≤ 0.4741) AND McNemar C-vs-D shows p=0.7463 (≥0.05, not significantly worse) → passes both gates
  - D itself always passes (it is `ao`)
  - Among passing candidates {C, D}, fewest total parameters: C=38918 < D=42886 → **adopted = C, baseline**
- **Verdict**: The `DECISAO:` sentence in `results/train_log_v33_gap3_arch_compare.txt` (adopted=C, reg=baseline, params=38918) is fully internally consistent with the raw numbers in `results/arch_compare_comparison.csv` — reproduced independently by hand, not merely trusted at face value.
- **Pitfall 3 (mean_epochs_to_stop) check across all 12 arms**: Config D at strong regularization is a clear, unambiguous positive case for RESEARCH.md's Pitfall 3. `mean_epochs_to_stop` for D/strong = 46.40, compared to D/light=76.80 and D/baseline=87.27 (roughly half), and far below every other architecture's own strong-regularization arm (A/strong=77.73, B/strong=83.93, C/strong=74.10). D/strong's macro_f1 also collapses to 0.2777 — the lowest of all 12 arms by a wide margin — and its McNemar-vs-MajorityClass result (p=0.1176) is the **only** one of the 12 arms where the MLP is not statistically distinguishable from the trivial majority-class baseline. This is direct, explicit evidence that "Config D is worse" is at minimum partly a "Config D + strong regularization is undertuned/miscalibrated" artifact rather than solely a depth-capacity limitation — combined dropout `[0.5,0.4,0.3]*1.4 = [0.70,0.56,0.42]` and L2 `0.001*1.4=0.0014` on an already-data-starved Expert network (down to a few hundred samples per fold/vowel) appears to be crippling training well before convergence. This does **not** change the adoption decision (D's own best regularization is light, not strong — D/strong was never a decision-procedure candidate), but is recorded here explicitly per the plan's Threat T-02-10 mitigation, for Plan 02-04's CLAUDE.md documentation to carry forward as an explicit limitation of this comparison.
- **Task 1 cross-reference**: Task 1's regression-check result (byte-identical `metrics_global.csv`/`bootstrap_ci.csv`/`mcnemar_vs_baselines.csv` vs the committed baseline) is fully consistent with this run's C/baseline arm using `SMOTE_BORDERLINE` producing a slightly higher macro_f1 (0.4587) than the SMOTE_STANDARD-based committed baseline (0.4514) — the expected direction and magnitude given Phase 1's already-adopted Borderline-SMOTE decision (delta ~+0.0235 to +0.0073 depending on exact comparison basis), not a new anomaly.

## Task Commits

Each task was committed atomically:

1. **Task 1: Regression-check the plain train/full CLI path** - `c2255f4` (test) — no diff to metrics_global.csv/bootstrap_ci.csv/mcnemar_vs_baselines.csv (byte-identical); only the new console log file committed
2. **Task 2: Run the full 12-arm architecture x regularization sweep** - `2a85ccc` (feat) — 51 new files (comparison CSV, report, console log, 48 per-arm diagnostics)
3. **Task 3: Verify the 1-SE + McNemar decision and Pitfall 3 check** - verification-only, no files modified; findings recorded in this SUMMARY

**Plan metadata:** (this commit, docs: complete plan)

## Files Created/Modified
- `results/train_log_v33_gap3_regression_check_console.txt` — full console output of Task 1's plain train/full run
- `results/arch_compare_comparison.csv` — 13-line (1 header + 12 arm rows) machine-readable comparison table, written incrementally arm-by-arm
- `results/train_log_v33_gap3_arch_compare.txt` — full Portuguese report: complete 12-row table, best-per-architecture summary, 1-SE band, McNemar-vs-`ao` results, `DECISAO:` sentence
- `results/train_log_v33_gap3_arch_compare_console.txt` — full console output of the 12-arm sweep (all fold/epoch/metrics logging)
- `results/metrics_global_borderline_{A,B,C,D}_{light,baseline,strong}.csv` (12 files) — per-arm class-level metrics
- `results/bootstrap_ci_borderline_{A,B,C,D}_{light,baseline,strong}.csv` (12 files) — per-arm bootstrap CI (N=1000, seed=42)
- `results/mcnemar_vs_baselines_borderline_{A,B,C,D}_{light,baseline,strong}.csv` (12 files) — per-arm McNemar vs MajorityClass/kNN/LogReg
- `results/smote_borderline_counts_{A,B,C,D}_{light,baseline,strong}.csv` (12 files) — per-arm safe/borderline/noise SMOTE pool counts

## Decisions Made

See `key-decisions` in frontmatter above for the full list. The most consequential: the adoption decision (Config C at baseline regularization — today's exact production configuration) requires **no architectural change** for Gap 3's outcome. This is a legitimate, mechanically-verified result of the fixed 1-SE + McNemar procedure, not a foregone conclusion — Config D actually scored higher on point-estimate Macro F1 (0.4741 vs 0.4587), but the procedure's statistical-parsimony rule correctly identifies that this difference is not large enough (relative to bootstrap uncertainty) or significant enough (by McNemar) to justify adopting a ~10% larger network.

## Deviations from Plan

None. Plan executed exactly as written; all three tasks' acceptance criteria were met without needing any Rule 1-4 deviations.

### Auto-fixed Issues

None required.

## Issues Encountered

**Stale background-loop process caused a false "still running" reading for ~10 minutes during Task 1.** An earlier Bash-tool foreground-timeout auto-backgrounded a `while pgrep -f "vocal_detect train" ...` polling loop; because that command's own process argv contains the literal string `"vocal_detect train"`, subsequent `pgrep -f "vocal_detect train"` checks self-matched the stale loop process itself even after the actual `vocal_detect` binary had exited. Diagnosed via `ps aux`/`pgrep -af` showing only the stale bash loop (PID 37396) with no `vocal_detect` binary in the process list, confirmed via `git diff --stat` showing the regenerated files were already present and unchanged. Resolved by killing the stale loop (`kill 37396`) and switching subsequent Task 2 polling checks to `ps aux | grep "build/vocal_detect arch-compare"` (matching the literal binary invocation, not a `pgrep -f` self-referencing pattern) to avoid recurrence. This did not affect Task 1's actual result or timing, only the accuracy of the polling script's own status reporting.

## User Setup Required

None — no external service configuration required. SVD audio directories and `overview_merged.csv` were already present at plan start (confirmed by orchestrator).

## Next Phase Readiness

- **Plan 02-04 can proceed directly** using this plan's verdict: adopted architecture = **C [128,64] at baseline regularization** (today's exact production configuration, no change required), with the explicit Pitfall 3 caveat (Config D collapses at strong regularization, `mean_epochs_to_stop` nearly halves, macro_f1 crashes to near-majority-class performance) to be documented in `CLAUDE.md`'s Gap 3 Outcome section per ARCH-06.
- All raw data needed for Plan 02-04's `CLAUDE.md` update is present in `results/arch_compare_comparison.csv` and `results/train_log_v33_gap3_arch_compare.txt` — no re-derivation needed.
- No blockers identified for Plan 02-04.

## Self-Check: PASSED

Both task commits (`c2255f4`, `2a85ccc`) verified present in `git log --oneline --all`. All required artifacts verified present and non-empty: `results/train_log_v33_gap3_arch_compare.txt` (contains exactly 1 `DECISAO:` line), `results/arch_compare_comparison.csv` (13 lines), all 48 per-arm diagnostic files (`metrics_global`/`bootstrap_ci`/`mcnemar_vs_baselines`/`smote_borderline_counts` x 4 arch x 3 reg — verified via loop, zero `MISSING:` lines emitted). No missing items.

---
*Phase: 02-gap-3-shallow-vs-deep-mlp-comparison-with-structural-refacto*
*Completed: 2026-07-28*
