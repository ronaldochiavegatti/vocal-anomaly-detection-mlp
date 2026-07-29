---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: complete
stopped_at: Completed 03-05-PLAN.md
last_updated: "2026-07-29T21:10:00.000Z"
last_activity: 2026-07-29
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 16
  completed_plans: 16
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-27)

**Core value:** Fechar, com rigor metodológico comprovável por comparação A/B (mesma seed, mesmos 5-folds), os 3 gaps entre a implementação atual e a proposta PIBIC original — sem piorar o baseline de referência (Macro F1 0,4423 / Acurácia 69,4%).
**Current focus:** Milestone complete — all 3 SPEC.md gaps closed (Gap 2 ADOTADO, Gap 3 ADOTADO/no-op, Gap 1 REJEITADO), see CLAUDE.md's Gap Adoption Status table.

## Current Position

Phase: 03 (gap-1-paraconsistent-feature-selection-final-reporting) — COMPLETE
Plan: 5 of 5
Status: Milestone complete
Last activity: 2026-07-29

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 11
- Average duration: - min
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 00 | 3 | - | - |
| 01 | 4 | - | - |
| 2 | 4 | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 00 P01 | 70min | 2 tasks | 3 files |
| Phase 00 P02 | 25min | 2 tasks | 1 files |
| Phase 00 P03 | 43min | 2 tasks | 4 files |
| Phase 01 P01 | 25min | 2 tasks | 1 files |
| Phase 01 P02 | 20min | 2 tasks | 1 files |
| Phase 01 P03 | 136min | 3 tasks | 14 files |
| Phase 01 P04 | 6min | 1 tasks | 1 files |
| Phase 02 P01 | 15min | 3 tasks | 7 files |
| Phase 02 P02 | 6min | 2 tasks | 1 files |
| Phase 02 P03 | 375min | 3 tasks | 51 files |
| Phase 02 P04 | 8min | 1 tasks | 1 files |
| Phase 03 P04 | 68min | 2 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Milestone-wide: Order is Gap 2 (Phase 1) → Gap 3 (Phase 2) → Gap 1 (Phase 3), preceded by Phase 0 infrastructure prerequisite — per SPEC.md and confirmed independently by research (structural refactor in Gap 3 must land before Gap 1's higher fan-out; Gap 3's parameter-count table would be invalidated if Gap 1's dimensionality changes came first).
- Milestone-wide: Baseline = v29 HEAD (commit e63483a), not the broken v30/v31 "nested stacked hierarchy" WIP.
- Phase 0: RNG race fix-vs-document is a judgment call to be made explicitly during Phase 0 planning, not left ambiguous (research flag).
- [Phase 00]: Fixed OpenMP RNG race in precalculate_augmentations() by removing #pragma omp parallel for (INFRA-02 sanctioned fallback), not merely documenting it — precalculate_augmentations() is not the pipeline's wall-clock bottleneck; verified deterministic via new verify-rng CLI mode (cmp exit code 0 across two runs)
- [Phase 00]: Baseline computation (MajorityClass/kNN/LogReg) placed BEFORE the per-vowel SMOTE+MLP training loop in each fold — lr_train's internal RNG draws happen first, deliberately shifting the subsequent SMOTE/dropout RNG sequence; documented as accepted ordering, not a bug
- [Phase 00]: Majority-class baseline counts only original (non-augmented) fold->n_train rows of train_y_all — Matches norm_fit's established precedent of fitting only on original training samples
- [Phase 00]: Baseline reconfirmation run MATCHES the SPEC-cited reference (Macro F1 0.4423 / Accuracy 0.6940) -- both fall within this run's own 95% bootstrap CI, so no RNG-divergence hypothesis section was needed
- [Phase 01]: mode_train_ex()'s result parameter is a nullable ownership-transfer switch -- NULL preserves today's train/full CLI byte-for-byte output; non-NULL transfers y_true/y_pred ownership to mode_smote_ab() and suffixes artifact filenames by SMOTE mode
- [Phase 01]: smote-ab A/B comparison runs both SMOTE arms sequentially in one process invocation (mode_smote_ab), relying on kfold_split()'s internal RNG reseed for identical fold assignments -- no manual RNG snapshot/restore
- [Phase 01]: Adopt/reject decision for Borderline-SMOTE is computed by a single fixed rule in write_smote_ab_report() (adopt iff borderline Macro F1 >= standard Macro F1) -- never asserted manually, closing off selective/cherry-picked reporting
- [Phase 01]: DECISAO: Borderline-SMOTE ADOTADO (point estimate: borderline=0.4587 >= padrao=0.4351, delta=+0.0235; bootstrap-mean: 0.4565 >= 0.4338) -- both value sources agree on sign, decision is unambiguous under the fixed rule
- [Phase 01]: Direct McNemar test between Borderline-SMOTE and standard SMOTE arms is NOT statistically significant (chi2=0.1928, p=0.6606) -- ADOTADO reflects a positive point-estimate delta, not a proven significant improvement; carry this caveat into CLAUDE.md documentation
- [Phase 01]: Plain train/full CLI path (mode_train_ex(...,NULL)) reconfirmed byte-identical to Phase 0 baseline (Macro F1 0.4514, Accuracy 0.6976) after plan 01-02's refactor -- zero regression
- [Phase 01]: Empty-borderline-pool fallback fired 0 times among the 60 real classification rows in this run's smote_borderline_counts.csv -- all 30 borderline==0 rows are structural placeholder rows for unused class slots (Master class=1, Expert class=0), not genuine fallback events
- [Phase 01]: CLAUDE.md's Borderline-SMOTE 'What Works' claim corrected to cite Han/Wang/Mao 2005 and results/train_log_v32_gap2_smote_ab.txt (macro F1 +0.0235 delta, McNemar p=0.66 not significant); new dated Gap 2 Outcome subsection added with full comparison table and empty-pool-fallback finding, satisfying SMOTE-05
- [Phase 02 planning]: Regularization-sweep scope resolved by explicit user decision (not left as a fixed-hyperparameter limitation): sweep BOTH dropout and L2 lambda jointly (not dropout alone), using 3 settings (not 2) via a single relative multiplier `REG_MULTIPLIER {0.6, 1.0, 1.4}` (light/baseline/strong) applied to both hyperparameters together — user chose the more expansive option each time over the recommended default, reasoning that architecture and regularization strength are confounded (a "shallow" config could look worse than a "deep" one purely because it was tested at the wrong regularization strength) and the extra compute (12 runs vs. 4) was accepted to remove that confound. Baseline L2 anchor uses config.h's actual `L2_LAMBDA=0.001f` (not CLAUDE.md's stale documented 0.003, itself corrected in Plan 02-04). Config D's 3rd hidden layer dropout is a hardcoded literal 0.3f (SPEC.md's proposed value), not config.h's dead unused `DROPOUT_RATE_HIDDEN3=0.0f` constant. Confirmed by gsd-plan-checker (2nd pass) as roadmap-sanctioned scope, faithfully implemented across all 4 Phase 2 plans.
- [Phase 02]: Plan 02-01: MLP_MAX_LAYERS=5 (one slot of headroom over Config D's exact 4 layers), covering the runtime-configurable-depth generalization needed for Gap 3's architecture comparison
- [Phase 02]: Plan 02-01: mlp_init_dynamic() kept as a behavior-identical thin wrapper around new mlp_init_multi(), avoiding call-site churn elsewhere in the codebase
- [Phase 02]: Plan 02-01: mlp_train() now accepts l2_lambda as a runtime parameter (both existing call sites pass L2_LAMBDA explicitly, zero behavior change), unblocking Plan 02-02's regularization sweep
- [Phase 02]: Plan 02-02: mode_train_ex() widened a 2nd time to (base_dir, smote_mode, arch, reg, result) -- ArchConfig/RegSetting/ARCH_CONFIGS/REG_MULTIPLIER added, ABResult extended with param_count_master/expert + mean_time_per_epoch_sec/mean_epochs_to_stop
- [Phase 02]: Plan 02-02: mode_arch_compare() runs all 12 (architecture x regularization) combos with SMOTE fixed at SMOTE_BORDERLINE, appending each arm to results/arch_compare_comparison.csv incrementally for partial-run durability -- NOT executed in this plan (reserved for Plan 02-03, 6-18+ hour operation)
- [Phase 02]: Plan 02-02: write_arch_compare_report() adoption decision is a fixed 5-step procedure (best reg per arch -> best arch overall -> 1-SE bootstrap-CI band -> McNemar gate -> fewest-params) -- SE derived from bootstrap CI half-width, not per-fold CART formula, since no per-fold macro_f1 array exists in this codebase
- [Phase 02]: Plan 02-02: fixed a real per-arm filename collision bug proactively (all 12 arch-compare arms share SMOTE_BORDERLINE) by suffixing metrics_global/bootstrap_ci/mcnemar_vs_baselines/smote_borderline_counts paths with arch->name/REG_NAME[reg] -- disclosed side effect: mode_smote_ab()'s 2 diagnostic filenames now suffix with _C_baseline, its actual required deliverables (smote_ab_comparison.csv, train_log_v32_gap2_smote_ab.txt) are unaffected
- [Phase 02]: Plan 02-03: Task 1's plain train/full CLI regression check reproduced results/metrics_global.csv, bootstrap_ci.csv, mcnemar_vs_baselines.csv byte-for-byte identical to the committed baseline (git diff = zero changes) -- Plan 02-02's mode_train_ex() widening confirmed fully behavior-preserving
- [Phase 02]: Plan 02-03: Real 12-arm arch-compare sweep executed to completion (~5h30min): best-per-arch = A/light 0.4365, B/light 0.4452, C/baseline 0.4587, D/light 0.4741 (=ao); 1-SE band=[0.4554,0.4741]; DECISAO = adopted architecture C at baseline regularization (today's exact production config, 38918 params) -- within 1-SE band and not significantly worse than D by McNemar (p=0.7463), fewer params than D (42886)
- [Phase 02]: Plan 02-03: Pitfall 3 (fixed-hyperparameter 'deeper is undertuned') confirmed present: Config D at strong regularization collapses (mean_epochs_to_stop=46.4 vs D's own light=76.8/baseline=87.3; macro_f1=0.2777, statistically indistinguishable from MajorityClass p=0.1176) -- does not change the adoption decision but must be documented as an explicit limitation in Plan 02-04's CLAUDE.md update
- [Phase 02]: No production config.h change required: adopted arm (Config C, baseline regularization) is today's exact compiled production configuration -- Gap 3's outcome is a validated no-op for production code, documented explicitly in CLAUDE.md's new Gap 3 Outcome subsection
- [Phase 02]: CLAUDE.md's L2 lambda documentation corrected (0.003 -> real config.h value 0.001) and Pitfall 3 caveat (Config D collapses at strong regularization, mean_epochs_to_stop=46.4, macro_f1=0.2777 statistically indistinguishable from MajorityClass) carried forward verbatim, satisfying ARCH-06
- [Phase 03]: Plan 03-04 real paraconsistent-ab run completed -- DECISAO: Selecao Paraconsistente REJEITADA (Macro F1 sem-selecao=0.4610 raw/0.4597 bootstrap-mean > com-selecao=0.4587 raw/0.4565 bootstrap-mean, delta=-0.0024, McNemar chi2=0.0506 p=0.8220 not significant, feature_reduction=0.0% far below SPEC.md's 30% trade-off floor) -- pipeline stays without paraconsistent feature selection in production

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 3 (deferred until then): Domain-specific precedent papers for the paraconsistent μ/λ derivation (Costa et al. 2019 DPM; 2025 wavelet+paraconsistent; 2021 grid-fault paper) are paywalled — plan to proceed with the ANOVA-F/η² substitute unless institutional access is obtained.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Tech debt | DEBT-01..04 (magic number, memory leak, CLAUDE.md/MEMORY.md drift, dead modules) | Deferred to v2 | Project init 2026-07-27 |
| Future work | FUTURE-01 (full factorial ablation), FUTURE-02 (external validation of paraconsistent thresholds) | Deferred to v2 | Project init 2026-07-27 |

## Session Continuity

Last session: 2026-07-29T19:38:11.534Z
Stopped at: Completed 03-04-PLAN.md
Resume file: None
