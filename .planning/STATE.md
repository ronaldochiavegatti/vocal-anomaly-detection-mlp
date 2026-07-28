---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 01-03-PLAN.md (smote-ab A/B execution + verification)
last_updated: "2026-07-28T01:31:56.156Z"
last_activity: 2026-07-28
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 7
  completed_plans: 6
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-27)

**Core value:** Fechar, com rigor metodológico comprovável por comparação A/B (mesma seed, mesmos 5-folds), os 3 gaps entre a implementação atual e a proposta PIBIC original — sem piorar o baseline de referência (Macro F1 0,4423 / Acurácia 69,4%).
**Current focus:** Phase 01 — gap-2-borderline-smote

## Current Position

Phase: 01 (gap-2-borderline-smote) — EXECUTING
Plan: 4 of 4
Status: Ready to execute
Last activity: 2026-07-28

Progress: [█████████░] 86%

## Performance Metrics

**Velocity:**

- Total plans completed: 3
- Average duration: - min
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 00 | 3 | - | - |

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

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 3 (deferred until then): Domain-specific precedent papers for the paraconsistent μ/λ derivation (Costa et al. 2019 DPM; 2025 wavelet+paraconsistent; 2021 grid-fault paper) are paywalled — plan to proceed with the ANOVA-F/η² substitute unless institutional access is obtained.
- Phase 2 (deferred until then): Whether a supplementary regularization-strength check across the 4 architecture configs is in scope, or the fixed-hyperparameter limitation is simply documented, needs an explicit decision during Phase 2 planning.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Tech debt | DEBT-01..04 (magic number, memory leak, CLAUDE.md/MEMORY.md drift, dead modules) | Deferred to v2 | Project init 2026-07-27 |
| Future work | FUTURE-01 (full factorial ablation), FUTURE-02 (external validation of paraconsistent thresholds) | Deferred to v2 | Project init 2026-07-27 |

## Session Continuity

Last session: 2026-07-28T01:31:56.150Z
Stopped at: Completed 01-03-PLAN.md (smote-ab A/B execution + verification)
Resume file: None
