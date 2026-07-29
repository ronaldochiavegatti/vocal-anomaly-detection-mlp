# Roadmap: Detecção de Anomalias Vocais (MLP em C) — Fechamento dos 3 Gaps PIBIC

## Overview

O pipeline v29 (Hierarchical Late Fusion, Macro F1 0,4423 / Acc 69,4%) já funciona e é o
baseline de referência. Este milestone fecha os 3 gaps prometidos na proposta PIBIC original
e não implementados: Borderline-SMOTE (Gap 2), comparação redes rasas × profundas (Gap 3) e
Seleção Paraconsistente de Características (Gap 1) — nessa ordem, por exigência do SPEC.md e
confirmação independente da pesquisa. Antes de qualquer gap, uma fase 0 conecta a
infraestrutura estatística já existente (bootstrap CI, McNemar) ao loop de treino e documenta/
mitiga a race condition de RNG sob OpenMP, tornando toda comparação A/B subsequente
verificável e reprodutível. Cada gap fecha com decisão explícita de adoção/rejeição (nunca por
inspeção visual) e atualização do `CLAUDE.md`, terminando em uma tabela consolidada
"Gap Adoption Status" pronta para a defesa perante a banca.

## Phases

**Phase Numbering:**

- Integer phases (0, 1, 2, 3): Planned milestone work, in strict dependency order
- Decimal phases (X.1, X.2): Urgent insertions (marked with INSERTED), none currently planned

- [x] **Phase 0: Statistical Infrastructure & RNG Reproducibility** - Bootstrap CI/McNemar wired into `mode_train()`, RNG race documented/mitigated, baseline reconfirmed with a fresh log (completed 2026-07-27)
- [x] **Phase 1: Gap 2 — Borderline-SMOTE** - `SmoteMode` (standard vs borderline) implemented and adopted/rejected via reproducible A/B comparison (completed 2026-07-28)
- [ ] **Phase 2: Gap 3 — Shallow vs Deep MLP Comparison** - Fold+vowel loop refactored into a reusable function; 4 architecture configs compared and the smallest non-inferior one adopted
- [ ] **Phase 3: Gap 1 — Paraconsistent Feature Selection** - New paraconsistent selection module integrated per (fold, vowel, network); final consolidated Gap Adoption Status report produced

## Phase Details

### Phase 0: Statistical Infrastructure & RNG Reproducibility Prerequisite

**Goal**: Every downstream gap's A/B comparison is statistically verifiable (bootstrap CI + McNemar available on every run) and reproducible (same `RANDOM_SEED=42` yields the same inputs), and the SPEC-cited baseline has a real backing artifact in the repo.
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03
**Success Criteria** (what must be TRUE):

  1. `mode_train()` calls `metrics_bootstrap_ci()` and `metrics_mcnemar()` on the aggregated out-of-fold predictions and prints/saves bootstrap CI + McNemar p-values (MLP vs MajorityClass, kNN, LogReg) for every training run.
  2. Running `make full` twice in a row (same `RANDOM_SEED=42`) produces identical augmented training inputs — the OpenMP RNG race in `precalculate_augmentations()` is fixed (per-thread RNG stream, or the loop is no longer parallel) rather than merely documented, unless a fix is explicitly judged infeasible and the caveat is recorded instead.
  3. `results/train_log_v29_baseline_reconfirmed.txt` exists with a freshly executed run's Macro F1/Accuracy, and either matches `results/metrics_global.csv` (Macro F1 0,4423 / Acc 69,4%) or the divergence is explicitly documented with a hypothesis for the cause.

**Plans:** 3/3 plans complete
Plans:
**Wave 1**

- [x] 00-01-PLAN.md — Fix the OpenMP RNG race in `precalculate_augmentations()` (drop the parallel pragma) and add a fast `verify-rng` determinism-check mode

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 00-02-PLAN.md — Wire `metrics_bootstrap_ci()`/`metrics_mcnemar()` into `mode_train()`, reconnecting MajorityClass/kNN/LogReg baselines for the 3-way comparison

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 00-03-PLAN.md — Run a fresh full pipeline execution and produce `results/train_log_v29_baseline_reconfirmed.txt` with an explicit match/divergence verdict

### Phase 1: Gap 2 — Borderline-SMOTE

**Goal**: Borderline-SMOTE (Han, Wang & Mao, 2005) is implemented as a selectable oversampling mode alongside standard SMOTE, and its adoption or rejection is decided by reproducible A/B comparison rather than assumption.
**Depends on**: Phase 0
**Requirements**: SMOTE-01, SMOTE-02, SMOTE-03, SMOTE-04, SMOTE-05
**Success Criteria** (what must be TRUE):

  1. `smote_oversample()` accepts a `SmoteMode {SMOTE_STANDARD, SMOTE_BORDERLINE}` parameter; borderline mode classifies each minority sample safe/borderline/noise using `find_knn_global()` with the untruncated boundary rule `2*m >= k` (never `m >= k/2` integer division).
  2. Synthetic-sample interpolation always uses `find_knn()` restricted to same-class neighbors — the global k-NN list used for safe/borderline/noise classification is never reused for interpolation.
  3. An empty borderline pool, and the pre-existing `n_class <= 1` case, both produce an explicit warning log with no crash and no degenerate synthetic samples.
  4. `results/` contains an A/B report comparing standard vs borderline SMOTE (same seed/folds): Macro F1 + per-class F1 table with McNemar/bootstrap CI (via Phase 0 infrastructure), a safe/borderline/noise count table per class/fold, and an explicit adopt/reject decision sentence.
  5. `CLAUDE.md` is updated (Optimization History / What Worked / What Didn't Work) with the Gap 2 outcome, regardless of which mode was adopted.

**Plans:** 4/4 plans complete
Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Core Borderline-SMOTE1 algorithm: SmoteMode, find_knn_global(), classify_borderline(), smote_oversample() rewired (SMOTE-01, SMOTE-02, SMOTE-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — mode_train_ex()/ABResult refactor + mode_smote_ab() CLI orchestration + write_smote_ab_report() (SMOTE-01, SMOTE-04)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03-PLAN.md — Execute the real `smote-ab` A/B run and verify the produced report (SMOTE-04)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 01-04-PLAN.md — Update CLAUDE.md with the Gap 2 outcome (SMOTE-05)

### Phase 2: Gap 3 — Shallow vs Deep MLP Comparison (with structural refactor)

**Goal**: The network's hidden-layer configuration is chosen by reproducible statistical comparison across 4 candidate depths, backed by a codebase that can represent variable-depth networks safely, not by inspection or accident.
**Depends on**: Phase 1
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04, ARCH-05, ARCH-06
**Success Criteria** (what must be TRUE):

  1. Documentation/report language is corrected before any comparison is presented: the current production network is labeled Config C `[128, 64]` (2 hidden layers), not the shallow Config A `[128]` originally assumed by SPEC.md.
  2. `mlp_init_multi()` (generalizing `mlp_init_dynamic()`) accepts configurable `hidden_sizes`/`dropout_rates`, and `MLP_MAX_LAYERS` (with corresponding fixed-size buffers in `mlp.c`/`mlp_train.c` — backprop deltas, checkpoints) is widened to safely hold the deepest config (D: 3 hidden layers) with no out-of-bounds writes.
  3. A single reusable fold+vowel training function trains all 4 configs (A/B/C/D, correctly relabeled) against identical 5-fold partitions — no duplicated per-config training loop in `main.c`.
  4. `results/` contains a 4-config comparison table (accuracy, Macro F1, per-class F1, parameter count, time/epoch) with McNemar/bootstrap CI between the best config and every simpler config — never a decision made by visual inspection alone.
  5. The "smallest complexity not statistically worse" decision rule (1-SE rule + McNemar) is applied and the adopted config is stated explicitly with its justification.
  6. `CLAUDE.md` is updated with the Gap 3 outcome, regardless of which config was adopted.

**Plans:** 3/4 plans executed
Plans:
**Wave 1**

- [x] 02-01-PLAN.md — ARCH-01 CLAUDE.md config-label correction + MLP_MAX_LAYERS/mlp_init_multi()/mlp_count_params()/dynamic mlp_backward() sizing + l2_lambda runtime parameter + ad hoc ASan Config D check

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-02-PLAN.md — ArchConfig/RegSetting/ARCH_CONFIGS + widened mode_train_ex() + write_arch_compare_report() (1-SE + McNemar) + mode_arch_compare() CLI orchestration

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 02-03-PLAN.md — Execute the real 12-arm (4 architecture x 3 regularization) arch-compare run and verify the produced report

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 02-04-PLAN.md — Update CLAUDE.md with the Gap 3 outcome + L2 lambda documentation correction

### Phase 3: Gap 1 — Paraconsistent Feature Selection & Final Reporting

**Goal**: Paraconsistent-logic feature selection (LPA2v) is implemented as an independent module, integrated per (fold, vowel, network) using the SMOTE mode and architecture config already adopted in Phases 1–2, and the full milestone closes with a committee-ready consolidated report.
**Depends on**: Phase 2
**Requirements**: PARA-01, PARA-02, PARA-03, PARA-04, PARA-05, PARA-06, CROSS-01, CROSS-02
**Success Criteria** (what must be TRUE):

  1. `src/feature_select_paraconsistent.c` + `include/feature_select_paraconsistent.h` compute μ via ANOVA F-statistic/η² (not SPEC's non-standard per-class ratio) and λ via global-variance-normalized dispersion (not `CV = std/mean`, which explodes near-zero-mean δMFCC features), with a `MIN_STD`-style variance-floor guard and a capped relaxation loop (no unbounded iteration when zero features would otherwise be selected).
  2. Selection runs independently per (fold, vowel, network) — Master (binary) and Expert (4-class) each receive their own selected-feature indices, persisted via `selected_save`/`selected_load`.
  3. `predict_hierarchical_late_fusion()` and the previously-duplicated inline slicing block in the validation loop both consume the same persisted indices consistently — discrete predictions and recorded probabilities never diverge.
  4. `results/` contains an aggregated selection-frequency table (feature, μ, λ, Gc, Gct, selected S/N) across all ~30 (fold × vowel × network) runs — not a single-run snapshot — plus Macro F1 before/after paraconsistent selection.
  5. A consolidated "Gap Adoption Status" table (decision / metric delta / citation status per gap) exists in `results/` or the final report, and no bibliographic citation is added for a technique not actually active in the final shipped model.
  6. `CLAUDE.md` is updated with the Gap 1 outcome, regardless of result.

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 0 → 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 0. Statistical Infrastructure & RNG Reproducibility | 3/3 | Complete   | 2026-07-27 |
| 1. Gap 2 — Borderline-SMOTE | 4/4 | Complete   | 2026-07-28 |
| 2. Gap 3 — Shallow vs Deep MLP Comparison | 3/4 | In Progress|  |
| 3. Gap 1 — Paraconsistent Feature Selection | 0/TBD | Not started | - |
