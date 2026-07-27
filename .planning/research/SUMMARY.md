# Project Research Summary

**Project:** Detecção de Anomalias Vocais (MLP em C) — PIBIC Academic-Rigor Gap Closure
**Domain:** Pure C99 numerical/ML pipeline — algorithmic-fidelity and academic-methodology research (no new libraries; the "stack" is exact formulas/pseudocode that must match cited literature, and the "features" are committee-facing reporting deliverables, not product capabilities)
**Researched:** 2026-07-27
**Confidence:** MEDIUM-HIGH overall (HIGH on architecture integration points and on Gap 2's algorithm fidelity, since primary sources/code were read in full; MEDIUM on Gap 1's μ/λ derivation, since no single canonical formula exists in the cited literature for deriving evidence degrees from continuous features)

## Executive Summary

This is not a greenfield build — it is a rigor-closure project on an already-working, already-deployed v29 Hierarchical Late Fusion voice-pathology classifier (Master binary + Expert 4-class, per-vowel, late-fused by probability averaging; 5-fold CV; baseline Macro F1 0.4423 / Accuracy 69.4%). The task is to add three specific, academically-cited techniques (Borderline-SMOTE, paraconsistent-logic feature selection, shallow-vs-deep MLP architecture comparison) that were promised in the original PIBIC proposal but never implemented, and to prove — via reproducible A/B comparison on the same seed/folds — that each either improves or is a documented, defensible non-improvement. Research across all four files converges on the same conclusion: the algorithms themselves are well-understood and low-risk to implement, but the actual risk lives in (a) silent numerical/statistical bugs that corrupt an A/B comparison without ever crashing (integer truncation in SMOTE's boundary rule, divide-by-zero in the Fisher-like μ formula, a fixed-size struct that silently overflows for deeper networks, and a pre-existing OpenMP RNG data race that already undermines "same seed" reproducibility today), and (b) academic-credibility failure modes specific to a committee defense (citing a technique not actually active in the shipped model — which this project has already done once — reporting only the metric that improved, or eyeballing a chart instead of running the McNemar/bootstrap-CI infrastructure that already exists in `metrics.c` but is currently disconnected from `mode_train()`).

The recommended approach is exactly SPEC.md's own stated order — Gap 2 (Borderline-SMOTE) → Gap 3 (shallow-vs-deep) → Gap 1 (paraconsistent selection) — confirmed independently by the architecture research as sound for a second, deeper reason: Gap 3 requires extracting the 83-line fold+vowel training loop in `main.c` into a reusable function (to run 4 architecture configs against the same 5 folds), and Gap 1 requires widening `predict_hierarchical_late_fusion()`'s signature to carry per-vowel-per-network selected-feature indices through to inference — doing the structural refactor (Gap 3) before the highest-fan-out change (Gap 1, up to 30 independent selection runs per full pipeline execution: 5 folds × 3 vowels × 2 networks) keeps `main.c` maintainable and avoids duplicating fragile per-vowel slicing logic across an ever-multiplying loop nest. Architecture research also found and corrected a factual drift in SPEC.md itself: the current production network already has 2 hidden layers ([128, 64]), which is SPEC.md's own "Config C," not the shallow "Config A" SPEC.md assumed as the baseline — this relabeling must happen before Gap 3 experiments are framed for the committee.

Key risks, in priority order: (1) a battery of silent-failure-mode bugs specific to each gap (documented exhaustively in PITFALLS.md with exact fixes — integer truncation, wrong-neighbor-pool interpolation, divide-by-zero, fixed-size layer arrays, unbounded relaxation loops) that must be guarded against in the *first* implementation pass, not discovered after a full expensive sweep; (2) the pre-existing, already-confirmed OpenMP RNG race in `precalculate_augmentations()` that means "same `RANDOM_SEED=42`" does not currently guarantee bit-identical inputs across the "before" and "after" runs each gap's A/B comparison depends on — this must be documented as a standing caveat in every gap's log, and ideally fixed as a lightweight prerequisite; and (3) the disconnected statistical-significance infrastructure (`metrics_bootstrap_ci`, `metrics_mcnemar` exist in `metrics.c` but are never called from `mode_train()`) which must be wired in before the *first* A/B comparison is run, since without it "adopt only if Macro F1 is equal or better" is unenforceable against fold-to-fold noise already documented to range from 0.28 to 0.70 for individual folds.

## Key Findings

### Recommended Stack

There is no new technology stack to adopt — this is a pure-C99 codebase with zero external ML dependencies, and the "stack" research instead verified the exact algorithms against primary/canonical sources. Borderline-SMOTE1 (Han, Wang & Mao, 2005) was read in full and confirmed to match SPEC.md's proposed implementation almost exactly, with two precision fixes needed (name `m` and `k` as independently-tunable constants rather than reusing `k=5` for both roles; label the DANGER-empty fallback explicitly as an engineering extension not in the original paper). The paraconsistent Gc/Gct lattice formulas (`Gc = μ − λ`, `Gct = μ + λ − 1`, ±0.5 12-region lattice) are canonical and correctly specified in SPEC.md, but the μ/λ-from-continuous-features derivation has no single universally-cited formula in the literature — SPEC.md's proposed μ formula (per-class Fisher-like ratio dividing only by the target class's own variance) is mathematically non-standard and biases toward whichever class has the tightest variance (likely "Normal," n=687); the standard one-way ANOVA F-statistic or η² (eta-squared) is the recommended, textbook-correct substitute. For Gap 3, the 1-standard-error rule (Hastie/Tibshirani/Friedman) combined with McNemar's test (Dietterich, 1998) is the recommended, doubly-citable decision protocol for "choose the smallest model not statistically worse."

**Core algorithms:**
- Borderline-SMOTE1 (Han, Wang & Mao, 2005): oversample only minority samples near the decision boundary — matches SPEC.md's proposal, lowest implementation risk of the three gaps
- Paraconsistent Annotated Logic / LPA2v (da Costa/Abe/Silva Filho): Gc/Gct/12-region lattice machinery is standard; the μ/λ derivation from continuous acoustic features must be presented as this project's own adaptation, not a literal citation
- 1-SE rule (Hastie et al., ESL 2nd ed. §7.10) + McNemar (Dietterich, 1998) + Cawley & Talbot (2010) selection-bias caveat: the standard citation set for "pick the smallest architecture not statistically worse than the best"

### Expected Features (Committee-Facing Deliverables)

"Features" here means the tables, plots, and disclosures a PIBIC defense committee will expect for each gap to be considered a credible, defensible implementation — not end-user product features.

**Must have (table stakes, per SPEC.md's own acceptance criteria):**
- Gap 2: per-class + Macro F1 A/B table (same seed/folds), explicit adopt/reject decision, citation only if adopted
- Gap 1: full (feature, μ, λ, Gc, Gct, selected S/N) table per network/vowel; retained-feature counts; Macro F1 before/after with the specific SPEC decision rule invoked stated explicitly; explicit train-only statistic disclosure
- Gap 3: 4-config × (accuracy, Macro F1, per-class F1, params, time/epoch) table with a stated cross-network/vowel aggregation method; McNemar or bootstrap CI between best config and every simpler config (never visual-only, per SPEC's own explicit ban); explicit final-config decision sentence
- Cross-cutting: no citation for a method not active in the final shipped model (this project has already made this mistake once per SPEC.md's own admission); CLAUDE.md/MEMORY.md updated per gap regardless of outcome

**Should have (differentiators, strengthen the defense beyond SPEC's minimum):**
- Gap 2: safe/borderline/noise breakdown per class/fold (turns a logged warning into a disclosed metric); formal significance test between SMOTE variants (closes an asymmetry — Gap 3 gets one, Gap 2 doesn't as currently specified)
- Gap 1: (Gct, Gc) 12-region scatter plot as visual proof of method fidelity; cross-fold feature-selection stability analysis (Jaccard similarity across the 30 fold/vowel/network runs); threshold sensitivity grid
- Gap 3: learning-curve overfitting comparison across configs (converts "shallow won" from an accident into a principled, mechanistic story); efficiency-normalized metric (Macro F1 per 1k params)
- Cross-cutting: a single consolidated "Gap Adoption Status" table (decision/metric-delta/citation-status per gap) — cheap, high-scanability, first thing a committee member will look for

**Defer (beyond current PIBIC defense scope):**
- Full factorial ablation across SMOTE mode × network depth × feature-selection setting (natural follow-up publication line, not needed now given the sequential Gap 2→3→1 order)
- External validation of paraconsistent thresholds on a second voice-pathology dataset

### Architecture Approach

The current v29 pipeline trains 30 independent MLPs per full run (5 folds × 3 vowels × 2 networks: Master binary + Expert 4-class), all hooking into a single ~369-line `main.c` orchestrator. All three gaps hook in strictly after normalization (`norm_fit`/`norm_transform`, already fold-scoped correctly) and before-or-during per-vowel/per-network MLP instantiation — no changes are needed to `FeatureMatrix`, `KFoldSplit`, or `NormParams`. Gap 2 is a pure two-call-site parameter-widening change (`smote_oversample()` gains a `SmoteMode` argument). Gap 3 requires the largest structural change: extracting the fold+vowel training loop body into a reusable function so 4 architecture configs can be trained against identical partitions, plus widening `MLP.layers[]` from a compile-time-fixed array (currently sized for exactly 3 layers) to a generous fixed maximum, since genuinely different network depths must coexist at runtime within the same process (nested-CV trains all 4 configs together). Gap 1 requires the highest-fan-out change: per-(fold, vowel, network) selected feature-index arrays that must be threaded all the way to inference time, including a currently-duplicated per-vowel slicing code path (`predict_hierarchical_late_fusion()` and a separate inline copy in the validation loop) that both need synchronized updates or the discrete predictions and recorded probabilities will silently disagree.

**Major components (existing, reused by all 3 gaps):**
1. `smote_oversample()` / `find_knn()` (`main.c:189-231`) — extended with a mode flag for Gap 2, called twice per vowel per fold (Master + Expert)
2. `mlp_init_dynamic()` (`mlp.c:220-241`) — generalized to `mlp_init_multi()` for Gap 3's variable-depth configs; runtime layer-count logic already correctly parameterized, only the array *sizing* is compile-time-bound
3. `predict_hierarchical_late_fusion()` (`main.c:39-69`) — must grow to accept per-vowel selected-index arrays for Gap 1; the single riskiest edit since it's called for every validation sample in every fold
4. `feature_select.h` (`selected_save`/`selected_load`) — generic persistence layer, already exists and matches Gap 1's assumed interface exactly, no new naming/collision
5. `metrics.c` (`metrics_bootstrap_ci`, `metrics_mcnemar`) — fully implemented but currently disconnected from `mode_train()`; must be wired in as a cross-cutting prerequisite before any gap's A/B comparison is trusted

### Critical Pitfalls

1. **Integer-division truncation silently shifts the Borderline-SMOTE safe/danger boundary** — `m >= k/2` truncates to `2` for odd `k=5` instead of `2.5`, misclassifying borderline samples. Fix: compare `2*m >= k` instead, never divide-then-truncate.
2. **Reusing the global (all-class) k-NN list for same-class interpolation** — the natural "avoid computing k-NN twice" refactor would silently produce cross-class synthetic samples carrying the minority label (silent label noise, no crash). Fix: keep `find_knn_global()` (for m-counting) and `find_knn()` (for interpolation, same-class only) as textually distinct calls.
3. **Divide-by-zero in the Fisher-ratio-like μ formula** — `var_c(j) == 0` for at least one (feature, class, fold) pair is plausible given 237 candidate features and no existing variance-floor guard; produces `inf`/`NaN` that poisons min-max normalization for every other feature. Fix: port the `MIN_STD`-style epsilon-clamp pattern already used in `normalize.c` into the new paraconsistent module.
4. **`Layer layers[MLP_NUM_LAYERS]` is a compile-time-fixed-size array** — Gap 3's deeper configs (up to 4 layers) need to coexist at runtime with shallower ones; writing past the current fixed size of 3 is a silent out-of-bounds heap/stack write with no crash guarantee. Fix: widen to a generous fixed `MLP_MAX_LAYERS` (e.g. 5) and rely on the already-correct runtime `net->num_layers` field used everywhere else.
5. **The pre-existing OpenMP RNG data race poisons every downstream "reproducible" random draw** — not just augmentation noise, but every SMOTE draw and every network weight initialization in the same process becomes non-reproducible under `RANDOM_SEED=42` once the racy parallel augmentation step has run. Must be documented as a standing caveat in every A/B log for all 3 gaps; ideally fixed (drop the `#pragma omp` from `precalculate_augmentations` or give each patient index its own RNG stream) as a lightweight prerequisite, since it directly undermines the "same seed, same folds" comparison protocol every gap's acceptance criterion depends on.

## Implications for Roadmap

Based on combined research, the roadmap should follow SPEC.md's own gap ordering (Gap 2 → Gap 3 → Gap 1), but with two cross-cutting prerequisite phases inserted first, since both pitfalls research and architecture research independently converge on "wire up the shared infrastructure before any gap-specific work, or every subsequent A/B comparison is unenforceable/unsafe."

### Phase 0: Statistical Infrastructure + RNG Reproducibility Prerequisite
**Rationale:** PITFALLS.md's Pitfall 4 (disconnected bootstrap CI/McNemar) and Pitfall 13 (RNG race) both independently block every downstream gap's acceptance criterion ("adopt only if Macro F1 equal or better," validated with McNemar/bootstrap CI, same seed/folds). Fixing or explicitly documenting these once, up front, avoids redoing ad-hoc comparison logic three times and avoids discovering non-reproducibility only after an "improvement" has already been reported to the committee.
**Delivers:** `metrics_bootstrap_ci()`/`metrics_mcnemar()` wired into `mode_train()`'s fold loop on the aggregated out-of-fold predictions; a documented reproducibility caveat (or a fix) for the OpenMP RNG race; a freshly re-confirmed baseline log (`results/train_log_v29_baseline_reconfirmed.txt`) since the SPEC-cited baseline numbers (0.4423/69.4%) have no backing artifact currently in-repo (the referenced log is empty).
**Avoids:** Pitfall 4, Pitfall 13, and the "committee cross-checks report against code and finds unenforced acceptance criteria" credibility failure from FEATURES.md's cross-cutting anti-features.

### Phase 1: Gap 2 — Borderline-SMOTE
**Rationale:** Lowest risk of the three gaps (confirmed by both architecture and pitfalls research — two call-site edits, no struct changes, no loop-nesting changes, signatures verified to match SPEC.md's assumptions exactly). Also validates the A/B comparison workflow itself before the two harder gaps need it.
**Delivers:** `SmoteMode` enum, `find_knn_global()`, safe/borderline/noise classification with the `2*m >= k` boundary (not truncated), per-class/fold safe/borderline/noise breakdown table, formal adopt/reject decision with McNemar/bootstrap CI backing.
**Addresses:** FEATURES.md's Gap 2 table-stakes + differentiator items (A/B table, fallback-rate disclosure, formal significance test).
**Avoids:** Pitfall 1 (boundary truncation), Pitfall 2 (wrong-neighbor-pool interpolation), Pitfall 3 (empty-borderline-set fallback compounding with the pre-existing `n_class <= 1` SMOTE degeneration — recommend fixing that pre-existing guard as a small sub-task here too).

### Phase 2: Gap 3 — Shallow vs. Deep MLP Comparison (with structural refactor)
**Rationale:** Architecture research identifies this as requiring the largest single-file structural change (extracting the fold+vowel loop into a reusable function, parameterized by architecture config) — doing this refactor now, before Gap 1 adds its own per-(fold,vowel,network) bookkeeping on top, keeps `main.c` maintainable. Also, per SPEC.md's own reasoning (confirmed sound), running this before Gap 1 avoids needing to redo the parameter-count table after Gap 1 changes input dimensionality.
**Delivers:** `mlp_init_multi()` generalization, `MLP_MAX_LAYERS` widening (with the corresponding `mlp_train.c` stack-array fixes and `mlp_backward`'s hardcoded delta-buffer-size fix, same commit), a corrected relabeling of SPEC.md's Config A/C (the current production config is actually SPEC's "Config C," not "Config A" — must fix before presenting to committee), the 4-config comparison table with McNemar/bootstrap CI vs. the best config, `make asan` target for pre-sweep verification.
**Uses:** 1-SE rule + McNemar decision protocol from STACK.md; existing `metrics.c` infra wired in Phase 0.
**Implements:** `run_fold_cv()` extraction (architecture component), `mlp_init_multi()`.

### Phase 3: Gap 1 — Paraconsistent Feature Selection
**Rationale:** Highest complexity and fan-out (up to 30 independent selection runs per pipeline execution), must incorporate whichever SMOTE mode and architecture config won in Phases 1-2 (per SPEC.md's own stated rationale, confirmed sound). Also requires the riskiest single edit of all three gaps: widening `predict_hierarchical_late_fusion()`'s signature and de-duplicating it against the validation loop's separate inline slicing copy.
**Delivers:** New `feature_select_paraconsistent.c`/`.h` module computing μ (ANOVA F-statistic or η², not SPEC's non-standard per-class ratio), λ (global-variance-normalized dispersion, not `CV=std/mean` which explodes for near-zero-mean delta-MFCC features), Gc/Gct, with variance-floor guards and a capped relaxation loop; per-(fold,vowel,network) selected-index persistence via existing `selected_save`/`selected_load`; a selection-frequency table (not a single-run snapshot) aggregating all 30 runs for the committee-facing deliverable.
**Addresses:** FEATURES.md's Gap 1 table-stakes (full μ/λ/Gc/Gct table, train-only statistic disclosure) and differentiators ((Gct,Gc) scatter plot, cross-fold stability analysis).
**Avoids:** Pitfall 5 (data-slice asymmetry from audio augmentation), Pitfall 6 (divide-by-zero), Pitfall 7 (unbounded relaxation loop masking Pitfall 6), Pitfall 8 (misleading single-run feature table).

### Phase Ordering Rationale

- **Dependency-driven:** Phase 0 must precede all gap work because Pitfalls 4 and 13 make every subsequent A/B comparison either unenforceable or silently non-reproducible — this is not optional infrastructure, it is a correctness precondition for the acceptance criteria SPEC.md defines for all three gaps.
- **Risk-ordering matches SPEC.md's own Gap 2→3→1 sequence**, independently confirmed sound by both architecture and pitfalls research for reasons beyond SPEC's original rationale: Gap 2 validates the A/B workflow cheaply; Gap 3's structural refactor (loop extraction, layer-array widening) must land before Gap 1 multiplies the loop-nesting complexity; Gap 1's input-dimensionality changes would otherwise invalidate Gap 3's parameter-count table if done first.
- **Avoids pitfalls by construction:** each gap's insertion point is scheduled specifically so its riskiest structural change (Gap 3's array widening, Gap 1's inference-signature widening) happens in isolation, verified (ASan for Gap 3, unit-tests for Gap 1's NaN guards) before being combined with the other gaps' changes.

### Research Flags

Needs deeper research/validation during planning:
- **Phase 3 (Gap 1):** The three domain-specific precedent papers (Costa et al. 2019 DPM; the 2025 wavelet+paraconsistent paper; the 2021 "Paraconsistent Feature Engineering" grid-fault paper) are paywalled and could not be read in full — their exact μ/λ derivation should be retrieved via institutional access (CAPES Periódicos) before finalizing the report's methodology section, since they are stronger, more domain-specific citations than the generic da Costa/Abe theoretical papers.
- **Phase 2 (Gap 3):** The "unfair hyperparameter reuse across depths" pitfall (Pitfall 11) has no settled mitigation — decide explicitly during planning whether a supplementary regularization-strength check for Config C/D is in scope, or whether the fixed-hyperparameter limitation will simply be stated in the report.

Phases with standard, well-documented patterns (research-phase likely unnecessary):
- **Phase 0:** Wiring already-implemented, already-tested functions (`metrics_bootstrap_ci`, `metrics_mcnemar`) into an existing call site — mechanical, no new algorithm design needed.
- **Phase 1 (Gap 2):** Borderline-SMOTE1's algorithm was read in full from the primary source and confirmed to match SPEC.md's plan almost exactly — implementation is a two-call-site parameter change with well-understood fixes for the known pitfalls.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH for Gap 2 (primary source read in full) and Gap 3 (canonical, well-established statistical citations); MEDIUM for Gap 1 (Gc/Gct/lattice machinery is canonical and cross-verified, but the μ/λ-from-continuous-features derivation has no single universally-cited formula in the literature) |
| Features | MEDIUM-HIGH (statistical-testing and SMOTE-variant reporting norms are well-established in mainstream ML literature — HIGH; paraconsistent-logic reporting conventions are thinner and rely more on domain-specific Brazilian academic literature — MEDIUM) |
| Architecture | HIGH (every claim verified directly against current `src/main.c`, `include/mlp.h`, `include/config.h`, `src/mlp.c`, `src/mlp_train.c` — not from SPEC.md's prose alone; a factual drift in SPEC.md itself was caught and corrected: current production is already 2-hidden-layer, not 1) |
| Pitfalls | HIGH for all findings tied to specific line numbers (verified by direct source reads); MEDIUM for general Borderline-SMOTE/OpenMP-RNG literature claims (WebSearch-sourced, cross-referenced against multiple independent secondary descriptions, no full-text primary access for some) |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Paywalled domain-specific precedent papers (Gap 1):** the three most directly-relevant applied papers on paraconsistent logic for voice-pathology/feature-selection could not be read in full. Handle during Phase 3 planning by retrieving via institutional access before finalizing the μ/λ formula citation language, or proceed with the ANOVA-F/η² substitute (already textbook-standard, independently defensible) and note the domain precedent as unverified.
- **SPEC.md's baseline reference numbers (0.4423 Macro F1 / 69.4% accuracy) have no backing artifact in the current repo** — the log that should contain them is empty. Handle in Phase 0 by re-running and freshly logging the baseline before any gap's A/B comparison claims a delta against it.
- **The RNG race's fix-vs-document tradeoff is a judgment call, not fully resolved by research** — pitfalls research recommends documenting as a caveat now and treating a full fix as separate tech debt unless a gap's acceptance criteria explicitly require bitwise reproducibility. This should be an explicit decision recorded in Phase 0 planning, not left ambiguous.
- **Gap 3's hyperparameter-fairness limitation (Pitfall 11) has no clean resolution** — decide during Phase 2 planning whether a supplementary regularization-strength check is in scope or whether the fixed-hyperparameter limitation is simply stated as a documented constraint in the final report.

## Sources

### Primary (HIGH confidence)
- Han, H., Wang, W.-Y., Mao, B.-H. (2005). "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning." *ICIC 2005*, LNCS 3644 — read in full via https://sci2s.ugr.es/keel/pdf/specific/congreso/han_borderline_smote.pdf
- Direct source reads: `src/main.c` (full, 369 lines), `include/mlp.h`, `src/mlp.c` (lines 215-241, 287, 550-592), `src/mlp_train.c` (lines 183-302), `include/config.h` (full), `include/feature_select.h`, `src/normalize.c`, `src/utils.c`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/CONCERNS.md`, `.planning/codebase/STRUCTURE.md`, `SPEC.md`, `CLAUDE.md`, `.planning/PROJECT.md`

### Secondary (MEDIUM confidence)
- Gc/Gct formulas and ±0.5 12-region lattice, cross-verified via a primary dissertation PDF (https://sites.unisanta.br/ppgmec/dissertacoes/dissertacao_fernando.pdf) plus multiple secondary Brazilian academic sources
- Dietterich, T.G. (1998), *Neural Computation* 10(7):1895–1923 — McNemar validity claim, confirmed via abstract + multiple independent secondary summaries
- Cawley, G.C. & Talbot, N.L.C. (2010), *JMLR* 11:2079–2107 — freely available, confirmed directly
- Hastie, Tibshirani, Friedman, *Elements of Statistical Learning* 2nd ed., §7.10 — confirmed via multiple independent secondary descriptions
- General OpenMP/parallel-RNG reproducibility principles — WebSearch, used only to confirm the general bug class, not project-specific claims

### Tertiary (LOW confidence, flagged for follow-up)
- Costa, S.C. et al. (2019), "Acoustic investigation of speech pathologies based on the discriminative paraconsistent machine (DPM)," *Biomedical Signal Processing and Control* — same SVD database, strongest domain precedent for μ/λ derivation, but paywalled (ScienceDirect 403), relevance confirmed via abstract only
- "Application of Wavelet Analysis and Paraconsistent Feature Extraction in the Classification of Voice Pathologies" (2025) — same access limitation
- "Wavelet-based features selected with Paraconsistent Feature Engineering..." (2021), *Measurement* — same access limitation, closest terminological match to "paraconsistent feature selection" specifically

---
*Research completed: 2026-07-27*
*Ready for roadmap: yes*
