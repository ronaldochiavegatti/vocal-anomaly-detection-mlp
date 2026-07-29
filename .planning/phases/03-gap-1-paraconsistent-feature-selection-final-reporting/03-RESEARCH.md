# Phase 3: Gap 1 — Paraconsistent Feature Selection & Final Reporting - Research

**Researched:** 2026-07-29
**Domain:** Paraconsistent Annotated Logic (PAL2v) feature selection in C99, applied to a hierarchical Master/Expert MLP ensemble; statistical A/B reporting infrastructure
**Confidence:** MEDIUM

## Summary

This phase adds a new, independent feature-selection module (`feature_select_paraconsistent.c/.h`) based on Paraconsistent Annotated Logic with two values (LPA2v/PAL2v — Da Costa 1990, formalized by Abe & Nakamatsu 2009), and closes the milestone with a consolidated Gap Adoption Status report. The **Gc = μ − λ** and **Gct = μ + λ − 1** formulas in `SPEC.md` are the textbook-standard PAL2v formulas and are confirmed correct by three independent sources (an MDPI/PMC paper, a 2025 arXiv PAL2v library paper, and its companion docs site) — **do not change them**. What `SPEC.md` gets wrong, and what PARA-01 explicitly calls out to fix, is how **μ** and **λ** are *derived from acoustic feature data*: SPEC's per-class Fisher ratio for μ is non-standard and its `CV = std/mean` for λ explodes for the project's δ-MFCC/δδ-MFCC features (documented near-zero mean in `CLAUDE.md`). This research recommends **μ = one-way ANOVA η² (eta-squared)** — naturally bounded in [0,1], no separate min-max normalization step needed — and **λ = unweighted mean of per-class σ_c / global σ** (a genuinely different aggregation than μ's sample-weighted SSB/SST, which avoids collapsing Gct into a deterministic function of Gc; verified by direct derivation that the "obvious" alternative — a weighted/pooled version of the same statistic — makes λ ≡ √(1−μ), destroying the two-independent-evidence-source premise of paraconsistent logic).

The bigger risk in this phase is not the math — it's integration. The codebase's fold+vowel loop in `mode_train_ex()` (`src/main.c`) already computes per-vowel per-network training/validation slices at a **fixed width of `nf_vowel = FEATURES_PER_VOWEL + NUM_METADATA_FEATURES = 85`** (not 251 — see the CLAUDE.md/config.h drift noted below). Introducing selection means every `mlp_init_multi()` call must use the *post-selection* width, and — this is PARA-04's real bug — the discrete-prediction path (`predict_hierarchical_late_fusion()`) and the probability-recording path (an inline duplicate slicing block in the validation loop, `src/main.c:557-564`) must be **collapsed into one function** so they can never apply different feature subsets. The safest fix is to make `predict_hierarchical_late_fusion()` the single owner of slicing+selection+forward-pass and have it also output probabilities via out-parameters, deleting the second block entirely.

A second, non-obvious finding that must inform the "before" baseline used in PARA-05's before/after report: **`mode_train()` (the plain `train`/`full` CLI path) still hard-codes `SMOTE_STANDARD`** (`src/main.c:690`), even though Gap 2 formally adopted Borderline-SMOTE and Gap 3's arch-compare fixed on it. The adopted-configuration reference run is `results/metrics_global_borderline_C_baseline.csv` (Macro F1 0.4587 point-estimate / 0.4565 bootstrap-mean), **not** `results/metrics_global.csv` (Macro F1 0.4514, which is still on SMOTE_STANDARD). Phase 3's new comparison mode should call `mode_train_ex(base_dir, SMOTE_BORDERLINE, &ARCH_CONFIGS[2], REG_BASELINE, result)` explicitly — mirroring `mode_smote_ab()`/`mode_arch_compare()` — not rely on or modify `mode_train()`'s default (flipping that default is out of this phase's SPEC.md-only scope per `REQUIREMENTS.md`'s explicit "no debt fixing this phase" constraint).

**Primary recommendation:** Implement `paraconsistent_select()` using η²-based μ and unweighted-ratio-based λ, computed only on each fold's **original (pre-audio-augmentation, pre-SMOTE) training rows** — mirroring the `norm_fit(..., fold->n_train, ...)` precedent already documented in `CLAUDE.md`'s Methodological Notes — persist per (fold, vowel, network) via the existing (currently dormant) `selected_save`/`selected_load` API unchanged, and refactor `predict_hierarchical_late_fusion()` into the single source of truth for both discrete predictions and recorded probabilities.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PARA-01 | New `feature_select_paraconsistent.c/.h` computing μ via ANOVA F/η² (not SPEC's per-class ratio) and λ via global-variance-normalized dispersion (not CV=std/mean) | See "LPA2v Algorithm" and "Code Examples" — η² formula, standard-ANOVA verification, λ design + degeneracy proof |
| PARA-02 | Gc/Gct with MIN_STD-style variance floor and a capped relaxation loop | See "Pitfall 2" and "Code Examples" — relaxation loop with hard iteration cap + all-features fallback, mirroring SMOTE-03's empty-pool precedent |
| PARA-03 | Selection independently per (fold, vowel, network) — Master and Expert get distinct indices | See "Architecture Patterns" — exact insertion point in `mode_train_ex()`'s fold/vowel loop, using `fold->n_train`-prefix rows |
| PARA-04 | `predict_hierarchical_late_fusion()` and the validation loop's duplicated inline slicing block consume the same persisted indices consistently | See "Pitfall 1" and "Architecture Patterns" — exact line numbers of the divergence risk (`src/main.c:39-69` vs `:557-564`) and the single-function refactor recommendation |
| PARA-05 | Aggregated selection-frequency table (feature, μ, λ, Gc, Gct, S/N) across ~30 runs, plus Macro F1 before/after | See "Architecture Patterns" (incremental-CSV pattern reused from `smote_borderline_counts.csv`) and "Open Questions" (correct "before" baseline identification) |
| PARA-06 | `CLAUDE.md` updated with Gap 1 outcome regardless of result | See "State of the Art" table and existing Gap 2/Gap 3 outcome sections in `CLAUDE.md` as the format precedent |
| CROSS-01 | Consolidated "Gap Adoption Status" table (decision / metric delta / citation status per gap) | See "Architecture Patterns" — reuse `write_smote_ab_report()`/`write_arch_compare_report()` DECISAO-sentence pattern for a 3-row summary table |
| CROSS-02 | No bibliographic citation for a technique not actually active in the final shipped model | See Summary's `mode_train()` default-SMOTE finding — this is a live instance of the exact risk CROSS-02 warns about and must be resolved/documented explicitly, not silently inherited |
</phase_requirements>

## Architectural Responsibility Map

This is a single-process, single-tier native C99 batch pipeline (no browser/server/API/CDN tiers — see `CLAUDE.md`'s Technology Stack). "Tiers" here are internal module boundaries within the one binary.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| μ/λ/Gc/Gct computation per feature | Feature Selection module (`feature_select_paraconsistent.c`, new) | — | Pure numeric function over a training matrix slice; no I/O, mirrors `dsp_utils.c`/`normalize.c` design |
| Selected-index persistence | Feature Selection module (reuses `feature_select.c`) | Model artifacts (`models/*.bin`) | Existing `selected_save`/`selected_load` API is already the right shape (`[int n][int idx_0]...`) — do not create a second format |
| Selection orchestration (per fold/vowel/network) | Training orchestration (`main.c: mode_train_ex()`) | — | Fold/vowel/network loop already owns SMOTE and MLP-init calls at the exact insertion point needed |
| Discrete prediction + probability computation | Training orchestration (`predict_hierarchical_late_fusion()`) | — | Must become the single owner of per-vowel slicing + index-selection + forward pass (PARA-04) |
| Selection-frequency table export | Reporting (`main.c`, new `write_*` function) | `results/*.csv` | Same incremental-CSV pattern as `smote_borderline_counts.csv` (SMOTE-04 precedent) |
| Gap Adoption Status consolidation | Reporting (`main.c`, new `write_*` function) | `CLAUDE.md` | Reuses the DECISAO-sentence + McNemar-gate pattern from `write_smote_ab_report()`/`write_arch_compare_report()` |

## Standard Stack

**Not applicable.** This project has zero external dependencies by design (`CLAUDE.md`: "sem dependências externas de ML"; only `libm`/`libgomp`/`libc`, no package manager, no lockfile). No npm/pip/cargo packages will be installed for this phase — the paraconsistent-logic algorithm is implemented from first principles in C99, matching every prior module in this codebase (kNN, LogReg, SMOTE, bootstrap CI, McNemar are all hand-rolled).

## Package Legitimacy Audit

**Not applicable — no external packages are installed in this phase.** The Package Legitimacy Gate protocol is skipped; there is nothing for `slopcheck`/registry verification to check. If, during planning, any auxiliary tooling (e.g. a Python plotting script for the poster) is proposed outside the C99 pipeline, it must go through this gate at that time — but it is out of this phase's stated success criteria.

## LPA2v Algorithm — Verified Formulas

Confirmed via three independent sources (see Sources) that the PAL2v formulas already in `SPEC.md` are the textbook-standard ones — **not** part of what needs correcting:

```
Dc  (Degree of Certainty)     = μ − λ,        Dc  ∈ [−1, 1]
Dct (Degree of Contradiction) = μ + λ − 1,     Dct ∈ [−1, 1]
```
`[CITED: PMC8234040 (MDPI Sensors 21(12):4219, 2021, "Paraconsistent Annotated Logic Algorithms Applied in Management and Control of Communication Network Routes")]` and independently confirmed by `[CITED: arXiv 2511.20700, "Paraconsistent-Lib: an intuitive PAL2v algorithm Python Library", Nov 2025]` and its docs site `eailab-ifsp.github.io/Paraconsistent-Lib/`.

The full PAL2v specification also defines a 12-region lattice classification (True/False/Inconsistent/Paracomplete plus 8 quasi-transition states) driven by four control values (Vcve/Vcfa/Vcic/Vcpa in the general literature; the Paraconsistent-Lib docs use a single `FtC` "Certainty Tolerance Factor," default 0.50) that maps a continuous evidence degree μER back to a ternary true/false/indeterminate decision. **This full 12-region classifier is not required by PARA-01/02** — the phase's stated success criteria only need threshold selection on `(Gc, Gct)` directly (`Gc >= gc_thresh AND |Gct| <= gct_max`), exactly as `SPEC.md` already specifies. Implementing the full lattice-region labeling is optional extra rigor, not a blocking requirement — flag this as a scope decision for the planner rather than assuming it's needed.

### μ: one-way ANOVA / η² (recommended, replaces SPEC's per-class Fisher ratio)

Standard one-way ANOVA decomposition per feature `j`, computed over `n` **original training samples** (see "Pitfall 3" on why augmented/SMOTE rows must be excluded), `C` classes, class counts `n_c`, global mean `x̄_j`, per-class mean `x̄_{c,j}`:

```
SSB_j = Σ_c  n_c * (x̄_{c,j} − x̄_j)²          (between-groups sum of squares)
SSW_j = Σ_c  Σ_{i∈c} (x_{i,j} − x̄_{c,j})²     (within-groups sum of squares)
SST_j = SSB_j + SSW_j
η²_j  = SSB_j / SST_j                          (naturally bounded [0,1])
```
`μ_j = η²_j` directly — **no min-max normalization step across features is needed**, unlike SPEC's Fisher-ratio approach (which required a separate per-fold min-max pass, making the same feature's score dependent on which other features happen to be in that fold's competing pool). `[CITED: GeeksforGeeks "Feature Selection using F-Anova"; easystats effectsize docs "F_to_eta2"]` — the relationship η² = (df_effect·F)/(df_effect·F + df_error) is standard and consistent across sources — `[VERIFIED: standard statistics formula, cross-checked across 2 independent tertiary sources, MEDIUM confidence]`.

Guard: if `SST_j < MIN_VAR` (a variance floor analogous to `normalize.c`'s `MIN_STD = 1e-8f`, but squared since SST is a sum-of-squares not a std), define `μ_j = 0` (a constant feature carries no separability evidence) rather than dividing by near-zero.

### λ: unweighted mean of per-class σ ratio to global σ (recommended, replaces SPEC's CV=std/mean)

```
σ_global_j = sqrt(SST_j / n),  floored at MIN_STD (1e-8f, reuse normalize.c's constant/pattern)
σ_c,j      = sqrt( Σ_{i∈c}(x_{i,j} − x̄_{c,j})² / n_c )     (per-class population std)
λ_j        = (1/C) * Σ_c  min(σ_c,j / σ_global_j, 1.0)      (unweighted mean, clipped to [0,1])
```

**Why this exact form, and not the "obvious" alternative:** PARA-01 requires λ to be "normalized by global variance" instead of the buggy per-class-mean CV. The most obvious way to do that is to reuse the ANOVA decomposition already computed for μ: a sample-count-weighted pooled within-class std is `sqrt(SSW_j/n) = σ_global_j · sqrt(1 − η²_j)`, giving `λ_j = sqrt(1 − η²_j)`. **This is a trap** — it makes λ a deterministic function of μ alone (`Gct = μ + √(1−μ) − 1`, a fixed 1-D curve), so two features with identical separability but very different actual within-class noise patterns become indistinguishable in `(Gc, Gct)` space, defeating the entire point of tracking two independent evidence sources in paraconsistent logic. The **unweighted** mean-of-ratios form above avoids this because it aggregates across classes without sample-count weighting — for this project's severely imbalanced dataset (687/140/91/112/68), weighted and unweighted aggregation diverge substantially, giving λ real (if partial) independence from μ. `[ASSUMED — this is a synthesized design, not sourced from a specific paper; the domain-specific precedent papers (Costa et al. 2019 DPM, a 2025 wavelet+paraconsistent paper, a 2021 grid-fault paper) that would normally ground this exact formula are paywalled — see `STATE.md`'s existing Phase 3 blocker note. Confirm the λ formula choice with the user/planner before treating it as locked.]`

### Relaxation loop (PARA-02)

`SPEC.md` already specifies: if zero features pass `Gc_j >= gc_thresh AND |Gct_j| <= gct_max`, relax `gc_thresh` by −0.05 and retry, logging a warning each time. PARA-02 requires this capped. Recommended cap: **10 iterations** (drops `gc_thresh` from a typical 0.35 starting point down to −0.15, covering effectively the entire useful range of Gc well before the cap is hit), with a **hard fallback to selecting all `nf_vowel` features** (never zero) if the cap is reached with still nothing selected — mirroring the SMOTE-03 precedent already established in this codebase (`smote_oversample()`'s empty-borderline-pool fallback, `src/main.c:313-315`, which falls back to the full pool + `log_warn`, never crashes or degenerates).

## Architecture Patterns

### Insertion point in `mode_train_ex()`'s fold/vowel loop

Current relevant code (`src/main.c`, inside `for (int v = 0; v < 3; v++)`, lines ~486-551):

```
tr_x_v[n_train_aug x nf_vowel]  ← sliced from train_x_all (includes fold->n_train
                                    original rows FOLLOWED BY audio-augmented rows;
                                    see collect_augmented_features(), which appends
                                    starting at out_idx = *n_ptr = fold->n_train)
vl_x_v[fold->n_val x nf_vowel]  ← sliced from val_x_all
tr_y_bin / vl_y_bin              ← binary labels (Master)
  ↓
smote_oversample(tr_x_v, tr_y_bin, n_train_aug, nf_vowel, 2, ...) → os_m_x/os_m_y (Master)
mlp_init_multi(&net_master[v], nf_vowel, 2, ...)
mlp_train(&net_master[v], os_m_x, os_m_y, ...)
  ↓
ex_tr_x/ex_tr_y  ← filtered from tr_x_v[all n_train_aug rows] where label != NORMAL (Expert)
smote_oversample(ex_tr_x, ex_tr_y, n_ex_tr, nf_vowel, 4, ...) → os_e_x/os_e_y (Expert)
mlp_init_multi(&net_expert[v], nf_vowel, 4, ...)
mlp_train(&net_expert[v], os_e_x, os_e_y, ...)
```

Recommended insertion (paraconsistent selection runs **before** `smote_oversample()` and **before** `mlp_init_multi()`, for both networks independently):

```
/* Master: use ONLY the original fold->n_train prefix of tr_x_v/tr_y_bin — same
 * precedent as norm_fit(train_x_all, fold->n_train, ...) already documented in
 * CLAUDE.md's Methodological Notes. Do NOT pass n_train_aug (includes augmented
 * rows) or os_m_x (includes SMOTE-synthesized rows) — both would bias μ/λ. */
int sel_m[FEATURES_PER_VOWEL + NUM_METADATA_FEATURES];
int ns_m = paraconsistent_select(tr_x_v, tr_y_bin, fold->n_train, nf_vowel, 2,
                                  PARA_GC_THRESH, PARA_GCT_MAX, sel_m,
                                  mu_out, lambda_out, gc_out, gct_out); /* optional out-buffers, size nf_vowel, for PARA-05's report */
selected_save(path_for(f, v, "master"), sel_m, ns_m);

/* apply sel_m to tr_x_v AND vl_x_v (both n_train_aug/n_val rows, full width) BEFORE
 * smote_oversample() and mlp_init_multi() -- selection reduces column count, SMOTE
 * interpolates in the already-reduced space */
slice_columns(tr_x_v, n_train_aug, nf_vowel, sel_m, ns_m, tr_x_v_sel);
slice_columns(vl_x_v, fold->n_val, nf_vowel, sel_m, ns_m, vl_x_v_sel);

smote_oversample(tr_x_v_sel, tr_y_bin, n_train_aug, ns_m, 2, ...);   /* nf_vowel -> ns_m */
mlp_init_multi(&net_master[v], ns_m, 2, ...);                        /* nf_vowel -> ns_m */
```

Expert follows the identical pattern but computed on the original-rows-only, non-NORMAL-filtered subset: filter `train_y_all[0:fold->n_train]` (not `[0:n_train_aug]`) for `!= CLASS_NORMAL`, extract those rows from `tr_x_v[0:fold->n_train]`, run `paraconsistent_select(..., n_classes=4, ...)` to get `sel_e`/`ns_e`, save via `selected_save(path_for(f, v, "expert"), sel_e, ns_e)`, then apply to the full `ex_tr_x`/`ex_vl_x` (which do include augmented rows) before `smote_oversample`/`mlp_init_multi`.

### PARA-04's core bug: two independent slicing implementations

`predict_hierarchical_late_fusion()` (`src/main.c:39-69`) slices `x_all` into per-vowel `x_v` and calls `mlp_forward` to get the discrete class. Separately, the validation loop (`src/main.c:553-569`) has its **own** inline slicing block (`float xv[251]; memcpy(...)`) that recomputes per-vowel vectors independently just to get `p_norm`/`p_exp` probabilities for `all_y_prob` (feeds ROC/PR export). Today these two blocks are harmless duplicates because both slice the same unselected `nf_vowel`-wide window. Once feature selection is introduced, **both blocks must apply the identical selected-index subset**, or the recorded discrete prediction and the recorded probability could come from different feature views of the same sample — exactly what PARA-04 forbids.

**Recommended fix:** extend `predict_hierarchical_late_fusion()`'s signature to also output the per-class probabilities (out-parameters `float *p_norm_out, float p_exp_out[4]`), thread the per-vowel `sel_m[3][]`/`ns_m[3]`/`sel_e[3][]`/`ns_e[3]` arrays into it (loaded once per fold via `selected_load()`, sized by `nf_vowel` as the upper bound — **not** the stale `251` literal used in the current `xv[251]` buffer, which is itself DEBT-01's known magic-number bug; do not propagate that literal into new code), and call it exactly once per validation sample. Delete the second inline block entirely. This is a genuine, in-scope refactor (not "debt fixing" — it is required to satisfy PARA-04's stated success criterion, which explicitly names both code paths).

### Aggregated selection-frequency report (PARA-05)

Reuse the incremental-CSV pattern already established for `smote_borderline_counts.csv` (open once per run with a header, append a block per fold/vowel/network iteration — see `src/main.c:420-434` and the `counts_f` fprintf calls at lines 506-511/534-539). Recommended columns: `fold,vowel,network,feature_idx,mu,lambda,gc,gct,selected`. With `nf_vowel=85` features × 5 folds × 3 vowels × 2 networks = 1275 rows — small, no incremental-durability concerns beyond what the existing pattern already provides.

### Gap Adoption Status table (CROSS-01)

Reuse the DECISAO-sentence pattern from `write_smote_ab_report()`/`write_arch_compare_report()`. A 3-row table works directly from data already on disk/in `CLAUDE.md`:

| Gap | Decision | Macro F1 delta | McNemar p | Citation status |
|-----|----------|----------------|-----------|------------------|
| Gap 2 (Borderline-SMOTE) | ADOTADO | +0.0235 (point) / bootstrap-mean 0.4565 vs 0.4338 | 0.6606 (not significant) | Han/Wang/Mao 2005 — cite only if wired into the actually-shipped default path (see Open Questions) |
| Gap 3 (Config C, 2 hidden layers) | ADOTADO (no config.h change) | — (validated as already-production) | 0.7463 vs Config D (not significant) | No new citation needed — architecture comparison is internal methodology, not a cited external technique |
| Gap 1 (Paraconsistent LPA2v) | TBD (this phase) | TBD | TBD | Da Costa 1990 / Abe & Nakamatsu 2009 (PAL2v) — cite only if the adopt rule in PARA-05/SPEC.md's acceptance criterion actually fires |

### Recommended file/module structure

```
include/feature_select_paraconsistent.h   # new — mirrors feature_select.h's terse doc-comment style
src/feature_select_paraconsistent.c       # new — paraconsistent_select(), one-way ANOVA helper, lambda helper
src/main.c                                 # modified — insertion in mode_train_ex()'s fold/vowel loop,
                                            #   predict_hierarchical_late_fusion() extended,
                                            #   new write_gap1_report() / write_gap_adoption_status(),
                                            #   new mode_paraconsistent_ab() CLI entry (mirrors mode_smote_ab/mode_arch_compare)
```
No changes needed to `include/feature_select.h`/`src/feature_select.c` (persistence API is already correctly shaped) or `Makefile` (wildcard-based `SRCS` picks up the new `.c` file automatically).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Selected-index persistence | A new binary format for paraconsistent indices | Existing `selected_save`/`selected_load` (`include/feature_select.h`) | Already exactly the right shape (`[int n][int idx...]`), already includes bounds validation against `TOTAL_FEATURES` on load — reuse as-is, per SPEC.md's own instruction |
| Bootstrap CI / McNemar significance testing | A new stats routine for the Gap 1 A/B comparison | `metrics_bootstrap_ci()` / `metrics_mcnemar()` (already wired via Phase 0/INFRA-01) | Identical pattern already proven correct across Gap 2 and Gap 3; reinventing risks silent inconsistency with the other two gaps' reported numbers |
| A/B orchestration + report-writing scaffolding | A bespoke comparison harness for Gap 1 | Copy the `mode_smote_ab()`/`write_smote_ab_report()` structure (single fixed adopt/reject rule computed in code, never asserted by inspection) | This is the established, reviewed pattern for exactly this kind of "run pipeline twice, compare, decide" task in this codebase |

**Key insight:** every piece of reusable infrastructure this phase needs (persistence, stats, reporting scaffolding) already exists and is proven by two prior gaps — the only genuinely new code is the μ/λ/Gc/Gct computation itself and its insertion point.

## Common Pitfalls

### Pitfall 1: Selection-index divergence between discrete prediction and recorded probability (PARA-04)
**What goes wrong:** `predict_hierarchical_late_fusion()` and the validation loop's separate inline slicing block silently apply different (or no) selected-feature subsets, so `all_y_pred[i]` (used for Macro F1/confusion matrix) and `all_y_prob[i]` (used for ROC/AUC/PR curves) are computed from different views of the same sample.
**Why it happens:** the two code paths were written independently before selection existed, and nothing forces them to share state.
**How to avoid:** collapse them into one function (see Architecture Patterns above) that owns slicing+selection+forward pass and returns both outputs from a single call site.
**Warning signs:** ROC-derived AUC and confusion-matrix-derived accuracy disagree on which samples are "confidently correct" in a way that doesn't reconcile with `argmax(all_y_prob[i]) == all_y_pred[i]` for every `i`.

### Pitfall 2: Unbounded or silently-empty relaxation loop (PARA-02)
**What goes wrong:** SPEC's relaxation loop ("relax gc_thresh by 0.05 and repeat") has no stated termination bound; a pathological fold/vowel/network combination (e.g., a near-constant feature block) could loop indefinitely or, if implemented naively, could still terminate with `n_selected == 0` and crash downstream `mlp_init_multi(net, 0, ...)`.
**Why it happens:** the SPEC pseudocode describes the happy path, not the degenerate case.
**How to avoid:** hard iteration cap (recommend 10) + guaranteed non-empty fallback (select all `nf_vowel` features) with `log_warn`, exactly mirroring the SMOTE-03 empty-borderline-pool precedent already in this codebase.
**Warning signs:** any fold/vowel/network log line showing the relaxation loop hitting its cap — should be rare; frequent occurrences suggest `gc_thresh`/`gct_max` defaults are miscalibrated for this feature set.

### Pitfall 3: Computing μ/λ on augmented or SMOTE-synthesized rows (data leakage / bias)
**What goes wrong:** if `paraconsistent_select()` is called on `tr_x_v`'s full `n_train_aug` rows (which include 8x audio-domain augmentation per minority-class original, per `N_AUG_PER_SAMPLE`) or on SMOTE-oversampled rows, the computed separability/dispersion statistics are biased — augmented rows are near-duplicates of their originals (inflating apparent within-class consistency), and SMOTE rows are literally interpolated from existing rows (mechanically reducing apparent within-class variance, inflating η²/μ artificially).
**Why it happens:** `tr_x_v` is already the augmented array by the time it reaches the SMOTE call in the existing loop; it's easy to reach for the same variable without noticing which rows are original.
**How to avoid:** slice on `fold->n_train` (the original-row prefix), exactly matching the already-documented `norm_fit(train_x_all, fold->n_train, ...)` precedent (`CLAUDE.md`'s Methodological Notes: "norm_fit: Must be called with fold->n_train ... NOT n_train_aug").
**Warning signs:** implausibly high μ (η² near 1.0) for features expected to be weak (e.g., wavelet-level statistics that showed little discriminative power in prior feature-importance analyses).

### Pitfall 4: Stale feature-count constants inherited from CLAUDE.md
**What goes wrong:** `CLAUDE.md`'s Feature Count table states 237 total features / 79 per vowel / 51 spectral features — these are **stale**. The actual current `config.h` (confirmed by direct read) defines `NUM_SPECTRAL_FEATURES=55` (includes 4 glottal-source features — Oq/Sq/NAQ/H1-H2 — not present in the documented table), `FEATURES_PER_VOWEL=83`, `NUM_METADATA_FEATURES=2`, `TOTAL_FEATURES=251`, so `nf_vowel = FEATURES_PER_VOWEL + NUM_METADATA_FEATURES = 85` (matches `SPEC.md`'s own "85/vogal" reference exactly — SPEC.md is *not* stale here, only CLAUDE.md's older table is).
**Why it happens:** the glottal-source features (Oq/Sq/NAQ/H1-H2) were added at some point (likely alongside the v29 Hierarchical Late Fusion commit `e63483a`) without updating the CLAUDE.md feature-count table.
**How to avoid:** always use `FEATURES_PER_VOWEL`/`NUM_METADATA_FEATURES`/`TOTAL_FEATURES` from `config.h` directly (never hard-code 79/237/251/85 as literals in new code — this is also why the existing `xv[251]` buffer at `src/main.c:325` is flagged as DEBT-01), and correct the CLAUDE.md table as part of this phase's PARA-06 documentation update (small, low-risk, directly relevant since Gap 1 depends on exactly these counts).
**Warning signs:** none at runtime (the cache-validation check in `features_load_csv()` already guards against a mismatched `TOTAL_FEATURES`) — this is a documentation-accuracy risk, not a functional one, but it will mislead planning if not called out.

### Pitfall 5: `mode_train()`'s default still on rejected SMOTE mode (CROSS-02 relevance)
**What goes wrong:** treating `results/metrics_global.csv` (produced by the plain `train`/`full` CLI path, which is hard-coded to `SMOTE_STANDARD`) as "the current production baseline" for Gap 1's before/after comparison — when the actually-adopted configuration (Borderline-SMOTE + Config C baseline) lives only in the arch-compare artifacts (`results/metrics_global_borderline_C_baseline.csv`).
**Why it happens:** `mode_train_ex()`'s `result == NULL` path (used by plain `train`/`full`) was deliberately kept byte-identical to the pre-Gap-2 baseline for regression-safety reasons documented in `STATE.md`'s Phase 1 decisions — this was the correct call for Phase 1, but it means the "adopted" SMOTE mode was never wired into the actual default entry point, only into the comparison-mode entry points (`smote-ab`, `arch-compare`).
**How to avoid:** Gap 1's new comparison mode should explicitly pass `SMOTE_BORDERLINE, &ARCH_CONFIGS[2], REG_BASELINE` (same as `mode_arch_compare()`'s C/baseline arm) rather than calling `mode_train()`/relying on its default. Flag this default-vs-adopted gap explicitly in the final Gap Adoption Status report (relevant to CROSS-02: if the poster/report ever cites "production uses Borderline-SMOTE" while `make train` still runs standard SMOTE, that's exactly the kind of citation-vs-reality mismatch CROSS-02 exists to prevent).
**Warning signs:** any report or citation claiming a specific SMOTE mode is "in production" should be checked against what `mode_train()`'s actual (unmodified) default does today.

## Code Examples

### One-way ANOVA (η²) computation per feature

```c
/* Source: standard one-way ANOVA decomposition (see GeeksforGeeks "Feature
 * Selection using F-Anova"; easystats effectsize F_to_eta2 docs) -- no direct
 * project precedent exists for this exact computation, this is new code. */
#define MIN_VAR 1e-16f  /* variance floor, analogous to normalize.c's MIN_STD=1e-8f squared */

static float feature_eta_squared(const float *x, const int *y, int n, int nf,
                                  int feat_idx, int n_classes)
{
    float global_mean = 0.0f;
    for (int i = 0; i < n; i++) global_mean += x[i * nf + feat_idx];
    global_mean /= (float)n;

    float *class_sum = (float *)safe_calloc(n_classes, sizeof(float));
    int   *class_n   = (int *)safe_calloc(n_classes, sizeof(int));
    for (int i = 0; i < n; i++) { class_sum[y[i]] += x[i * nf + feat_idx]; class_n[y[i]]++; }

    float ssb = 0.0f;
    for (int c = 0; c < n_classes; c++) {
        if (class_n[c] == 0) continue;
        float class_mean = class_sum[c] / (float)class_n[c];
        float diff = class_mean - global_mean;
        ssb += (float)class_n[c] * diff * diff;
    }

    float ssw = 0.0f;
    for (int i = 0; i < n; i++) {
        int c = y[i];
        float class_mean = class_sum[c] / (float)class_n[c];
        float diff = x[i * nf + feat_idx] - class_mean;
        ssw += diff * diff;
    }
    free(class_sum); free(class_n);

    float sst = ssb + ssw;
    if (sst < MIN_VAR) return 0.0f;  /* constant feature: no separability evidence */
    return ssb / sst;                /* eta-squared, already in [0,1] */
}
```

### λ via unweighted per-class σ / global σ ratio

```c
/* Design note: deliberately UNWEIGHTED across classes (not sample-count-weighted
 * like SSB/SSW), so lambda does not collapse to sqrt(1-eta_squared) -- see
 * RESEARCH.md "LPA2v Algorithm" section for the degeneracy proof. */
static float feature_lambda(const float *x, const int *y, int n, int nf,
                             int feat_idx, int n_classes, float global_std)
{
    if (global_std < MIN_STD) global_std = MIN_STD;

    float *class_sum = (float *)safe_calloc(n_classes, sizeof(float));
    float *class_sq  = (float *)safe_calloc(n_classes, sizeof(float));
    int   *class_n   = (int *)safe_calloc(n_classes, sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = y[i]; float v = x[i * nf + feat_idx];
        class_sum[c] += v; class_sq[c] += v * v; class_n[c]++;
    }

    float lambda_sum = 0.0f; int active_classes = 0;
    for (int c = 0; c < n_classes; c++) {
        if (class_n[c] < 2) continue;  /* need >=2 samples for a std estimate */
        float mean_c = class_sum[c] / (float)class_n[c];
        float var_c  = class_sq[c] / (float)class_n[c] - mean_c * mean_c;
        if (var_c < 0.0f) var_c = 0.0f;  /* numeric guard */
        float std_c = sqrtf(var_c);
        float ratio = std_c / global_std;
        if (ratio > 1.0f) ratio = 1.0f;  /* clip to [0,1] */
        lambda_sum += ratio; active_classes++;
    }
    free(class_sum); free(class_sq); free(class_n);
    return (active_classes > 0) ? (lambda_sum / (float)active_classes) : 0.0f;
}
```

### Gc/Gct with capped relaxation loop and guaranteed-non-empty fallback

```c
/* Source: SPEC.md's relaxation-loop pseudocode, capped per PARA-02. Mirrors the
 * SMOTE-03 empty-borderline-pool fallback pattern (src/main.c:313-315). */
#define PARA_MAX_RELAX_ITERS 10
#define PARA_GC_RELAX_STEP   0.05f

int paraconsistent_select(const float *x, const int *y, int n, int nf, int n_classes,
                           float gc_thresh, float gct_max, int *selected,
                           float *mu_out, float *lambda_out,
                           float *gc_out, float *gct_out)
{
    float *mu = (float *)safe_malloc(nf * sizeof(float));
    float *lambda = (float *)safe_malloc(nf * sizeof(float));
    float *gc = (float *)safe_malloc(nf * sizeof(float));
    float *gct = (float *)safe_malloc(nf * sizeof(float));

    for (int j = 0; j < nf; j++) {
        float global_mean = 0.0f, global_sq = 0.0f;
        for (int i = 0; i < n; i++) { float v = x[i * nf + j]; global_mean += v; global_sq += v * v; }
        global_mean /= (float)n;
        float global_var = global_sq / (float)n - global_mean * global_mean;
        if (global_var < 0.0f) global_var = 0.0f;
        float global_std = sqrtf(global_var);

        mu[j] = feature_eta_squared(x, y, n, nf, j, n_classes);
        lambda[j] = feature_lambda(x, y, n, nf, j, n_classes, global_std);
        gc[j] = mu[j] - lambda[j];
        gct[j] = mu[j] + lambda[j] - 1.0f;
    }

    int n_selected = 0;
    float cur_gc_thresh = gc_thresh;
    for (int iter = 0; iter <= PARA_MAX_RELAX_ITERS; iter++) {
        n_selected = 0;
        for (int j = 0; j < nf; j++) {
            if (gc[j] >= cur_gc_thresh && fabsf(gct[j]) <= gct_max) selected[n_selected++] = j;
        }
        if (n_selected > 0) break;
        if (iter < PARA_MAX_RELAX_ITERS) {
            log_warn("paraconsistent_select: 0 features selecionadas com gc_thresh=%.3f -- relaxando para %.3f (iter %d/%d)",
                      cur_gc_thresh, cur_gc_thresh - PARA_GC_RELAX_STEP, iter + 1, PARA_MAX_RELAX_ITERS);
            cur_gc_thresh -= PARA_GC_RELAX_STEP;
        }
    }
    if (n_selected == 0) {
        log_warn("paraconsistent_select: relaxamento esgotado (%d iteracoes) sem selecionar nenhuma feature -- "
                  "fallback para todas as %d features (mesmo padrao de fallback do SMOTE-03)", PARA_MAX_RELAX_ITERS, nf);
        for (int j = 0; j < nf; j++) selected[j] = j;
        n_selected = nf;
    }

    if (mu_out) memcpy(mu_out, mu, nf * sizeof(float));
    if (lambda_out) memcpy(lambda_out, lambda, nf * sizeof(float));
    if (gc_out) memcpy(gc_out, gc, nf * sizeof(float));
    if (gct_out) memcpy(gct_out, gct, nf * sizeof(float));
    free(mu); free(lambda); free(gc); free(gct);
    return n_selected;
}
```

## State of the Art

| SPEC.md's original (non-standard) | This phase's replacement | Why changed |
|---|---|---|
| μ = mean over classes of one-vs-rest Fisher ratio `F_c(j) = (mean_c - mean_global)² / var_c`, then min-max normalized across features per fold | μ = one-way ANOVA η² (SSB/SST) | η² is naturally bounded [0,1] without a separate cross-feature normalization step whose result depends on the competing feature pool; it is the textbook-standard normalized effect-size statistic for this exact use case |
| λ = 1 − 1/(1 + mean-CV), CV = std/mean per class | λ = unweighted mean of (per-class σ / global σ), clipped [0,1] | CV explodes when a class's mean is near zero — documented in `CLAUDE.md` as a known property of δ-MFCC/δδ-MFCC features (std-of-delta computation, near-zero mean for stable vowels); the replacement normalizes by global std, which is bounded away from zero by a `MIN_STD` floor regardless of any class's mean |
| Gc = μ − λ, Gct = μ + λ − 1, unbounded relaxation loop | Same Gc/Gct formulas (confirmed standard), capped 10-iteration relaxation with guaranteed all-features fallback | Formulas were already correct; only the termination behavior needed a bound (PARA-02) |

**Deprecated/not applicable:** the full 12-region PAL2v lattice classification (True/False/Inconsistent/Paracomplete + 8 quasi-states) exists in the broader PAL2v literature but is not required by this phase's stated success criteria — direct `(Gc, Gct)` threshold selection, as `SPEC.md` already specifies, is sufficient and matches the phase's scope.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | λ formula (unweighted mean of per-class σ/global-σ ratio) is a reasonable, non-degenerate substitute for SPEC's buggy CV — this is a researcher-synthesized design, not sourced from the paywalled domain-specific precedent papers (Costa et al. 2019 DPM; a 2025 wavelet+paraconsistent paper; a 2021 grid-fault paper) | LPA2v Algorithm — λ section | If the actual precedent papers define a materially different λ formula, the planner may need to revise PARA-01's implementation after institutional access is obtained; low risk to phase completion (any well-motivated normalized-dispersion measure satisfies PARA-01's literal requirement) but could affect academic-defense answers about "why this exact formula" |
| A2 | μ = η² (rather than the raw, unbounded F-statistic) is the correct interpretation of "μ via ANOVA F-statistic/η²" in PARA-01's wording, since PARA-01 explicitly offers both as acceptable | LPA2v Algorithm — μ section | Low risk — F and η² are monotonically related for fixed degrees of freedom; if the planner/user prefers reporting raw F in the PARA-05 table alongside η²-derived μ, both are cheap byproducts of the same computation |
| A3 | Paraconsistent selection must be computed on `fold->n_train`-prefix (original, pre-augmentation, pre-SMOTE) rows, mirroring `norm_fit`'s precedent, rather than on the full augmented/SMOTE'd training set | Architecture Patterns; Pitfall 3 | Medium risk if wrong — computing on augmented/SMOTE data would silently bias μ/λ and could produce misleadingly "cleaner" separability metrics that don't reflect the real acoustic data; this is a methodological-rigor question the PIBIC committee is likely to probe, given the project's existing emphasis on this exact norm_fit precedent |
| A4 | The "before" baseline for PARA-05's Macro F1 before/after comparison should be `results/metrics_global_borderline_C_baseline.csv` (Gap 3's adopted-arm reference), not `results/metrics_global.csv` (still on `SMOTE_STANDARD` via `mode_train()`'s unmodified default) | Summary; Pitfall 5 | High relevance to CROSS-02 if wrong — citing/reporting the wrong "before" number would misrepresent what Gap 1 actually improves upon, and could reintroduce exactly the citation-vs-reality mismatch CROSS-02 is designed to prevent |
| A5 | Relaxation-loop cap of 10 iterations and step size of 0.05 (unchanged from SPEC.md) are reasonable defaults; no literature-sourced value exists for this specific cap | LPA2v Algorithm — Relaxation loop; Code Examples | Low risk — any finite cap satisfies PARA-02's literal requirement; the exact number is a tuning/engineering choice, not a correctness question |
| A6 | `CLAUDE.md`'s Feature Count table (237 total / 79 per vowel / 51 spectral) is stale relative to the actual current `config.h` (251 total / 83 per vowel / 55 spectral, including 4 glottal-source features not in the documented table) | Pitfall 4 | Medium risk if not corrected — new code that hard-codes the stale numbers (or a planner that plans around them) would introduce off-by-N buffer bugs; low risk to this phase's own correctness since this research reads `config.h` directly rather than trusting the stale table |

**If this table is empty:** Not applicable — see entries above.

## Open Questions

1. **Should the paywalled precedent papers (Costa et al. 2019 DPM; 2025 wavelet+paraconsistent; 2021 grid-fault) be pursued via institutional access before implementation, or is the ANOVA-η²/global-variance-ratio substitute sufficient to proceed?**
   - What we know: `STATE.md`'s existing Phase 3 blocker note already anticipates this — "plan to proceed with the ANOVA-F/η² substitute unless institutional access is obtained."
   - What's unclear: whether the exact λ formula in those papers differs materially enough from this research's synthesized A1 design to matter for the committee defense.
   - Recommendation: proceed with the substitute (already the documented fallback plan); the planner should not block implementation on paper access, but PARA-06's CLAUDE.md update should explicitly note the substitution and its rationale (already drafted in this research's "State of the Art" table) so the committee defense has a ready answer.

2. **Should `mode_train()`'s default flip from `SMOTE_STANDARD` to `SMOTE_BORDERLINE` as part of this phase, or strictly stay out of scope?**
   - What we know: `REQUIREMENTS.md`'s Out of Scope table explicitly defers "Corrigir bugs de débito técnico do mapeamento de codebase nesta fase" (fix tech-debt bugs) to v2, and `STATE.md` documents the current default as an intentional Phase 1 decision (byte-identical regression safety), not an oversight discovered mid-Phase-3.
   - What's unclear: whether leaving `mode_train()` un-flipped, while Gap 1's new comparison mode explicitly passes `SMOTE_BORDERLINE`, creates a confusing "two different meanings of production" state that undermines CROSS-02's spirit even if it satisfies CROSS-02's letter (no *citation* is added for an inactive technique, but the plain CLI's actual behavior still doesn't match what's documented as adopted).
   - Recommendation: keep `mode_train()` unchanged (respects the explicit scope boundary), but require the final Gap Adoption Status report / CLAUDE.md update to state this explicitly and unambiguously (e.g., "the `train`/`full` CLI default remains on SMOTE_STANDARD/Config-C-baseline for regression-safety reasons; the actually-adopted configuration is only reachable via `arch-compare`'s C/baseline arm or a future default-flip, tracked as tech debt") — this turns a potential future confusion into a documented, defensible statement.

3. **Should PARA-05's acceptance rule (adopt paraconsistent selection if Macro F1 equal-or-better, OR up to −0.01 worse with ≥30% feature reduction) be implemented as a second fixed decision rule in code (like Gap 2/3's DECISAO pattern), or left as a narrative judgment call in the final report?**
   - What we know: `SPEC.md`'s Gap 1 acceptance criterion already specifies this exact trade-off rule in words; Gap 2 and Gap 3 both implemented their adopt/reject rules as fixed code logic (`write_smote_ab_report()`/`write_arch_compare_report()`), never asserted by inspection.
   - What's unclear: whether the −0.01/30% trade-off threshold needs to be a compile-time constant (like `PARA_GC_RELAX_STEP`) for consistency with the project's established "no cherry-picked decisions" methodology.
   - Recommendation: implement it as fixed code logic, following the established precedent exactly (this is a strong signal from two prior gaps, not really ambiguous — flagging here mainly so the planner allocates a task for it rather than treating it as narrative-only).

## Environment Availability

Skipped — this phase adds pure C99 source files to an already-building project (gcc 13.3.0 + GNU Make + OpenMP, all already verified working across Phases 0-2, including a repo hook that runs `make` after every `.c`/`.h` edit). No new external tools, services, or runtimes are introduced.

## Validation Architecture

Skipped — `.planning/config.json` has `workflow.nyquist_validation: false`.

## Security Domain

`security_enforcement` is absent from `.planning/config.json` (treated as enabled per the default rule), but this project has no network-facing surface, no authentication, no user-supplied untrusted input beyond local WAV/CSV files already validated by existing loaders (`wav_io.c`, `csv_parser.c`) — it is an offline, single-user, locally-run research CLI (`CLAUDE.md`'s Platform Requirements: "No deployment target"). Most ASVS categories do not apply.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | No user-facing auth surface exists |
| V3 Session Management | No | No sessions — single-process batch CLI |
| V4 Access Control | No | No multi-user/multi-tenant boundary |
| V5 Input Validation | Partial | Already handled by existing loaders: `features_load_csv()` validates column count against `TOTAL_FEATURES`; `selected_load()` validates `n <= TOTAL_FEATURES`; NaN/Inf from degenerate audio already swept to 0.0 post-extraction. New code in `feature_select_paraconsistent.c` should apply the same `MIN_VAR`/`MIN_STD`-style guards already used elsewhere (this phase's PARA-02 requirement is itself an input-validation control against degenerate/zero-variance feature columns) |
| V6 Cryptography | No | No cryptographic operations anywhere in this codebase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Malformed/truncated `.bin` persistence files (e.g. a corrupted `selected_master_fold{k}_v{vowel}.bin`) causing an out-of-bounds read on load | Tampering / DoS (local, not adversarial in this context — accidental corruption is the realistic threat, not an attacker) | Already mitigated: `selected_load()` bounds-checks `*n` against `TOTAL_FEATURES` before the subsequent `fread` into a caller-provided buffer sized to at least `TOTAL_FEATURES` — reuse unchanged, do not weaken this check when wiring in the new per-(fold,vowel,network) file naming |
| Division by near-zero variance in μ/λ computation causing NaN/Inf propagation into the MLP (silent corruption of a "committee-ready" report) | Tampering (data integrity, not security in the adversarial sense) | `MIN_VAR`/`MIN_STD` floors as specified in PARA-02 and implemented in the Code Examples above |

## Sources

### Primary (HIGH confidence)
- None — Context7 was not queried (this phase has no library/framework dependency to look up; it is pure algorithmic C99). No official docs site exists for this bespoke academic algorithm beyond the sources below.

### Secondary (MEDIUM confidence)
- [Paraconsistent Annotated Logic Algorithms Applied in Management and Control of Communication Network Routes (PMC8234040 / MDPI Sensors 21(12):4219, 2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8234040/) — confirmed Dc=μ−λ, Dct=μ+λ−1 formulas, [-1,1] range, 4-vertex lattice (True/False/Paracomplete/Inconsistent)
- [Paraconsistent-Lib: an intuitive PAL2v algorithm Python Library (arXiv 2511.20700, Nov 2025)](https://arxiv.org/abs/2511.20700) and its [docs site](https://eailab-ifsp.github.io/Paraconsistent-Lib/) — independently confirmed the same Dc/Dct formulas, 12-region lattice with quasi-states, FtC/default-0.50 decision rule
- [Acoustic investigation of speech pathologies based on the discriminative paraconsistent machine (DPM), Fonseca/Guido et al., Biomedical Signal Processing and Control, 2019-2020](https://www.sciencedirect.com/science/article/abs/pii/S174680941930196X) — the domain-specific precedent for paraconsistent methods applied to Saarbrücken-Voice-Database-style vocal pathology data; abstract confirms 95% accuracy on an SVD subset; full text paywalled, matches `STATE.md`'s existing blocker note
- [Feature Selection using F-Anova (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/feature-selection-using-f-anova/) — standard one-way ANOVA F formula
- [F_to_eta2 — effectsize R package docs](https://easystats.github.io/effectsize/reference/F_to_eta2.html) — η² ↔ F relationship, [0,1]-bounded by construction

### Tertiary (LOW confidence)
- General WebSearch summaries on PAL2v applications (robotics, medical diagnosis, edge computing) — background context only, not load-bearing for any implementation decision in this research

## Metadata

**Confidence breakdown:**
- LPA2v Gc/Gct formulas: HIGH — confirmed identical across 3 independent sources (MDPI/PMC paper, 2025 arXiv paper, its docs site), and match `SPEC.md` exactly
- μ/λ derivation formulas (the actual PARA-01 fix): MEDIUM — η² formula is standard/verified; the specific λ design is a researcher synthesis (flagged `[ASSUMED]`, see A1) because the domain-specific precedent papers are paywalled
- Integration points in `main.c` (insertion point, PARA-04's slicing-divergence bug, PARA-05's reporting pattern): HIGH — derived from direct reading of the current `src/main.c`/`include/*.h`, not from external sources
- `mode_train()` default-SMOTE finding (Pitfall 5 / A4): HIGH — directly verified by reading `src/main.c:690` and comparing `results/metrics_global.csv` vs `results/metrics_global_borderline_C_baseline.csv` contents

**Research date:** 2026-07-29
**Valid until:** No expiry driver — this is a closed, offline C99 codebase with no upstream dependency churn; treat as valid until Phase 3 planning/execution completes (the μ/λ formula choice (A1) should be re-confirmed with the user before being treated as a locked decision, independent of any time-based staleness)
