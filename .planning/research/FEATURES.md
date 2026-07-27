# Feature Research

**Domain:** Academic-rigor reporting requirements for a PIBIC (Brazilian undergraduate
scientific-initiation) evaluation committee — not product features. "Features" here are
the deliverable artifacts (tables, plots, statistical tests, disclosures) that a report/
poster/abstract must contain for each of the 3 gaps in `SPEC.md` to be considered a
credible, defensible implementation of the original proposal.
**Researched:** 2026-07-27
**Confidence:** MEDIUM-HIGH (statistical-testing and SMOTE-variant reporting norms are
well-established in the ML literature — HIGH; paraconsistent-annotated-logic reporting
conventions are thinner in mainstream ML sources and rely more on the domain-specific
[Da Costa/Abe] literature already cited in the proposal — MEDIUM)

## Feature Landscape

Organized per SPEC.md gap, since the "features" requested are gap-specific reporting
deliverables, not generic product capabilities. Each subsection cross-references what
`SPEC.md`'s "Critério de aceite" already mandates and flags what it is silent on.

---

### GAP 2 — Borderline-SMOTE A/B Comparison

**SPEC.md already requires** (§ Gap 2, Critério de aceite): run 5-fold with
`SMOTE_STANDARD` vs `SMOTE_BORDERLINE`, same seed, adopt only if Macro F1 is equal or
better, "atenção especial ao F1 de Disfonia Psicogênica/Funcional", and forbids citing
Han/Wang/Mao 2005 unless the method is actually adopted in the final model.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Global Macro F1 delta, mean over 5 folds, identical `RANDOM_SEED=42`/fold split for both arms | A committee member with any ML background will ask "same folds?" first — an A/B comparison on different splits is not a comparison. SPEC already mandates this. | LOW | Verify programmatically that `kfold_split()` is called once and its output reused for both SMOTE modes in the same run, not re-derived per mode. |
| Per-class F1 delta for **all 5 classes**, not just macro average | Macro F1 can move for reasons unrelated to the borderline mechanism (e.g. Normal class shifting). SPEC explicitly calls out DisfPsicog/DisfFunc as the classes theoretically expected to benefit — the committee will check whether the theory-predicted effect actually shows up there specifically, not just in the aggregate number. | LOW | Already produced by existing `metrics.c` per-class output — just needs to be tabulated for both runs side by side. |
| Explicit final adopt/reject decision statement, tied to SPEC's stated rule | "We tried it" is not a result; the committee needs to know what shipped and why, using SPEC's own pre-registered criterion (equal-or-better Macro F1), not a post-hoc rationalization. | LOW | One sentence: "Adopted / not adopted because [Macro F1 X vs Y]." |
| Safe / borderline / noise sample counts per class per fold | This is the actual mechanism claim of Borderline-SMOTE. If most minority samples end up classified "safe" (falling back to standard-SMOTE-like behavior) or "noise" (discarded), the method may be doing almost nothing different from standard SMOTE — that is a first-order finding, not a footnote. SPEC's spec text mentions logging this fallback as a warning but does not explicitly require reporting it as a result. | LOW-MEDIUM | **Gap vs SPEC**: SPEC only requires logging a warning when `class_idx_borderline[c]` is empty; it does not require reporting the safe/borderline/noise breakdown as a table. Recommend adding it — it is cheap (already computed internally) and is exactly the kind of mechanistic detail a committee asks about when a bolted-on citation is suspected. |
| Both accuracy AND Macro F1 reported together, never accuracy alone | CLAUDE.md/PROJECT.md already establish Macro F1 as the primary metric for this severely imbalanced problem; reverting to accuracy-only for one specific gap comparison would look inconsistent with the rest of the report. | LOW | — |

**Differentiators (strengthen the defense beyond SPEC's minimum):**

| Feature | Value Proposition | Complexity | Notes |
|---------|--------------------|------------|-------|
| McNemar test (or bootstrap CI overlap, both already implemented in the codebase) between standard-SMOTE and borderline-SMOTE final predictions | A raw Macro F1 delta of e.g. +0.01–0.02 on classes with only 91–140 samples is well within noise. Pairing the delta with a formal test (or CI overlap) preempts the obvious committee question "is that difference real or noise?" — the same rigor SPEC already mandates for Gap 3 but does not explicitly require here. | LOW (infra already exists: `metrics.c` McNemar + bootstrap CI) | Recommend adding to close the asymmetry: Gap 3 gets a formal test in SPEC, Gap 2 doesn't — a committee that reads both sections back to back may notice the inconsistency. |
| Fallback-rate summary: % of (class × fold) combinations where the borderline set was empty and standard behavior was used instead | Turns SPEC's "logar aviso" into an actual disclosed metric, defusing the possibility that borderline-SMOTE silently behaved like standard SMOTE most of the time while still being cited as the active method. | LOW | Directly protects against the anti-feature below (false attribution). |
| Simple 2D visualization (PCA of top-2 components) contrasting where standard-SMOTE vs borderline-SMOTE synthetic points land relative to class boundaries | One qualitative figure that visually explains *why* the quantitative result happened — strong poster material, low cost given feature matrix is already cached. | MEDIUM | Optional; only worth the time if Gap 2 is adopted (no point illustrating a discarded method beyond a brief note). |

**Anti-features (would hurt credibility):**

| Feature/Practice | Why it seems fine | Why it's problematic | Alternative |
|---|---|---|---|
| Citing Han/Wang/Mao 2005 (Borderline-SMOTE) in the abstract/poster while the final adopted model actually kept standard SMOTE | "We implemented and tested it, worth mentioning" | This is the exact mistake SPEC.md already flags as previously committed ("erro já corrigido uma vez") — citing a technique not active in the final model is a credibility-destroying discovery for a committee to make. | Cite it only in a "methods explored" subsection explicitly labeled as not adopted, with the reason; do not cite it as if it were the production method. |
| Reporting only the metric that improved (e.g. "accuracy went up") while omitting one that got worse (e.g. Macro F1 dropped) | Makes the change look like an unambiguous win | Selective reporting is the single fastest way to lose credibility with a committee that has the full metrics.csv available to cross-check | Report both metrics regardless of direction; if they disagree, say so and explain which one governs the SPEC decision rule (Macro F1). |
| Basing the "SMOTE improved things" conclusion on a single best fold instead of the mean across the 5 folds | Best fold looks more impressive | SPEC's acceptance criterion is about the aggregate/mean 5-fold result, not a cherry-picked fold; MEMORY.md's own history shows this project has previously reported "best individual fold" numbers in a way that could be mistaken for the primary result | Report fold-mean ± CI as the decision number; a best-fold anecdote may appear only as a clearly-labeled aside, never as the headline number. |
| Re-running kfold_split with a different call/seed "by accident" between the two SMOTE-mode runs | Easy mistake when running two separate CLI invocations | Silently invalidates the entire A/B comparison — the two arms are no longer on the same data partitions | Log the fold composition (e.g. a hash of patient IDs per fold) at run time and diff it between runs before trusting the comparison. |

---

### GAP 1 — Paraconsistent Feature Selection (μ, λ, Gc, Gct)

**SPEC.md already requires** (§ Gap 1, Critério de aceite): run with/without selection,
same seed; report retained-feature counts per network/vowel, Macro F1, per-class F1;
adopt only if Macro F1 is equal/better OR (≤ −0.01 Macro F1 with ≥30% feature reduction,
documented explicitly as a parsimony trade-off); add a full feature/μ/λ/Gc/Gct/selected
table to the final report to preempt committee questions.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Full (feature, μ, λ, Gc, Gct, selected S/N) table exactly as SPEC mandates | This is the single most explicitly-required artifact in the entire SPEC — Gap 1 is called out as the item "a banca avaliadora tende a perguntar especificamente sobre" because it has dedicated citations [12][13]. Not producing this table is an automatic flag. | LOW (values already computed internally by `paraconsistent_select()`) | Must be generated per network (Master/Expert) × per vowel, since SPEC explicitly allows/expects different selected indices per network. |
| Explicit statement that μ/λ are computed **only on training-fold data**, never validation/test | This is a leakage question every ML-literate committee member asks reflexively, and SPEC itself states the rule ("calculado só com dados de treino, nunca com validação/teste — mesma regra de norm_fit"). | LOW | One sentence in methods + a note that it mirrors the existing `norm_fit` convention already documented in CLAUDE.md — reuses established credibility. |
| Retained-feature counts per network/vowel/fold, reported as counts (not just percentages) | SPEC requires this by name. Raw counts reveal instability that a percentage alone can hide (e.g. "40%" could mean 30-of-75 in one fold and 12-of-30 in another). | LOW | — |
| Macro F1 before/after with the exact SPEC decision rule applied and stated (either "equal/better" or "≤0.01 drop + ≥30% reduction — trade-off accepted") | SPEC pre-registers two different acceptance conditions; if the trade-off clause is the one actually invoked, the report must say so explicitly rather than implying a clean win. | LOW | Conflating "improved" with "acceptable trade-off" is listed as an anti-feature below. |
| gc_thresh / gct_max values actually used, plus how often the "relax by 0.05 if empty" fallback (SPEC step 5) was triggered | SPEC specifies the fallback mechanism but does not explicitly require *reporting* how often it fires. If it fires frequently, the initially chosen thresholds were poorly calibrated — a fact the committee should be told, not discover by re-running the code. | LOW-MEDIUM | **Gap vs SPEC**: add this as an explicit reported statistic; it is a one-line counter increment in the existing algorithm. |

**Differentiators:**

| Feature | Value Proposition | Complexity | Notes |
|---------|--------------------|------------|-------|
| (Gct, Gc) scatter plot with the 12 paraconsistent lattice regions drawn, each feature plotted as a point colored by selected/rejected | This is the natural visual proof that the [12][13] framework (Avron/Arieli/Zamansky; Abe) was actually implemented as described, not just approximated numerically. Given Gap 1 is explicitly the most-cited-in-the-proposal item, this single figure is high leverage for the defense. | MEDIUM | Directly reuses the μ/λ/Gc/Gct values already computed; only needs a plotting step. |
| Cross-fold stability analysis: which features are selected in all 5 folds ("core" features) vs only some | Strengthens the interpretability claim SPEC itself makes ("alimenta uma resposta pronta para a banca sobre quais features o método considerou mais relevantes") — a single static table from one fold could otherwise be misleading if selection is unstable across folds, since selection runs independently per fold. | LOW-MEDIUM | **Gap vs SPEC**: SPEC's acceptance criterion implies one table; it does not explicitly ask whether that table is stable across folds. A committee member could ask "did you check if this is the same every time?" — worth pre-empting. |
| Sensitivity analysis over a small grid of (gc_thresh, gct_max) pairs (e.g. 0.30/0.30, 0.35/0.30, 0.40/0.25) showing retained-feature-count and Macro F1 trade-off curve | Demonstrates the thresholds were not arbitrary "magic numbers" — SPEC's own text flags them as illustrative ("ex.: 0.3-0.5") rather than derived from the cited papers, so showing the sensitivity strengthens the defense of an otherwise ad hoc choice. | MEDIUM | Reuses the same 5-fold infrastructure, run 3-4 times with different thresholds. |
| Mapping the selected "certainty-region" features back to acoustic/phonetic meaning (e.g. jitter, CPP, δMFCC std) with a short interpretive paragraph | A domain-literate committee (voice pathology / phonetics adjacent) will value a plausibility check — "does the paraconsistent method select features that make phonetic sense," not just "does the number go up." | LOW | Ties back to CLAUDE.md's existing feature taxonomy — no new computation needed. |

**Anti-features:**

| Feature/Practice | Why it seems fine | Why it's problematic | Alternative |
|---|---|---|---|
| Presenting the (μ,λ,Gc,Gct) table for only a curated subset of "clean-looking" features, omitting inconsistent/indeterminate ones | Table looks tidier | The entire epistemic point of paraconsistent annotated logic is representing uncertainty/contradiction explicitly — hiding the messy/indeterminate features defeats the method's own stated purpose and looks like cherry-picking to anyone who knows the framework | Show the full table (or a representative sample across all four lattice categories: true, false, inconsistent, indeterminate), explicitly noting how many features fell in each region |
| Presenting gc_thresh=0.35 / gct_max=0.3 as if derived from Avron/Arieli/Zamansky or Abe | Sounds more rigorous, avoids admitting a tuned parameter | These specific numeric thresholds are not standard, universally-defined values from that literature — SPEC's own text marks them as illustrative examples. Claiming a citation basis for a number that was actually empirically tuned is a citation-accuracy failure, structurally identical to the SMOTE citation mistake this project has already made once | State explicitly: "thresholds tuned empirically on training folds, inspired by but not directly specified by [12][13]" |
| Reporting "paraconsistent selection improved Macro F1" when the trade-off clause (≤0.01 drop + ≥30% reduction) was actually what applied | Sounds like a stronger result | Misrepresents which of SPEC's two pre-registered decision rules was actually invoked; a careful reader (or the student under cross-examination) will be caught in an inconsistency | State the applicable rule explicitly, e.g. "Macro F1 −0.008, feature count −34% → accepted under the parsimony trade-off clause" |
| Computing μ/λ statistics using validation or test fold data (even inadvertently, e.g. reusing a global CSV load instead of the training split) | Easy to get wrong if refactoring the existing `select_features_variance`-style code as a style reference (SPEC explicitly warns "referência de estilo, não de lógica") | Data leakage inflates apparent separability and invalidates the entire feature-selection result — exactly the kind of bug an ML-literate committee member probes for first | Unit-test `paraconsistent_select()` against a synthetic dataset where the training/validation split is known, verifying no validation indices are touched |

---

### GAP 3 — Shallow vs Deep Network Comparison

**SPEC.md already requires** (§ Gap 3, Critério de aceite): a table with the 4 configs
(A:[128], B:[64], C:[128,64], D:[128,64,32]) × 5 metrics (accuracy, Macro F1, per-class
F1, time/epoch, param count); choice must use McNemar or bootstrap CI, explicitly "nunca
por inspeção visual"; document the winning config with a complexity-vs-performance
justification.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Full 4-config × metrics table (accuracy, Macro F1, per-class F1, params, time/epoch), aggregated with a clearly stated method across Master/Expert × 3 vowels | SPEC requires this exact table. With 2 networks × 3 vowels × 4 configs = 24 raw cells, the report needs an explicit, stated aggregation rule (e.g. report the fused-ensemble Macro F1 per config, with per-network/per-vowel numbers in an appendix) or the table becomes unreadable and undermines the "concise justification" the proposal asks for. | LOW-MEDIUM | **Gap vs SPEC**: SPEC doesn't specify how to aggregate across networks/vowels — must be decided and stated explicitly before presenting results, or the table will look ad hoc. |
| Parameter counts computed for the **actual** final input dimensionality used in that run, not the SPEC table's illustrative ~85-input case | SPEC's own table (11k/5.5k/19k/21k params) is explicitly computed "85→·→2" as an example. Per PROJECT.md's mandated gap order (Gap 2 → Gap 3 → Gap 1), Gap 3 runs before Gap 1's feature selection, so its actual input size may differ from the SPEC illustration. Presenting the SPEC's example numbers as if they were the measured numbers for this run would be inaccurate. | LOW | Recompute and report actual params per config given the input size active at the time Gap 3 is executed; state that size explicitly. |
| Formal test (McNemar with Edwards continuity correction, already implemented) or bootstrap CI overlap between the best config and each simpler config | SPEC explicitly bans a visual-only decision ("nunca por inspeção visual"). The acceptance criterion literally requires "escolher a configuração de menor complexidade cujo Macro F1 não seja estatisticamente pior que a melhor" — this cannot be satisfied without a reported p-value or CI comparison. | LOW (infra exists) | Must compare **every** simpler config against whichever config is empirically best, not just A vs D. |
| Explicit, single-sentence final decision tied to the numbers: "chosen config X: Macro F1 not significantly different from best config Y (p=…), Z% fewer parameters, W% less training time" | This sentence is literally what the original PIBIC proposal asked for ("justificativa da escolha por complexidade") — a table without this closing sentence fails to answer the actual proposal requirement. | LOW | — |

**Differentiators:**

| Feature | Value Proposition | Complexity | Notes |
|---------|--------------------|------------|-------|
| Bootstrap CI width reported alongside the McNemar p-value, not instead of it | A non-significant McNemar p-value can mean "truly equivalent" or "test underpowered by small minority-class n" — pairing it with a visibly overlapping bootstrap CI is a more defensible, harder-to-refute version of the same claim, and shows statistical maturity beyond SPEC's minimum ask (SPEC allows "McNemar **or** bootstrap CI"; using both is stronger). | LOW (both already implemented) | — |
| Learning-curve comparison (train/val loss & F1 per epoch, from the already-existing `results/learning_curves.csv` pipeline) across the 4 configs | Directly tests the a-priori hypothesis (already stated in CLAUDE.md's "Fundamental bottleneck" section) that deeper configs on a ~1098-patient, severely imbalanced dataset will overfit — showing the train/val gap widening for config D turns "shallow won" from an empirical accident into a principled, explainable result. | MEDIUM | High value for the defense: converts a result into a story with a mechanism, not just a table. |
| "Macro F1 per 1k parameters" or similar efficiency-normalized metric | Single synthesized number that communicates the complexity/performance trade-off in one column, easy to put on a poster. | LOW | Purely derived from already-collected numbers. |
| Multiple-comparison correction (e.g. Holm-Bonferroni) noted when running 3 pairwise McNemar tests against the best config | With 4 configs there are multiple pairwise comparisons; a committee member familiar with statistics may ask whether multiple testing was considered. Correcting makes it *harder* to reject H0, which actually reinforces (rather than undermines) a "no significant difference, pick simpler" conclusion — reporting it shows the student anticipated the objection. | LOW | Optional given n=5 folds is already a small-sample setting where correction has limited power either way — but stating "no correction applied, or Holm applied, because…" preempts the question. |

**Anti-features:**

| Feature/Practice | Why it seems fine | Why it's problematic | Alternative |
|---|---|---|---|
| Choosing the "winning" config by eyeballing which bar is tallest in a chart, without ever running McNemar/bootstrap CI | Faster, and the numbers often look obviously different | SPEC explicitly forbids this ("nunca por inspeção visual") — it is a direct, named violation of the project's own stated acceptance rule, not just a generic best practice | Always accompany any config-selection chart with the corresponding statistical test result in the same figure/caption |
| Running the 4-config comparison on a different fold split or seed than the documented reference baseline (Macro F1 0.4423 / 69.4% accuracy) | Easy to happen if re-invoking the pipeline with slightly different CLI args over multiple sessions | Breaks comparability with PROJECT.md's own reference numbers and with Gap 2's SMOTE comparison, undermining the whole "reproducible A/B, same seed/folds" methodology the project is built on | Confirm `RANDOM_SEED=42` and the same `kfold_split()` invocation are used across every gap's experiments; log a fold-composition fingerprint to check |
| Deciding the config with the single best individual fold rather than the 5-fold mean/CI | Best fold makes the chosen config look stronger | SPEC's decision rule is about the aggregate result across folds; MEMORY.md shows this project has a documented history of reporting "best individual fold" numbers prominently (Fold 3 Macro F1=0.7045 in a past PRD) — repeating that framing here as the *decision* metric (rather than a labeled aside) would misrepresent the actual comparison | Use the 5-fold mean ± bootstrap CI as the decision number; a best-fold highlight may appear separately, explicitly labeled as non-decisive |
| Skipping training of configs C and/or D "because the dataset is small and deeper will obviously overfit," and asserting the shallow-wins conclusion without the data | Saves significant compute time (4 configs × 5 folds × 3 vowels × 2 networks is expensive) | SPEC explicitly requires testing all 4 configs even when the expected outcome (shallow wins) is likely correct — asserting an untested conclusion, however plausible, is an unsupported claim a committee can directly probe ("did you actually run config D?") | If compute time is genuinely prohibitive, reduce scope transparently (e.g., run the 4-config sweep only on the Master network, not Master+Expert×3 vowels) and state the reduced scope explicitly rather than skipping silently |

---

## Cross-Cutting Anti-Features (apply to all 3 gaps — Question 4)

These are the general credibility failure modes most likely to occur precisely because
the project has already committed a version of one of them once (per SPEC.md's own
admission: "erro já corrigido uma vez nesta sessão" re: SMOTE citation).

| Practice | Surface Appeal | Why It Hurts Credibility | Alternative |
|---|---|---|---|
| Citing a technique (Borderline-SMOTE, or any paraconsistent-selection variant) in the abstract/poster/final report that is not the one actually active in the shipped model | Sounds more sophisticated / matches the original PIBIC proposal's language | This is the exact class of error the project has already made once; a committee cross-checking the report against the code (or asking the student to explain the method live) will expose it immediately, and it damages trust in every other claim in the document | Maintain a single "Gap adoption status" table (recommended below) as the one source of truth for what is cited vs what is only "explored" |
| Visual-only comparisons (bar charts, line plots) presented as sufficient evidence for a methodological decision | Faster to produce, intuitive for a general audience (poster session) | SPEC's own general acceptance rule bans this explicitly ("nunca por inspeção visual") for good reason — visual differences at this sample size (91-140 patients in minority classes) are frequently within noise | Every decision-bearing figure must be paired with the numeric test result (p-value or CI) that justified the decision, even in the poster |
| Reporting accuracy as the headline metric anywhere in the 3-gap comparisons | Accuracy is easier to explain to a general/lay committee member | CLAUDE.md and PROJECT.md already establish Macro F1 (+ per-class F1) as the metric of record for this severely imbalanced 5-class problem (687/140/91/112/68); reverting to accuracy-only framing in any one gap's section would look methodologically inconsistent with the rest of the work | Always lead with Macro F1 and per-class F1; accuracy may appear as a secondary/familiar reference number, never as the sole decision criterion |
| Omitting a gap's results because the change didn't help (negative result) | Makes the final report look cleaner / more successful | SPEC's own "Critérios de Aceite Gerais" explicitly requires documenting a non-working gap ("inclusive se um gap não funcionar, isso deve ficar documentado") — a committee that reads the original proposal will ask about all 3 items regardless of outcome, and silently dropping a negative result is a worse look than reporting a well-documented null finding | Every gap gets a full comparison section regardless of outcome; a documented "no significant improvement, kept baseline, here's why" is legitimate PIBIC output |
| Reporting a statistical test (McNemar/bootstrap CI) without stating N (bootstrap resamples), which specific predictions were compared (paired, same test set), or the correction method used | Statistical machinery is present, "looks rigorous" at a glance | Statistics without methodological detail look pasted-in rather than understood — a committee member with any stats background will ask for exactly these details, and not having them ready undermines the appearance of rigor the numbers were meant to create | State explicitly: bootstrap N=1000 seed=42 (already the project convention per CLAUDE.md), which classifier pair, Edwards continuity correction for McNemar (already the project convention) |

**Differentiator (applies across all 3 gaps):** a single consolidated "Gap Adoption
Status" table in the final report/poster — one row per gap, columns: Decision
(adopted/kept baseline/trade-off), primary metric delta, citation status (added/kept/
removed), pointer to the detailed section. SPEC.md does not ask for this consolidated
view explicitly, but it directly serves SPEC's own general rule about citation accuracy
and is the first thing a committee member scanning the report will look for. Low cost,
high leverage — recommended as a v1 (table-stakes-adjacent) addition, not merely a
nice-to-have.

## Feature Dependencies

```
[Baseline reference locked: Macro F1 0.4423 / Acc 69.4%, same seed/folds]
    └──required-before──> [Gap 2: Borderline-SMOTE A/B]
                               └──required-before──> [Gap 3: Shallow vs Deep, 4 configs]
                                                          └──required-before──> [Gap 1: Paraconsistent selection]

[McNemar / bootstrap CI infra (already implemented, metrics.c)]
    └──enables──> [Gap 3 formal config comparison] (table-stakes, SPEC-mandated)
    └──enables──> [Gap 2 formal SMOTE-variant comparison] (differentiator, SPEC-silent)

[learning_curves.csv infra (already implemented)]
    └──enables──> [Gap 3 overfitting-story differentiator]

[Gap 1 feature selection] ──conflicts-with-naive-reuse-of──> [Gap 3's SPEC-table param counts]
    (Gap 3's illustrative ~85-input param table must be recomputed if run after/with
     Gap 1's selection changes input dimensionality — see Gap 3 table-stakes notes)

["Gap Adoption Status" consolidated table] ──synthesizes──> [Gap 1, Gap 2, Gap 3 sections]
```

### Dependency Notes

- **Ordering (Gap 2 → Gap 3 → Gap 1) is already fixed by SPEC.md/PROJECT.md** and is
  sound for a different reason than pure risk management: Gap 3's parameter-count table
  is only accurate for whatever feature-selection state is active when it runs. Running
  Gap 3 before Gap 1 (as already planned) avoids having to redo the param-count table
  after Gap 1 changes the input dimensionality — but the final report must still state
  explicitly which feature-selection state was active during the Gap 3 experiments, in
  case a committee member asks whether Gap 3's conclusion still holds post-Gap-1.
- **McNemar/bootstrap CI infra is a shared dependency** for both Gap 3 (mandatory per
  SPEC) and Gap 2 (recommended differentiator, not mandated by SPEC as written) — since
  the code already exists (`metrics.c`), extending its use to Gap 2 is low cost and
  closes an asymmetry in rigor between the two gaps.
- **The consolidated adoption-status table conflicts with nothing** — it is purely a
  synthesis layer over the three gap-specific sections and should be written last, after
  all three gaps' individual decisions are finalized.

## MVP Definition

### Launch With (v1) — must be present or the committee will flag it

- [ ] Gap 2: full per-class + Macro F1 A/B table (same seed/folds), explicit adopt/
      reject decision matching SPEC's stated rule, citation only if adopted
- [ ] Gap 1: full (feature, μ, λ, Gc, Gct, selected S/N) table per network/vowel,
      retained-feature counts, Macro F1 before/after with the specific SPEC decision
      rule invoked stated explicitly, explicit train-only statistic disclosure
- [ ] Gap 3: 4-config × (accuracy, Macro F1, per-class F1, params, time/epoch) table
      with a stated aggregation method across networks/vowels, McNemar or bootstrap CI
      between best config and each simpler config, explicit final-config decision
      sentence
- [ ] Cross-cutting: no citation for a method not active in the final shipped model
      (all 3 gaps); CLAUDE.md/MEMORY.md "What Worked/Didn't Work" updated per gap
      regardless of outcome (already required by SPEC's general acceptance criteria)

### Add After Validation (v1.x) — strengthens the defense, not blocking

- [ ] Gap 2: safe/borderline/noise breakdown per class/fold; formal test (McNemar/
      bootstrap CI) between SMOTE variants
- [ ] Gap 1: (Gct, Gc) 12-region scatter plot; cross-fold feature-stability analysis;
      threshold sensitivity grid
- [ ] Gap 3: learning-curve overfitting comparison across configs; efficiency-
      normalized metric (Macro F1 per 1k params); multiple-comparison correction note
- [ ] Cross-cutting: consolidated "Gap Adoption Status" summary table for the final
      report/poster

### Future Consideration (v2+) — beyond PIBIC defense scope

- [ ] Full factorial ablation across SMOTE mode × network depth × feature-selection
      setting (the sequential Gap 2 → Gap 3 → Gap 1 order avoids needing this now, but
      it is a natural "future work" line for a possible follow-up publication)
- [ ] External validation of the paraconsistent thresholds on a second voice-pathology
      dataset, to test whether gc_thresh/gct_max generalize beyond this SVD subset

## Feature Prioritization Matrix

| Feature | Committee Value | Implementation Cost | Priority |
|---------|------------------|----------------------|----------|
| Gap 2: per-class A/B table + adopt/reject decision | HIGH | LOW | P1 |
| Gap 1: full (μ,λ,Gc,Gct) table per network/vowel | HIGH | LOW | P1 |
| Gap 3: 4-config table + formal test + decision sentence | HIGH | MEDIUM | P1 |
| Gap 1: train-only statistic disclosure | HIGH | LOW | P1 |
| Gap 2: safe/borderline/noise breakdown | MEDIUM | LOW | P2 |
| Gap 2: formal test between SMOTE variants | MEDIUM | LOW | P2 |
| Gap 1: (Gct,Gc) scatter plot | HIGH (visual centerpiece for most-cited gap) | MEDIUM | P2 |
| Gap 1: cross-fold feature stability | MEDIUM | LOW-MEDIUM | P2 |
| Gap 3: learning-curve overfitting story | MEDIUM-HIGH | MEDIUM | P2 |
| Consolidated Gap Adoption Status table | HIGH (cheap, high scanability) | LOW | P2 |
| Gap 1: threshold sensitivity grid | MEDIUM | MEDIUM | P3 |
| Gap 3: efficiency-normalized metric | LOW-MEDIUM | LOW | P3 |
| Full factorial ablation (all 3 gaps combined) | LOW for PIBIC defense (HIGH for a paper) | HIGH | P3 |

**Priority key:**
- P1: Must have — omission is a defensible-flag risk for the committee
- P2: Should have — meaningfully strengthens the defense, moderate cost
- P3: Nice to have — future-work material, not needed for the current defense

## Sources

- `SPEC.md` (this repository) — primary source of the 3 gaps' existing acceptance
  criteria; all "table stakes" items above are cross-referenced against it directly.
- `.planning/PROJECT.md` (this repository) — confirms gap ordering, reference baseline
  (Macro F1 0.4423 / Acc 69.4%), and out-of-scope boundaries.
- `CLAUDE.md` (this repository) — confirms existing statistical infra (McNemar with
  Edwards continuity correction, bootstrap CI N=1000 seed=42, per-class metrics) that
  the differentiator recommendations above reuse rather than propose building new.
- Han, H.; Wang, W.-Y.; Mao, B.-H. (2005). *Borderline-SMOTE: A New Over-Sampling
  Method in Imbalanced Data Sets Learning* — confirms the safe/borderline/noise
  classification mechanism and that the original paper's own evaluation reports
  per-class TP rate / F-value specifically for the minority class, supporting the
  "per-class delta, not just aggregate" table-stakes requirement for Gap 2.
  https://www.researchgate.net/publication/225129029_Borderline-SMOTE_A_New_Over-Sampling_Method_in_Imbalanced_Data_Sets_Learning
- Paraconsistent Annotated Logic (LPA2v) sources confirming Gc = μ − λ, Gct = μ + λ − 1
  formulas and the 12-region lattice as SPEC.md describes them (MEDIUM confidence —
  WebSearch-sourced Brazilian academic literature, not a single canonical reference):
  https://www.researchgate.net/publication/358107843_APLICACOES_ESTADO_DA_ARTE_DA_LOGICA_PARACONSISTENTE_ANOTADA_EVIDENCIAL_E_NA_MEDICINA
  https://www.sciencedirect.com/science/article/abs/pii/S0952197623015269
- mlxtend McNemar test documentation and general statistical-testing-for-ML-comparison
  sources — confirm the standard framing that a non-significant McNemar p-value
  justifies preferring the simpler/cheaper model, directly supporting Gap 3's
  table-stakes requirement:
  https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
  https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/

---
*Feature research for: PIBIC academic-rigor reporting requirements (SPEC.md gaps 1-3)*
*Researched: 2026-07-27*
