# Stack Research — Algorithmic Fidelity for 3 PIBIC Gaps (C99, no ML libraries)

**Domain:** Pure C99 numerical/ML pipeline — algorithm-fidelity research (not a technology-stack
research in the usual sense; there are no new libraries to install, only exact formulas/pseudocode
that must match cited literature).
**Researched:** 2026-07-27
**Confidence:** HIGH for Gap 2 (primary source read in full) and Gap 3 methodology (canonical,
well-established citations); MEDIUM for Gap 1 (canonical lattice/Gc/Gct math verified against
multiple sources, but the μ/λ-from-continuous-features derivation has no single universally-cited
formula — flagged explicitly below).

---

## Recommended Stack

### Core Algorithms

| Technique | Canonical Source | Purpose | Why This Exact Form |
|---|---|---|---|
| Borderline-SMOTE1 | Han, Wang & Mao (2005), *ICIC 2005*, LNCS 3644, pp. 878–887 | Oversample only minority samples near the decision boundary | This is the variant SPEC.md actually describes (same-class-only interpolation neighbors) — confirmed against the original paper's Step 1–4 pseudocode, verbatim |
| LPA2v / Análise Paraconsistente de Evidências | da Costa, Subrahmanian & Vago (1991) "Annotated Paraconsistent Logic" (theoretical foundation); Abe, J.M. (1992) doctoral thesis *Fundamentos da Lógica Anotada*, USP (LPA2v bivalued specialization, advised by da Costa); Silva Filho, J.I. (1999) — 12-region "Para-Analyzer" lattice engineering formalization | Feature scoring/selection via degrees of belief/disbelief | Gc/Gct formulas and the ±0.5-threshold 12-region lattice are the standard, repeatedly-published construction (verified against 3 independent secondary sources + 1 primary dissertation PDF) |
| 1-SE rule + McNemar / bootstrap CI for architecture selection | Hastie, Tibshirani & Friedman, *Elements of Statistical Learning* 2nd ed. (2009), §7.10 (one-standard-error rule); Dietterich, T. (1998), *Neural Computation* 10(7):1895–1923 (McNemar validity for single-run classifier comparison); Cawley & Talbot (2010), *JMLR* 11:2079–2107 (selection-bias / nested-CV caveats) | Choose "smallest model not statistically worse" among shallow/deep MLP configs | These three papers together are the standard citation set for exactly this decision rule in the ML methodology literature |

### Directly On-Point Prior Work (same database, same domain — cite these)

| Work | Relevance |
|---|---|
| Costa, S.C. et al., "Acoustic investigation of speech pathologies based on the discriminative paraconsistent machine (DPM)" (2019), *Biomedical Signal Processing and Control* | Applies paraconsistent annotated logic (μ/λ, favorable/contrary evidence) directly to voice-pathology acoustic features (energy, ZCR, entropy) from the **Saarbrücken Voice Database** — the same corpus this project uses. **Strongest available precedent for how μ/λ are derived from acoustic features in this exact domain.** Paywalled (ScienceDirect); could not extract exact μ/λ formula — get via CAPES Periódicos/university library before writing the final report, since this is a far more defensible citation than a generic PAE textbook reference. |
| "Application of Wavelet Analysis and Paraconsistent Feature Extraction in the Classification of Voice Pathologies" (2025), *Biomedical Signal Processing and Control* (or similar) | Even more directly on-point: wavelet features + paraconsistent logic for voice pathology classification. Same access caveat as above. |
| "Wavelet-based features selected with Paraconsistent Feature Engineering successfully classify events in low-voltage grids" (2021), *Measurement* | Uses the term "Paraconsistent Feature Engineering" for an actual **feature selection** (not just classification) task — closest terminological match to Gap 1's stated goal. Worth obtaining for its μ/λ derivation method, which is likely closer to a genuine "selection" precedent than the voice-pathology papers (which use PAE for classification/fusion, not attribute pruning). |

**Action item for the student:** these three papers are not accessible via public web search/WebFetch
(ScienceDirect paywall). Before finalizing Gap 1's report language, retrieve them via institutional
access (CAPES Periódicos is free for Brazilian public-university PIBIC students) and check whether
the μ/λ derivation reported there differs from what is proposed below — if so, prefer the domain-specific
precedent since it is a stronger, more specific citation for the committee than the generic da Costa/Abe
theoretical papers.

---

## GAP 2 — Borderline-SMOTE: Exact Algorithm (verified against primary source)

I read Han, Wang & Mao (2005) in full. The paper defines **two** variants; SPEC.md implements
(correctly) borderline-SMOTE1's classification+generation logic, but the SPEC's prose slightly
under-specifies two things that matter for a faithful citation. Details below.

### Step 1 — Classification rule (verified verbatim against the paper)

For every minority-class sample `p_i`, compute its `m` nearest neighbors **from the whole training
set T** (all classes, not just minority) — this matches SPEC.md's `find_knn_global()`. Let `m'`
(0 ≤ m' ≤ m) be the number of those `m` neighbors that belong to majority classes:

```
m' == m            → NOISE       (discard, do not oversample)
m/2 <= m' < m      → DANGER      (borderline — this is the set that gets oversampled)
0 <= m' < m/2      → SAFE        (already well-represented, do not oversample)
```

**This is exactly what SPEC.md proposes.** Confirmed correct — no deviation here.

### Step 2 — Generation formula for borderline-SMOTE1 (the variant SPEC.md implements)

For each `p'_i` in DANGER, compute its `k` nearest neighbors **restricted to the minority class P**
(same-class-only — this is the defining difference of SMOTE1 vs SMOTE2). Randomly select `s`
(1 ≤ s ≤ k) of those `k` neighbors, then for each selected neighbor:

```
dif_j      = neighbor_j - p'_i
r_j        ~ Uniform(0, 1)
synthetic_j = p'_i + r_j * dif_j
```

This is algebraically identical to standard SMOTE's interpolation formula — the only difference
from plain SMOTE is which base points are eligible (`DANGER` instead of the full minority class).
**SPEC.md's proposed generation step ("vizinhos para interpolação continuam sendo buscados dentro
da própria classe") is correct and matches Borderline-SMOTE1 exactly.**

### Deviations/imprecisions in SPEC.md to fix before writing up the citation

1. **`m` (classification k-NN) vs `k` (generation k-NN) are two different, independently-tunable
   parameters in the original paper — not the same value.** The paper explicitly calibrates `m`
   per-dataset so that "the number of minority examples in DANGER is about half of the minority
   class," while `k` (generation neighbor pool) is fixed at 5, matching standard SMOTE. SPEC.md's
   proposal reuses `k=5` for both roles. This is a common simplification (e.g., several open-source
   implementations default both to 5), but **is not literally what the paper does** — flag this
   explicitly in the methodology section rather than silently presenting `k=5/m=5` as "the Han et al.
   parameterization." Minimum fix: name them as two separate constants in code
   (`BORDERLINE_M_NEIGHBORS`, `SMOTE_K_NEIGHBORS`) even if both start at 5, and ideally run a small
   sensitivity check (m ∈ {5, 7, 9, 11}) reporting how many samples land in DANGER per class —
   this turns an implementation shortcut into a documented, defensible design choice.
2. **The paper's DANGER-empty fallback does not exist in Han et al. (2005) at all.** SPEC.md's
   "if `class_idx_borderline[c]` is empty, fall back to all `class_idx[c]`" is a reasonable
   engineering safeguard for very small/isolated minority classes, but must be labeled explicitly
   as **an extension not in the cited paper**, not attributed to Han/Wang/Mao. Log how often this
   fires per class per fold — if it fires frequently for Edema de Reinke (n=68, smallest class,
   and the smallest at the Expert-level 4-class stage where per-fold training subsets shrink to
   roughly ~54 samples), the method effectively degrades to plain SMOTE for that class in practice,
   which would materially weaken the "we implemented Borderline-SMOTE, not standard SMOTE" claim to
   the committee. This should be reported as a finding either way (works / degrades to plain SMOTE
   for the smallest class), consistent with SPEC.md's own "document what doesn't work" rule.
3. **Precision in citation:** since the paper defines two variants, cite the implemented one
   specifically as *Borderline-SMOTE1* (Han, Wang & Mao, 2005) in the final report/poster, not
   generic "Borderline-SMOTE" — this shows the committee the citation was actually read, and avoids
   implying SMOTE2 behavior (which additionally interpolates toward majority neighbors with a
   0–0.5 gap, specifically to keep synthetics closer to the minority side — **not** implemented
   here and should not be claimed).
4. **Stability risk given this dataset's class sizes**: with `k=m=5` and Expert-level per-fold
   minority training counts as low as ~54 (Reinke) to ~90 (DisfPsicog), the SAFE/DANGER/NOISE
   partition can be high-variance across folds (a handful of neighbor-membership flips changes
   which samples qualify as DANGER). Recommend logging the SAFE/DANGER/NOISE counts per class per
   fold as a diagnostic table in the results — this is cheap to add and gives the committee hard
   evidence the method is behaving as described, not just a pass/fail Macro F1 comparison.

---

## GAP 1 — Paraconsistent Feature Selection: Exact Formulas and Where SPEC.md Is Imprecise

### What IS standard/canonical (verified against 3+ independent sources)

```
Gc  = μ − λ                    (Grau de Certeza / Degree of Certainty,     ∈ [-1, 1])
Gct = μ + λ − 1                (Grau de Contradição / Degree of Contradiction, ∈ [-1, 1])
```

These two formulas are correct and match SPEC.md exactly — no changes needed here. They originate
in da Costa, Subrahmanian & Vago's (1991) Annotated Paraconsistent Logic and Abe's (1992) LPA2v
bivalued specialization, and are used identically across the entire Silva Filho "Para-Analyzer"
literature.

### The 12-region lattice — confirmed structure

The (Gct, Gc) plane ∈ [-1,1]×[-1,1] is discretized using **±0.5 as the control thresholds on both
axes**, producing 4 "extreme" corner regions plus 8 intermediate "quase" (near-) regions = 12 total.
The 4 extreme regions and their canonical meaning:

| Region | Condition | Meaning |
|---|---|---|
| **Verdadeiro (V) — "certeza verdadeira"** | Gc ≥ 0.5 and \|Gct\| < 0.5 | High belief, low disbelief, low contradiction → strong, consistent evidence the feature is discriminative |
| Falso (F) | Gc ≤ -0.5 and \|Gct\| < 0.5 | Strong evidence AGAINST usefulness |
| Inconsistente (⊤ / T) | Gct ≥ 0.5 and \|Gc\| < 0.5 | μ and λ both high — contradictory evidence (feature looks useful AND unreliable simultaneously) |
| Indeterminado (⊥) | Gct ≤ -0.5 and \|Gc\| < 0.5 | μ and λ both low — no real evidence either way |

The remaining 8 regions ("quase-verdadeiro tendendo a...", etc.) are transitional zones between two
adjacent extremes and are not needed for a binary select/reject decision — **SPEC.md's plan to use
only the "certeza verdadeira" (V) region as the acceptance criterion, and reject everything else
(false, inconsistent, indeterminate, and all 8 transitional states), is a defensible simplification**
of the full 12-state machinery, common in applied feature-selection use of this lattice (most
applied papers only ever act on the 4 extreme states and treat all 8 transitional ones as "not
selected" rather than trying to assign each of the 8 a distinct action). Document this simplification
explicitly as a deliberate scope reduction from 12 states to a binary select/reject rule, citing
Silva Filho (1999) for the full lattice and noting only the "V" quadrant is operationalized.

**One correction to SPEC.md's boundary syntax:** SPEC.md's pseudocode selects on
`Gc_j >= gc_thresh && fabs(Gct_j) <= gct_max` with tunable `gc_thresh` (0.3–0.5) and `gct_max`
(≈0.3) rather than the fixed canonical 0.5/0.5 split. This is fine and arguably better practice
(treating the boundary as a hyperparameter to tune via the inner/outer CV, rather than hard-coding
the textbook 0.5) — but be explicit in the writeup that 0.5/0.5 is the "textbook" region boundary
from Silva Filho, and the project is using a *relaxed, tunable* version of it, not the literal
canonical thresholds. Do not present `gc_thresh=0.35` as if it were the standard boundary.

### What is NOT standardized — the μ/λ-from-continuous-features step (the actual risk area)

This is the part of Gap 1 with no single universally-cited formula. Da Costa/Abe's original theory
was built for **fusing evidence from independent expert/sensor sources** (e.g., two different
measuring instruments' opinions about the same proposition), not for scoring a single continuous
numeric feature automatically from a dataset. Every applied paper that uses LPA2v/PAE for automatic
feature scoring (routing, fault detection, voice pathology, network anomaly detection, etc.) has to
invent its own domain-specific mapping from raw statistics to μ and λ — there is no "the" canonical
formula analogous to how Gc/Gct themselves are fixed. **Present this honestly to the committee**:
cite da Costa/Abe/Silva Filho for the Gc/Gct/lattice machinery (that part is genuinely standard),
but describe the μ/λ derivation as an adaptation designed for this feature-selection task, following
the same general pattern used in domain precedents (see the on-point voice-pathology papers above).

Given that framing, here is a concrete assessment of SPEC.md's proposed derivation, with fixes:

**μ_j (proposed: average of per-class one-vs-rest Fisher-like ratios)**

SPEC.md's formula:
```
F_c(j) = (mean_c(j) - mean_global(j))^2 / variance_c(j)
mu_j = mean over c of F_c(j), then min-max normalized across features
```

Problem: dividing only by `variance_c(j)` (the variance of the target class, ignoring the spread of
"the rest") is **not the standard Fisher/discriminant ratio** and is not the correct one-vs-rest
generalization of it either. It systematically inflates the score for any class with naturally tight
within-class variance, independent of true separability. In this dataset, "Normal" (n=687, the most
"typical"/least variable class for many features) would then dominate μ regardless of whether the
feature actually separates the 5 classes well — the exact opposite of what Fisher scoring is meant
to measure.

**Recommended fix — use the standard multi-class ANOVA F-ratio (or equivalently η², eta-squared)
as a single per-feature score, not an average of asymmetric per-class ratios:**

```
Between-class sum of squares:  SSB_j = Σ_c  n_c * (mean_c(j) - mean_global(j))^2
Within-class sum of squares:   SSW_j = Σ_c  Σ_{i in class c}  (x_ij - mean_c(j))^2

F_j   = ( SSB_j / (C - 1) ) / ( SSW_j / (n - C) )     — standard one-way ANOVA F-statistic
```
or, as a bounded [0,1] alternative that skips the min-max normalization step entirely:
```
eta2_j = SSB_j / (SSB_j + SSW_j)      — eta-squared, effect size, already in [0,1]
mu_j   = eta2_j
```
Both are textbook (Fisher, 1936 for the F-test; Cohen, 1988 for η² as effect size), computable in a
single pass over the training data per fold (two accumulator loops: per-class sums/sums-of-squares,
then a final combine step — trivial in C, no `malloc` needed beyond `n_classes` accumulators).
`eta2_j` is strongly preferred in this codebase because it needs **no cross-feature min-max
normalization step** (already bounded), removing one extra source of fold-to-fold instability that
SPEC.md's F_c(j)-then-normalize approach would otherwise introduce.

**Guard against division by zero:** `SSW_j` can be exactly 0 if a feature is constant within every
class (rare but possible for some wavelet-level statistics on nearly-silent frames). Add an epsilon
(`SSW_j + 1e-8`) — do not skip the feature silently, since that would make `eta2_j` computation
branch inconsistently across folds.

**λ_j (proposed: 1 − 1/(1 + mean coefficient of variation within classes))**

The general design idea — deriving λ from a *different* statistical property than μ, so that μ and
λ are not algebraically forced into a fixed relationship — is actually the right high-level
instinct: if λ were defined as `1 − μ` (or any deterministic function of μ alone), then
`Gct = μ + λ − 1` would collapse to a constant (0), and the "contradiction" axis of the whole
paraconsistent apparatus would become degenerate — i.e., the method would reduce to a single
threshold on μ and would not actually need paraconsistent logic at all. SPEC.md's CV-based λ avoids
this trap because it comes from a genuinely different statistic (intra-class dispersion) than the
between/within-class ratio used for μ. **Keep this general design**, but fix a concrete, dataset-specific
correctness bug:

**Bug: `CV = std/mean` is undefined or explodes for near-zero-mean features.** CLAUDE.md's own
documented findings state that mean delta-MFCCs are "near-zero for sustained vowels" (which is
exactly why the project already uses std-dev of delta-MFCCs instead of the mean as a *feature*).
The same near-zero-mean issue will recur when computing CV *of* several of this project's 237
features directly (delta/delta-delta MFCC-derived quantities, several wavelet-level mean
coefficients, and any feature that can legitimately take negative values so its within-class mean
can pass through zero). A naive `std_c(j) / mean_c(j)` will produce huge or `inf`/`NaN` CV values for
exactly these features, silently corrupting λ_j (and downstream Gct) for a nontrivial fraction of the
237-feature set.

**Recommended fix:** replace CV (scale-relative dispersion) with a **scale-free-but-not-mean-relative**
dispersion measure, e.g. intra-class variance normalized by the *global* feature range or global
variance instead of the *local class mean*:

```
lambda_j = mean over c of [ variance_c(j) / (variance_global(j) + eps) ]
```
clipped to [0,1] (values >1 truncated to 1, meaning "this class is more spread out than the whole
dataset on this feature" is treated as maximal disbelief). This keeps the same semantic intent
(high intra-class spread → high disbelief) without ever dividing by something that can be zero or
near-zero for a real feature in this dataset. Whatever formula is finally chosen, explicitly unit-test
it against known near-zero-mean features (any delta-MFCC-derived column) before running the full
5-fold pipeline, since a silent `NaN`/`inf` here would propagate into `Gc_j`/`Gct_j` and could
select/reject features essentially at random for that column.

### Persisting selection artifacts

SPEC.md's plan to persist `(feature, μ, λ, Gc, Gct, selected)` per fold/network via
`selected_save`/`selected_load` and present it as a table is good practice and directly answers the
likely committee question "which features did the method consider relevant" — no changes recommended
there.

---

## GAP 3 — Shallow vs. Deep MLP Comparison Methodology

### What SPEC.md gets right

- Training all 4 architecture configs (A/B/C/D) on the **same 5 outer folds** and comparing Macro F1
  is standard practice for comparing a small number of discrete, pre-specified model classes.
- Using McNemar's test or bootstrap CI overlap to decide "not statistically worse" before picking the
  smallest adequate config is exactly the right kind of decision rule, and both methods already exist
  (dormant) in `src/metrics.c` (`metrics_mcnemar`, `metrics_bootstrap_ci`) — no new implementation
  needed, only wiring them into the fold loop for this specific comparison.
- Reporting number-of-parameters and time/epoch alongside Macro F1 is the correct way to make the
  "complexity vs. performance" tradeoff explicit and defensible to the committee.

### One terminology/rigor correction

SPEC.md's protocol (train each of 4 configs across the same 5 outer folds, no additional inner
resampling loop) is **not, technically, "nested cross-validation"** in the sense defined by Cawley &
Talbot (2010) — nested CV specifically means an *inner* CV loop performs model/hyperparameter
selection independently within each outer training fold, so that the *outer* test folds never
influence the selection decision, giving an unbiased outer estimate of "the pipeline that includes
architecture selection." What SPEC.md actually describes is a simpler (and for only 4 discrete,
pre-specified candidates, entirely adequate) **repeated K-fold comparison**: the same 5 outer folds
serve double duty as both the basis for choosing the winning architecture and the basis for reporting
that architecture's performance. This carries a small optimism/selection-bias risk exactly of the
kind Cawley & Talbot warn about (the folds "helped pick" the reported winner). The risk is modest
here because there are only 4 candidates (not a continuous hyperparameter grid), and it is further
mitigated by SPEC.md's own requirement to publish **all 4 configs' full metrics**, not just the
winner's — this transparency is precisely the mitigation Cawley & Talbot recommend when full nested
CV is impractical. **Action:** rename "nested CV" to "K-fold architecture comparison" (or similar) in
the final report so the methodology section doesn't overclaim, and add one sentence citing Cawley &
Talbot (2010) to justify why reporting all 4 configs' out-of-fold metrics (rather than only the
winner's) is the chosen mitigation for the small residual selection-bias risk.

### Recommended addition: the one-standard-error (1-SE) rule as the primary, simplest justification

Hastie, Tibshirani & Friedman (*Elements of Statistical Learning*, 2nd ed., 2009, §7.10) formalize
exactly the decision SPEC.md wants ("choose the smallest model not statistically worse than the
best") as: compute mean and standard error of the CV metric for each candidate, then select the
**most parsimonious candidate whose mean is within one standard error of the best candidate's mean**.
This is simpler to compute and explain than a full pairwise McNemar test, uses statistics the project
already collects (per-fold Macro F1 → mean and std across the 5 folds), and is a widely-cited,
textbook-standard rule that a committee will recognize immediately. Recommended protocol,
combining both citable methods for redundancy:

1. Compute mean ± SE of Macro F1 across the 5 folds for each of the 4 configs (cheap, already have
   the data).
2. Apply the 1-SE rule to get a first-pass "smallest adequate config" recommendation.
3. Confirm with McNemar's test (Dietterich, 1998) on the concatenated out-of-fold predictions between
   the 1-SE-rule winner and the overall best config — Dietterich's paper is the specific justification
   for why McNemar (rather than a paired t-test on fold accuracies, which Dietterich shows has
   unacceptably high Type I error) is the statistically valid choice **for this exact use case**:
   comparing two classifiers that are each trained/evaluated once (not resampled/repeated), which
   matches this project's 5-fold-CV-with-concatenated-oof-predictions setup exactly.
4. Report both the 1-SE table and the McNemar p-value in the final writeup — two independent,
   well-cited justifications strengthens the answer to a likely committee question ("how did you
   decide the shallow network was good enough?").

### Known pitfalls of adding depth on a dataset this size (project-specific)

- **Parameter-to-sample ratio.** Config D (~21k params) trained on the Expert network (4 pathological
  classes only, ~80% of 411 patients ≈ 329 real training rows before SMOTE, per fold) implies a
  parameters:real-samples ratio on the order of 60:1 even before considering that SMOTE-synthesized
  rows do not add genuinely new information. This is a much worse ratio than the already-tight
  Master network (binary, ~878 real training rows/fold, ~19:1 for Config C). Expect Config C/D to
  show a *larger* gap between training and validation Macro F1 (visible directly in
  `results/learning_curves.csv`, which the project already exports) — report this gap explicitly as
  evidence for or against the deeper configs, not just the final validation number.
- **No BatchNorm, single early-stopping signal.** CLAUDE.md already documents that BatchNorm hurts on
  this dataset and is disabled; Dropout + L2 + early stopping on val Macro F1 are the only active
  regularizers. Depth compounds dropout's per-batch capacity variance (more stacked dropout layers ⇒
  more variance in which sub-network is "active" per mini-batch of size 32), which can slow or
  destabilize convergence for Configs C/D relative to A/B — if C/D need more epochs to reach a
  comparable loss, but patience=30 was tuned for the shallow baseline, they may be cut off before
  converging. Recommend either (a) using the same patience for a fair architecture comparison and
  reporting if deeper configs were early-stopped before plateauing (visible in learning curves), or
  (b) explicitly noting that patience was NOT re-tuned per architecture as a documented limitation.
- **SMOTE-interacting risk (reasoned inference, not independently verified in the literature search —
  flag as LOW confidence / present as a hypothesis to test empirically, not a settled fact):** with
  more capacity, a deeper network has more ability to fit the specific interpolated/synthetic minority
  samples produced by Borderline-SMOTE rather than the true underlying class-conditional distribution.
  If Configs C/D show high train Macro F1 but no val Macro F1 improvement over A specifically on the
  post-SMOTE minority classes, this is the likely mechanism — worth a specific sentence in the
  discussion section either way.
- **CV variance already dominates the signal.** The project's own bootstrap CIs (already computed) on
  Macro F1 have documented widths around 0.07–0.09 in prior runs (see CLAUDE.md's v26 results,
  0.374–0.449). Any point-estimate difference between configs A/B/C/D smaller than this is very
  likely fold-partition noise, not a real architecture effect — this is exactly why step 3 above
  (McNemar on paired oof-predictions, not a naive "which mean is higher" comparison) is mandatory,
  not optional, before choosing a winner.

---

## What NOT to Use

| Avoid | Why | Use Instead |
|---|---|---|
| Attributing the DANGER-empty fallback behavior to Han, Wang & Mao (2005) | Not in the paper — the paper doesn't discuss this edge case at all | Present it explicitly as "an engineering extension for this project's extreme minority classes," not a cited technique |
| Citing generic "Borderline-SMOTE" without specifying variant 1 | The paper defines two variants with materially different generation behavior (SMOTE2 pulls toward majority neighbors) | Cite "Borderline-SMOTE1 (Han, Wang & Mao, 2005)" specifically, matching what's actually implemented |
| Using `CV = std/mean` as the sole basis for λ on this feature set | Several of this project's 237 features (delta-MFCC-derived, some wavelet-level means) are documented as near-zero-mean; CV blows up/becomes `NaN` for these, silently corrupting λ and thus Gct for those columns | Use a global-variance-normalized dispersion ratio (`variance_c(j) / variance_global(j)`) that never divides by a class mean |
| SPEC.md's per-class-averaged one-vs-rest "Fisher ratio" (`(mean_c - mean_global)^2 / variance_c`) as μ | Not the standard Fisher/ANOVA formula; divides only by the target class's own variance, biasing μ toward whichever class happens to have the tightest spread (likely "Normal," n=687) regardless of true multi-class separability | Standard one-way ANOVA F-statistic or η² (eta-squared) computed once per feature from between/within sums of squares across all classes simultaneously |
| Calling SPEC.md's Gap-3 protocol "nested cross-validation" | Cawley & Talbot (2010) define nested CV as requiring an inner resampling loop for selection, decoupled from the outer test folds; SPEC.md's protocol reuses the same 5 outer folds for both selection and reporting | Call it "K-fold architecture comparison" and cite Cawley & Talbot only to justify why reporting all 4 configs' full metrics (transparency) mitigates the smaller selection-bias risk of comparing just 4 discrete candidates |
| Treating 0.5/0.5 as "the" paraconsistent lattice threshold once `gc_thresh`/`gct_max` are tuned away from it | Silva Filho's canonical 12-region lattice uses exactly ±0.5; SPEC.md's tunable thresholds (0.3–0.5) are a relaxation of that boundary, not the boundary itself | State explicitly in the report that 0.5/0.5 is the textbook boundary and the project uses a tuned/relaxed version, selected via CV like any other hyperparameter |

---

## Stack Patterns by Variant

**If Reinke (n=68) or another very small class shows the DANGER-fallback firing on most folds:**
- Report this explicitly as a finding ("Borderline-SMOTE degrades to standard SMOTE for the smallest
  class due to insufficient local class density") rather than silently keeping the fallback silent.
- This is directly reusable in the final report's "what didn't work" section per CLAUDE.md's own
  documentation convention.

**If λ (paraconsistent disbelief) produces `NaN`/`inf` for any feature during a debug run:**
- It is almost certainly a delta-MFCC-derived or wavelet-mean feature with a near-zero within-class
  mean — switch to the global-variance-normalized λ formula above rather than adding an ad hoc
  epsilon to the CV denominator (an epsilon patches the crash but doesn't fix the underlying
  scale-instability of CV as a dispersion measure for these features).

**If the committee specifically asks "why not the literal da Costa/Abe formulas for μ/λ":**
- Answer: da Costa/Abe's original PAE formalism assumes μ/λ are supplied by independent evidence
  sources (e.g., two sensors/experts); there is no standard, citable procedure in that literature for
  deriving them automatically from a single continuous feature's class-conditional distribution. The
  Gc/Gct/lattice machinery is used exactly as published; the μ/λ derivation is this project's own
  adaptation for the feature-selection use case, informed by the same general pattern used in
  domain-specific applied papers (voice-pathology DPM, Paraconsistent Feature Engineering for grid
  event classification).

---

## Sources

**HIGH confidence (primary source read in full):**
- Han, H., Wang, W.-Y., Mao, B.-H. (2005). "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced
  Data Sets Learning." *ICIC 2005*, LNCS 3644, pp. 878–887. Full text obtained and read via
  https://sci2s.ugr.es/keel/pdf/specific/congreso/han_borderline_smote.pdf — Steps 1–4 and the
  SMOTE1/SMOTE2 distinction quoted/verified directly from this PDF.

**MEDIUM confidence (verified across 2–3 independent secondary sources + 1 primary dissertation PDF,
formulas cross-checked and consistent):**
- Gc/Gct formulas and ±0.5 12-region lattice: cross-verified via
  https://sites.unisanta.br/ppgmec/dissertacoes/dissertacao_fernando.pdf (primary dissertation PDF,
  fetched and read), plus Dialnet entries on LPA2v (da Silva Filho), and multiple secondary
  descriptions of the "Para-Analyzer" 12-state lattice.
- da Costa/Subrahmanian/Vago (1991) and Abe (1992) as founding LPA/LPA2v citations — confirmed via
  web search of USP thesis records and the Abe/da Costa academic lineage.
- Dietterich, T.G. (1998). "Approximate Statistical Tests for Comparing Supervised Classification
  Learning Algorithms." *Neural Computation* 10(7):1895–1923. McNemar validity claim confirmed via
  MIT Press abstract and multiple secondary summaries (paper itself paywalled, but the specific
  claims used here — McNemar has acceptable Type I error for single-run classifier comparison —
  are consistently reported identically across independent secondary sources).
- Cawley, G.C. & Talbot, N.L.C. (2010). "On Over-fitting in Model Selection and Subsequent Selection
  Bias in Performance Evaluation." *JMLR* 11:2079–2107. Freely available at
  https://www.jmlr.org/papers/v11/cawley10a.html — abstract/summary confirmed via JMLR page directly
  (open access journal, not paywalled).
- Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed.,
  §7.10 (one-standard-error rule) — confirmed via multiple independent secondary descriptions
  (R `mgcv` package docs, course notes) that consistently describe the same rule identically.

**LOW confidence / could not verify directly (paywalled, flagged for the student to verify before
citing in the final report):**
- Costa, S.C. et al. (2019), "Acoustic investigation of speech pathologies based on the discriminative
  paraconsistent machine (DPM)," *Biomedical Signal Processing and Control* — relevance (same SVD
  database, PAE applied to acoustic features) confirmed via abstract/title search only; exact μ/λ
  formula NOT verified (ScienceDirect 403).
- "Application of Wavelet Analysis and Paraconsistent Feature Extraction in the Classification of
  Voice Pathologies" (2025) — same access limitation.
- "Wavelet-based features selected with Paraconsistent Feature Engineering successfully classify
  events in low-voltage grids" (2021), *Measurement* — same access limitation; flagged as the
  closest terminological match to "paraconsistent feature selection" specifically (as opposed to
  paraconsistent classification/fusion).
- The "SMOTE-interacting-with-depth" overfitting hypothesis in the Gap 3 pitfalls section is a
  reasoned inference from general imbalanced-learning + overfitting principles, not a specific paper
  finding — presented as a hypothesis to test empirically, not a citable fact.

---
*Stack research for: PIBIC academic-rigor gaps (Borderline-SMOTE, paraconsistent feature selection,
shallow-vs-deep MLP comparison) in a pure C99 vocal-anomaly-detection pipeline*
*Researched: 2026-07-27*
