# Pitfalls Research

**Domain:** Academic ML pipeline in hand-rolled C99 — adding Borderline-SMOTE, paraconsistent
(evidence-based) feature selection, and a shallow-vs-deep MLP comparison to an existing
small-sample (1098 patients), severely-imbalanced (687/140/91/112/68), hierarchical
late-fusion voice-pathology classifier.
**Researched:** 2026-07-27
**Confidence:** HIGH for all findings tied to specific line numbers below (verified by
reading `src/main.c`, `src/mlp.c`, `include/mlp.h`, `include/config.h`,
`src/normalize.c`, `src/utils.c` directly). MEDIUM for general Borderline-SMOTE/ML
literature claims (WebSearch, cross-referenced against the original Han/Wang/Mao 2005
algorithm description — no full-text PDF access, definition confirmed via multiple
secondary sources).

This file assumes the reader has also read `.planning/codebase/CONCERNS.md` and
`SPEC.md`. It does not repeat those findings shallowly — it cross-references them and
goes one level deeper into the specific failure mechanisms that will bite Gap 1/2/3
implementation.

---

## Critical Pitfalls

### Pitfall 1: Integer-division truncation silently shifts the Borderline-SMOTE safe/danger boundary

**What goes wrong:**
SPEC.md's own rule (§ GAP 2) is: `m == k` → noise, `k/2 <= m < k` → borderline/danger,
`m < k/2` → safe, where `k` is the number of neighbors (SPEC uses `k=5` for parity with
the existing `smote_oversample`'s `k`). If this is implemented literally as C integer
arithmetic (`if (m >= k/2)`), `k/2` for the odd default `k=5` truncates to `2` instead of
`2.5`. That means a sample with `m=2` majority-class neighbors out of 5 (only 40%
majority) gets classified as "borderline/danger" and used as a synthetic-sample base,
when the continuous form of the rule (`m >= k/2.0`, i.e. `>= 50%`) would classify it as
"safe" and skip it. This silently over-generates synthetic samples from less-borderline
points than the paper intends, biasing the oversampled distribution toward the class
interior rather than the true decision boundary — undermining the entire premise of
Borderline-SMOTE ("only borderline") without producing any error or warning.

**Why it happens:**
C's `/` operator on two `int`s truncates toward zero. Nobody deliberately chooses to
truncate here — the bug is invisible in code review because `m >= k/2` reads as
mathematically correct and compiles cleanly; the divergence from the paper's ≥50% rule
only shows up if you expand `k/2` by hand for odd `k`.

**How to avoid:**
Compare `2*m >= k` instead of `m >= k/2` (or `m >= k/2` with `k`/`m` cast to `float`
before dividing). `2*m >= k` is exact for both even and odd `k` and avoids the
divide-then-truncate step entirely. Add a one-line comment stating explicitly which
convention is used, since the boundary choice is a real modeling decision (not just an
implementation detail) that the academic report needs to be able to state precisely.

**Warning signs:**
- Unit-test (or a quick standalone script) with `k=5`, `m=2` should NOT appear in the
  borderline set if you intend the strict ≥50% rule — if it does, you have the
  truncation bug.
- If the fraction of samples classified "borderline" per class is suspiciously close to
  identical to the fraction under a naive `m > k/2` (strict) rule for even `k` values
  but visibly different for `k=5` specifically, that's the truncation signature.

**Phase to address:** Gap 2 (Borderline-SMOTE) implementation phase — fix at the point
the `safe`/`borderline`/`noise` classification function is written, before any A/B run
is executed (a wrong boundary invalidates the whole A/B comparison, not just a corner
case).

---

### Pitfall 2: Conflating the "which neighbors count as majority" step with the "which neighbor to interpolate with" step

**What goes wrong:**
SPEC.md correctly specifies two *different* neighbor searches: (1) `find_knn_global()`
over **all classes** to count `m` (how many of a sample's k-NN are majority-class, used
only to classify safe/borderline/noise), and (2) the existing `find_knn()` restricted to
**the same class** (`class_idx[c]`) to pick the actual interpolation partner for
synthesis (unchanged from today's `smote_oversample`, `src/main.c:189-203`). The most
natural refactoring mistake is to reuse the neighbor list already computed in step (1)
for step (2) as well — since both are "k nearest neighbors of the same base point" and
it looks redundant to compute it twice. If a same-class-only interpolation neighbor is
picked from the global (all-class) list without re-filtering, the synthetic sample
becomes a convex combination of two *different* classes' feature vectors, which is
injected into the training set carrying the minority class's label — i.e., silent label
noise with no crash and no obviously wrong output (the resulting float vector still
"looks like data").

**Why it happens:**
SPEC.md itself calls this out defensively ("os vizinhos para interpolação continuam
sendo buscados dentro da própria classe... só a escolha do ponto-base muda") — the fact
that the spec had to say this explicitly is itself evidence that this is the natural
mistake an implementer (human or LLM) would otherwise make, since deduplicating "two
kNN searches on the same point" is an obvious-looking micro-optimization.

**How to avoid:**
Keep the two neighbor searches as textually and structurally distinct function calls
(`find_knn_global` for counting `m`; `find_knn` — unchanged — for interpolation), and
add an assertion or comment at the interpolation call site stating "must search only
`class_idx[c]`, never the global index list used for m-counting." Do not refactor them
into a shared helper unless the helper takes an explicit "restrict to these indices"
parameter that is different at each call site.

**Warning signs:**
- Spot-check: pick a few generated synthetic samples per fold/class and verify their two
  parent points (base + interpolation neighbor) both belong to the same original class
  label before oversampling.
- A regression where Macro F1 for the two acoustically-overlapping classes (Disfonia
  Psicogênica / Disfonia Funcional — see CLAUDE.md's "Fundamental Bottleneck") gets
  *worse* under Borderline-SMOTE than under standard SMOTE is a plausible symptom, since
  cross-class interpolation would specifically blur the boundary between exactly these
  two already-hard-to-separate classes.

**Phase to address:** Gap 2 implementation phase, code-review checklist item before
merging — verify by reading the diff, not just by running the A/B comparison (a
regression this subtle could also be masked by fold-to-fold variance, see Pitfall 4).

---

### Pitfall 3: Empty "borderline" set on already-degenerate small classes compounds with the existing `n_class <= 1` SMOTE bug

**What goes wrong:**
CONCERNS.md already documents that `smote_oversample()` degenerates when a class has
`n_class <= 1` samples in a given fold/vowel split (the only "neighbor" is the point
itself, so `find_knn` yields the point as its own neighbor and SMOTE silently produces
an exact duplicate — see CONCERNS.md "Fragile Areas"). Borderline-SMOTE adds a second,
independent way to end up with zero usable base points: if **every** sample of a class
is classified "noise" (`m == k`, i.e., completely surrounded by majority-class
neighbors — plausible for the smallest classes, Edema de Reinke n≈68 total, further
split 5-fold × train/val × 4-class-Expert-only, potentially single digits per
fold/vowel), `class_idx_borderline[c]` is empty. SPEC.md's own fallback ("cair de volta
para todos os `class_idx[c]`") re-enters the pre-existing `n_class <= 1` degenerate path
from CONCERNS.md — so the two bugs can now trigger *together* in the same run, and any
debugging session investigating a bad synthetic sample won't immediately know which of
the two independent code paths produced it.

**Why it happens:**
The two failure modes were introduced at different times by different code (original
SMOTE fragility documented pre-existing; Borderline's noise/safe/danger split is new),
and neither has a unit test (CONCERNS.md "Test Coverage Gaps" — no tests exist for
`smote_oversample`/`find_knn` at all). Small absolute class counts (68-91 patients)
combined with 5-fold CV and a 4-class Expert split make single-digit per-fold-per-vowel
class counts a *normal*, not edge-case, occurrence in this dataset — not a rare event
worth deferring.

**How to avoid:**
Fix the CONCERNS.md `n_class <= 1` guard *first*, as a prerequisite task before starting
Gap 2 (log a warning and make the fallback-to-duplication behavior explicit rather than
silent), so that when Borderline-SMOTE's fallback re-enters this path during Gap 2
testing, its behavior is already known-and-logged rather than a second unknown. Add an
explicit log line whenever `class_idx_borderline[c]` is empty and the code falls back to
`class_idx[c]` (SPEC.md asks for this — "logar aviso" — enforce it, don't skip it as
"just a fallback").

**Warning signs:**
- Grep training logs for the "falling back to standard behavior" warning per
  class/fold/vowel — if it fires for Edema de Reinke or Disfonia Psicogênica on most
  folds, Borderline-SMOTE is not actually doing borderline selection for those classes
  most of the time, which should be reported as a limitation, not hidden.
- Any fold producing `Macro F1` outliers far outside the documented v29 baseline range
  (0.28–0.70 across folds per CLAUDE.md) is a candidate for having hit this compound
  degenerate path.

**Phase to address:** Fix the `n_class <= 1` guard as a small prerequisite task; address
the empty-borderline-set fallback logging as part of Gap 2 itself.

---

### Pitfall 4: "Adopt only if Macro F1 is equal or better" is unenforceable today — the statistical infrastructure exists but is disconnected

**What goes wrong:**
SPEC.md's acceptance criteria for all three gaps rely on comparing Macro F1 (and, for
Gap 3, McNemar/bootstrap CI) between two configurations run on the same folds/seed.
CONCERNS.md already found that `metrics_bootstrap_ci()` and `metrics_mcnemar()` are
fully implemented (`src/metrics.c:167`, `:206`) but **never called** from
`mode_train()` in the current v29 HEAD — only the point-estimate `metrics_export_csv`
runs. Given CLAUDE.md's own documented fold-to-fold variance (individual fold Macro F1
ranging from ~0.28 to ~0.70 in past experiments), comparing two configurations by their
5-fold-mean point estimate alone is exactly the kind of noisy signal that produces false
positives ("Borderline-SMOTE is better!") and false negatives ("the deeper net is
worse") that a bootstrap CI or McNemar test — sitting unused in the codebase right now —
was built specifically to guard against.

**Why it happens:**
The infrastructure was written for an earlier architecture (per CONCERNS.md, "these
functions... were exercised by an earlier architecture version, per git history
`2e5af63`") and was never re-wired into the current hierarchical late-fusion
`mode_train()`. It is easy to assume "we have bootstrap CI in this codebase" (true) and
skip verifying "we call it on the aggregated predictions of *this* run" (currently
false).

**How to avoid:**
Before running the first A/B comparison for any of the 3 gaps, add call sites in
`mode_train`'s fold loop for `metrics_bootstrap_ci()` on the aggregated
`all_y_true`/`all_y_pred` (the arrays already being accumulated at
`src/main.c:246-248`), and for `metrics_mcnemar()` comparing the two candidate models'
per-sample predictions on the same out-of-fold indices. Treat this wiring as a
zero-risk, mechanical prerequisite task (not a new gap) — it's needed by all three SPEC
gaps' acceptance criteria, so doing it once up front avoids redoing ad-hoc comparison
logic three times.

**Warning signs:**
- If a gap's "adopted" decision in `results/train_log_vXX_<gap>.txt` cites only a single
  Macro F1 number per configuration with no confidence interval or significance test,
  the acceptance criterion in SPEC.md has not actually been honored, regardless of what
  the log says.

**Phase to address:** Cross-cutting prerequisite, ideally its own small phase/task
before Gap 2/3/1 A/B runs begin (matches SPEC.md's stated implementation order:
Gap 2 → Gap 3 → Gap 1).

---

### Pitfall 5: Feature-selection statistics computed on the wrong data slice — order-of-operations, not a leakage bug per se

**What goes wrong:**
CLAUDE.md documents a *previously fixed* leakage class: `norm_fit` must use
`fold->n_train` (original samples) not `n_train_aug` (audio-augmented), and this is
correctly honored today (`src/main.c:267`: `norm_fit(train_x_all, fold->n_train, ...)`,
confirmed by reading the current source). SPEC.md's own worked example for Gap 1,
however, calls `paraconsistent_select(tr_x_v, tr_y_bin, n_train_aug, nf_vowel, 2, ...)`
using `n_train_aug` — i.e., train + 8x audio-domain-augmented pathological samples, but
crucially **before** `smote_oversample()` runs (SMOTE happens after, at
`src/main.c:293`). This ordering is actually fine with respect to test/val leakage (no
val/test data is touched), but it introduces a *different*, more subtle statistical
problem specific to the Fisher-ratio-like `μ` formula: audio-domain augmentation is
applied only to pathological classes (`precalculate_augmentations` explicitly skips
`CLASS_NORMAL`, `src/main.c:139`), so the per-class sample counts feeding the
variance/mean computation in `F_c(j) = (mean_c - mean_global)^2 / var_c` are wildly
unbalanced across classes (Normal: ~1x its fold count; every pathological class: ~9x its
fold count, from the original + 8 augmented copies each). This differentially changes
each class's variance estimate (the augmented copies add correlated near-duplicate
noise, likely *inflating* `var_c` for pathological classes relative to Normal), which
feeds directly into μ_j and therefore into which features get selected — a bias that
has nothing to do with train/val leakage but everything to do with an implicit,
undocumented asymmetry in the input the formula sees.

**Why it happens:**
"No test/val leakage" and "statistically comparable class-conditional estimates" are
two different correctness properties, and CLAUDE.md's documented leakage rule
(`norm_fit` on `n_train`, not `n_train_aug`) only guards the first one. It is easy to
treat "matches the norm_fit rule" as sufficient without noticing the *class-count
asymmetry* problem is new to Gap 1's specific statistic (mean/variance/class-Fisher
ratio), which `norm_fit`'s global not-per-class Z-score never had to worry about.

**How to avoid:**
Decide explicitly and document which of these three options is used, and why:
(a) Compute μ/λ using **only the true original training samples** (`fold->n_train`,
same rule as `norm_fit`), excluding both audio augmentation and SMOTE — simplest,
matches the existing documented precedent exactly, avoids the asymmetry entirely; or
(b) if augmented samples are intentionally included to give minority classes more
stable variance estimates, explicitly downweight or subsample so effective N per class
is comparable; or (c) compute μ/λ per-class using each class's *own* augmentation
multiplier as a documented, deliberate design choice (defensible to a review board only
if stated as such). Option (a) is the recommended default — it is the same rule
CLAUDE.md already established and requires the least new justification.

**Warning signs:**
- If selected feature sets change dramatically when toggling audio augmentation on/off
  for the same fold, that is a direct signal the augmentation asymmetry is driving
  selection rather than genuine class separability.
- Report the number of samples-per-class actually fed into `paraconsistent_select` per
  fold as a debug log line — if pathological classes show ~9x the Normal class's
  effective N, this pitfall is live.

**Phase to address:** Gap 1 design step, before writing `paraconsistent_select()` —
this is a specification decision (which data slice) that should be pinned down and
documented alongside the function signature, not discovered after the fact by comparing
confusing per-fold feature-selection outputs.

---

### Pitfall 6: Divide-by-zero in the Fisher-ratio-like `μ` formula for near-zero-variance features — no guard specified

**What goes wrong:**
SPEC.md's formula for `μ_j` is `F_c(j) = (mean_c(j) - mean_global(j))^2 / var_c(j)`,
averaged over classes `c`. With 237 raw acoustic features (many highly correlated
MFCC/delta-MFCC/wavelet-statistic families per CLAUDE.md's feature table) computed per
fold's training slice, it is entirely plausible for `var_c(j)` to be exactly zero or
extremely small for at least one (feature, class) pair in at least one fold — e.g., a
wavelet-energy statistic that happens to be constant across the handful of Reinke
training samples in a given fold, or any feature after a prior thresholding step already
removed near-constant columns (this codebase currently has none — CONCERNS.md notes
`select_features()`/variance thresholding is *dead code*, `src/feature_select.c`
contains only save/load stubs — so **nothing today filters near-zero-variance features
before they would reach `paraconsistent_select`**). `F_c(j)` with `var_c(j) == 0`
produces `inf` or `NaN` in C float arithmetic, which then propagates through the
class-average into `μ_j`, and through `Gc_j = μ_j - λ_j` into the selection decision —
silently marking that feature as either always-selected (`inf > gc_thresh` is true) or
poisoning downstream min-max normalization of `μ` across all `j` features in the fold
(a single `inf`/`NaN` in a min-max-normalized array corrupts the normalization for
*every other feature* in that array, not just the offending one).

**Why it happens:**
SPEC.md's formula is stated in pure mathematical notation without a numerical-stability
clause, and the codebase's only existing precedent for this exact problem
(`src/normalize.c:15,42`: `#define MIN_STD 1e-8f` ... `if (params->std[j] < MIN_STD)
params->std[j] = MIN_STD;`) is in a different file (`normalize.c`, not
`feature_select_paraconsistent.c`) and easy to forget to port over when writing a new,
structurally different formula.

**How to avoid:**
Apply the exact same pattern already established in `src/normalize.c:15,42`: define a
`MIN_VARIANCE` epsilon (e.g., `1e-8f`) and clamp `var_c(j)` to it before dividing,
`var_c(j) = fmaxf(var_c(j), MIN_VARIANCE)`. This makes near-zero-variance features
produce a large-but-finite `F_c(j)` (correctly signaling "this class is a tight,
distinctive cluster on this feature" rather than corrupting the computation) instead of
`inf`/`NaN`. Additionally, after computing `μ_j` for all `j`, assert/guard that no `NaN`
survives before the min-max normalization step — one bad feature should not be able to
corrupt every other feature's normalized `μ`.

**Warning signs:**
- Log a count of `(feature, class)` pairs where `var_c(j)` was clamped, per fold — if
  this count is non-trivial (it likely will be, given 237 candidate features × 4-5
  classes × 5 folds × up to 2 networks/vowel), it confirms the guard is load-bearing,
  not theoretical.
- Any fold reporting `0` or `nf` (all) features selected is a direct symptom of `NaN`
  propagation through the min-max step (everything ties at `NaN` or every comparison
  involving `NaN` evaluates false, which in C's `>=`/`<=` comparisons used for the
  `Gc_j >= gc_thresh` selection rule silently excludes the feature — the *opposite*
  failure mode from `inf`, so both directions need checking).

**Phase to address:** Gap 1 implementation — this guard must be written into
`paraconsistent_select()`'s first version, not added reactively after a fold produces
an empty or full feature set (which SPEC.md's own §4-5 already anticipates needing a
"relax gc_thresh" fallback for — see Pitfall 7 — but that fallback does not fix a
NaN-corrupted `μ` array, it only adjusts the threshold applied to it).

---

### Pitfall 7: The "relax threshold if zero features survive" fallback can mask a NaN/variance bug instead of a genuinely strict threshold

**What goes wrong:**
SPEC.md step 5 specifies: "Se nenhuma feature sobrar, relaxar `gc_thresh` em 0.05 e
repetir... logar um aviso quando isso ocorrer." This is a reasonable safety net for the
*intended* failure mode (threshold genuinely too strict for a given fold's class
separability). But if the real cause of zero-features-selected is Pitfall 6 (NaN
corrupting every feature's `Gc_j` to a value that never satisfies `>= gc_thresh`
regardless of how low it is relaxed), the relaxation loop will either (a) loop
indefinitely / need an explicit termination clause SPEC.md doesn't specify (what happens
if `gc_thresh` relaxes below 0, or below `gct_max`, without ever finding a feature?), or
(b) eventually relax so far that it accepts features based on `NaN`-comparison quirks
rather than genuine signal, silently selecting an arbitrary/wrong feature subset instead
of surfacing the real bug.

**Why it happens:**
A "keep relaxing until something is selected" loop is, by construction, designed to
never report total failure — which is exactly the property that makes it dangerous as
the *only* line of defense against a numerical bug upstream. SPEC.md doesn't specify a
maximum relaxation count or a hard failure path, likely because the spec's author was
reasoning about the intended failure mode (strict threshold) rather than the numerical
one (Pitfall 6).

**How to avoid:**
Implement Pitfall 6's variance-clamping guard *first*, so the relaxation loop only ever
needs to handle its intended case. Additionally, cap the relaxation loop at a small
fixed number of iterations (e.g., 5, i.e., `gc_thresh` down to as low as `0.35 - 5*0.05
= 0.10`) and if still zero features survive after the cap, fail loudly (log an error and
either select the single best-`Gc` feature as an explicit last resort, or abort that
fold with a clear message) rather than looping unboundedly or silently returning an
empty selection that a downstream `mlp_init_dynamic(net, 0, ...)` would then choke on
with a zero-sized input layer.

**Warning signs:**
- Any fold's log showing more than 2-3 relaxation iterations before finding a surviving
  feature is worth investigating as a possible symptom of Pitfall 6, not just "this
  fold's classes happen to overlap a lot."
- A crash or garbage output from `mlp_init_dynamic` with `input_size == 0` is the
  downstream symptom of an unguarded total-selection-failure path.

**Phase to address:** Gap 1 implementation, same task as Pitfall 6's guard — write the
termination condition and the variance guard together, since they're two halves of the
same defensive-programming concern.

---

### Pitfall 8: Per-fold, per-network feature subsets make a single "committee-facing" importance table misleading

**What goes wrong:**
SPEC.md explicitly requires (§ GAP 1 integration) that Master and Expert networks — and,
implicitly, each of the 3 vowels × 5 folds — can select **different** feature subsets
("os índices selecionados para Mestra e Especialista podem (e devem) ser diferentes").
SPEC.md's acceptance criteria then ask for "uma tabela: feature, μ, λ, Gc, Gct,
selecionada (S/N)" for the final report/poster. If this table is generated from a single
arbitrary fold/network's run (e.g., "fold 0, Master, vowel A" — the easiest thing to
print), it will misrepresent the method to the review board: a feature marked "not
selected" in that one table might be selected in 4 of the other 5 folds, and vice versa.
Given this codebase's already-documented high fold-to-fold variance (CLAUDE.md), feature
selection *instability* across folds is a near-certainty, not an edge case — the 30
independent selection runs (5 folds × 3 vowels × 2 networks) will very plausibly not
agree on a single canonical subset.

**Why it happens:**
SPEC.md's acceptance criteria describe "a table" in the singular, which reads naturally
as "one table" during implementation, but the method as specified genuinely produces up
to 30 different subsets per full pipeline run. The mismatch between "one table the
committee can look at" and "30 potentially-different subsets" is a scope gap in the spec
itself, not an implementation bug — but it will surface as a pitfall the moment someone
tries to fill in that table.

**How to avoid:**
Report a **selection-frequency table** instead of a single-run table: for each feature
`j`, report how many of the 30 (fold × vowel × network) runs selected it (e.g.,
"selected in 27/30 runs"), plus the mean/std of `μ_j`, `λ_j`, `Gc_j`, `Gct_j` across
those runs. This is both more honest (communicates instability instead of hiding it)
and more defensible to a review board asking "which features did the method consider
most relevant" (CLAUDE.md/SPEC.md's own stated goal for this table) — a feature selected
in 29/30 runs is a much stronger answer than a feature selected in one arbitrarily-shown
run. Persist all 30 selection outcomes (SPEC.md already specifies per-fold/per-vowel
`.bin` files for this) specifically so this aggregate table can be built after the full
run, not guessed from one log.

**Warning signs:**
- If the final report/poster table has no fold/vowel/network dimension mentioned at all
  ("feature X was selected"), that's the misleading-single-run version.
- Compute the Jaccard similarity of selected-feature-index sets across folds for the
  same network/vowel — a low overlap (<50-60%) is expected here given the small class
  sizes, and should be *reported as a finding* (feature-selection stability under small
  N), not smoothed over.

**Phase to address:** Gap 1's reporting/deliverable step (end of Gap 1 work, per SPEC's
own "Critério de aceite" item about the report table) — but the *persistence* of
per-run selection results (the `.bin` files) must be wired in from the start of Gap 1
implementation, since you cannot retroactively reconstruct a frequency table if only the
final fold's selection was kept.

---

### Pitfall 9: `Layer layers[MLP_NUM_LAYERS]` is a compile-time-fixed-size array — a true multi-depth comparison cannot coexist with it unmodified

**What goes wrong:**
`include/mlp.h:63` declares `Layer layers[MLP_NUM_LAYERS];` as a **fixed-size embedded
array inside the `MLP` struct**, where `MLP_NUM_LAYERS` (`include/config.h:81`) is a
single, global, compile-time constant currently `3`. `mlp_init_dynamic()`
(`src/mlp.c:220-241`) already branches on `#if MLP_NUM_LAYERS == 3` vs. `#else` to choose
between a 3-layer and 4-layer sizing scheme, but both branches still write into the same
statically-sized `layers[3]` array — the `#else` branch (`src/mlp.c:229-240`, 4 layers:
3 hidden + output) is dead code today specifically *because* `MLP_NUM_LAYERS` is `3` at
compile time, so that branch never executes, and if it did, it would need `layers[4]`,
one more than the array actually has room for (SPEC.md's own text flags this exact
concern: "Verificar que `net->layers[4]` (tamanho fixo hoje) comporta `n_hidden+1`
camadas"). Gap 3's `mlp_init_multi(net, ..., hidden_sizes, n_hidden, ...)` needs to
support `n_hidden` up to 3 (SPEC's Config D: `[128, 64, 32]`, i.e., `num_layers = 4`)
*at runtime*, with different configs coexisting in the same binary/run (nested CV trains
configs A, B, C, D within the same process). A fixed `MLP_NUM_LAYERS` compile-time macro
cannot represent "this particular `MLP` instance has 4 layers, that other instance has
2" — writing `net->layers[3]` for a Config D instance when the struct's array is sized
for `MLP_NUM_LAYERS == 3` (i.e., valid indices 0-2 only) is an out-of-bounds write into
whatever memory follows the `layers` array in the struct (`num_layers`, `timestep`) or
past the end of a heap-allocated `MLP`, corrupting adjacent memory silently (undefined
behavior, no compiler warning, since C does no bounds checking on array members).

**Why it happens:**
The current codebase's only two configurations ever exercised (the `#if`/`#else`
branches) happen to both be selected entirely at *compile time*, so nobody had to
confront the fact that `layers[MLP_NUM_LAYERS]` cannot vary per-instance. Gap 3 is the
first requirement to instantiate genuinely different depths *simultaneously at
runtime*, which is exactly the assumption this struct layout violates.

**How to avoid:**
Change `Layer layers[MLP_NUM_LAYERS]` to either (a) `Layer layers[MAX_MLP_LAYERS]` where
`MAX_MLP_LAYERS` is a generous fixed upper bound (e.g., 5, covering up to 4 hidden
layers + output) with `net->num_layers` as the actual runtime-used count — simplest, one
extra unused-but-harmless `Layer` slot's worth of struct size when `num_layers` is
small; or (b) `Layer *layers` with `layers = malloc(num_layers * sizeof(Layer))` in
`mlp_init_multi`, freed in `mlp_free`. Option (a) is lower-risk for this codebase: it
requires no change to `mlp_free`/`mlp_save`/`mlp_load`/`mlp_backward` (all already
iterate `for (i = 0; i < net->num_layers; i++)`, which is correct regardless of whether
`layers` is a bigger-than-needed fixed array or a malloc'd exact-size one) and avoids
introducing a new heap allocation whose lifetime must be tracked (see Pitfall 12).

**Warning signs:**
- Build with `-fsanitize=address` (not currently in the Makefile — CONCERNS.md already
  recommends adding a `make asan` target for this exact class of bug) and run any config
  with `n_hidden >= MLP_NUM_LAYERS - 1` (i.e., total layers > current compile-time `3`)
  — ASan will report a stack/heap-buffer-overflow on the `layers[]` write immediately.
  Without a sanitizer, this bug can run silently for an entire training session,
  corrupting `num_layers`/`timestep` or adjacent heap metadata, producing plausible-but-
  wrong results (e.g., `mlp_free` iterating a corrupted `num_layers` value) rather than
  an obvious crash.
- Any Config D-style test that trains for one epoch and produces obviously-corrupted
  loss values (`NaN`/huge numbers appearing immediately, not after many epochs) with no
  other explanation is a candidate symptom.

**Phase to address:** Must be fixed *before* any Config C/D (`n_hidden >= 2`) is trained
— i.e., a prerequisite sub-task at the very start of Gap 3, not something to discover
mid-comparison. Configs A/B (`n_hidden <= 1`, ≤2 total layers) would not trigger this bug
even if left unfixed, which is exactly what makes it dangerous: the first two configs of
a 4-config sweep could pass cleanly, creating false confidence right before Config C/D
silently corrupts memory.

---

### Pitfall 10: Backprop delta buffer sized from a macro, not from the actual network being backpropagated (CONCERNS.md finding, deepened)

**What goes wrong:**
`mlp_backward()` (`src/mlp.c:287`) does `int max_size = MLP_HIDDEN1_SIZE;` (hardcoded
128) rather than computing the true maximum hidden-layer width of the specific `net`
instance passed in. CONCERNS.md already flags this as safe-today-because-128-is-still-
the-max-in-both-existing-configs. For Gap 3 specifically: SPEC.md's four configs (A:
`[128]`, B: `[64]`, C: `[128,64]`, D: `[128,64,32]`) all keep every hidden layer ≤128, so
none of the four *specified* configs trigger this bug — but the danger is that this fact
is coincidental, not enforced. If the comparison is later extended (a very likely
follow-up request once the committee sees the Config A-D table and asks "what about
wider layers", or if `mlp_init_multi`'s `hidden_sizes` parameter is reused for an
unrelated future experiment), any hidden layer wider than 128 silently overflows
`delta`/`delta_next`, corrupting whatever heap memory follows those two `safe_malloc`
allocations — again, no crash guaranteed, no compiler warning, purely a heap corruption
bug gated on a config choice nobody is prevented from making.

**Why it happens:**
Once Pitfall 9's struct-sizing generalization is in place, it becomes *easier*, not
harder, to reach for `mlp_init_multi(net, in, out, (int[]){256, 128}, 2, ...)` in a quick
follow-up experiment — the struct will happily accept it, and `mlp_forward` will run
correctly (it doesn't have a fixed-size intermediate buffer), masking the fact that
`mlp_backward`'s delta buffer is still tied to a stale assumption.

**How to avoid:**
Compute `max_size` in `mlp_backward()` from the actual instance:
`for (i = 0; i < net->num_layers - 1; i++) if (net->layers[i].output_size > max_size)
max_size = net->layers[i].output_size;` (only hidden layers matter, i.e., exclude the
final output layer, matching the existing logic's intent). This is a small, local,
low-risk fix — do it in the same commit as the Pitfall 9 struct fix, since both stem
from the same underlying stale assumption ("no hidden layer is ever wider than
`MLP_HIDDEN1_SIZE`") and reviewing them together makes the shared root cause explicit in
the diff/PR description (useful for the academic report's "what we found while
implementing Gap 3" narrative).

**Warning signs:** Same as Pitfall 9 — an ASan/Valgrind run is the only reliable
detector; a config with any hidden layer > 128 is the trigger condition. Since none of
SPEC.md's 4 configs trigger it, this pitfall will very likely NOT manifest during Gap 3
itself — flagging it now is specifically to prevent it from being reintroduced by a
*future* config sweep that reuses `mlp_init_multi` without re-reading this constraint.

**Phase to address:** Fix alongside Pitfall 9, same commit, even though none of the 4
specified configs currently exercise it — cheap insurance, and leaving a known landmine
undocumented/unfixed after having just found it would be worse than fixing it now.

---

### Pitfall 11: Comparing architectures with hyperparameters tuned for a different architecture biases the "shallow vs deep" conclusion

**What goes wrong:**
SPEC.md explicitly instructs: "Usar os mesmos hiperparâmetros de otimização já
validados (Adam, LR com cosine annealing, class weights por rede, early stop por
Macro-F1, `RANDOM_SEED=42`)" for all four configs. This is reasonable as a controlled-
comparison baseline, but Configs C/D have meaningfully more parameters (~19-21k vs
~11k for Config A per SPEC's own table) on a dataset where the Expert network alone
trains on a few hundred samples per fold/vowel after the hierarchical split. A deeper
network trained with a dropout schedule, L2 lambda (`L2_LAMBDA=0.001f`,
`include/config.h:102`), and early-stopping patience all *tuned for the shallow
config's capacity* is very likely to overfit faster and get early-stopped sooner —
producing a "deeper is worse" result that is really "deeper needs different
regularization strength, which we didn't search for." Presenting this as "shallow wins"
without that caveat would be a methodologically weak claim to defend to a review board
that understands basic overfitting/regularization tradeoffs — precisely the kind of
question SPEC.md itself anticipates being asked (the whole point of Gap 3 is to
*justify* the architecture choice with real comparative evidence).

**Why it happens:**
Holding hyperparameters fixed is the simplest way to get an apples-to-apples "does depth
help" comparison, and SPEC.md's instruction to do so is not wrong — but "same
hyperparameters" and "fair comparison of what each architecture can achieve" are
different claims, and conflating them is a very common pitfall in architecture-ablation
papers generally (not specific to this codebase, but directly applicable here given the
already-small dataset makes overfitting sensitivity to depth much higher than on
typical benchmark-sized datasets).

**How to avoid:**
Keep SPEC.md's fixed-hyperparameter run as the primary, reported comparison (it is
simpler, reproducible, and matches the spec), but explicitly document in the final
report that L2/dropout were *not* re-tuned per config, and — time permitting — run one
supplementary check: increase dropout/L2 slightly for Config C/D only (a config-specific
override, not a full hyperparameter search) and note whether the shallow-vs-deep
ranking changes. Even if this supplementary check is skipped for time, stating the
limitation explicitly ("hyperparameters were held fixed across configs; a
depth-specific regularization search was out of scope") pre-empts the review board
question rather than leaving it to be discovered as a gap during defense.

**Warning signs:**
- If Config C/D's training curves (`learning_curves.csv`, if the CONCERNS.md-flagged
  disconnected `train_history_export_csv` call is wired back in for this comparison —
  it should be, since Gap 3's whole point is a training-cost/behavior comparison) show
  early stopping firing markedly earlier (many fewer epochs) than Config A/B, that's a
  direct symptom of undertuned regularization for the deeper configs, not necessarily
  evidence depth doesn't help this problem.

**Phase to address:** Gap 3's comparison-protocol design step, before running the 4×5
nested-CV sweep — decide and document the fixed-hyperparameter tradeoff up front so it
appears as a stated methodological choice in the report, not an unexamined default.

---

### Pitfall 12: Memory-management regressions when generalizing `MLP` from 1-hidden-layer-shaped assumptions to N hidden layers

**What goes wrong:**
Beyond the struct-sizing issue (Pitfall 9), generalizing to `mlp_init_multi` touches
every function that currently assumes the specific 2-config (`#if`/`#else`) shape:
`layer_init`/`layer_free` (`src/mlp.c:123-168`) are already correctly parameterized by
`input_size`/`output_size` per call and iterate via `net->num_layers`, so they are
low-risk. The higher-risk spots are (a) `mlp_save`/`mlp_load` (`src/mlp.c:550-592`),
which serialize/deserialize exactly `net->num_layers` layers in a simple sequential
format with no length-prefix per layer beyond what `layer_init` already fixed at
construction time — if a saved Config-A checkpoint (2 layers: `[128, output]` — actually
3 per the current `#if MLP_NUM_LAYERS==3` build) is ever accidentally loaded into a
Config-D-shaped `net` (4 layers) via `mlp_load`, the function will read fewer
weights/biases than the file contains (silently truncating the trailing tensor's data)
without any format/version check — `mlp_load`'s only validation is per-`fread` return-
code checks, which cannot detect "this file was written by a differently-shaped
network," since the byte counts for a *smaller* saved network are still a valid prefix
of what a *larger* network's load loop expects to read only up to `net->num_layers`
(i.e., reading stops early with no length mismatch error if the smaller file simply runs
out of layers to serialize — actually the loop would fail on `fread` returning fewer
elements than expected on the *last* layer it saved for a smaller net being loaded into
a bigger `net`, which IS caught — but the *reverse* case, loading a bigger saved net's
file into a smaller in-memory `net`, silently succeeds while ignoring the extra trailing
bytes in the file, which is the dangerous silent case). And (b) if you switch to a
malloc'd `layers` pointer (Pitfall 9, option b) rather than the recommended fixed
oversized array, `mlp_free` must add a `free(net->layers)` after the existing per-layer
free loop — a missed final free here is a genuine new leak (distinct from, and
additive to, the pre-existing `mode_train` leak of `Dataset`/`FeatureMatrix` already
noted in CONCERNS.md), and this leak would now be exercised **repeatedly within a single
process** (4 configs × 5 folds × 3 vowels × 2 networks = up to 120 `MLP` instances
created per `make full` run for Gap 3's sweep, vs. the current 30), turning a
previously-harmless "leaks reclaimed on process exit" pattern into an actually-
significant cumulative leak during the run itself — exactly the scenario CONCERNS.md's
"Manual memory management" section already predicted ("would become a real leak if
`mode_train` were ever called in a loop... exactly the kind of change likely to turn
this into a real, cumulative leak").

**How to avoid:**
Prefer Pitfall 9's option (a) (fixed oversized `layers[MAX_MLP_LAYERS]` array, no new
heap allocation) specifically to sidestep this entire class of new-leak risk — it
requires no change to `mlp_free`. If a malloc'd-pointer approach is chosen anyway for
memory-efficiency reasons, add the `free(net->layers)` call and immediately verify it
under the `make asan`/`-fsanitize=address` build CONCERNS.md already recommends adding.
Separately, and regardless of which struct-sizing option is chosen: add a per-file
header to `mlp_save`'s format (`net->num_layers` and each layer's `input_size`/
`output_size` written first, verified against the in-memory `net` on load, failing
loudly on mismatch) — this is good practice generally but becomes load-bearing the
moment more than one architecture shape can exist in `models/*.bin` files
simultaneously (Gap 3 produces exactly that situation: 4 differently-shaped checkpoints
per fold/vowel/network instead of 1).

**Warning signs:**
- Any `make full` run for the Gap 3 sweep that shows steadily increasing RSS memory
  across configs/folds (check with `/usr/bin/time -v` or a periodic `ps` sample) rather
  than roughly flat memory use is the leak signature.
- Loading `models/selected_fold{k}.bin`-style checkpoint files across a config change
  without an explicit "was this file written by the same architecture" check is a
  latent footgun even if it doesn't manifest during Gap 3 itself (e.g., a future script
  that loads "the best model" without recording which of the 4 configs produced it).

**Phase to address:** Gap 3 implementation, specifically the `mlp_init_multi`/
`mlp_save`/`mlp_load`/`mlp_free` generalization sub-task — add the `make asan` Makefile
target (CONCERNS.md already recommends this) as part of this same sub-task so the new
code path is sanitizer-checked before the full 4-config × 5-fold sweep is run (an ASan
run on 1 fold takes seconds; the full sweep takes much longer per SPEC.md's own
"custo computacional não trivial" framing — cheaper to catch memory bugs before the
long run, not after).

---

### Pitfall 13: The confirmed global-RNG data race doesn't just corrupt augmentation noise — it poisons every subsequent "reproducible" random draw in the same process

**What goes wrong (deepening CONCERNS.md's finding):**
CONCERNS.md already confirms a real, unsynchronized data race on `rng_state`
(`src/utils.c:52`) during `precalculate_augmentations()`'s
`#pragma omp parallel for schedule(dynamic, 1)` loop (`src/main.c:137`, confirmed by
reading the current source at line 251 in the present HEAD). Reading `main.c`'s actual
call order makes the blast radius bigger than "just the noise-augmented samples":
`rng_seed(RANDOM_SEED)` runs once (`src/main.c:245`), immediately followed by
`kfold_split()` (deterministic, single-threaded, safe) on the *same* line, and then —
still before the fold loop — `precalculate_augmentations(&ds, fm.num_features,
aug_cache)` runs (`src/main.c:251`), which is the parallel, RNG-racing call. Every
single random draw that happens **after** this point in the same process — SMOTE's
`rng_int`/`rng_uniform` calls inside the fold loop (`src/main.c:222,224`), every
He-initialization `rng_normal()` call in `layer_init` (`src/mlp.c:147`) for every one of
the 30 (or, post-Gap-3, up to 120) `MLP` instances created for the run, and every
dropout-mask draw in `mlp_forward` (`src/mlp.c:266`) — all read from the *same* global
`rng_state`, which by the time the fold loop starts has already been left in a
thread-schedule-dependent (non-reproducible) state by the completed parallel
augmentation step. In other words, the bug is not confined to "the noise-augmented
audio samples are slightly different between runs" (CONCERNS.md's framing) — it means
**no random draw anywhere in the rest of the program, for the rest of the process, is
actually reproducible under `RANDOM_SEED=42`**, including SMOTE's synthetic-sample
choices and every network's weight initialization, both of which are exactly the kind
of randomness Gap 1/2/3's A/B comparisons need to hold constant between the "before" and
"after" run for a valid controlled comparison.

**Why this is the general class of bug (for future edits):**
Any global mutable state (`rng_state` here, but the same applies to any `static`
variable, or any struct field mutated through a pointer shared across threads) that is
read-modify-written without synchronization inside an OpenMP (or any threaded) parallel
region is a data race, full stop — the fact that it "doesn't crash" (as CONCERNS.md
notes: "x86 doesn't fault on unsynchronized 32-bit read/modify/write in practice") is
precisely what makes this bug class dangerous: it produces plausible, non-crashing,
silently-wrong (here: silently non-reproducible) output instead of an obvious failure.
**The general rule for this codebase going forward: before adding `#pragma omp parallel
for` around any loop body, grep that loop body (and everything it calls, transitively)
for calls to `rng_*`, or any other access to file-scope `static` state — if found, the
loop is not safely parallelizable as written**, regardless of how unrelated the new
loop's purpose seems to be to "randomness." Gap 2 (Borderline-SMOTE's new
`find_knn_global()` computing `m` for every minority sample) and Gap 3 (training up to
120 `MLP` instances, each calling `rng_normal()` during `layer_init` and `rng_uniform()`
during dropout) are both areas where a well-intentioned future optimization ("let's
`#pragma omp parallel for` the per-config or per-fold training loop since configs are
independent") would reintroduce exactly this bug, at larger scale, if attempted without
first giving each thread its own RNG stream.

**How to avoid:**
Do not fix this as part of Gap 1/2/3 unless a gap's acceptance criteria explicitly
depend on run-to-run bitwise reproducibility being restored (worth flagging to whoever
prioritizes work, since it currently undermines the "same seed, same folds" comparison
protocol all three gaps rely on per SPEC.md's own text: "Toda mudança é validada por
comparação A/B com os mesmos folds... nunca por inspeção visual"). If/when it is fixed,
CONCERNS.md's two proposed options remain the right choices: (a) simplest — drop
`#pragma omp parallel for` from `precalculate_augmentations` (this loop is not the
pipeline's bottleneck; extraction and training dominate wall-clock time), restoring full
determinism at a modest, one-time speed cost; or (b) give each thread (or, more
robustly, each patient index, so the result doesn't depend on the number of OpenMP
threads at all — thread-count-independence is a stronger and more useful property than
mere thread-safety) its own deterministic RNG stream derived from `RANDOM_SEED +
patient_index`, avoiding any shared mutable state. Regardless of which fix is chosen (or
if it is deferred), **document in `results/train_log_vXX_<gap>.txt` for every A/B
comparison run for Gap 1/2/3 that the audio-augmentation stage's exact numeric output is
not currently seed-reproducible**, so a reviewer (human or future agent) doesn't
mistakenly treat "same `RANDOM_SEED=42`" as a guarantee of bit-identical inputs across
the "before" and "after" runs being compared — the *feature values fed into the MLP
after augmentation* have a race-dependent component today, even though
`results/features.csv` itself (pure extraction, no RNG) remains correctly deterministic
per CONCERNS.md.

**Warning signs:**
- Run `make clean && make full` twice in a row with `OMP_NUM_THREADS` fixed, and diff
  `results/metrics_global.csv` — CONCERNS.md predicts these can already differ between
  runs today; if you're about to trust a "Macro F1 improved by 0.01" conclusion for
  Gap 1/2/3, first confirm that re-running the *unchanged baseline* twice produces a
  smaller difference than the improvement you're crediting to the gap.
- Run with `OMP_NUM_THREADS=1` vs. the default (multi-core) and diff results — if they
  differ even with the same seed, that's the race's fingerprint (thread-count-dependent
  output is the classic symptom of unsynchronized shared state under OpenMP).

**Phase to address:** Not gated to any single gap — flag as a cross-cutting
methodological caveat to state explicitly in the final report regardless of whether it
gets fixed, and treat "grep for `rng_*` calls before parallelizing" as a standing review
checklist item for every future `#pragma omp` addition (Gap 2's `find_knn_global`
computation over many samples is a plausible future parallelization target that would
need this same scrutiny if someone tries to speed it up later — it does not call `rng_*`
today as specified, so it is safe *as specified*, but would stop being safe the moment
someone "helpfully" adds randomized sampling to speed up the O(n²) neighbor search
CONCERNS.md already flags as a performance bottleneck).

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| `find_knn`/`find_knn_global` O(n²) distance recompute (CONCERNS.md, deepened for Gap 2) | SMOTE/Borderline-SMOTE stage takes noticeably longer once `find_knn_global` (all-class, not just same-class) is added for every minority sample's m-count | Precompute a pairwise distance matrix or k-NN adjacency list once per class/fold before the synthesis loop; this becomes strictly more important once Gap 2 adds a second, all-class neighbor search on top of the existing per-class one | Not a problem at current class sizes (≤700/class); would matter if dataset grows 5-10x or if Gap 3's 4-config sweep multiplies the number of times SMOTE runs per `make full` invocation (4x more networks trained = 4x more SMOTE calls) |
| Paraconsistent selection re-run per (fold × vowel × network) = up to 30 independent O(n·nf·C) passes | `make full` wall-clock time increases proportionally to nf (237) × n_classes computations per fold, on top of existing extraction/training time | Nothing to prevent per se (this is the correct, intended computational cost) — but budget for it explicitly when scheduling Gap 1 last (per SPEC's own recommended order) so its added wall-clock time doesn't collide with time-boxed academic deadlines | Becomes noticeable once combined with Gap 3's 4-config sweep if both gaps are combined into the same run (30 selection calls × 4 configs' worth of retraining, if selection is naively redone per config instead of cached) |
| 4-config × 5-fold × 3-vowel × 2-network Gap 3 sweep = up to 120 full training runs per `make full` | Total wall-clock time for `make full` increases roughly 4x versus the current single-config baseline (already documented as "custo computacional não trivial" in SPEC.md) | Run Gap 3's sweep as its own dedicated `make` target/mode distinct from the default `make full`, so iterating on Gap 1/Gap 2 during their own development doesn't pay Gap 3's full 4x cost every time | Immediately relevant — this is the primary reason SPEC.md orders Gap 3 before Gap 1 (avoid re-tuning everything under all 4 configs) |

---

## "Looks Done But Isn't" Checklist

- [ ] **Borderline-SMOTE (Gap 2):** Often "done" once it compiles and produces a
  different Macro F1 number — verify the safe/borderline/noise boundary uses `2*m >= k`
  (not truncated `m >= k/2`), verify interpolation neighbors are re-searched per-class
  (not reused from the global m-counting search), and verify the A/B comparison used
  bootstrap CI/McNemar (Pitfall 4), not a single point-estimate diff.
- [ ] **Paraconsistent feature selection (Gap 1):** Often "done" once it selects a
  non-trivial subset of features without crashing — verify a `MIN_VARIANCE`-style guard
  exists (Pitfall 6), verify the relaxation loop has a hard iteration cap and a loud
  failure path (Pitfall 7), and verify the reported "feature table" is a
  selection-frequency aggregate across all 30 fold/vowel/network runs, not one run's
  snapshot (Pitfall 8).
- [ ] **Shallow-vs-deep comparison (Gap 3):** Often "done" once all 4 configs train
  without crashing on the current dataset size — this is exactly the failure mode where
  Pitfall 9 (fixed `layers[MLP_NUM_LAYERS]` array) can silently corrupt memory without
  crashing on a given run/machine/allocator state and produce plausible-looking but
  subtly wrong numbers; verify with an ASan/Valgrind build (`make asan`, currently
  missing per CONCERNS.md) before trusting any Config C/D result.
- [ ] **Any A/B comparison across the 3 gaps:** Often "done" once one run's
  `results/train_log_vXX.txt` shows a better Macro F1 than the baseline — verify the
  baseline was *also* re-run under the exact same code/seed close in time (not compared
  against an old, possibly stale, log from a different commit), and verify
  reproducibility itself hasn't silently degraded further due to Pitfall 13 (RNG race)
  making even the "baseline" non-reproducible run-to-run.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|-----------------|
| Pitfall 1 (k/2 truncation) | LOW | One-line fix (`2*m >= k`); re-run the A/B comparison for Gap 2 from scratch — cheap since it's an isolated function |
| Pitfall 3 (empty borderline set / degenerate SMOTE) | LOW-MEDIUM | Add the logging/guard retroactively; re-run only the affected folds/classes if logs show the fallback fired — full re-run only needed if the silent duplication already contaminated an "adopted" result |
| Pitfall 5 (feature-selection data-slice asymmetry) | MEDIUM | Requires re-deriving μ/λ with the corrected data slice and re-running all 30 fold/vowel/network selections — moderate cost since Gap 1 is scheduled last (per SPEC's order), so no downstream work depends on the wrong subset yet if caught early |
| Pitfall 9/10 (struct sizing / delta buffer) | MEDIUM-HIGH if undetected until after the full Gap 3 sweep | If caught via ASan before the sweep: near-zero cost (fix + re-run). If caught only after a full 120-model sweep produced silently-corrupted results: discard all Config C/D results and re-run — this is exactly why the Recovery Steps recommend running ASan on one fold/config first (see Pitfall 12) |
| Pitfall 13 (RNG race) | HIGH to fully fix (touches a documented "what works" assumption), LOW to just document as a caveat | Full fix requires re-validating that removing OpenMP from `precalculate_augmentations` doesn't unacceptably slow `make full`, or implementing per-thread/per-index RNG streams and re-validating all downstream "what works"/"what doesn't work" experiments in CLAUDE.md that were run under the racy version — recommend documenting-as-caveat now, fixing later as dedicated tech-debt work outside the 3 gaps' scope |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|--------------|
| 1. k/2 boundary truncation | Gap 2 implementation | Unit-check `k=5, m=2` classifies as intended before running any A/B |
| 2. Wrong-class interpolation neighbor | Gap 2 implementation, code review | Spot-check synthetic sample parent classes match |
| 3. Empty borderline set / degenerate SMOTE | Prerequisite fix + Gap 2 | Fallback events are logged and reviewed per class/fold |
| 4. Missing statistical A/B infra | Cross-cutting prerequisite (before any gap's A/B run) | `metrics_bootstrap_ci`/`metrics_mcnemar` call sites exist and are exercised in every `train_log_vXX_<gap>.txt` |
| 5. Feature-selection data-slice asymmetry | Gap 1 design step | Log effective per-class N fed into `paraconsistent_select`; confirm it matches the documented choice |
| 6. Fisher-ratio divide-by-zero | Gap 1 implementation | Log count of variance-clamped (feature, class) pairs per fold; confirm no NaN in `μ`/`Gc` arrays |
| 7. Unbounded relaxation loop | Gap 1 implementation (same task as #6) | Relaxation iteration count logged per fold; hard cap + loud failure path present |
| 8. Misleading single-run feature table | Gap 1 reporting step (persistence wired from Gap 1 start) | Final table shows selection frequency across all 30 runs, not one |
| 9. Fixed `layers[MLP_NUM_LAYERS]` array | Gap 3 prerequisite (before Config C/D) | ASan-clean run of Config D on at least one fold/vowel/network |
| 10. Hardcoded backprop delta buffer size | Gap 3 prerequisite (same commit as #9) | `max_size` computed from `net->layers[i].output_size`, not a macro |
| 11. Unfair hyperparameter reuse across depths | Gap 3 comparison-protocol design step | Report explicitly states hyperparameters were held fixed; early-stopping epoch counts reviewed for premature-stop bias |
| 12. Memory management generalizing to N layers | Gap 3 implementation | `make asan` target added and passes for all 4 configs before the full sweep runs |
| 13. Global RNG race under OpenMP | Cross-cutting; document now, fix as separate tech debt | Two consecutive `make clean && make full` runs (fixed `OMP_NUM_THREADS`) diffed; caveat stated in every A/B log for Gap 1/2/3 |

---

## Sources

- `.planning/codebase/CONCERNS.md` (2026-07-27 audit) — primary source for the confirmed
  RNG data race, the `mlp_evaluate()` n==0 guard gap, the hardcoded `xv[251]` buffer, the
  disconnected bootstrap-CI/McNemar infrastructure, the `n_class <= 1` SMOTE
  degeneration, and the `mlp_backward` delta-buffer sizing note — all cross-referenced
  and deepened above rather than repeated verbatim.
- `SPEC.md` — algorithm specifications and acceptance criteria for all three gaps,
  directly quoted where its own wording anticipates or misses a pitfall.
- `CLAUDE.md` — "Methodological Notes" (`norm_fit`/`n_train` vs `n_train_aug` rule),
  "Current Best Results" (fold-to-fold variance figures used in Pitfall 4/8/11), "What
  works/doesn't work" (used in Pitfall 11's regularization framing).
- Direct source reads (this session): `src/main.c` (SMOTE, `find_knn`,
  `precalculate_augmentations`, `mode_train` fold loop, RNG seeding order),
  `src/mlp.c` (struct layout, `mlp_backward`, `mlp_init_dynamic`, `layer_init`,
  `mlp_save`/`mlp_load`), `include/mlp.h` (`Layer layers[MLP_NUM_LAYERS]` struct
  definition), `include/config.h` (`MLP_NUM_LAYERS`, `MLP_HIDDEN*_SIZE`, `L2_LAMBDA`,
  class weights), `src/normalize.c` (`MIN_STD` guard precedent for Pitfall 6),
  `src/utils.c` (`xorshift32`/`rng_state` implementation for Pitfall 13).
- [Han, Wang & Mao (2005), Borderline-SMOTE — algorithm summary via Scientific Research
  Publishing reference listing](https://www.scirp.org/reference/referencespapers?referenceid=1603045)
  — MEDIUM confidence, used to confirm the `m/k` safe/danger/noise boundary convention
  (no full-text primary access; corroborated by multiple independent secondary
  descriptions returned in the same search, including
  [themis R package's `bsmote` documentation](https://themis.tidymodels.org/reference/bsmote.html)
  and a [TDS Archive summary of Borderline-SMOTE variants](https://medium.com/data-science/class-imbalance-from-smote-to-smote-n-759d364d535b)).
- General OpenMP/parallel-RNG reproducibility principles (MEDIUM confidence, WebSearch,
  used only to confirm the general bug class described in Pitfall 13, not any
  project-specific claim): [dqrng parallel RNG usage vignette](https://cran.r-project.org/web/packages/dqrng/vignettes/parallel.html),
  [OpenRAND: reproducible parallel RNG library paper](https://arxiv.org/pdf/2310.19925),
  [Intel Community: keeping the same seeds for random numbers with OpenMP loop](https://community.intel.com/t5/Intel-Fortran-Compiler/Keeping-the-same-seeds-for-random-numbers-with-OpenMP-loop/td-p/1141087).

---
*Pitfalls research for: vocal-anomaly-detection C99 MLP pipeline — Gap 1 (paraconsistent
feature selection), Gap 2 (Borderline-SMOTE), Gap 3 (shallow-vs-deep MLP comparison)*
*Researched: 2026-07-27*
