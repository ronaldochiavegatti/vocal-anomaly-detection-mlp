# Phase 1: Gap 2 — Borderline-SMOTE - Research

**Researched:** 2026-07-27
**Domain:** Imbalanced-data oversampling (Borderline-SMOTE1, Han/Wang/Mao 2005) implemented as a hand-rolled C99 extension of an existing SMOTE function, plus a reproducible statistical A/B comparison protocol, in a pure-C academic ML pipeline with zero external dependencies.
**Confidence:** HIGH — every claim about the current codebase is verified by direct source reads (`src/main.c`, `src/kfold.c`, `src/utils.c`, `include/metrics.h`). The Borderline-SMOTE1 algorithm itself is verified against the primary source (Han, Wang & Mao, 2005, read in full by prior project research in `.planning/research/STACK.md`). No new external research was needed for the algorithm — this phase's risk is entirely in faithful C implementation and rigorous A/B protocol design, not in discovering an unfamiliar library or technology.

## Summary

Gap 2 modifies exactly one existing function, `smote_oversample()` (`src/main.c:205-231`), called from exactly two call sites inside `mode_train()`'s per-vowel loop (Master 2-class at `src/main.c:318`, Expert 4-class at `src/main.c:335`). `find_knn_global()` — the function named in every requirement (SMOTE-01/02) and in `SPEC.md` — **does not exist anywhere in the codebase today** (confirmed by grep across `src/` and `include/`); it must be written from scratch as a sibling to the existing `find_knn()` (`src/main.c:189-203`), which stays unmodified and is reused, unchanged, for same-class interpolation. This is a genuinely low-risk phase: no struct changes, no change to loop nesting, no change to `mlp_init_dynamic()` call arguments, and (per prior architecture research) no drift between what `SPEC.md` proposed and what the current source actually looks like.

The two things that make this phase easy to get subtly wrong, both already diagnosed by prior project research (`.planning/research/PITFALLS.md` Pitfalls 1-3) and repeated here with implementation-ready detail: (1) the safe/borderline/noise boundary must use `2*m >= k` comparison logic, never `m >= k/2` with integer truncation, and the `m == k` "noise" case must be checked as its own branch, not folded into the `2*m >= k` inequality; and (2) the neighbor list used to classify a sample (searched across **all** classes via the new `find_knn_global()`) must never be reused for the neighbor used to interpolate a synthetic sample (searched within **the same class only**, via the existing, unmodified `find_knn()`) — these are two structurally distinct searches over two different candidate sets and must remain two separate function calls, not a shared/deduplicated helper.

The bigger open design question this research surfaces — not fully specified by `REQUIREMENTS.md`/`ROADMAP.md`/`SPEC.md` — is **how the "same seed, same folds" A/B comparison is actually invoked**. There is no CLI flag mechanism beyond `<mode> [base_dir]` today (Phase 0 added exactly one precedent for this: a new `verify-rng` CLI mode for a fast, dedicated check). This research recommends threading a `SmoteMode` parameter through `mode_train()` itself and adding a new CLI mode (e.g. `smote-ab`) that runs the entire fold loop twice in one process invocation — once per mode — reusing the project's own established RNG-reset-via-`kfold_split()` behavior to guarantee bit-identical fold assignments across both arms. This also lets the phase produce the SMOTE-04 A/B report from a single command instead of manually diffing two separate 60-90 minute `make full` logs.

**Primary recommendation:** Add `SmoteMode` (`SMOTE_STANDARD`/`SMOTE_BORDERLINE`) as a parameter to `smote_oversample()` directly (per `ROADMAP.md`'s success-criterion wording — not `SPEC.md`'s originally-sketched `smote_oversample_ex()` rename, which predates the finalized requirements); add a new `find_knn_global()` function that performs an all-class k-NN search with **no RNG calls** (preserving the project's RNG-draw-count parity between the two SMOTE modes); classify safe/borderline/noise with `m == k → NOISE`, else `2*m >= k → BORDERLINE`, else `SAFE`; keep interpolation on the existing unmodified `find_knn()` restricted to `class_idx_borderline[c]` (falling back to `class_idx[c]` with a logged warning when the borderline pool is empty); and drive the A/B comparison through a new dedicated CLI mode that runs both arms in one process and writes the comparison report to `results/train_log_v32_gap2_smote_ab.txt` (next available version number — v30/v31/v31_intel are already consumed by the discarded "nested stacked hierarchy" WIP experiment per `git status`, do not reuse those numbers).

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SMOTE-01 | `smote_oversample()` gains a `SmoteMode {SMOTE_STANDARD, SMOTE_BORDERLINE}` parameter; borderline mode uses a new `find_knn_global()` (all-class k-NN) to classify each minority sample safe/borderline/noise using the untruncated rule `2*m >= k` (never `m >= k/2` integer division) | Architecture Patterns (Pattern 1, code sketch of `find_knn_global()`/`classify_borderline()`); Common Pitfalls (Pitfall 1); Sources (Han/Wang/Mao 2005 Step 1, verified against primary source) |
| SMOTE-02 | Synthetic-sample interpolation always uses the existing `find_knn()` restricted to same-class neighbors — the global k-NN list from SMOTE-01's classification step is never reused for interpolation | Architecture Patterns (Pattern 1, system diagram distinguishing the two search calls); Common Pitfalls (Pitfall 2); Anti-Patterns |
| SMOTE-03 | Empty borderline pool, and the pre-existing `n_class <= 1` case, both produce an explicit warning log with no crash and no degenerate synthetic samples | Common Pitfalls (Pitfall 3); Open Questions (Q1 — exact semantics of "no degenerate synthetic samples" flagged as a decision the plan must pin down) |
| SMOTE-04 | `results/` contains an A/B report (standard vs. borderline SMOTE, same seed/folds): Macro F1 + per-class F1 table with McNemar/bootstrap CI, a safe/borderline/noise count table per class/fold, and an explicit adopt/reject decision sentence | Architecture Patterns (Pattern 2, `mode_smote_ab()` orchestration); Don't Hand-Roll (reuse of Phase-0-wired `metrics_bootstrap_ci`/`metrics_mcnemar`); Open Questions (Q2 — single-process vs. two-invocation A/B design) |
| SMOTE-05 | `CLAUDE.md` updated (Optimization History / What Worked / What Didn't Work) with the Gap 2 outcome, regardless of which mode was adopted | State of the Art (existing `CLAUDE.md` drift already misclaims Borderline-SMOTE was validated — must be corrected, not just appended to, once the real A/B result is known) |
</phase_requirements>

## Architectural Responsibility Map

This is a single-process C99 CLI pipeline, not a multi-tier web app — "tiers" here map to the project's own documented architectural layers (`CLAUDE.md` "Layers" / `.planning/research/ARCHITECTURE.md`).

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `SmoteMode` enum + CLI/mode plumbing | Orchestration (`src/main.c`, `main()`/`mode_train()`) | — | Mode selection is a run-level choice, not per-fold/per-vowel state; belongs where CLI dispatch already lives |
| Safe/borderline/noise classification (`find_knn_global()`) | Fold/Resampling layer (`src/main.c` SMOTE section) | — | New sibling function to existing `find_knn()`/`smote_oversample()`, same file/section, no new module needed |
| Same-class interpolation (existing `find_knn()`) | Fold/Resampling layer | — | Unchanged — reused as-is, per SMOTE-02's explicit "never reuse the global list" requirement |
| Per-class/fold safe/borderline/noise count tracking | Fold/Resampling layer (computation) | Metrics/Reporting layer (aggregation + CSV export) | Counts are a byproduct of classification; exporting them as a table is a reporting concern |
| A/B statistical comparison (McNemar, bootstrap CI) | Metrics/Reporting layer (`src/metrics.c`, already wired in Phase 0) | Orchestration (`mode_train()` call sites) | Infrastructure already exists and is already connected (`INFRA-01`, Phase 0 complete) — Gap 2 only needs to invoke it twice (once per mode) and diff the two result sets |
| A/B report + adopt/reject decision | Reporting/Documentation layer (`results/*.csv`, `results/train_log_v32_*.txt`, `CLAUDE.md`) | — | Pure output-artifact concern, no new runtime logic |

## Standard Stack

**No external stack applies.** This is a 100% pure C99 codebase with zero third-party dependencies (`libm`, `libgomp`/OpenMP, and the C standard library only — per `CLAUDE.md` "Key Dependencies"). Nothing is installed via a package manager (there is no package manager for this project). The "stack" for this phase is exclusively existing in-repo code:

| Component | Location | Role for Gap 2 |
|-----------|----------|-----------------|
| `find_knn()` | `src/main.c:189-203` | Unchanged — reused for same-class interpolation only |
| `smote_oversample()` | `src/main.c:205-231` | Modified — gains `SmoteMode` parameter |
| `rng_int()`, `rng_uniform()` | `src/utils.c:76-80,71-74` | Unchanged — consumed by SMOTE's synthesis loop; each call advances the global RNG stream by exactly one `xorshift32()` draw regardless of argument, which is why draw-count parity between modes is achievable (see Pitfalls) |
| `kfold_split()` | `src/kfold.c:35-37` | Unchanged — calls `rng_seed(seed)` internally at its own start, which is the mechanism this research recommends leaning on to guarantee identical fold assignments across the two A/B arms |
| `metrics_bootstrap_ci()`, `metrics_mcnemar()` | `src/metrics.c` (wired into `mode_train()` in Phase 0) | Reused as-is — call twice (once per SMOTE mode) on each arm's aggregated out-of-fold predictions |
| `log_warn()` | `include/utils.h:60` | Reused — required by SMOTE-03 for the empty-pool and `n_class <= 1` fallback warnings |

### Alternatives Considered
None — the phase explicitly targets a single, named algorithm (Borderline-SMOTE1, Han/Wang/Mao 2005) inside a single existing function. There is no library alternative to evaluate; the only "alternative" ever relevant is standard SMOTE, which remains available as `SMOTE_STANDARD` for the A/B comparison itself, not as a competing implementation choice.

**Installation:** None required — no new dependency, no new file, pure in-place C source edits to `src/main.c` (and possibly a small addition to `main()`'s dispatch table and `include/`-level constants if `BORDERLINE_M_NEIGHBORS`/`SMOTE_K_NEIGHBORS` are pulled into `config.h`, matching the codebase's "all tunables live in `config.h`" convention).

## Package Legitimacy Audit

**N/A — no packages are installed in this phase.** Zero external dependencies, per `CLAUDE.md` Tech Stack ("no `pip`, `npm`, `cargo`... All dependencies are system libraries linked directly by the compiler"). This section is intentionally empty; the Package Legitimacy Gate protocol does not apply to a pure-C99, dependency-free codebase.

## Architecture Patterns

### System Architecture Diagram (Gap 2 scope only)

```
mode_train(base_dir, smote_mode)              [existing, gains a parameter]
  │
  ├─ rng_seed(RANDOM_SEED) + kfold_split()     [existing — internally reseeds RNG,
  │                                              this is what guarantees identical
  │                                              fold assignments across A/B arms]
  │
  ├─ precalculate_augmentations()              [existing, Phase-0-fixed, single-
  │                                              threaded, deterministic — runs
  │                                              identically regardless of smote_mode]
  │
  └─ for each fold (5):
       for each vowel (3):
         ├─ build tr_x_v / tr_y_bin (Master, 2-class)   [existing, unchanged]
         │    └─ smote_oversample(..., smote_mode)       [MODIFIED — new mode param]
         │         │
         │         ├─ if SMOTE_BORDERLINE:
         │         │    for each sample in class c needing synthesis:
         │         │      find_knn_global(x_in, i, ALL n_in samples, k=5)  [NEW FUNCTION]
         │         │        → count m = neighbors NOT in class c
         │         │        → classify: m==k→NOISE, 2*m>=k→BORDERLINE, else→SAFE
         │         │    class_idx_borderline[c] = { i : classified BORDERLINE }
         │         │    if empty: log_warn(...); fall back to class_idx[c]
         │         │
         │         └─ synthesis loop (existing structure, base pool changes only):
         │              base_idx = borderline_pool[c][rng_int(pool_size)]   [POOL CHANGES:
         │                                                                    class_idx_borderline[c]
         │                                                                    instead of class_idx[c]]
         │              find_knn(x_in, base_idx, class_idx[c], ..., k=5, neighbors)  [UNCHANGED —
         │                                                                             same-class only]
         │              alpha = rng_uniform(); interpolate                    [UNCHANGED]
         │
         ├─ build ex_tr_x / ex_tr_y (Expert, 4-class)     [existing, unchanged]
         │    └─ smote_oversample(..., smote_mode)         [same modification, 2nd call site]
         │
         └─ mlp_train(...) × 2 (Master + Expert)           [existing, fully unchanged]

  → aggregate all_y_true / all_y_pred per mode
  → metrics_bootstrap_ci() + metrics_mcnemar() per mode    [existing infra, Phase 0]
  → NEW: diff the two modes' Macro F1 / per-class F1 / CI, safe-borderline-noise
    count table → results/train_log_v32_gap2_smote_ab.txt + CSV artifacts
  → NEW: explicit adopt/reject decision sentence
```

### Recommended Project Structure

No new files. All changes land in `src/main.c` (and, if the codebase's "everything tunable lives in `config.h`" convention is followed strictly, two new constants in `include/config.h`: `BORDERLINE_M_NEIGHBORS` and `SMOTE_K_NEIGHBORS`, both defaulting to 5 but named independently per the Stack research finding that these are two different parameters in the original paper even though this implementation starts them at the same value).

```
src/main.c
├── typedef enum { SMOTE_STANDARD = 0, SMOTE_BORDERLINE = 1 } SmoteMode;   [NEW]
├── find_knn()                          [UNCHANGED — src/main.c:189-203]
├── find_knn_global()                   [NEW — sibling function, same section]
├── smote_oversample()                  [MODIFIED — gains SmoteMode parameter]
├── mode_train()                        [MODIFIED — gains SmoteMode parameter,
│                                         threaded to both smote_oversample() call sites]
├── mode_smote_ab()                     [NEW — runs mode_train() twice, once per
│                                         mode, produces the comparison report]
└── main()                              [MODIFIED — one new dispatch line for
                                          the new CLI mode, mirroring "verify-rng"]
```

### Pattern 1: Two-phase k-NN search (classification vs. interpolation)

**What:** Compute two conceptually different k-NN queries against two different candidate sets from the same base point, and never let one substitute for the other.
**When to use:** Any time an algorithm needs "which class does this point's neighborhood suggest" (global search) as a *gate*, separate from "which same-class point do I blend with" (restricted search) as the actual *generation* mechanism. This is the structural core of Borderline-SMOTE1 vs. plain SMOTE.
**Example** (adapted from Han, Wang & Mao 2005, Step 1 — classification — verified against primary source in `.planning/research/STACK.md`. **Note: this code sketch is illustrative pseudocode only, not literal text to commit verbatim** — comments actually committed to `src/main.c` must be fully Portuguese per `CLAUDE.md`'s Comments convention; see 01-01-PLAN.md's Task 1 for the exact Portuguese wording to use):
```c
/* Classification: search ALL n_in samples, count non-same-class neighbors.
 * NO rng_* calls in this function -- preserves RNG draw-count parity between
 * SMOTE_STANDARD and SMOTE_BORDERLINE runs (see Pitfalls, RNG stream parity). */
static int find_knn_global(const float *x, const int *y, int base, int n_in,
                            int nf, int k, int *neighbors)
{
    float *dists = (float *)safe_malloc(n_in * sizeof(float));
    int *order = (int *)safe_malloc(n_in * sizeof(int));
    for (int i = 0; i < n_in; i++) {
        order[i] = i;
        if (i == base) { dists[i] = 1e30f; continue; }
        float dist = 0.0f;
        for (int f = 0; f < nf; f++) {
            float diff = x[base * nf + f] - x[i * nf + f];
            dist += diff * diff;
        }
        dists[i] = dist;
    }
    int kk = (k < n_in - 1) ? k : n_in - 1; if (kk < 1) kk = 1;
    for (int i = 0; i < kk; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n_in; j++)
            if (dists[order[j]] < dists[order[min_idx]]) min_idx = j;
        int tmp = order[i]; order[i] = order[min_idx]; order[min_idx] = tmp;
        neighbors[i] = order[i];
    }
    free(dists); free(order);
    return kk; /* actual neighbor count used (may be < k for tiny n_in) */
}

/* Safe/borderline/noise classification -- 2*m >= k, never m >= k/2 truncated.
 * m == k checked FIRST as its own branch (2*m>=k is also true when m==k,
 * so order of these checks matters). */
static int classify_borderline(int m, int k)
{
    if (m == k) return 2;       /* NOISE */
    if (2 * m >= k) return 1;   /* BORDERLINE (danger) */
    return 0;                   /* SAFE */
}
```

### Pattern 2: RNG-stream parity across A/B arms via existing `kfold_split()` reseed

**What:** `kfold_split()` (`src/kfold.c:35`) calls `rng_seed(seed)` internally at its own entry point, before any shuffling. Because of this, calling `mode_train()`/the fold-split step fresh for each SMOTE mode — rather than trying to snapshot/restore RNG state manually — is sufficient to guarantee both A/B arms see identical fold assignments, as long as no `rng_*` call happens between `main()` startup and this point that differs between the two arms.
**When to use:** Any time this codebase needs two runs to be "on the same seed, same folds" — this is the existing, already-correct mechanism; do not invent a second one.
**Example** (structuring the new `mode_smote_ab()`):
```c
static int mode_smote_ab(const char *base_dir)
{
    log_info("=== MODO: A/B BORDERLINE-SMOTE (Gap 2) ===");
    ABResult res_standard, res_borderline;
    if (mode_train_ex(base_dir, SMOTE_STANDARD, &res_standard) != 0) return -1;
    if (mode_train_ex(base_dir, SMOTE_BORDERLINE, &res_borderline) != 0) return -1;
    /* Both calls independently call kfold_split(fm.labels, fm.count, RANDOM_SEED, &splits)
     * internally, which reseeds the global RNG to RANDOM_SEED at that exact point --
     * the two runs' fold assignments are therefore guaranteed identical regardless of
     * what happened earlier in the process. */
    write_ab_report(&res_standard, &res_borderline, "results/train_log_v32_gap2_smote_ab.txt");
    return 0;
}
```

### Anti-Patterns to Avoid

- **Reusing the `find_knn_global()` neighbor list for interpolation:** The single most tempting "optimization" in this phase — both searches start from the same base point, so it looks redundant to compute k-NN twice. Doing so silently produces cross-class synthetic samples (label noise) with no crash. Keep the two calls textually distinct (SMOTE-02, Pitfall 2).
- **`m >= k/2` with integer division:** Compiles cleanly, reads as correct, silently truncates `k/2` to 2 for the standard `k=5`, shifting the safe/danger boundary. Always compare `2*m >= k` (Pitfall 1).
- **Treating the "relax to standard SMOTE on empty borderline pool" fallback as a citable part of Han/Wang/Mao (2005):** It is not in the original paper (confirmed in `.planning/research/STACK.md`) — it is a necessary engineering extension for this project's very small minority classes. Label it as such in any report; do not attribute it to the cited authors.
- **Adding `#pragma omp parallel for` to `find_knn_global()`'s per-sample loop to speed it up:** `find_knn_global()` itself calls no `rng_*` functions (by design, see Pattern 1), so it is safe to parallelize *today* — but if a future change adds any randomized sampling to speed up the O(n²) search, this would reintroduce exactly the RNG data race Phase 0 just fixed elsewhere (Pitfall 13 in `.planning/research/PITFALLS.md`). Flag this in code comments as a standing constraint, mirroring the comment already left in `precalculate_augmentations()`.
- **Citing generic "Borderline-SMOTE"** instead of the specific variant implemented: this codebase implements **Borderline-SMOTE1** specifically (same-class-only interpolation neighbors). Borderline-SMOTE2 (which also draws from majority-class neighbors with a capped interpolation ratio) is a different, unimplemented variant — do not imply it was used.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical significance between two model variants | A custom p-value/significance calculation for the A/B comparison | `metrics_mcnemar()` (`src/metrics.c`, already implemented and wired into `mode_train()` since Phase 0) | Already correct (Edwards continuity correction), already tested by the Phase 0 baseline reconfirmation, avoids re-deriving McNemar's test from scratch |
| Confidence intervals on Macro F1 / per-class F1 | A custom bootstrap loop | `metrics_bootstrap_ci()` (`src/metrics.c`, N=1000, seed=`RANDOM_SEED`, already wired) | Same infrastructure Phase 0 already validated against the reconfirmed baseline; reusing it for Gap 2 is exactly what Phase 0 was built to enable |
| Deterministic, seed-controlled fold splitting for the A/B comparison | A manual snapshot/restore of RNG state between the two arms | `kfold_split()`'s existing internal `rng_seed(seed)` reseed (`src/kfold.c:35`) | Already deterministic and already the established mechanism in this codebase; manually managing RNG state around it is unnecessary complexity that could itself introduce a new reproducibility bug |
| Fast, low-cost "did this change work" verification before committing to a 60-90 min full pipeline run | A shell script wrapping partial pipeline stages | A dedicated CLI mode, following the `verify-rng` precedent from Phase 0 (`mode_verify_rng()`, `src/main.c:446-466`) | This codebase already established the pattern of "no test framework exists, so add a small dedicated CLI mode for fast verification" — Gap 2 should follow the same convention for its own comparison (`mode_smote_ab()`), not invent a new verification style |

**Key insight:** Everything Gap 2 needs from a "statistical rigor" perspective was already built and wired in Phase 0 specifically so Gaps 1-3 would not need to reinvent it. The only genuinely new code in this phase is the Borderline-SMOTE1 classification/generation logic itself (~40-60 lines) and the orchestration to run it twice under controlled conditions.

## Common Pitfalls

*(Full detail already researched and documented in `.planning/research/PITFALLS.md` — summarized here with the specific fix each requires; do not re-derive independently, this section is the actionable digest for planning.)*

### Pitfall 1: Integer-division truncation silently shifts the safe/danger boundary
**What goes wrong:** `if (m >= k/2)` truncates `k/2` to `2` for `k=5` instead of `2.5`, misclassifying a 2-of-5-majority sample as borderline when the paper's rule (`>= 50%`) would call it safe.
**How to avoid:** Compare `2*m >= k`. Check `m == k` (noise) as a separate, earlier branch — `2*m >= k` is also true when `m == k`, so branch order matters.
**Warning signs:** A unit-style check with `k=5, m=2` classifying as borderline is the truncation signature.

### Pitfall 2: Conflating the classification neighbor list with the interpolation neighbor list
**What goes wrong:** Reusing `find_knn_global()`'s result for the actual synthesis step produces synthetic samples that are convex combinations of two *different* classes' feature vectors, silently injected as the minority label — cross-class label noise with no crash.
**How to avoid:** Keep `find_knn_global()` (all-class, classification only) and `find_knn()` (same-class, interpolation only, unchanged) as textually and structurally separate calls. Do not deduplicate into a shared helper unless it takes an explicit "restrict to these indices" parameter that differs per call site.
**Warning signs:** Spot-check a handful of generated synthetic samples per fold/class — both parent points (base + interpolation neighbor) must share the same original class label. A *worse* Macro F1 for Disfonia Psicogênica/Funcional under borderline mode vs. standard mode is a plausible symptom (these two classes are already the acoustically-overlapping ones per `CLAUDE.md`'s "Fundamental Bottleneck" — cross-class interpolation would specifically blur that exact boundary further).

### Pitfall 3: Empty borderline pool compounds with the pre-existing `n_class <= 1` SMOTE bug
**What goes wrong:** `class_idx_borderline[c]` can legitimately end up empty if every sample of a small class (e.g. Edema de Reinke, n≈68, further reduced by 5-fold × train/val × 4-class-Expert split to potentially single digits per fold/vowel) is classified NOISE. SPEC's own fallback ("fall back to `class_idx[c]`") re-enters the CONCERNS.md-documented `n_class <= 1` degeneration (the only "neighbor" is the point itself, producing an exact duplicate labeled as a synthetic sample) — the two bugs can now trigger together.
**How to avoid:** `SMOTE-03` explicitly requires both cases to log an explicit warning and avoid crash/degenerate output. Recommended interpretation (see Open Questions below for the exact semantics to pin down during planning): log a `log_warn()` line identifying class/fold/vowel/mode whenever (a) `class_idx_borderline[c]` is empty and the code falls back to `class_idx[c]`, and (b) `n_class <= 1` for a class needing synthesis.
**Warning signs:** Grep training logs for the fallback warning per class/fold — if it fires for Reinke or Disfonia Psicogênica on most folds, borderline-SMOTE is not actually doing borderline selection for those classes most of the time; this must be reported as a limitation in the A/B report (SMOTE-04's safe/borderline/noise count table makes this visible directly).

### Pitfall 4 (new, this session): RNG-draw-count parity between the two SMOTE modes is achievable and worth preserving
**What goes wrong (if not preserved):** If `find_knn_global()` or any new code path introduced by this phase calls any `rng_*` function, the two SMOTE modes will consume a different number of RNG draws before reaching subsequent network-initialization/dropout code, meaning "same seed" no longer implies "identical weight initialization and dropout masks" between the two A/B arms — undermining the rigor of the comparison beyond what fold-assignment identity alone guarantees.
**Why it's avoidable here:** `find_knn()` (existing) and the proposed `find_knn_global()` are both purely deterministic distance computations with no randomness. `rng_int()` consumes exactly one `xorshift32()` draw regardless of its argument `n` (verified in `src/utils.c:76-80`), and the synthesis loop's iteration count (`n_synthetic = max_count - n_class`) is identical between modes — only the *pool* the base index is drawn from differs, not the *number* of `rng_int()`/`rng_uniform()` calls made.
**How to avoid:** Verify (by code review, not just by testing) that `find_knn_global()` introduces zero new `rng_*` calls, and that the borderline-vs-standard branch only changes which array (`class_idx_borderline[c]` vs `class_idx[c]`) `rng_int()` indexes into — never how many times `rng_int()`/`rng_uniform()` are called in total.
**Warning signs:** If a future edit adds any randomized subsampling to `find_knn_global()` (e.g. to speed up its O(n²) cost on a larger dataset), this parity is silently broken — flag this constraint in a code comment at the function definition, mirroring the existing comment in `precalculate_augmentations()`.

## Code Examples

See "Architecture Patterns" above for the two load-bearing code sketches (`find_knn_global()` + `classify_borderline()`, and the `mode_smote_ab()` orchestration pattern). No additional external code examples apply — this is a self-contained, single-file C extension with no library API to demonstrate.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Standard SMOTE only (`smote_oversample()`, all minority samples eligible as synthesis base) | Standard SMOTE retained as `SMOTE_STANDARD`, Borderline-SMOTE1 added as `SMOTE_BORDERLINE`, selectable, decided by A/B comparison | This phase (Gap 2) | Whichever mode is *not* adopted becomes a documented negative/neutral result in `CLAUDE.md`, not a silent removal — per `SMOTE-05`, the outcome must be documented "independentemente do resultado" |
| `CLAUDE.md`'s current "What Worked" text already claims "Borderline-SMOTE (better than regular SMOTE)" | This claim predates any actual Borderline-SMOTE implementation in the code (confirmed by `CONCERNS.md`: "a documentation inaccuracy... SPEC.md's audit contradicts by inspecting the actual code") | Discovered during project-init audit, 2026-07-27 | This existing claim in `CLAUDE.md` must be corrected (not just appended to) once Gap 2's real A/B result is known — it currently describes something that was never actually run |

**Deprecated/outdated:** None specific to this phase — the algorithm being implemented (Han/Wang/Mao 2005) is 20 years old and stable; there is no newer "Borderline-SMOTE" revision to track. The one thing to actively correct is the codebase's own documentation drift described above.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `smote_oversample()`'s signature should be widened in place (matching `ROADMAP.md`'s literal success-criterion wording) rather than introduced as a new `smote_oversample_ex()` function (as `SPEC.md`'s earlier draft sketched) | Summary, Standard Stack | Low — both approaches are functionally equivalent; choosing the wrong one only affects naming/diff size, not correctness. Flagging because `SPEC.md` and `REQUIREMENTS.md`/`ROADMAP.md` literally disagree on the function name, and `REQUIREMENTS.md` is the newer, authoritative source. |
| A2 | The A/B comparison should be driven by a new dedicated CLI mode (`mode_smote_ab()`) that runs `mode_train()` twice in one process, rather than two separate manual `make full` invocations with a config edit in between | Summary, Pattern 2 | Medium — if the planner instead chooses "two separate manual runs," the fold-assignment-identity guarantee still holds (via `kfold_split()`'s internal reseed) as long as no other seed-affecting code changes between runs, but the *convenience* and *single-command reproducibility* of SMOTE-04's "A/B report exists" requirement becomes harder to satisfy cleanly, and there is more risk of accidentally comparing against a stale/different-commit baseline log (an anti-pattern explicitly flagged in `.planning/research/FEATURES.md`). |
| A3 | `BORDERLINE_M_NEIGHBORS` and `SMOTE_K_NEIGHBORS` should both default to 5 but be named as independently-tunable constants, per the Stack research finding that the original paper treats them as separate parameters even though this implementation starts them equal | Architecture Patterns, Anti-Patterns | Low — functionally identical to hardcoding `k=5` for both roles at first; only matters for citation precision and future tunability. Not doing this risks slightly overstating fidelity to Han/Wang/Mao (2005) in the academic report. |
| A4 | For the pre-existing `n_class <= 1` case (SMOTE-03's second clause), the recommended fix is to log a warning and continue producing an explicit, labeled duplicate (not to skip synthesis entirely and leave the class under-sampled) | Pitfall 3 | Medium — "no crash and no degenerate synthetic samples" in `ROADMAP.md`'s exact wording could also be read as "produce zero synthetic samples for that class/fold instead of a disguised duplicate." This is a genuine design decision the plan must make explicitly (see Open Questions) — training on an under-balanced class relies on the existing class-weight mechanism (`cw_binary`/`cw_expert` in `mode_train()`) to compensate, which already exists and already handles imbalance to some degree. |

## Open Questions (RESOLVED)

1. **What exactly does "no degenerate synthetic samples" mean for the pre-existing `n_class <= 1` case (SMOTE-03)?**
   - What we know: `find_knn`/`find_knn_global` with `n_class <= 1` can only return the sample itself as its "neighbor" (guarded to `dist=1e30f` but still selected, since it is the only candidate), producing an interpolated point that is mathematically identical to the base point — a disguised duplicate, not a genuine synthetic sample.
   - What's unclear: Whether the required fix is (a) log a warning and continue producing the duplicate anyway (making today's silent behavior explicit, per `CONCERNS.md`'s own suggested fix), or (b) log a warning and skip synthesis entirely for that class/fold, leaving it under `max_count` (relying on `cw_binary`/`cw_expert` class weights to compensate for the residual imbalance).
   - Recommendation: Default to (a) — log and continue with an explicitly-labeled duplicate — since it is the smaller, more localized change (a warning line, no change to `max_count`/downstream array sizing math) and matches CONCERNS.md's own stated fix approach; but this must be an explicit decision recorded in the plan, not left implicit, since it changes what "no degenerate synthetic samples" is interpreted to mean when the plan is later checked against SMOTE-03.
   - **RESOLVED:** Option (a) adopted — the pre-existing `n_class <= 1` case logs an explicit warning (`log_warn`, "smote_oversample: classe %d com apenas %d amostra(s)...") and continues, producing a documented, explicit duplicate rather than skipping synthesis or altering `max_count`/array-sizing math. See `01-01-PLAN.md` Task 2, Block A.

2. **Should the A/B comparison run both SMOTE modes in a single process invocation, or as two separate CLI invocations?**
   - What we know: `kfold_split()`'s internal `rng_seed(seed)` reseed guarantees identical fold assignments either way, as long as nothing else that consumes `rng_*` differs between the two runs before that point (confirmed: `precalculate_augmentations()` is now deterministic and mode-independent per Phase 0's fix).
   - What's unclear: Whether a single new CLI mode (`smote-ab`, this research's recommendation) or two invocations of the existing `train`/`full` mode with an added optional 3rd CLI argument (e.g. `./build/vocal_detect train . borderline`) is the better fit for this codebase's existing conventions and for satisfying SMOTE-04's "report exists in `results/`" requirement cleanly.
   - Recommendation: Prefer the single dedicated mode (mirrors the `verify-rng` precedent from Phase 0, produces the comparison report from one command, avoids any risk of comparing across accidentally-different commits/configs) — but note that adding an optional 3rd CLI argument to the existing `train`/`full` dispatch is also useful independently (for ad hoc single-mode debugging runs) and the two are not mutually exclusive; the plan should decide whether both are in scope or just the dedicated `smote-ab` mode.
   - **RESOLVED:** Single dedicated CLI mode adopted, not two separate invocations. A new `mode_smote_ab()` mirrors the `mode_verify_rng()` precedent and runs both arms — `mode_train_ex(base_dir, SMOTE_STANDARD, &res_standard)` then `mode_train_ex(base_dir, SMOTE_BORDERLINE, &res_borderline)` — sequentially within one process invocation of `./build/vocal_detect smote-ab .`. No optional 3rd CLI argument to `train`/`full` was added; that ad hoc debugging convenience is out of scope. See `01-02-PLAN.md` Task 2.

3. **Where should `BORDERLINE_M_NEIGHBORS`/`SMOTE_K_NEIGHBORS` live — `config.h` or local constants in `main.c`?**
   - What we know: The project convention (`CLAUDE.md` Configuration section) is "all tunables live in `include/config.h`", and every other SMOTE-adjacent constant (`RANDOM_SEED`, `K_FOLDS`) already lives there.
   - What's unclear: Whether this phase's scope includes touching `config.h` at all, given the existing `k = 5` in `smote_oversample()` is currently a local variable, not a `config.h` constant.
   - Recommendation: Add both as local `#define`s near the top of the SMOTE section in `main.c` (minimal-diff, consistent with the existing `k=5` being local today) unless the plan explicitly decides to migrate all SMOTE tunables to `config.h` as part of this phase — a larger-than-necessary change for this gap's stated scope.
   - **RESOLVED:** Local `#define`s adopted in `src/main.c`, not `config.h`. `SMOTE_K_NEIGHBORS`, `BORDERLINE_M_NEIGHBORS`, and `MAX_SMOTE_CLASSES` are all defined as local `#define` constants near the top of the SMOTE section in `main.c`, consistent with the existing local `k = 5` convention. No SMOTE tunables were migrated to `config.h` as part of this phase. See `01-01-PLAN.md` Task 1, item 2.

## Environment Availability

Skipped — this phase has no external dependencies beyond the existing build toolchain (`gcc`, `make`, `libm`, OpenMP), already verified present and working by every prior phase (Phase 0's `make full` run succeeded). No new tool, service, or runtime is introduced.

## Security Domain

**N/A for this phase.** Per `.planning/codebase/CONCERNS.md`'s existing Security Considerations audit: "This is an offline, single-user academic C pipeline with no network exposure, no external service calls, and no user-supplied input beyond local file paths." Gap 2 adds no new I/O, no new network surface, and no new input parsing beyond an optional CLI mode string — the same trust model as every existing CLI mode (`extract`/`train`/`full`/`verify-rng`) applies unchanged. No ASVS category is meaningfully applicable to a local, single-researcher, offline batch-processing tool with no authentication, sessions, or externally-facing input.

## Sources

### Primary (HIGH confidence)
- Direct source reads (this session): `src/main.c` (`smote_oversample`, `find_knn`, `mode_train`, `main()` dispatch, `precalculate_augmentations`), `src/kfold.c` (`kfold_split`'s internal `rng_seed` reseed), `src/utils.c` (`rng_int`, `rng_uniform`, `rng_shuffle_int` implementations — confirms exactly one `xorshift32()` draw per `rng_int()` call regardless of argument), `include/metrics.h` (`metrics_bootstrap_ci`, `metrics_mcnemar` signatures, already wired per Phase 0), `include/utils.h` (`log_warn` availability), `Makefile` (existing CLI mode dispatch pattern), `SPEC.md` §GAP 2 (algorithm specification, acceptance criteria), `.planning/REQUIREMENTS.md` (SMOTE-01..05, authoritative over `SPEC.md`'s earlier draft where they differ), `.planning/ROADMAP.md` (Phase 1 exact success-criterion wording), `.planning/STATE.md` (Phase 0 completion status, decision log)
- `.planning/research/STACK.md` §"GAP 2 — Borderline-SMOTE: Exact Algorithm" — Han, Wang & Mao (2005), *Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning*, ICIC 2005, LNCS 3644, pp. 878-887, read in full by prior project research (https://sci2s.ugr.es/keel/pdf/specific/congreso/han_borderline_smote.pdf) — verified Step 1 (classification rule) and Step 2 (generation formula) against the primary source; confirms `2*m >= k` boundary, confirms same-class-only interpolation (Borderline-SMOTE1, not SMOTE2), confirms the empty-DANGER fallback is a project-specific engineering extension not present in the original paper

### Secondary (MEDIUM confidence)
- `.planning/research/PITFALLS.md` Pitfalls 1-4, 13 — prior project research cross-referencing `CONCERNS.md` and direct source reads; confidence HIGH for line-number-anchored codebase claims, MEDIUM for the general Borderline-SMOTE citation-precision claims (corroborated by multiple independent secondary sources: themis R package's `bsmote` docs, a TDS Archive SMOTE-variants summary)
- `.planning/research/FEATURES.md` §"GAP 2 — Borderline-SMOTE A/B Comparison" — reporting-requirement gap analysis (safe/borderline/noise count table, McNemar/CI symmetry with Gap 3, anti-features around selective/cherry-picked reporting)
- `.planning/research/ARCHITECTURE.md` §"Gap 2 — Borderline-SMOTE" — confirms no drift between `SPEC.md`'s proposed signatures and the actual current `smote_oversample()`/`find_knn()` code, character-for-character
- `.planning/codebase/CONCERNS.md` — "Fragile Areas" (pre-existing `n_class <= 1` SMOTE degeneration), "Performance Bottlenecks" (`find_knn()` O(n²) cost, relevant to `find_knn_global()`'s added cost)

### Tertiary (LOW confidence)
- None — this phase required no new WebSearch/Context7 lookups; all algorithmic and codebase claims were resolvable from existing project research artifacts and direct source reads.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no external stack exists to be uncertain about; all claims verified by direct source reads
- Architecture: HIGH — verified line-by-line against current `src/main.c`; no drift from prior architecture research
- Pitfalls: HIGH — all four pitfalls tied to specific, verified line numbers or verified algorithm-primary-source details; the one genuinely open design question (the `n_class <= 1` fallback semantics) is flagged explicitly as an Open Question, not asserted as settled

**Research date:** 2026-07-27
**Valid until:** No expiry driver — this is a static, dependency-free C99 codebase and a 20-year-old, stable published algorithm; re-research only if `src/main.c`'s SMOTE section or `SPEC.md`/`REQUIREMENTS.md` change materially before planning begins.
