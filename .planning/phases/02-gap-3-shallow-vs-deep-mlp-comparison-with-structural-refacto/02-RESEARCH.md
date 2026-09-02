# Phase 2: Gap 3 — Shallow vs Deep MLP Comparison (with structural refactor) - Research

**Researched:** 2026-07-27
**Domain:** Hand-rolled C99 MLP architecture generalization (compile-time-fixed → runtime-configurable depth) + reproducible statistical model-selection protocol (1-SE rule + McNemar) for a hierarchical late-fusion voice-pathology classifier
**Confidence:** HIGH — every claim about current code behavior below is verified by direct reading of `src/mlp.c`, `include/mlp.h`, `src/mlp_train.c`, `include/config.h`, and the current `src/main.c` (post-Phase-0/Phase-1 `mode_train_ex()` refactor) in this session, not from SPEC.md's prose or training-data assumptions about "typical" MLP code.

## Summary

The phase's SPEC.md framing is doubly wrong about the current architecture, and this research corrects both errors before any comparison table can be built. First, `mlp_init_dynamic()` does **not** take a `hidden_size` parameter or a 1-hidden-layer/2-hidden-layer ternary as SPEC.md claims — it is a zero-argument-beyond-input/output-size function that **always** builds exactly 2 hidden layers (`[128, 64]`) via a `#if MLP_NUM_LAYERS == 3` compile-time branch, because `config.h`'s `MLP_NUM_LAYERS` is permanently `3`. This means today's production network already **is** SPEC.md's "Config C" — the "Config A ([128], shallow)" baseline SPEC.md assumes has never been trained. Second, `MLP.layers` is a fixed-size compile-time array (`Layer layers[MLP_NUM_LAYERS]`, `include/mlp.h:63`) and `src/mlp_train.c` independently declares three more arrays sized off the same macro (`best_weights`/`best_biases`, `best_bn_gamma/beta/mean/var`, `swa_weights/swa_biases` — lines 183-205 in the version read this session). Widening to hold Config D (3 hidden + 1 output = 4 layers) requires touching all four array declarations, not just the struct — but every loop that walks these arrays already iterates by `net->num_layers` (a runtime field), never by the macro, so the fix is a pure array-sizing bump with no loop-logic changes. A second, independent hazard exists in `mlp_backward()`, which sizes its backprop delta buffers from the hardcoded constant `MLP_HIDDEN1_SIZE` (128) rather than the actual widest hidden layer of the network instance being backpropagated — harmless today only because all 4 specified configs happen to cap out at 128.

Verified parameter counts (computed directly from `layer_init()`'s `nw = output_size*input_size` + `output_size` biases formula, input=85 per vowel): Config A ≈11.4k (Master)/11.7k (Expert), Config B ≈5.6k/5.8k, Config C (current production) ≈19.5k/19.7k, Config D ≈21.5k/21.6k — closely matching SPEC.md's approximate table, confirming its numbers are usable as-is once the labels are corrected. Phase 1 already established the exact reusable pattern this phase needs: `mode_train_ex(base_dir, smote_mode, result)` + `ABResult` struct + a thin orchestration wrapper (`mode_smote_ab()`) that runs the full pipeline twice under identical seed/folds and writes a code-computed adopt/reject report. The lowest-risk design for ARCH-03 is to widen `mode_train_ex()`'s signature once more (add an `ArchConfig` parameter) rather than extract a new `run_fold_cv()` function — this reuses working, already-tested infrastructure instead of introducing a second refactor path.

**Primary recommendation:** Fix the two structural array-sizing hazards (`MLP_MAX_LAYERS` bump in `mlp.h`+`mlp_train.c`, `mlp_backward()`'s dynamic `max_size` computation) in one prerequisite commit verified under `-fsanitize=address` on Config D before touching the comparison protocol; then add `mlp_init_multi()` (with `mlp_init_dynamic()` kept as a compatibility wrapper), widen `mode_train_ex()` with an `ArchConfig` parameter, and orchestrate a new `arch-compare` CLI mode that runs all 4 configs sequentially (reusing Phase 1's `kfold_split()`-reseed-for-identical-folds pattern) and applies a code-computed 1-SE-rule-plus-McNemar decision rule — never a manual/visual pick.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Variable-depth MLP struct/array sizing | Model layer (`mlp.h`/`mlp.c`) | — | Struct layout is the sole owner of how many `Layer` slots exist per instance |
| Configurable network construction (`mlp_init_multi`) | Model layer (`mlp.c`) | — | Must replace/wrap `mlp_init_dynamic()`, the only current entry point |
| Backprop buffer sizing | Model layer (`mlp.c`, `mlp_backward`) | — | Internal to forward/backward pass, no caller-visible API change |
| Checkpoint/SWA/BN buffer sizing | Training loop (`mlp_train.c`) | — | Declared locally in `mlp_train()`, sized off the same macro as the struct |
| 4-config × 5-fold × 3-vowel × 2-network orchestration | Orchestration layer (`main.c`) | — | `mode_train_ex()`/new `mode_arch_compare()` is the only place that owns the fold+vowel loop and RNG-reseed discipline |
| Parameter counting / time-per-epoch measurement | Model layer (new `mlp_count_params()`) + Orchestration layer (`timer_now()` around `mlp_train()`) | — | Param count is a pure function of `MLP` struct state; timing is a call-site concern in the training loop |
| 1-SE rule + McNemar decision logic | Orchestration layer (`main.c`, new `write_arch_compare_report()`) | Statistics layer (`metrics.c`, unchanged) | Reuses existing `metrics_mcnemar()`/`metrics_bootstrap_ci()` verbatim; only the comparison/decision logic is new |
| `CLAUDE.md` update | Documentation | — | Must reflect actual adopted config, not SPEC.md's mislabeled table |

## Project Constraints (from CLAUDE.md)

- C99 only (`-std=c99`), compiled with `gcc -O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -fopenmp -lm`; zero new warnings tolerated (repo hook `.claude/hooks/compile-check.sh` runs `make` after every `.c`/`.h` edit).
- No external ML/DL dependencies of any kind — the MLP, Adam optimizer, kNN, and logistic-regression baselines are all hand-implemented; this phase must not introduce any third-party library.
- All prose comments in Portuguese, `/* ... */` block style only — never `//` line comments (zero occurrences in the codebase today; do not introduce the first one).
- Section-header banners use the `/* ========== Nome ========== */` convention in files that already use it (`config.h`, `utils.h`).
- Every non-trivial public function in a `.h` gets a `/* ... */` doc comment describing parameter shapes (`[n]`/`[n x m]`) and, where applicable, its academic citation (see `metrics.h`'s McNemar/permutation-importance docstrings as the house style to match for any new `mlp_init_multi()`/`mlp_count_params()` declarations).
- **Methodological (SPEC.md-derived, restated in CLAUDE.md):** no change is adopted without a reproducible A/B (same `RANDOM_SEED=42`, same 5 folds) with a log saved to `results/train_log_vXX_<gap-name>.txt`; any change that drops global Macro F1 below 0.42 must be reverted or kept only as a documented experiment; `CLAUDE.md` must be updated with the outcome regardless of which config is adopted (ARCH-06).
- Implementation order is fixed: Gap 2 → Gap 3 → Gap 1 (already Phase 1 → Phase 2 → Phase 3). Gap 3 (this phase) must incorporate whatever SMOTE mode Phase 1 adopted (Borderline-SMOTE, per `STATE.md`'s recorded decision) as the fixed SMOTE setting for all 4 architecture configs — this phase is not a re-run of the Gap 2 A/B.
- `make asan`/AddressSanitizer is explicitly **not** a required permanent CI target (out of scope per `.planning/REQUIREMENTS.md`'s "Out of Scope" table) but **is** recommended as a one-off pre-sweep verification step — do not add a permanent Makefile target, but do run an ad hoc `-fsanitize=address` build/execution once on Config D before the full 4-config sweep.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ARCH-01 | Correct the SPEC.md config label before any comparison is presented: current production = Config C `[128,64]`, not Config A `[128]` | Confirmed via direct read of `src/mlp.c:220-241`: `mlp_init_dynamic()` always builds 2 hidden layers via the `#if MLP_NUM_LAYERS==3` branch (only branch ever compiled, since `config.h:81` fixes `MLP_NUM_LAYERS=3`). See Summary and "Config Label Correction" pattern below. |
| ARCH-02 | `mlp_init_dynamic()` generalized to `mlp_init_multi()` with configurable `hidden_sizes`/`dropout_rates`; `MLP_MAX_LAYERS` widened with corresponding fixed-size buffers in `mlp.c`/`mlp_train.c` safely holding Config D, no OOB writes | See "Don't Hand-Roll" / "Common Pitfalls" (Pitfalls 1-3, mirroring PITFALLS.md's Pitfalls 9-10) and "Code Examples" section for exact array declarations to change and the `mlp_backward()` dynamic `max_size` fix |
| ARCH-03 | Single reusable fold+vowel training function trains all 4 configs against identical 5-fold partitions — no duplicated per-config loop in `main.c` | See "Architecture Patterns" — recommends widening `mode_train_ex()` (Phase 1 precedent) with an `ArchConfig` parameter rather than a fresh `run_fold_cv()` extraction |
| ARCH-04 | `results/` contains a 4-config comparison table (accuracy, Macro F1, per-class F1, parameter count, time/epoch) with McNemar/bootstrap CI between best and every simpler config | See "Code Examples" for `mlp_count_params()`, `timer_now()`-based time/epoch capture, and the pairwise-McNemar-vs-best pattern extending `write_smote_ab_report()`'s single-pair McNemar to a 4-arm comparison |
| ARCH-05 | "Smallest complexity not statistically worse" decision rule (1-SE rule + McNemar) applied and documented | See "State of the Art" section for the 1-SE rule's definition/citation and how it composes with McNemar here |
| ARCH-06 | `CLAUDE.md` updated with the Gap 3 outcome regardless of adopted config | Follows the exact precedent of Phase 1's SMOTE-05 CLAUDE.md update (01-04-PLAN.md) — same pattern, new section |
</phase_requirements>

## Standard Stack

### Core
No new external libraries. This phase is a pure C99 internal refactor + statistical comparison, using only what's already linked (`libm`, `libgomp`). Per CLAUDE.md's tech stack and the project's zero-dependency constraint, introducing any third-party library (including a header-only C stats/argparse library) would violate an explicit project constraint.

### Supporting
| Component (existing, reused) | Location | Purpose |
|---|---|---|
| `metrics_bootstrap_ci()` | `src/metrics.c` (wired into `mode_train_ex` since Phase 0) | Bootstrap CI (N=1000, seed=42) on aggregated out-of-fold predictions per config |
| `metrics_mcnemar()` | `src/metrics.c` | Pairwise significance test, best-config-vs-each-simpler-config |
| `timer_now()` | `src/utils.c`/`include/utils.h` | High-resolution monotonic timer, already used in `feature_extract.c` — reuse for time/epoch measurement, do not add a second timer abstraction |
| `ABResult` struct + `mode_train_ex()`/`mode_smote_ab()` pattern | `src/main.c` (Phase 1) | Direct precedent for "run pipeline N times under identical seed/folds, capture structured results, decide by fixed rule" — extend rather than reinvent |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Fixed oversized `Layer layers[MLP_MAX_LAYERS]` array | `Layer *layers` malloc'd to exact `num_layers` | Malloc'd pointer "saves" a few `Layer` structs' worth of memory (~200-300 bytes/instance) but requires adding `free(net->layers)` to `mlp_free()`, auditing `mlp_save`/`mlp_load` for a matching change, and mixing heap-bound with the 3 remaining stack-bound arrays in `mlp_train.c` — net risk increase for negligible gain given megabyte-scale feature matrices already dominate memory use. **Recommendation: fixed oversized array (`MLP_MAX_LAYERS=5`).** |
| Widening `mode_train_ex()` with an `ArchConfig` parameter | Extracting a brand-new `run_fold_cv()` function per the older (pre-Phase-1) `.planning/research/ARCHITECTURE.md` suggestion | The `run_fold_cv()` suggestion predates Phase 1's actual `mode_train_ex()`/`ABResult` refactor and was written against an earlier, un-refactored `main.c`. Since Phase 1 already turned the fold+vowel loop into a single reusable, parameterized function (`mode_train_ex`), adding one more parameter to it is lower-risk and more consistent with established precedent than introducing a second, differently-shaped reusable-function pattern. **Recommendation: widen `mode_train_ex()`.** |
| Per-fold-array 1-SE rule (classic Breiman/CART: SE = std(5 fold scores)/√5) | Bootstrap-CI-derived SE (`(ci.upper - ci.lower) / (2×1.96)`, reusing the already-computed `ConfidenceInterval`) | The classic 1-SE rule needs each fold's raw Macro F1 captured into an array (`float fold_macro_f1[K_FOLDS]`) — a new field not currently tracked anywhere (today only a running `macro_f1_sum` accumulator exists). The bootstrap-SE approximation reuses the `ConfidenceInterval` struct already computed per config with zero new tracking. Both are defensible; document whichever is chosen explicitly in the report. **Recommendation: bootstrap-CI-derived SE**, for consistency with the project's already-established uncertainty-quantification method and zero new struct fields — but flag this as a documented methodological choice, not silently substituted for the "classic" rule. |

**Installation:** None — no new packages.

## Package Legitimacy Audit

**Not applicable.** This phase installs zero external packages. The project has no package manager (no `pip`/`npm`/`cargo`), and CLAUDE.md's Technology Stack section confirms all dependencies are system libraries (`libm`, `libgomp`) already linked. The Package Legitimacy Gate protocol is skipped entirely for this phase.

## Architecture Patterns

### System Architecture Diagram

```
                         mode_arch_compare(base_dir)   [NEW — this phase]
                                    |
          for cfg in { A[128], B[64], C[128,64](current prod), D[128,64,32] }:
                                    |
                                    v
          mode_train_ex(base_dir, SMOTE_BORDERLINE /* Phase-1-adopted */, cfg, &result[cfg])
                                    |
              (existing, unchanged pipeline shape — Phase 0/1 already built this)
                                    |
          rng_seed(42) -> kfold_split() [identical folds across all 4 cfg iterations,
                                          guaranteed by kfold_split's internal reseed,
                                          same mechanism mode_smote_ab() already relies on]
                                    |
          for f in 0..4 (fold loop):
              baselines (MajorityClass/kNN/LogReg) computed once per fold, cfg-independent
              for v in 0..2 (vowel loop):
                  mlp_init_multi(&net_master[v], 85, 2,  cfg.hidden_sizes, cfg.n_hidden, cfg.dropout_rates)  [was mlp_init_dynamic]
                  t0 = timer_now(); mlp_train(&net_master[v], ...); dt = timer_now()-t0  [NEW: time/epoch]
                  mlp_init_multi(&net_expert[v], 85, 4, cfg.hidden_sizes, cfg.n_hidden, cfg.dropout_rates)
                  t0 = timer_now(); mlp_train(&net_expert[v], ...); dt = timer_now()-t0
              predict_hierarchical_late_fusion(...) -> all_y_true/all_y_pred (per cfg)
                                    |
          metrics_compute + metrics_bootstrap_ci + metrics_mcnemar(vs 3 baselines)  [existing, unchanged]
                                    |
          result[cfg] = { accuracy, macro_f1, f1_per_class, ci, param_count, mean_time_per_epoch, y_true, y_pred }
                                    |
                                    v
          write_arch_compare_report(result[A..D])   [NEW]
              -> pick best_cfg = argmax(macro_f1) among {A,B,C,D}
              -> for each other cfg: metrics_mcnemar(y_true, best.y_pred, other.y_pred, ...)
              -> 1-SE band = [best.macro_f1 - SE(best), best.macro_f1]
              -> adopted_cfg = simplest (fewest params) cfg whose macro_f1 falls inside the 1-SE band
                               AND is not significantly worse than best_cfg by McNemar
              -> results/arch_compare_comparison.csv (4 rows x metrics)
              -> results/train_log_v33_gap3_arch_compare.txt (Portuguese report, adoption sentence)
              -> CLAUDE.md update (ARCH-06, separate task/commit)
```

### Recommended Project Structure

No new files are strictly required (unlike Gap 1, which needs a new `.c`/`.h` pair). All changes land in the existing files:

```
include/config.h      # MLP_MAX_LAYERS (new), keep MLP_NUM_LAYERS=3 as mlp_init_dynamic's implicit "current prod" default
include/mlp.h         # Layer layers[MLP_MAX_LAYERS]; mlp_init_multi() declaration; mlp_count_params() declaration
src/mlp.c             # mlp_init_multi() (new, generalized); mlp_init_dynamic() becomes a thin wrapper; mlp_backward()'s
                       #   max_size computed dynamically; mlp_count_params() (new)
src/mlp_train.c       # best_weights/biases, best_bn_*, swa_weights/biases arrays: MLP_NUM_LAYERS -> MLP_MAX_LAYERS
src/main.c            # ArchConfig struct + static ARCH_CONFIGS[4] table; mode_train_ex() gains an ArchConfig* parameter;
                       #   mode_arch_compare() (new, mirrors mode_smote_ab()); write_arch_compare_report() (new,
                       #   mirrors write_smote_ab_report()); new "arch-compare" CLI dispatch line in main()
CLAUDE.md             # Gap 3 Outcome section (ARCH-06), config-label correction in Architecture/Current Best Results
```

### Pattern 1: Config-label correction before any new code (ARCH-01)

**What:** Before writing any comparison code, correct every place that calls the current `[128,64]` network "Config A"/"the shallow baseline." This includes SPEC.md is not to be edited (it's a historical artifact document, not project truth) but any *new* report language, `CLAUDE.md` text, and code comments this phase produces must call the current production network **Config C**.
**When to use:** Immediately, as the first task of this phase — a documentation-only, zero-risk change that unblocks correct labeling of everything that follows.
**Example:**
```c
/* Config C = producao atual (2 camadas ocultas [128, 64]) -- NAO e "Config A" como o
 * SPEC.md original assumia. mlp_init_dynamic() (abaixo) sempre construiu 2 camadas
 * ocultas desde a v29 HEAD; nunca houve uma configuracao de 1 camada oculta em producao. */
```

### Pattern 2: `mlp_init_multi()` generalization, `mlp_init_dynamic()` as compatibility wrapper

**What:** Add a new function taking `hidden_sizes[]`/`n_hidden`/`dropout_rates[]` explicitly, replacing the compile-time `#if MLP_NUM_LAYERS == 3` branch with a runtime loop building `net->num_layers = n_hidden + 1` layers.
**When to use:** Core ARCH-02 requirement.
**Example (structure, not final code — planner should verify against actual `layer_init` signature):**
```c
/* mlp.h */
#define MLP_MAX_LAYERS 5   /* 1 (input) + up to 4 hidden + 1 output = covers Config D (3 hidden) with headroom */

typedef struct {
    Layer layers[MLP_MAX_LAYERS];  /* was: Layer layers[MLP_NUM_LAYERS] */
    int num_layers;                /* set per-instance: n_hidden + 1 (1 for A/B... wait A/B have n_hidden=1 -> 2 total) */
    int timestep;
} MLP;

void mlp_init_multi(MLP *net, int input_size, int output_size,
                     const int *hidden_sizes, int n_hidden,
                     const float *dropout_rates);
int  mlp_count_params(const MLP *net);   /* NEW: sum(output_size*input_size + output_size) over net->num_layers */

/* mlp.c */
void mlp_init_multi(MLP *net, int input_size, int output_size,
                     const int *hidden_sizes, int n_hidden,
                     const float *dropout_rates)
{
    net->num_layers = n_hidden + 1;
    net->timestep = 0;
    int sizes[MLP_MAX_LAYERS + 1];
    sizes[0] = input_size;
    for (int i = 0; i < n_hidden; i++) sizes[i + 1] = hidden_sizes[i];
    sizes[n_hidden + 1] = output_size;

    for (int i = 0; i < net->num_layers; i++) {
        layer_init(&net->layers[i], sizes[i], sizes[i + 1], /*use_bn=*/0);
        net->layers[i].dropout_rate = (i < n_hidden) ? dropout_rates[i] : 0.0f;
    }
}

/* Thin wrapper preserving exact current behavior -- Gap 1/Gap 2 code and any other
 * caller keep working with zero changes. */
void mlp_init_dynamic(MLP *net, int input_size, int output_size)
{
    int hidden[] = { MLP_HIDDEN1_SIZE, MLP_HIDDEN2_SIZE };
    float drop[] = { DROPOUT_RATE_HIDDEN1, DROPOUT_RATE_HIDDEN2 };
    mlp_init_multi(net, input_size, output_size, hidden, 2, drop);
}
```
**Verified fact backing this design:** every loop in `mlp.c`/`mlp_train.c` that walks `net->layers[]` already uses `for (i = 0; i < net->num_layers; i++)` (confirmed by direct read of `mlp_forward`, `mlp_backward`, `mlp_zero_gradients`, `mlp_adam_update`, `mlp_l2_regularization`, `mlp_save_checkpoint`, `mlp_load_checkpoint`, `mlp_free` in `src/mlp.c`, and the checkpoint/SWA loops in `src/mlp_train.c`) — never by the macro. This is what makes the fixed-oversized-array approach safe: no loop-body changes are needed, only the four array **declarations**.

### Pattern 3: `mlp_backward()`'s delta-buffer sizing must become dynamic, not a second macro bump

**What:** `mlp_backward()` (`src/mlp.c:287`, verified this session) currently does `int max_size = MLP_HIDDEN1_SIZE;` — a hardcoded 128. This is safe today (and for all 4 SPEC.md configs, since none exceed a 128-wide hidden layer) purely by coincidence, not by construction.
**When to use:** Fix in the same commit as the `MLP_MAX_LAYERS` bump — same root cause ("no hidden layer is assumed wider than 128"), same risk class.
**Example:**
```c
/* was: int max_size = MLP_HIDDEN1_SIZE; */
int max_size = 0;
for (int i = 0; i < net->num_layers - 1; i++) {   /* exclude output layer, matches existing intent */
    if (net->layers[i].output_size > max_size) max_size = net->layers[i].output_size;
}
float *delta = (float *)safe_malloc(max_size * sizeof(float));
float *delta_next = (float *)safe_malloc(max_size * sizeof(float));
```

### Pattern 4: Extending Phase 1's `ABResult`/`mode_train_ex` pattern to N configs (ARCH-03)

**What:** Phase 1 already solved "run the identical pipeline N times under the same seed/folds, capture structured comparable results, decide by a fixed rule." Reuse it directly rather than inventing a second pattern.
**When to use:** ARCH-03/ARCH-04's core requirement.
**Example:**
```c
/* main.c -- new struct, sibling to the existing SmoteMode enum */
typedef struct {
    const char *name;              /* "A", "B", "C", "D" */
    int hidden_sizes[3];           /* max 3 hidden layers (Config D) */
    int n_hidden;
    float dropout_rates[3];
} ArchConfig;

static const ArchConfig ARCH_CONFIGS[4] = {
    { "A", {128},         1, {0.5f} },
    { "B", {64},          1, {0.5f} },
    { "C", {128, 64},     2, {0.5f, 0.4f} },        /* current production, ARCH-01 */
    { "D", {128, 64, 32}, 3, {0.5f, 0.4f, 0.3f} },  /* dropout 0.3 for the 3rd hidden layer is a NEW
                                                        value -- config.h's existing DROPOUT_RATE_HIDDEN3
                                                        is 0.0f (leftover from dead #else branch); confirm
                                                        0.3 vs 0.0 explicitly during planning, see Open
                                                        Questions */
};

/* mode_train_ex() signature widened once more (Phase-1 precedent: it already grew from
 * mode_train() -> mode_train_ex(base_dir, smote_mode, result) once before) */
static int mode_train_ex(const char *base_dir, SmoteMode smote_mode,
                          const ArchConfig *arch, ABResult *result)
{
    /* ... unchanged fold/vowel loop body, except: */
    mlp_init_multi(&net_master[v], nf_vowel, 2, arch->hidden_sizes, arch->n_hidden, arch->dropout_rates);
    double t0 = timer_now();
    mlp_train(&net_master[v], ...);
    double dt_master = timer_now() - t0;
    /* accumulate dt_master / h_m.num_epochs into a running mean-time-per-epoch for this config */
    /* ... same for net_expert[v] ... */
}

static int mode_arch_compare(const char *base_dir)
{
    ABResult results[4] = {0};
    for (int i = 0; i < 4; i++) {
        /* Use whichever SmoteMode Phase 1 adopted (Borderline-SMOTE, per STATE.md) -- fixed
         * across all 4 configs, NOT re-compared here. */
        if (mode_train_ex(base_dir, SMOTE_BORDERLINE, &ARCH_CONFIGS[i], &results[i]) != 0) return -1;
    }
    write_arch_compare_report(results, ARCH_CONFIGS, 4,
                               "results/train_log_v33_gap3_arch_compare.txt",
                               "results/arch_compare_comparison.csv");
    for (int i = 0; i < 4; i++) { free(results[i].y_true); free(results[i].y_pred); }
    return 0;
}
```
**Verified precedent this mirrors exactly:** `mode_smote_ab()` (`src/main.c`, Phase 1) already runs `mode_train_ex()` twice sequentially in one process, relying on `kfold_split()`'s internal reseed for identical folds across both arms, with no manual RNG snapshot/restore — the same mechanism extends cleanly to 4 sequential calls instead of 2.

### Anti-Patterns to Avoid
- **Nesting the 4 configs inside the fold loop instead of outside it:** SPEC.md's protocol ("mesma partição de dados em todas") requires configs to be the **outer** loop with folds nested inside (each config gets its own full run through all 5 folds) — not folds outer with configs inner, which would require either 4x the in-memory network instances alive simultaneously per fold or an awkward interleaving that breaks the existing `mode_train_ex()`-per-arm pattern.
- **Sizing `MLP.layers[]` exactly per-config with `malloc`** instead of a fixed oversized array — see "Alternatives Considered" above; this creates a new leak-tracking surface (`mlp_free()` would need a new `free(net->layers)`) for negligible memory savings, and 4 configs × 5 folds × 3 vowels × 2 networks = up to 120 `MLP` instances created **within a single process run** (vs. today's 30), meaning any new leak here is no longer "reclaimed harmlessly at process exit" — it's a real, cumulative leak during the run itself.
- **Re-running the Gap 2 SMOTE A/B inside Gap 3's sweep:** this phase must hold SMOTE mode fixed at whatever Phase 1 adopted (Borderline-SMOTE) and vary only architecture depth — conflating the two comparisons would 2x the sweep's cost for no additional methodological rigor and violates the fixed-order dependency (Gap 2's decision must already be locked in before Gap 3 starts).
- **Picking the winning config by eyeballing the Macro F1 column** — ARCH-04/ARCH-05 explicitly require this decision to be computed by a fixed rule in code (mirroring `write_smote_ab_report()`'s `if (bl_res->macro_f1 >= std_res->macro_f1)` precedent), not asserted in report prose.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bootstrap confidence intervals | A new resampling routine | `metrics_bootstrap_ci()` (`src/metrics.c`, already wired into `mode_train_ex` since Phase 0) | Already implemented, tested, and seeded correctly (N=1000, seed=42) — reuse verbatim per config run |
| Paired-classifier significance test | A custom chi-square/t-test | `metrics_mcnemar()` (`src/metrics.c`) — Edwards continuity-corrected McNemar, already used for MLP-vs-baselines and for Phase 1's SMOTE-arm-vs-arm comparison | Exact same statistical need (paired predictions on identical samples); a hand-rolled second implementation would risk a different (and undocumented) continuity-correction convention |
| High-resolution timing for time/epoch | `clock()`/`gettimeofday` ad hoc in `main.c` | `timer_now()` (`src/utils.c`, `CLOCK_MONOTONIC`-based, already used in `feature_extract.c`) | One timer abstraction in the codebase; a second one invites drift (e.g. wall-clock vs monotonic mismatches across configs run at different times of day) |
| "Which fold/vowel/network won" bookkeeping | A new ad hoc struct-of-arrays | `ABResult` (Phase 1, `src/main.c`) — already the project's one "structured pipeline run result" data holder | Reuse the exact struct shape (or a minimal superset adding `param_count`/`mean_time_per_epoch`) rather than inventing a second parallel result type |

**Key insight:** Every statistical primitive this phase needs (bootstrap CI, McNemar, seeded reproducible multi-run orchestration) was already built and validated in Phase 0/Phase 1. The only genuinely new code is (a) the struct/array-sizing generalization inside `mlp.c`/`mlp_train.c`/`mlp.h`, and (b) the decision-rule composition (1-SE band + McNemar) glueing existing primitives together — there is no new statistics to implement from scratch.

## Common Pitfalls

> The following pitfalls were identified and verified with HIGH confidence (specific line-number citations) in an earlier research pass (`.planning/research/PITFALLS.md`, Pitfalls 9-12, and `.planning/codebase/CONCERNS.md`) before Phase 0/Phase 1 executed. This session re-verified all of them against the **current** `src/mlp.c`/`include/mlp.h`/`src/mlp_train.c` (unchanged by Phase 0/1, which only touched `src/main.c`) and confirms every claim below still holds exactly as originally documented.

### Pitfall 1: `Layer layers[MLP_NUM_LAYERS]` cannot represent Config D without an out-of-bounds write
**What goes wrong:** `include/mlp.h:63` declares `Layer layers[MLP_NUM_LAYERS]` where `MLP_NUM_LAYERS` (`config.h:81`) is a compile-time `3`. Config D needs `n_hidden=3` → `num_layers=4`. Writing `net->layers[3]` (0-indexed 4th slot) into a struct array sized for indices 0-2 is undefined behavior — a silent heap/stack corruption with no compiler warning, no crash guarantee.
**Why it happens:** The only two configurations ever exercised historically (`#if MLP_NUM_LAYERS==3` vs. the dead `#else` branch) were both selected at *compile* time, so nobody had to confront a struct that varies *per instance* at runtime — which is exactly what training 4 configs within one process, in the same binary, requires.
**How to avoid:** Introduce `MLP_MAX_LAYERS` (recommend `5`, covering Config D with one slot of headroom) used **only** for array sizing (`Layer layers[MLP_MAX_LAYERS]` in `mlp.h`; the three `mlp_train.c` stack arrays). Leave `net->num_layers` as the runtime-set field every existing loop already uses.
**Warning signs:** Build with `-fsanitize=address` and run Config D for one fold/vowel/network — ASan reports the overflow immediately if unfixed. Without a sanitizer, this can run silently for an entire multi-hour sweep, corrupting `num_layers`/`timestep` (the fields immediately following `layers[]` in the struct) or heap metadata past a heap-allocated `MLP`.
**Phase to address:** Prerequisite sub-task, first thing in this phase, before any Config C/D training happens.

### Pitfall 2: `mlp_backward()`'s delta buffer is sized from a stale macro, not the actual network
**What goes wrong:** `mlp_backward()` (`src/mlp.c:287`, confirmed this session: `int max_size = MLP_HIDDEN1_SIZE;`) allocates `delta`/`delta_next` at a hardcoded 128 floats regardless of the actual widest hidden layer of the `net` instance passed in. None of SPEC.md's 4 configs exceed 128 in any hidden layer, so this coincidentally never triggers for this phase's specific scope — but the danger is exactly that coincidence: nothing prevents a future config (or a copy-paste follow-up experiment reusing `mlp_init_multi`) from using a layer wider than 128 and silently corrupting adjacent heap memory.
**Why it happens:** Written when only one architecture (`[128,64]`) existed, so "widest possible hidden layer = `MLP_HIDDEN1_SIZE`" was true by definition, not by any actual bound-checking.
**How to avoid:** Compute `max_size` from `net->layers[i].output_size` for `i` in `[0, num_layers-1)` (excluding the output layer) at the top of `mlp_backward()`. Small, local, zero-risk fix — do it in the same commit as Pitfall 1's array-sizing bump, since both stem from the identical stale assumption.
**Warning signs:** Same as Pitfall 1 — only an ASan/Valgrind run reliably detects this; none of the 4 specified configs trigger it today, which is exactly why it must be fixed proactively rather than left until a future config does trigger it.
**Phase to address:** Same commit as Pitfall 1.

### Pitfall 3: Fixed hyperparameters across depths can produce a "deeper is worse" artifact that is really "deeper is undertuned"
**What goes wrong:** SPEC.md instructs holding Adam/LR/class-weights/dropout-schedule/L2/patience fixed across all 4 configs for a controlled comparison. Configs C/D have ~2x the parameters of A/B on a dataset where the Expert network trains on a few hundred samples per fold/vowel after the hierarchical split — a deeper network using regularization tuned for the shallow config's capacity is likely to overfit faster and get early-stopped sooner, producing a result that looks like "depth doesn't help" but is really "depth needs different regularization strength, which was not searched."
**Why it happens:** "Same hyperparameters" (apples-to-apples comparison) and "fair comparison of what each architecture can achieve" (best case per architecture) are different claims, easy to conflate — especially on a small, high-variance dataset (documented per-fold Macro F1 range in prior experiments: ~0.28-0.70) where overfitting sensitivity to depth is much higher than on larger benchmark datasets.
**How to avoid:** Keep the fixed-hyperparameter run as the primary, reported comparison (matches SPEC.md, is simpler and reproducible) but explicitly state the limitation in the final report/`CLAUDE.md` update ("hyperparameters held fixed across configs; a depth-specific regularization search was out of scope for this comparison"). If early stopping fires markedly earlier for C/D than A/B in the exported learning-curve data, note it as a caveat on the conclusion, not silently omit it.
**Warning signs:** Compare `TrainHistory.num_epochs`/`best_epoch` across configs (already captured by the existing `mlp_train()` return value) — if Config D consistently stops at a much lower epoch count than Config A/B, that is direct evidence of this pitfall, not necessarily evidence depth doesn't help.
**Phase to address:** Comparison-protocol design step (decide and document up front), not a code fix.

### Pitfall 4: Memory-management regressions when generalizing 30 → up to 120 `MLP` instances per run
**What goes wrong:** `mode_train()`/`mode_train_ex()` today never calls `dataset_free()`/`features_free()` before returning (a pre-existing, currently-harmless leak since the process exits immediately after one call). A 4-config sweep still calls `mode_train_ex()` only 4 times total (once per config, each looping its own 5-fold×3-vowel×2-network internally, same as today) — so this specific leak does **not** multiply 4x; it remains a single per-process leak, reclaimed at exit, same as today. The genuinely new risk is **within** each `mode_train_ex()` call: `mlp_free()` must be called for every `net_master[v]`/`net_expert[v]` at the end of every fold iteration (confirmed already present at `src/main.c:514` in the current code) — if a future edit to thread `ArchConfig` through the loop accidentally short-circuits this free (e.g. an early `continue`/`return` added without auditing the ~15 paired `malloc`/`free` calls per vowel iteration), the leak becomes real and cumulative across all folds/vowels/configs within a single sweep, not just harmless-until-exit.
**Why it happens:** The per-vowel block (`src/main.c`, current `mode_train_ex`'s vowel loop) already has 15+ manually paired allocations with terse variable names (`tr_x_v`, `os_e_y`, etc.) — CONCERNS.md already flags this as fragile to modify without full-frame auditing.
**How to avoid:** Do not add any early return/continue inside the vowel loop without re-auditing every live allocation in that loop iteration first. Prefer the "widen `mode_train_ex()`'s signature, keep the loop body shape" approach (Pattern 4) specifically because it minimizes new control-flow paths inside the fragile loop.
**Phase to address:** Implementation review checklist item for the `ArchConfig`-threading task.

## Runtime State Inventory

*Not applicable — this is a greenfield architectural-refactor phase (adding configurability), not a rename/refactor/migration of existing identifiers, stored data, or external service state.*

## Code Examples

### Verified parameter counts (computed from `layer_init()`'s actual formula, input=85=`FEATURES_PER_VOWEL(83)+NUM_METADATA_FEATURES(2)`)

`layer_init(input_size, output_size)` allocates `nw = output_size * input_size` weights + `output_size` biases per layer (confirmed, `src/mlp.c:123-157`; BN is disabled everywhere so contributes 0 extra params). Total per network = sum over all layers.

| Config | Hidden layers | Master (output=2) params | Expert (output=4) params |
|--------|--------------|---------------------------|---------------------------|
| A | [128] | 11,136 + 258 = **11,394** | 11,136 + 516 = **11,652** |
| B | [64] | 5,504 + 130 = **5,634** | 5,504 + 260 = **5,764** |
| C (current prod) | [128, 64] | 11,136 + 8,256 + 130 = **19,522** | 11,136 + 8,256 + 260 = **19,652** |
| D | [128, 64, 32] | 11,136 + 8,256 + 2,080 + 66 = **21,538** | 11,136 + 8,256 + 2,080 + 132 = **21,604** |

These match SPEC.md's approximate table (~11k/~5.5k/~19k/~21k) closely enough to confirm SPEC.md's *parameter estimates* are usable as-is — only the *config labels* (which one is "current production") needed correction (ARCH-01).

```c
/* mlp.h / mlp.c — new function for ARCH-04's parameter-count column */
int mlp_count_params(const MLP *net)
{
    int total = 0;
    for (int i = 0; i < net->num_layers; i++) {
        const Layer *l = &net->layers[i];
        total += l->output_size * l->input_size + l->output_size;
    }
    return total;
}
```

### Time-per-epoch capture (reusing existing `timer_now()`)

```c
/* Source: existing pattern in src/feature_extract.c (timer_now() used around extraction) */
double t0 = timer_now();
mlp_train(&net_master[v], os_m_x, os_m_y, os_n_m, vl_x_v, vl_y_bin, fold->n_val, nf_vowel, 2, cw_binary, &h_m);
double elapsed = timer_now() - t0;
double time_per_epoch = (h_m.num_epochs > 0) ? elapsed / h_m.num_epochs : 0.0;
/* Accumulate time_per_epoch into a per-config running mean (across all 5 folds x 3 vowels x 2 networks
 * for that config) for the final comparison table's "tempo/epoca" column. */
```

## State of the Art

### The 1-SE rule (one-standard-error rule), and how it composes with McNemar here

The 1-SE rule originates from Breiman, Friedman, Olshen & Stone's CART methodology (1984) for cross-validated model selection: rather than picking the model with the single best CV score, pick the **simplest** model whose CV score is within one standard error of the best model's score — the reasoning being that scores within one SE of each other are not reliably distinguishable given the sample's own variability. It is the model-selection analogue of Occam's razor and is the same principle `glmnet::cv.glmnet`'s `lambda.1se` implements for penalized regression. `[CITED: Breiman et al. 1984, widely re-described in modern ML tooling docs — e.g. scikit-learn's "Balance model complexity and cross-validated score" example, cross-referenced via WebSearch this session]`

**How this phase should compose it with McNemar (per SPEC.md's own instruction, "usar teste de McNemar OU comparação de ICs de bootstrap"):**
1. Compute each config's point-estimate Macro F1 and its bootstrap CI (already available per config via `metrics_bootstrap_ci()`).
2. Identify `best_cfg` = config with the highest point-estimate Macro F1.
3. Compute an approximate standard error for `best_cfg` from its bootstrap CI: `SE ≈ (ci.upper - ci.lower) / (2 × 1.96)` (95% CI half-width ÷ 1.96), OR from the classic per-fold definition `SE = std(5 fold macro_f1 values) / sqrt(5)` if per-fold values are captured (see "Alternatives Considered" — this phase does not currently track per-fold Macro F1 in an array, only a running sum; document whichever SE source is chosen).
4. The "1-SE band" is `[best_cfg.macro_f1 - SE, best_cfg.macro_f1]`.
5. Among all configs whose point-estimate Macro F1 falls inside that band, additionally require McNemar's test (best vs. that config, on the paired out-of-fold predictions) to show **no** statistically significant difference (`p >= 0.05`) — this is the "AND McNemar" refinement SPEC.md asks for, since the 1-SE band alone is a rougher, distribution-agnostic heuristic while McNemar directly tests the paired-prediction difference.
6. `adopted_cfg` = the config with the **fewest parameters** among those passing both the 1-SE-band and McNemar-non-significance filters. If only `best_cfg` itself passes (i.e., every simpler config either falls outside the 1-SE band or is McNemar-significantly worse), `adopted_cfg = best_cfg`.

This composition is not something this codebase (or, to this research's knowledge, a single canonical published source) has pre-packaged — it is a reasonable, explicitly-documented synthesis of two individually well-established statistical tools, and must be presented in the final report as exactly that (a documented decision procedure), not cited as if it is a single named method from the literature.

### Prior fixed-hyperparameter architecture comparisons in this codebase

CLAUDE.md's own "Optimization History" already documents that wider layers (128+), LeakyReLU, and dropout "worked," and that this project has never previously run a systematic depth sweep — Gap 3 is the first controlled, statistically-adjudicated version of a comparison this project has only ever done informally (ad hoc `results/train_log_vN.txt` version bumps).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Config D's third hidden layer dropout should be `0.3` (per SPEC.md's table) rather than `config.h`'s existing (but currently-dead) `DROPOUT_RATE_HIDDEN3 = 0.0f` | Pattern 4 / Code Examples | Low-medium — affects only Config D's regularization strength; if `0.0` is used instead, Config D is more prone to overfitting than SPEC.md intended, which could shift the "deeper is worse" conclusion for reasons unrelated to genuine architecture capacity (compounds with Pitfall 3) |
| A2 | Recommending bootstrap-CI-derived SE (`(upper-lower)/(2×1.96)`) over the classic per-fold-array SE for the 1-SE rule, since per-fold Macro F1 values are not currently captured in an array (only summed) | "Alternatives Considered", State of the Art | Low — both are defensible; if the classic per-fold SE is later deemed necessary by the committee, a `float fold_macro_f1[K_FOLDS]` field must be added to whatever result struct is used, a small but non-zero rework |
| A3 | `MLP_MAX_LAYERS = 5` provides sufficient headroom (Config D needs exactly 4) — this session recommends 5 for one slot of margin, following SPEC.md's own suggestion and the (pre-Phase-1) `.planning/research/ARCHITECTURE.md`'s identical recommendation | Pattern 2, Common Pitfalls | Very low — even `MLP_MAX_LAYERS=4` (exact fit) would satisfy ARCH-02 for the specified 4 configs; using 5 only adds ~1 unused `Layer` struct's worth of memory (~150-250 bytes) per instance, harmless at this codebase's memory scale |
| A4 | Widening `mode_train_ex()`'s signature (adding an `ArchConfig` parameter) is lower-risk than extracting a new `run_fold_cv()` function, given Phase 1 already validated the widen-in-place pattern | "Alternatives Considered", Architecture Patterns | Low — if `mode_train_ex()` becomes too unwieldy after this addition (it will grow from 3 to 4 parameters and gain per-config timing/param-count bookkeeping), a future extraction may still be warranted; this is a design preference, not a correctness risk either way |

**If this table is empty:** N/A — see entries above; all are genuine open design decisions surfaced by direct code reading this session, not fabricated claims.

## Open Questions

1. **Is a supplementary regularization-strength check across the 4 configs in scope, or is the fixed-hyperparameter limitation simply documented?**
   - What we know: `STATE.md`'s "Blockers/Concerns" section already records this as an explicit unresolved item deferred to Phase 2 planning: *"Whether a supplementary regularization-strength check across the 4 architecture configs is in scope, or the fixed-hyperparameter limitation is simply documented, needs an explicit decision during Phase 2 planning."*
   - What's unclear: SPEC.md's protocol only asks for the fixed-hyperparameter run; Pitfall 3 above documents why a supplementary check would strengthen the conclusion but at real additional time cost (potentially another full sweep, or at minimum 2 extra config×fold×vowel×network runs for C/D with adjusted L2/dropout).
   - Recommendation: Default to documenting the limitation explicitly (SPEC.md's literal ask) unless time budget allows the supplementary check; either way, this must be an explicit stated decision in the phase's plan, not silently defaulted.

2. **Which SE source backs the 1-SE rule — bootstrap-CI-derived or classic per-fold-array?**
   - What we know: Both are statistically defensible; the codebase currently only supports the bootstrap-CI-derived one without new struct fields (see Assumption A2).
   - What's unclear: Whether the academic committee/report expects the "textbook" per-fold-CV-SE definition specifically (closer to Breiman's original CART formulation) vs. this project's already-established bootstrap-CI convention.
   - Recommendation: Use bootstrap-CI-derived SE for consistency with the rest of the project's statistical infrastructure, but state explicitly in the report which SE definition was used and why — do not let this go undocumented.

3. **How long will the full 4-config sweep actually take, and should it run as one CLI invocation or be split across multiple execution waves?**
   - What we know: Phase 1's 2-arm `smote-ab` sweep (60 total MLP trainings: 2 configs × 5 folds × 3 vowels × 2 networks) was documented as "~60-180 min." Gap 3's 4-config sweep trains up to 120 MLPs (4× the base single-config cost), so a naive extrapolation suggests **2-6+ hours** of wall-clock time for one full `arch-compare` run, though shallower configs (A/B) likely train faster per epoch than C/D given fewer parameters (partially offsetting the 4x multiplier).
   - What's unclear: Exact wall-clock time cannot be known until run; this affects whether the phase's plan should execute the sweep as a single long-running background task or split it into per-config plans/waves for incremental checkpointing/verification.
   - Recommendation: Design `mode_arch_compare()` so each config's `ABResult` is captured and could, if needed, be persisted/logged incrementally (e.g., a log line after each config completes, before the final report is written) so a very long run's partial progress remains inspectable — and plan the execution step itself as a background/monitored task, not a blocking foreground wait.

4. **Should `mlp_save`/`mlp_load` (disk checkpoint persistence) gain a format-version/shape header as part of this phase, per PITFALLS.md Pitfall 12's original recommendation?**
   - What we know: Confirmed this session — `mlp_save()`/`mlp_load()` are **never called** anywhere in the current `src/main.c` pipeline (verified via `grep -n "mlp_save\|mlp_load\b" src/main.c` returning zero matches). The `models/*.bin` files present in the repo's `models/` directory are stale artifacts from an earlier (pre-hierarchical-fusion) architecture version, not regenerated by any current code path. Only the in-RAM `mlp_save_checkpoint()`/`mlp_load_checkpoint()` (best-epoch tracking within `mlp_train()`) are actually exercised.
   - What's unclear: Whether this phase should proactively add the format-version header anyway (cheap insurance, matches PITFALLS.md's original recommendation) or explicitly leave it out of scope since the disk-persistence functions are dead code from the training pipeline's perspective.
   - Recommendation: **Out of scope for this phase** — since `mlp_save`/`mlp_load` are not invoked by any Gap 3 code path, there is no actual risk of cross-config-shape checkpoint loading during this phase's execution. Note this finding in the phase's plan so a future phase that *does* wire up disk persistence knows to add the header check then, not now.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| gcc (C99 + OpenMP) | Entire build | ✓ | 13.3.0 (Ubuntu 24.04) | — |
| GNU Make | Build orchestration | ✓ | 4.3 | — |
| `-fsanitize=address` (ASan) support in gcc | Pre-sweep verification of Config D array-sizing fix (recommended, not a permanent target) | ✓ (confirmed this session: a trivial ASan-instrumented binary built and ran cleanly) | Built into gcc 13.3.0 | If ever unavailable on a different machine: fall back to careful manual review of the array-sizing diff + a single Config D fold/vowel/network dry run with `valgrind --tool=memcheck` if installed, or line-by-line review as last resort |
| SVD dataset WAV files + `overview_merged.csv` | Full pipeline execution (already required by all prior phases) | Not verified in this research session (outside scope — assumed present per CLAUDE.md's Prerequisites, unchanged from Phase 0/1) | — | — |

**Missing dependencies with no fallback:** None identified.

**Missing dependencies with fallback:** None currently missing — ASan fallback listed above is precautionary only (ASan itself is confirmed available on this machine).

## Security Domain

> `security_enforcement` is absent from `.planning/config.json` — treated as enabled per protocol, but this is a genuinely low-relevance domain for this phase, restated from `.planning/codebase/CONCERNS.md`'s prior finding (still accurate, unchanged by this phase's scope).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | Offline single-user CLI tool, no auth surface |
| V3 Session Management | No | No sessions of any kind |
| V4 Access Control | No | No multi-user/access boundaries |
| V5 Input Validation | Marginal | `base_dir`/CLI mode string parsing already exists (`main.c`); this phase adds one new CLI mode string (`"arch-compare"`) — validate via the same `strcmp` dispatch pattern already used for `"smote-ab"`/`"verify-rng"`, no new validation surface introduced |
| V6 Cryptography | No | No cryptographic operations anywhere in this codebase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Out-of-bounds array write (`Layer layers[]` sizing, Pitfalls 1-2 above) | Tampering (memory corruption, not an external-attacker threat model but a genuine memory-safety bug) | Fixed-size headroom array (`MLP_MAX_LAYERS`) + one-off ASan verification before the full sweep, per this phase's own recommended practice |
| Untrusted `base_dir` CLI argument (pre-existing, unchanged) | Tampering/path traversal | Already documented as accepted-risk in CONCERNS.md ("local CLI tool run by its own author on trusted input, no action needed unless wrapped in a service") — this phase does not change that risk profile |

## Sources

### Primary (HIGH confidence)
- Direct reads this session of `src/mlp.c` (600 lines, full), `include/mlp.h` (full), `src/mlp_train.c` (348 lines, full), `include/config.h` (full), `src/main.c` (788 lines, full, current post-Phase-0/1 state), `include/metrics.h` (full), `Makefile` (full)
- `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/STATE.md` (full reads, current state)
- `.planning/phases/00-statistical-infrastructure-rng-reproducibility-prerequisite/00-02-SUMMARY.md`, `.planning/phases/01-gap-2-borderline-smote/01-02-SUMMARY.md` (full reads — Phase 0/1 precedent patterns)
- `.claude` project CLAUDE.md (tech stack, conventions, architecture, constraints — full)
- `SPEC.md` lines 195-265 (GAP 3 section, full read)
- Direct verification this session: `gcc --version` (13.3.0), `make --version` (4.3), `gcc -fsanitize=address` trivial-program build+run (confirmed working), `grep -n "mlp_save\|mlp_load\b" src/main.c` (zero matches, confirming disk persistence is dead code in the current pipeline)

### Secondary (MEDIUM confidence)
- `.planning/research/PITFALLS.md` Pitfalls 9-12 (written 2026-07-27, before Phase 0/1 executed) — re-verified this session against the *current* `mlp.c`/`mlp.h`/`mlp_train.c` (unchanged by Phase 0/1) and confirmed still fully accurate
- `.planning/research/ARCHITECTURE.md` (written 2026-07-27, before Phase 0/1's `main.c` refactor — its `main.c` line-number citations are stale, but its structural analysis of `mlp.c`/`mlp_train.c`/`mlp.h` remains accurate since those files were untouched by Phase 0/1)
- `.planning/codebase/CONCERNS.md` (written 2026-07-27, pre-Phase-0/1 — RNG race and bootstrap-CI-disconnection findings are now resolved per Phase 0, noted as historical context only)
- WebSearch this session: "one standard error rule model selection cross-validation" — cross-referenced against multiple sources (Breiman/CART origin, scikit-learn documentation example, statistics blog explanations) for the 1-SE rule definition

### Tertiary (LOW confidence)
- None — all findings in this research were either directly verified against current source code or cross-referenced against multiple secondary sources for the one external statistical-methodology claim (1-SE rule).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new dependencies, entire "stack" is existing project infrastructure verified by direct code reading
- Architecture: HIGH — every structural claim (fixed-size arrays, `#if` branch behavior, loop iteration patterns, parameter-count formula) verified against current source this session, not assumed
- Pitfalls: HIGH — all four documented pitfalls trace to specific, re-verified line-level code facts; no speculative pitfalls included
- 1-SE rule / decision-procedure composition: MEDIUM — the 1-SE rule itself is well-established (Breiman/CART), but its specific composition with McNemar for this phase is a documented synthesis this research proposes, not a single citable canonical method

**Research date:** 2026-07-27
**Valid until:** Effectively indefinite for the structural/architectural findings (tied to source code that only changes when this phase's own tasks modify it) — 30 days for the "current best results"/CLAUDE.md-drift framing, since a parallel phase or hotfix could change `main.c` again before this phase executes.
