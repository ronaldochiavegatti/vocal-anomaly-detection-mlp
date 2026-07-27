# Phase 1: Gap 2 — Borderline-SMOTE - Pattern Map

**Mapped:** 2026-07-27
**Files analyzed:** 1 (single-file modification, per RESEARCH.md — no new files)
**Analogs found:** 4 / 4 (all analogs are functions/sections within the same file, `src/main.c`, since this phase modifies exactly one existing translation unit)

This phase does not add any new file to the repository. All "files to create/modify" are actually **new/modified functions inside the single existing file `src/main.c`**. Classification below is therefore at function granularity, not file granularity, which is the correct level of detail for this phase's scope (confirmed by RESEARCH.md's "Recommended Project Structure").

## File Classification

| New/Modified Unit | File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|------|-----------|-----------------|----------------|
| `SmoteMode` enum | `src/main.c` (new, near top of SMOTE section) | config/type-definition | N/A (compile-time constant) | none needed — trivial 2-value enum, no analog required | n/a |
| `find_knn_global()` | `src/main.c` (new function) | utility (numeric transform) | batch (O(n²) distance computation, no I/O, no RNG) | `find_knn()` — `src/main.c:189-203` | exact (same algorithm shape, different candidate set) |
| `classify_borderline()` | `src/main.c` (new function) | utility (pure classifier, 3-way branch) | transform | none in codebase — new logic; branch-order pitfall documented in RESEARCH.md Pitfall 1 | no analog — see "No Analog Found" |
| `smote_oversample()` (modified — gains `SmoteMode` param) | `src/main.c:205-231` (existing, modified in place) | service (CRUD-like: reads samples, produces expanded sample set) | batch/transform | itself (pre-modification version is its own best analog — signature widened in place, per RESEARCH.md Assumption A1) | exact |
| `mode_train()` (modified — gains `SmoteMode` param, threaded to both call sites) | `src/main.c:235-435` (existing, modified in place) | controller (CLI-mode orchestrator) | batch/event-driven (one-shot pipeline run per invocation) | itself (pre-modification version) | exact |
| `mode_smote_ab()` (new CLI mode function) | `src/main.c` (new function) | controller (CLI-mode orchestrator) | batch (runs `mode_train()`-equivalent twice, diffs results) | `mode_verify_rng()` — `src/main.c:446-466` | exact (same "dedicated fast-verification CLI mode" pattern established in Phase 0) |
| `main()` dispatch (modified — one new `strcmp` branch) | `src/main.c:474-484` | route/CLI-dispatch | request-response (argv parsing → mode function call) | itself (pre-modification version — the `verify-rng` branch at line 482 is the literal template for the new `smote-ab` branch) | exact |

## Pattern Assignments

### `find_knn_global()` (utility, batch/transform)

**Analog:** `find_knn()` — `src/main.c:189-203`

**Full existing function** (copy structure exactly, only widen candidate set from `class_indices[n_class]` to "all `n_in` samples"):
```c
static void find_knn(const float *x, int base, const int *class_indices, int n_class, int nf, int k, int *neighbors)
{
    float *dists = (float *)safe_malloc(n_class * sizeof(float));
    int *order = (int *)safe_malloc(n_class * sizeof(int));
    for (int i = 0; i < n_class; i++) {
        order[i] = i; if (class_indices[i] == base) { dists[i] = 1e30f; continue; }
        float dist = 0.0f; for (int f = 0; f < nf; f++) { float diff = x[base * nf + f] - x[class_indices[i] * nf + f]; dist += diff * diff; }
        dists[i] = dist;
    }
    for (int i = 0; i < k && i < n_class; i++) {
        int min_idx = i; for (int j = i + 1; j < n_class; j++) if (dists[order[j]] < dists[order[min_idx]]) min_idx = j;
        int tmp = order[i]; order[i] = order[min_idx]; order[min_idx] = tmp; neighbors[i] = class_indices[order[i]];
    }
    free(dists); free(order);
}
```

**Key structural facts to preserve when writing `find_knn_global()`:**
- Same partial-selection-sort pattern (selection sort over just the first `k` slots, not a full sort) — O(k·n) not O(n log n), matches project style.
- `dists[i] = 1e30f` sentinel for `i == base` so the base point never selects itself as its own neighbor — reuse this exact sentinel value for consistency.
- Allocates with `safe_malloc` (not raw `malloc`) — this project-wide convention (see `include/utils.h`) must be followed; `find_knn_global()` should use `safe_malloc(n_in * sizeof(...))` since it operates over **all** `n_in` samples, not just one class's `n_class` subset.
- Frees `dists`/`order` before returning — no leak, mirror exactly.
- **Divergence from the analog required by RESEARCH.md Pattern 1 / Pitfall 4:** `find_knn_global()` must call **zero `rng_*` functions** (the analog `find_knn()` also calls none today — this is already true, just must not regress) and must return the **count `kk` of neighbors actually found** (may be `< k` for tiny `n_in`), unlike `find_knn()` which has no return value. See RESEARCH.md's full code sketch (Architecture Patterns, Pattern 1) for the exact recommended signature: `static int find_knn_global(const float *x, const int *y, int base, int n_in, int nf, int k, int *neighbors)`.

### `classify_borderline()` (utility, transform — no analog, new logic)

No existing codebase analog — this is a genuinely new 3-way branch. RESEARCH.md's own code sketch (verified against Han/Wang/Mao 2005 primary source) is the pattern to copy verbatim, since branch **order** is load-bearing (Pitfall 1):
```c
static int classify_borderline(int m, int k)
{
    if (m == k) return 2;       /* NOISE */
    if (2 * m >= k) return 1;   /* BORDERLINE (danger) */
    return 0;                   /* SAFE */
}
```
Style note: matches the project's terse, single-purpose static-helper convention already visible in `map_to_binary_labels()` (`src/main.c:32-37`) — small, no logging inside the pure classifier itself (logging happens at the call site per SMOTE-03).

### `smote_oversample()` (service, modified in place)

**Analog:** itself, pre-modification (`src/main.c:205-231`) — this is a widen-in-place change, not a rewrite.

**Full existing function body** (the modification target — every line below either stays unchanged or gets a `smote_mode`-conditional branch inserted at the two marked points):
```c
static void smote_oversample(const float *x_in, const int *y_in, int n_in, int nf, int num_classes, float **x_out, int **y_out, int *n_out)
{
    int k = 5; int *counts = (int *)safe_calloc(num_classes, sizeof(int));
    for (int i = 0; i < n_in; i++) counts[y_in[i]]++;
    int max_count = 0; for (int c = 0; c < num_classes; c++) if (counts[c] > max_count) max_count = counts[c];
    *n_out = max_count * num_classes;
    *x_out = (float *)safe_malloc(*n_out * nf * sizeof(float));
    *y_out = (int *)safe_malloc(*n_out * sizeof(int));
    int **class_idx = (int **)safe_malloc(num_classes * sizeof(int *));
    int *class_pos = (int *)safe_calloc(num_classes, sizeof(int));
    for (int c = 0; c < num_classes; c++) class_idx[c] = (int *)safe_malloc(counts[c] * sizeof(int));
    for (int i = 0; i < n_in; i++) class_idx[y_in[i]][class_pos[y_in[i]]++] = i;
    int out_idx = 0;
    for (int c = 0; c < num_classes; c++) {
        int n_class = counts[c]; int knn = (k < n_class - 1) ? k : n_class - 1; if (knn < 1) knn = 1;
        for (int i = 0; i < n_class; i++) { memcpy(&(*x_out)[out_idx * nf], &x_in[class_idx[c][i] * nf], nf * sizeof(float)); (*y_out)[out_idx++] = c; }
        int n_synthetic = max_count - n_class; int *neighbors = (int *)safe_malloc(knn * sizeof(int));
        for (int s = 0; s < n_synthetic; s++) {
            /* INSERTION POINT A: if SMOTE_BORDERLINE, build class_idx_borderline[c] here
             * (once per class, before this synthesis loop) via find_knn_global()+classify_borderline(),
             * falling back to class_idx[c] with log_warn() if empty (SMOTE-03). */
            int base_idx = class_idx[c][rng_int(n_class)]; find_knn(x_in, base_idx, class_idx[c], n_class, nf, knn, neighbors);
            /* INSERTION POINT B: base_idx must be drawn from class_idx_borderline[c] instead of
             * class_idx[c] when smote_mode == SMOTE_BORDERLINE -- find_knn() call itself stays
             * UNCHANGED (still same-class interpolation only, per SMOTE-02). */
            int neighbor_idx = neighbors[rng_int(knn)]; float alpha = rng_uniform();
            for (int f = 0; f < nf; f++) (*x_out)[out_idx * nf + f] = x_in[base_idx * nf + f] + alpha * (x_in[neighbor_idx * nf + f] - x_in[base_idx * nf + f]);
            (*y_out)[out_idx++] = c;
        }
        free(neighbors); free(class_idx[c]);
    }
    free(class_idx); free(class_pos); free(counts);
}
```
**Critical invariant to preserve (from RESEARCH.md Pitfall 4):** the number of `rng_int()`/`rng_uniform()` calls per synthesis iteration must stay identical between `SMOTE_STANDARD` and `SMOTE_BORDERLINE` — only which array `rng_int(...)` indexes into changes. Do not add any `rng_*` call inside the classification step (Insertion Point A).

**Call sites to update (both in `mode_train()`, unchanged in structure otherwise):**
```c
/* src/main.c:318 -- Master, 2-class */
smote_oversample(tr_x_v, tr_y_bin, n_train_aug, nf_vowel, 2, &os_m_x, &os_m_y, &os_n_m);
/* src/main.c:335 -- Expert, 4-class */
smote_oversample(ex_tr_x, ex_tr_y, n_ex_tr, nf_vowel, 4, &os_e_x, &os_e_y, &os_n_e);
```
Both calls gain a trailing `smote_mode` argument; no other argument changes.

### `mode_smote_ab()` (controller, new CLI mode)

**Analog:** `mode_verify_rng()` — `src/main.c:446-466`

**Full existing analog function** (this is the established "dedicated fast-verification CLI mode" pattern from Phase 0 — copy its shape: log a mode banner, do the work, write artifact(s) to `results/`, log a summary, clean up, return 0/-1):
```c
static int mode_verify_rng(const char *base_dir)
{
    Dataset ds; char csv_p[1024]; snprintf(csv_p, 1024, "%s/%s", base_dir, CSV_METADATA);
    if (dataset_load(base_dir, csv_p, &ds) != 0) return -1;

    rng_seed(RANDOM_SEED);

    float *aug_cache = (float *)safe_calloc((size_t)ds.count * N_AUG_PER_SAMPLE * TOTAL_FEATURES, sizeof(float));
    precalculate_augmentations(&ds, TOTAL_FEATURES, aug_cache);

    char out_p[1024]; snprintf(out_p, 1024, "%s/aug_cache_verify.bin", RESULTS_DIR);
    FILE *f = fopen(out_p, "wb");
    if (!f) { free(aug_cache); dataset_free(&ds); return -1; }
    fwrite(aug_cache, sizeof(float), (size_t)ds.count * N_AUG_PER_SAMPLE * TOTAL_FEATURES, f);
    fclose(f);

    log_info("verify-rng: %d pacientes, %d aumentacoes/paciente, %d features -> %s",
              ds.count, N_AUG_PER_SAMPLE, TOTAL_FEATURES, out_p);

    free(aug_cache); dataset_free(&ds); return 0;
}
```
**What `mode_smote_ab()` should copy from this:** the "no separate translation unit, no CLI flags beyond `<mode> [base_dir]`, artifact written under `results/`, single `log_info` summary line" shape. **What it must differ on:** `mode_verify_rng()` does a single lightweight pass; `mode_smote_ab()` runs the equivalent of the full `mode_train()` pipeline twice (once per `SmoteMode`), per RESEARCH.md's `mode_train_ex()`-returning-`ABResult` sketch (Pattern 2, Architecture Patterns section) — this is a heavier, ~60-90-minute-class operation, not a fast check, so its log banner and expectations should say so explicitly (unlike `verify-rng`'s implicit "this is fast" framing).

**RESEARCH.md's recommended orchestration shape to follow (Pattern 2):**
```c
static int mode_smote_ab(const char *base_dir)
{
    log_info("=== MODO: A/B BORDERLINE-SMOTE (Gap 2) ===");
    ABResult res_standard, res_borderline;
    if (mode_train_ex(base_dir, SMOTE_STANDARD, &res_standard) != 0) return -1;
    if (mode_train_ex(base_dir, SMOTE_BORDERLINE, &res_borderline) != 0) return -1;
    write_ab_report(&res_standard, &res_borderline, "results/train_log_v32_gap2_smote_ab.txt");
    return 0;
}
```

### `main()` dispatch (route, modified)

**Analog:** itself, pre-modification — `src/main.c:474-484`

**Full existing dispatch block** (the new branch is a one-line insertion mirroring the `verify-rng` line exactly):
```c
int main(int argc, char *argv[])
{
    if (argc < 2) { fprintf(stderr, "Uso: %s <modo> [diretorio]\n", argv[0]); return 1; }
    const char *mode = argv[1]; const char *base_dir = (argc >= 3) ? argv[2] : ".";
    log_set_level(LOG_INFO); log_info("Classificador Vocals - Hierarchical Late Fusion");
    if (strcmp(mode, "extract") == 0) return mode_extract(base_dir) == 0 ? 0 : 1;
    if (strcmp(mode, "train") == 0 || strcmp(mode, "full") == 0) return mode_train(base_dir) == 0 ? 0 : 1;
    if (strcmp(mode, "external") == 0) return mode_validate_external(base_dir) == 0 ? 0 : 1;
    if (strcmp(mode, "verify-rng") == 0) return mode_verify_rng(base_dir) == 0 ? 0 : 1;
    return 1;
}
```
New line to add (mirrors `verify-rng` exactly, same ternary-to-exit-code idiom):
```c
if (strcmp(mode, "smote-ab") == 0) return mode_smote_ab(base_dir) == 0 ? 0 : 1;
```
Note: since `train`/`full` currently call `mode_train(base_dir)` with no mode argument, and RESEARCH.md Open Question 2 leaves open whether a 3rd optional CLI arg (`./build/vocal_detect train . borderline`) is also in scope, if the plan adopts that too, this line becomes `mode_train(base_dir) == 0` → `mode_train(base_dir, SMOTE_STANDARD) == 0` (default), with an optional `argv[3]` parse inserted between the `base_dir` assignment and the dispatch `if` chain — follow the same `argc >= 3 ? argv[2] : "."` idiom already used for `base_dir` (i.e. `argc >= 4 ? ... : SMOTE_STANDARD`).

## Shared Patterns

### Memory allocation convention
**Source:** used throughout `src/main.c` (e.g. `find_knn()` line 191-192, `smote_oversample()` lines 207-215)
**Apply to:** `find_knn_global()`, any new buffer in `mode_smote_ab()`/`ABResult` handling
```c
float *dists = (float *)safe_malloc(n_class * sizeof(float));
int *counts = (int *)safe_calloc(num_classes, sizeof(int));
```
Never use raw `malloc`/`calloc` — this project wraps allocation failure checks inside `safe_malloc`/`safe_calloc`/`safe_realloc` (declared in `include/utils.h`, used unconditionally across every `.c` file read in this session).

### Logging convention (for SMOTE-03's required warnings)
**Source:** `include/utils.h:59-61`, called via `log_info(...)`/`log_warn(...)`/`log_error(...)` throughout `src/main.c`
**Apply to:** the empty-borderline-pool fallback and the `n_class <= 1` case (SMOTE-03)
```c
log_warn("smote_oversample: classe %d sem amostras borderline (fold/vogal atual) -- usando class_idx[c] completo como fallback", c);
```
All log calls in this codebase are Portuguese-language, printf-style variadic — match this exactly (see CLAUDE.md "Comments" convention — this extends to log message language too, confirmed by every `log_info`/`log_warn` call read in `mode_train()`).

### RNG reseed as the single source of A/B determinism
**Source:** `src/main.c:245` (`rng_seed(RANDOM_SEED); ... kfold_split(fm.labels, fm.count, RANDOM_SEED, &splits);`), reinforced internally inside `kfold_split()` (`src/kfold.c:35`, calls `rng_seed(seed)` again at its own entry)
**Apply to:** `mode_smote_ab()` — do not manually snapshot/restore RNG state between the two arms; rely on `kfold_split()`'s own internal reseed, exactly as `mode_train()` already does today.
```c
rng_seed(RANDOM_SEED); KFoldSplits splits; kfold_split(fm.labels, fm.count, RANDOM_SEED, &splits);
```

### Statistical reporting infrastructure (already wired, Phase 0)
**Source:** `src/main.c:369-430` (existing `mode_train()` tail — bootstrap CI + McNemar + CSV export), signatures in `include/metrics.h`
**Apply to:** SMOTE-04's A/B report — reuse verbatim, called once per arm, not reimplemented
```c
ConfidenceInterval ci[CI_N_METRICS];
metrics_bootstrap_ci(all_y_true, all_y_pred, all_count, 1000, RANDOM_SEED, ci);

float chi2_maj, p_maj, chi2_knn, p_knn, chi2_lr, p_lr;
metrics_mcnemar(all_y_true, all_y_pred, all_y_pred_majority, all_count, &chi2_maj, &p_maj);
```
**Ordering constraint carried over from the existing comment at `src/main.c:372-375`:** `metrics_bootstrap_ci()` must be the *last* RNG-consuming call in each arm of `mode_smote_ab()`'s two `mode_train_ex()` invocations — do not add any `rng_*` call after it inside either arm.

### CSV/report file writing convention
**Source:** `src/main.c:406-430` (`results/bootstrap_ci.csv`, `results/mcnemar_vs_baselines.csv` writers)
**Apply to:** SMOTE-04's safe/borderline/noise count table and A/B comparison CSV(s)
```c
FILE *ci_f = fopen("results/bootstrap_ci.csv", "w");
if (ci_f) {
    fprintf(ci_f, "metric,mean,ci_lower,ci_upper\n");
    fprintf(ci_f, "accuracy,%.6f,%.6f,%.6f\n", ci[CI_ACCURACY].mean, ci[CI_ACCURACY].lower, ci[CI_ACCURACY].upper);
    fclose(ci_f);
} else {
    log_error("Falha ao abrir results/bootstrap_ci.csv para escrita");
}
```
Convention: `fopen` guarded with `if (ci_f)`, `log_error()` (Portuguese) on failure, no crash — mirror for any new CSV artifact this phase produces.

## No Analog Found

| Unit | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `classify_borderline()` | utility (pure classifier) | transform | No 3-way classification helper exists anywhere in the codebase today — this is genuinely new logic. Use RESEARCH.md's own Han/Wang/Mao-verified code sketch (reproduced above under "Pattern Assignments") rather than searching further; the project convention to follow is only the *style* (terse static helper, matching `map_to_binary_labels()`), not a structural analog. |
| `ABResult` struct / `mode_train_ex()` (if the plan adopts RESEARCH.md's exact sketch) | model/data-holder | N/A | No existing struct in this codebase currently packages "one full pipeline run's aggregated predictions + metrics" for programmatic reuse — `mode_train()` today prints/exports everything inline and returns only an `int` status code. If the plan adopts the `mode_train_ex()` refactor, this is new struct design, not a copy from an existing analog; the closest partial precedent is `MetricsResult`/`ConfidenceInterval` (`include/metrics.h`) as a *style* reference for a flat, plain-data struct with no methods. |

## Metadata

**Analog search scope:** `src/main.c` (full file read across two non-overlapping ranges: lines 1-60 and 134-485), `src/utils.c` (RNG + logging implementations, lines 54-84, 161-185), `include/utils.h` (RNG/logging declarations), `include/metrics.h` (bootstrap CI / McNemar signatures), `include/config.h` (constant-placement convention, lines 1-70, 110-120)
**Files scanned:** 1 primary modification target (`src/main.c`, 485 lines, read in full) + 4 supporting headers/sources for signature/convention verification
**Pattern extraction date:** 2026-07-27
