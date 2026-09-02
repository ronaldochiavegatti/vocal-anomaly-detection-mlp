# Phase 3: Gap 1 — Paraconsistent Feature Selection & Final Reporting - Pattern Map

**Mapped:** 2026-07-29
**Files analyzed:** 3 (2 new, 1 modified-in-place with 5 distinct integration points)
**Analogs found:** 3 / 3 (no file lacks an analog; this codebase has two prior "gap" implementations — SMOTE A/B and arch-compare — that are near-exact structural precedents for everything this phase needs)

No CONTEXT.md exists for this phase (user chose to continue without one, per orchestrator note) — the file list below is derived entirely from `03-RESEARCH.md`'s Architecture Patterns / Recommended file/module structure section and phase requirements table (PARA-01..06, CROSS-01/02).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `include/feature_select_paraconsistent.h` | utility (module header) | transform (pure function, no I/O) | `include/feature_select.h` | exact (sibling module, same directory, same terse doc-comment convention) |
| `src/feature_select_paraconsistent.c` | utility/service (numeric feature-selection algorithm) | batch/transform | `src/normalize.c` (`norm_fit`, per-feature stats loop) + `src/main.c`'s `smote_oversample()` relaxation/fallback block (lines 279-334) | role-match (normalize.c) + data-flow-match (smote_oversample's capped-relaxation idiom) |
| `src/main.c` — insertion in `mode_train_ex()`'s fold/vowel loop (PARA-03) | controller/orchestration (training loop) | CRUD-like (compute → persist → consume) | `src/main.c`'s own `smote_oversample()` call site at lines 503-516 (same loop, same "compute then feed into mlp_init") | exact (same function, same loop, immediately adjacent insertion point) |
| `src/main.c` — `predict_hierarchical_late_fusion()` refactor + deletion of duplicate slicing block (PARA-04) | controller (inference/prediction) | request-response (one sample in, prediction+probabilities out) | itself, `src/main.c:39-69` (discrete-prediction half) vs. `src/main.c:553-569` (probability-recording half, to be deleted) | exact (refactor target, not an external analog) |
| `src/main.c` — `write_gap1_report()` / `write_gap_adoption_status()` (PARA-05, CROSS-01) | reporting/service (report writer) | batch (aggregate → CSV/txt) | `write_smote_ab_report()` (`src/main.c:727-813`) and `write_arch_compare_report()` (`src/main.c:839-948`) | exact (same DECISAO-sentence + CSV-export pattern, same project) |
| `src/main.c` — `mode_paraconsistent_ab()` (new CLI entry) | controller (CLI mode dispatcher) | batch/orchestration | `mode_smote_ab()` (`src/main.c:819-832`) and `mode_arch_compare()` (`src/main.c:955-1003`) | exact |
| `results/paraconsistent_selection_freq.csv` (or similar, PARA-05 output) | reporting output (generated artifact, not authored code) | batch (incremental append) | `results/smote_borderline_counts.csv` generation pattern (`src/main.c:420-434`, `506-511`, `534-539`) | exact |
| `models/selected_master_fold{k}_v{vowel}.bin` / `selected_expert_fold{k}_v{vowel}.bin` (PARA-03 persistence) | model artifact (binary persistence) | file-I/O | Existing `selected_save()`/`selected_load()` in `src/feature_select.c` — reuse unchanged, do not create a new format | exact (explicitly mandated reuse, per SPEC.md and RESEARCH.md "Don't Hand-Roll") |
| `CLAUDE.md` (PARA-06 update) | documentation | — | The existing "Gap 2 Outcome" and "Gap 3 Outcome" sections already in `CLAUDE.md` | exact (format precedent already in the same file) |

## Pattern Assignments

### `include/feature_select_paraconsistent.h` (utility header, transform)

**Analog:** `include/feature_select.h` (full file, 20 lines — reproduced below in full since it is the entire analog)

```c
#ifndef FEATURE_SELECT_H
#define FEATURE_SELECT_H

/*
 * feature_select.h - Persistencia dos indices de features selecionadas por fold
 *
 * Formato binario: [int n_selected][int idx_0]...[int idx_{n-1}]
 */

/* Salva n indices em path. Retorna 0 em sucesso, -1 em erro. */
int selected_save(const char *path, const int *indices, int n);

/*
 * Carrega indices de path. Preenche indices[] e *n.
 * Retorna 0 em sucesso, -1 em erro (incluindo se o n lido for <= 0 ou > TOTAL_FEATURES).
 * O buffer indices[] deve ter capacidade para pelo menos TOTAL_FEATURES elementos.
 */
int selected_load(const char *path, int *indices, int *n);

#endif /* FEATURE_SELECT_H */
```

**Copy exactly:**
- Terse top-of-file `/* modulename.h - one-line purpose */` banner comment, followed by a short paragraph explaining the on-disk/data format when relevant.
- `#ifndef MODULE_H` / `#define MODULE_H` / `#endif /* MODULE_H */` guard style (matches every header in `include/`).
- Doc comment directly above each function prototype (`/* ... */` block, describing parameters + return code semantics, `0`=success / `-1`=error convention).
- No struct needed here (this module is stateless: pure function `paraconsistent_select()` + optional out-buffers) — do **not** invent a `ParaconsistentResult` struct unless the planner decides one is needed for the PARA-05 report data (mu/lambda/gc/gct arrays can be plain `float *` out-parameters, exactly as specified in SPEC.md's own signature and RESEARCH.md's Code Examples section).

**Signature to declare** (from `SPEC.md` lines 89-90, extended per RESEARCH.md's Code Examples with optional out-buffers for PARA-05's report):
```c
int paraconsistent_select(const float *x, const int *y, int n, int nf, int n_classes,
                           float gc_thresh, float gct_max, int *selected,
                           float *mu_out, float *lambda_out,
                           float *gc_out, float *gct_out);
```

---

### `src/feature_select_paraconsistent.c` (utility/service, batch transform)

**Primary analog:** `src/normalize.c` (full file read, 100 lines) — for the "single pass over an `[n x num_features]` row-major training matrix, compute per-feature statistics" shape.

**Imports pattern** (`src/normalize.c` lines 8-13):
```c
#include "normalize.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
```
Apply the same import shape to the new file: `#include "feature_select_paraconsistent.h"`, `#include "utils.h"` (for `safe_malloc`/`safe_calloc`/`log_warn`), `#include "config.h"` (only if any constant like `TOTAL_FEATURES` is needed — likely not, since `nf` is passed in), plus `<math.h>` for `sqrtf`/`fabsf`, `<string.h>` for `memcpy` (out-buffer copies).

**Variance-floor guard pattern** (`src/normalize.c` lines 15, 40-43 — this is the direct precedent for `MIN_VAR`/`MIN_STD` guards PARA-02 requires):
```c
#define MIN_STD 1e-8f
...
for (int j = 0; j < num_features; j++) {
    params->std[j] = sqrtf(params->std[j] / n);
    if (params->std[j] < MIN_STD) params->std[j] = MIN_STD;
}
```
Reuse this exact `MIN_STD = 1e-8f` constant name/value (RESEARCH.md explicitly calls this out: "a variance floor analogous to `normalize.c`'s `MIN_STD = 1e-8f`, but squared since SST is a sum-of-squares"). Define a sibling `MIN_VAR 1e-16f` in the new file for the SST/eta-squared guard, and reuse `MIN_STD 1e-8f` (or import it) for the lambda global-std floor — RESEARCH.md's Code Examples section already has both constants worked out (see below).

**Core numeric pattern — one-way ANOVA η² per feature** (from RESEARCH.md's "Code Examples", verified standard formula, no direct in-repo precedent for the *statistic* itself, only for the *loop shape* via `normalize.c`):
```c
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

**Lambda pattern (unweighted per-class σ/global-σ ratio):**
```c
static float feature_lambda(const float *x, const int *y, int n, int nf,
                             int feat_idx, int n_classes, float global_std)
{
    if (global_std < MIN_STD) global_std = MIN_STD;
    /* ... accumulate per-class mean/var, ratio = std_c/global_std clipped to [0,1],
     * lambda = unweighted mean of ratio over classes with class_n[c] >= 2 ... */
}
```
(Full body already worked out in `03-RESEARCH.md` lines 260-293 — copy verbatim, it is already vetted against the degeneracy trap described in that same section.)

**Relaxation-loop + guaranteed-non-empty-fallback pattern — analog:** `smote_oversample()`'s empty-borderline-pool fallback, `src/main.c:313-315`:
```c
if (n_borderline == 0) {
    log_warn("smote_oversample: classe %d sem amostras borderline (fold/vogal atual) -- usando class_idx[c] completo como fallback", c);
}
```
And the pool-selection fallback that immediately follows it (`src/main.c:318-322`):
```c
int *pool = class_idx[c]; int pool_size = n_class;
if (smote_mode == SMOTE_BORDERLINE && n_borderline > 0) { pool = class_idx_borderline; pool_size = n_borderline; }
```
**Copy this exact idiom**: compute a candidate result, check if it's empty, `log_warn` with a message describing what triggered the fallback and what the fallback does, then fall back to a value that is *guaranteed non-degenerate* (full pool here; full `nf` feature set in the paraconsistent case). RESEARCH.md's Code Examples section already applies this idiom to `paraconsistent_select()`'s relaxation loop (10-iteration cap, `PARA_GC_RELAX_STEP=0.05f`, fallback = select all `nf` features) — copy that block verbatim, it mirrors the SMOTE-03 precedent line-for-line in spirit.

**Persistence — do NOT write new code, reuse as-is:**
```c
/* include/feature_select.h — already the correct shape, call unchanged: */
int selected_save(const char *path, const int *indices, int n);
int selected_load(const char *path, int *indices, int *n);
```
Call `selected_save(path_for(f, v, "master"), sel_m, ns_m)` / `selected_save(path_for(f, v, "expert"), sel_e, ns_e)` after each `paraconsistent_select()` call, one file per (fold, vowel, network) combination — no new `.c`/`.h` needed for persistence.

---

### `src/main.c` — insertion in `mode_train_ex()`'s fold/vowel loop (PARA-03)

**Analog:** the existing `smote_oversample()` call site in the same loop, `src/main.c` lines 503-516 (Master) and 532-544 (Expert) — this is the established idiom for "compute a per-(fold,vowel,network) transform on `tr_x_v`/`ex_tr_x`, feed the result into `mlp_init_multi`/`mlp_train`".

**Exact current Master block to insert before** (`src/main.c:503-518`):
```c
SmoteBorderlineCounts sbc_master = {0}, sbc_expert = {0};
float *os_m_x; int *os_m_y, os_n_m;
smote_oversample(tr_x_v, tr_y_bin, n_train_aug, nf_vowel, 2, smote_mode, &sbc_master, &os_m_x, &os_m_y, &os_n_m);
if (counts_f) {
    for (int c = 0; c < 2; c++) {
        fprintf(counts_f, "%d,%d,master,%d,%d,%d,%d\n", f, v, c,
                sbc_master.safe[c], sbc_master.borderline[c], sbc_master.noise[c]);
    }
}
mlp_init_multi(&net_master[v], nf_vowel, 2, arch->hidden_sizes, arch->n_hidden, eff_dropout);
if (f == 0 && v == 0) { param_count_master = mlp_count_params(&net_master[v]); }
TrainHistory h_m;
double t0_m = timer_now();
mlp_train(&net_master[v], os_m_x, os_m_y, os_n_m, vl_x_v, vl_y_bin, fold->n_val, nf_vowel, 2, cw_binary, eff_l2, &h_m);
```

**Insertion point (per RESEARCH.md Architecture Patterns):** `paraconsistent_select()` runs on **`tr_x_v`'s first `fold->n_train` rows only** (not `n_train_aug`, which includes 8x audio-augmented rows appended by `collect_augmented_features()` — see Pitfall 3), *before* `smote_oversample()` and *before* `mlp_init_multi()`. After selection, `slice_columns()`-style application must reduce `tr_x_v`/`vl_x_v` from `nf_vowel` columns down to `ns_m` columns, and every subsequent `nf_vowel` literal in that block (the `smote_oversample(...)` call, `mlp_init_multi(...)` call) becomes `ns_m`/`ns_e`.

**Directly reuse `norm_fit`'s "original rows only" precedent** — this is already documented in `CLAUDE.md`'s Methodological Notes and exercised at `src/main.c:458`:
```c
NormParams norm; norm_fit(train_x_all, fold->n_train, nf_all, &norm);
norm_transform(train_x_all, n_train_aug, &norm);
```
Note the asymmetry: `norm_fit` is called with `fold->n_train` (fit only on original rows) but `norm_transform` is called with `n_train_aug` (applied to all rows, original + augmented). `paraconsistent_select()` must follow the identical asymmetry: **compute** on `fold->n_train` rows, **apply** (column-slice) to all `n_train_aug`/`fold->n_val` rows.

**Expert-side filtering precedent** (`src/main.c:520-531`, already filters by "original rows only" implicitly through `n_train_aug` — for PARA-03's Expert computation, this must be tightened to `fold->n_train`-prefix rows specifically, per RESEARCH.md line 140):
```c
int n_ex_tr = 0; for (int i = 0; i < n_train_aug; i++) if (train_y_all[i] != CLASS_NORMAL) n_ex_tr++;
float *ex_tr_x = (float *)safe_malloc(n_ex_tr * nf_vowel * sizeof(float));
int *ex_tr_y = (int *)safe_malloc(n_ex_tr * sizeof(int));
int cur = 0; for (int i = 0; i < n_train_aug; i++) if (train_y_all[i] != CLASS_NORMAL) {
    memcpy(&ex_tr_x[cur * nf_vowel], &tr_x_v[i * nf_vowel], nf_vowel * sizeof(float)); ex_tr_y[cur++] = train_y_all[i] - 1;
}
```
For the paraconsistent computation specifically, filter `train_y_all[0:fold->n_train]` (not `[0:n_train_aug]`) — a *separate*, narrower loop than the one shown above, which stays as-is for the SMOTE/training step.

---

### `src/main.c` — `predict_hierarchical_late_fusion()` refactor (PARA-04)

**This is a refactor of the file's own two divergent blocks, not an external analog.** Both blocks are reproduced here in full since PARA-04's entire risk is their divergence.

**Block 1 — discrete prediction** (`src/main.c:39-69`, current full function):
```c
static int predict_hierarchical_late_fusion(MLP master[3], MLP expert[3], 
                                            const float *x_all)
{
    float prob_pathology = 0.0f;
    float prob_expert[4] = {0, 0, 0, 0};
    int meta_offset = NUM_VOWELS * FEATURES_PER_VOWEL;

    for (int v = 0; v < 3; v++) {
        float x_v[FEATURES_PER_VOWEL + NUM_METADATA_FEATURES];
        memcpy(x_v, &x_all[v * FEATURES_PER_VOWEL], FEATURES_PER_VOWEL * sizeof(float));
        memcpy(&x_v[FEATURES_PER_VOWEL], &x_all[meta_offset], NUM_METADATA_FEATURES * sizeof(float));

        float out_m[2];
        mlp_forward(&master[v], x_v, out_m, 0);
        prob_pathology += out_m[1];

        float out_e[4];
        mlp_forward(&expert[v], x_v, out_e, 0);
        for (int c = 0; c < 4; c++) prob_expert[c] += out_e[c];
    }

    if ((prob_pathology / 3.0f) < 0.5f) {
        return CLASS_NORMAL;
    } else {
        int best_c = 0;
        for (int c = 1; c < 4; c++) {
            if (prob_expert[c] > prob_expert[best_c]) best_c = c;
        }
        return best_c + 1;
    }
}
```

**Block 2 — duplicated inline slicing for probability recording, TO BE DELETED** (`src/main.c:553-569`, inside the fold's validation loop):
```c
for (int i = 0; i < fold->n_val; i++) {
    const float *x_samp = &val_x_all[i * nf_all];
    all_y_pred[all_count] = predict_hierarchical_late_fusion(net_master, net_expert, x_samp);
    all_y_true[all_count] = fm.labels[fold->val_indices[i]];
    float p_norm = 0, p_exp[4] = {0};
    for (int v = 0; v < 3; v++) {
        float xv[251]; memcpy(xv, &x_samp[v * FEATURES_PER_VOWEL], FEATURES_PER_VOWEL * sizeof(float));
        memcpy(&xv[FEATURES_PER_VOWEL], &x_samp[3 * FEATURES_PER_VOWEL], 2 * sizeof(float));
        float om[2], oe[4]; mlp_forward(&net_master[v], xv, om, 0); mlp_forward(&net_expert[v], xv, oe, 0);
        p_norm += om[0]; for(int c=0; c<4; c++) p_exp[c] += om[1] * oe[c];
    }
    all_y_prob[all_count * 5 + 0] = p_norm / 3.0f; for(int c=1; c<5; c++) all_y_prob[all_count * 5 + c] = p_exp[c-1] / 3.0f;
    all_y_pred_majority[all_count] = majority_class;
    all_y_pred_knn[all_count] = knn_pred_buf[i];
    all_y_pred_logreg[all_count] = logreg_pred_buf[i];
    all_count++;
}
```
Note the `float xv[251]` magic-number stack buffer (DEBT-01, flagged in RESEARCH.md Pitfall 4) — **do not propagate `251` as a literal into the refactored function**; use `nf_vowel` (already computed once at `src/main.c:415` as `FEATURES_PER_VOWEL + NUM_METADATA_FEATURES`) or `FEATURES_PER_VOWEL + NUM_METADATA_FEATURES` directly.

**Recommended refactored signature** (per RESEARCH.md's "Recommended fix"):
```c
static int predict_hierarchical_late_fusion(MLP master[3], MLP expert[3],
                                             const float *x_all,
                                             const int sel_m[3][/*ns_m[v] upper bound: nf_vowel*/], const int ns_m[3],
                                             const int sel_e[3][/*ns_e[v] upper bound: nf_vowel*/], const int ns_e[3],
                                             float *p_norm_out, float p_exp_out[4]);
```
Collapse both blocks into this single function (owns slicing + selection + forward pass, returns both the discrete class *and* the probabilities via out-parameters), then delete Block 2 entirely and call the refactored function once per validation sample at the former Block 2 call site.

---

### `src/main.c` — `write_gap1_report()` / `write_gap_adoption_status()` (PARA-05, CROSS-01)

**Analog 1 — DECISAO-sentence pattern:** `write_smote_ab_report()`, `src/main.c:727-813` (full function). Key excerpt — the fixed adopt/reject rule, never asserted by inspection (`src/main.c:775-781`):
```c
if (bl_res->macro_f1 >= std_res->macro_f1) {
    fprintf(f, "DECISAO: Borderline-SMOTE ADOTADO (Macro F1 borderline=%.4f >= padrao=%.4f, delta=%+.4f, McNemar chi2=%.4f p=%.4f)\n",
            bl_res->macro_f1, std_res->macro_f1, bl_res->macro_f1 - std_res->macro_f1, chi2_ab, p_ab);
} else {
    fprintf(f, "DECISAO: Borderline-SMOTE REJEITADO (Macro F1 borderline=%.4f < padrao=%.4f, delta=%+.4f, McNemar chi2=%.4f p=%.4f) -- mantendo SMOTE padrao em producao\n",
            bl_res->macro_f1, std_res->macro_f1, bl_res->macro_f1 - std_res->macro_f1, chi2_ab, p_ab);
}
```
For PARA-05, the fixed rule is different (SPEC.md's acceptance criterion, Open Question 3): adopt if `Macro F1` equal-or-better, **or** up to −0.01 worse with ≥30% feature reduction — implement this exact trade-off as fixed `if`/`else if`/`else` code, mirroring the structure above (never a narrative-only judgment call, per RESEARCH.md's recommendation on Open Question 3).

**Analog 2 — direct McNemar-between-two-branches pattern** (`src/main.c:735-736`):
```c
float chi2_ab, p_ab;
metrics_mcnemar(std_res->y_true, bl_res->y_pred, std_res->y_pred, std_res->n, &chi2_ab, &p_ab);
```
Reuse identically for "with paraconsistent selection" vs "without" — call `metrics_mcnemar()` on the two branches' out-of-fold predictions (valid because both branches share the same seed/fold order, exactly as documented in the comment above this call).

**Analog 3 — CSV comparison export** (`src/main.c:785-812`, the `metric,standard,standard_ci_lower,...` fprintf block) — reuse the same column pattern (`metric,without_selection,...ci_lower,...ci_upper,with_selection,...`) for `results/paraconsistent_ab_comparison.csv`.

**Analog 4 — CROSS-01's Gap Adoption Status 3-row table:** `write_arch_compare_report()`'s multi-step decision structure (`src/main.c:839-948`) is the closest analog for combining *three* separate gap outcomes into one consolidated table — reuse its `fprintf(f, "arch,reg,...")`-style CSV row-per-decision pattern, but with columns `gap,decision,macro_f1_delta,mcnemar_p,citation_status` as specified in RESEARCH.md's CROSS-01 table (already drafted with real Gap 2/Gap 3 numbers — Gap 1's row is `TBD` until this phase's comparison runs).

---

### `src/main.c` — `mode_paraconsistent_ab()` (new CLI entry)

**Analog:** `mode_smote_ab()`, `src/main.c:819-832` (full function):
```c
static int mode_smote_ab(const char *base_dir)
{
    log_info("=== MODO: A/B BORDERLINE-SMOTE (Gap 2) ===");
    log_info("Atencao: modo de longa duracao (~60-180 min) -- executa o pipeline hierarquico completo duas vezes (uma por modo SMOTE)");
    ABResult res_standard = {0}, res_borderline = {0};
    if (mode_train_ex(base_dir, SMOTE_STANDARD, &ARCH_CONFIGS[2], REG_BASELINE, &res_standard) != 0) return -1;
    if (mode_train_ex(base_dir, SMOTE_BORDERLINE, &ARCH_CONFIGS[2], REG_BASELINE, &res_borderline) != 0) return -1;
    write_smote_ab_report(&res_standard, &res_borderline,
                           "results/train_log_v32_gap2_smote_ab.txt",
                           "results/smote_ab_comparison.csv");
    free(res_standard.y_true); free(res_standard.y_pred);
    free(res_borderline.y_true); free(res_borderline.y_pred);
    return 0;
}
```
**Critical detail to copy exactly (Pitfall 5 / RESEARCH.md Summary):** both calls pass `SMOTE_BORDERLINE, &ARCH_CONFIGS[2], REG_BASELINE` explicitly (mirroring `mode_arch_compare()`'s C/baseline arm) — **never** call `mode_train()` or rely on its unmodified `SMOTE_STANDARD` default. `mode_paraconsistent_ab()` must explicitly pass `mode_train_ex(base_dir, SMOTE_BORDERLINE, &ARCH_CONFIGS[2], REG_BASELINE, &res_with_selection/&res_without_selection)` for both its "before" and "after" arms — the *only* varying factor between the two arms should be whether `paraconsistent_select()` runs inside `mode_train_ex()`'s fold/vowel loop, which likely requires a new boolean/enum parameter threaded through `mode_train_ex()`'s signature (analogous to how `smote_mode`/`arch`/`reg` are already threaded through).

**CLI dispatch registration analog** (`src/main.c:1020-1021`, inside `main()`):
```c
if (strcmp(mode, "smote-ab") == 0) return mode_smote_ab(base_dir) == 0 ? 0 : 1;
if (strcmp(mode, "arch-compare") == 0) return mode_arch_compare(base_dir) == 0 ? 0 : 1;
```
Add `if (strcmp(mode, "paraconsistent-ab") == 0) return mode_paraconsistent_ab(base_dir) == 0 ? 0 : 1;` in the same style, same location.

---

### Selection-frequency incremental CSV (PARA-05 report data)

**Analog:** the `smote_borderline_counts.csv` incremental-write pattern.

**Open-with-header** (`src/main.c:420-434`):
```c
FILE *counts_f = NULL;
if (smote_mode == SMOTE_BORDERLINE) {
    char counts_path[160];
    if (result != NULL) {
        snprintf(counts_path, sizeof(counts_path), "results/smote_borderline_counts_%s_%s.csv", arch->name, REG_NAME[reg]);
    } else {
        snprintf(counts_path, sizeof(counts_path), "results/smote_borderline_counts.csv");
    }
    counts_f = fopen(counts_path, "w");
    if (counts_f) {
        fprintf(counts_f, "fold,vowel,network,class,safe,borderline,noise\n");
    } else {
        log_error("Falha ao abrir %s para escrita", counts_path);
    }
}
```

**Append-per-iteration** (`src/main.c:506-511`):
```c
if (counts_f) {
    for (int c = 0; c < 2; c++) {
        fprintf(counts_f, "%d,%d,master,%d,%d,%d,%d\n", f, v, c,
                sbc_master.safe[c], sbc_master.borderline[c], sbc_master.noise[c]);
    }
}
```

**Close at end of function** (`src/main.c:664`): `if (counts_f) fclose(counts_f);`

Apply this identical open/append/close idiom for PARA-05's `feature,mu,lambda,gc,gct,selected` rows (columns per RESEARCH.md line 150: `fold,vowel,network,feature_idx,mu,lambda,gc,gct,selected`) — open once at the top of `mode_train_ex()` (only when the new paraconsistent-selection flag is active, mirroring the `smote_mode == SMOTE_BORDERLINE` guard), append one block per (fold, vowel, network) iteration immediately after each `paraconsistent_select()` call, close at the end.

## Shared Patterns

### Guard-against-degenerate-input (variance/pool floors)
**Source:** `src/normalize.c:15,40-43` (`MIN_STD 1e-8f`) and `src/main.c:313-322` (SMOTE-03 empty-pool fallback)
**Apply to:** `feature_eta_squared()` (SST floor `MIN_VAR`), `feature_lambda()` (global-std floor `MIN_STD`), and `paraconsistent_select()`'s relaxation loop (all-features fallback)
```c
#define MIN_STD 1e-8f   /* normalize.c precedent, reuse the same name/value */
if (params->std[j] < MIN_STD) params->std[j] = MIN_STD;
```

### Fixed-rule adopt/reject decisions (never narrative)
**Source:** `write_smote_ab_report()` (`src/main.c:775-781`) and `write_arch_compare_report()`'s 5-step procedure (`src/main.c:841-890`)
**Apply to:** `write_gap1_report()`'s adopt/reject rule (SPEC.md's trade-off criterion) and `write_gap_adoption_status()`'s consolidated 3-row table
```c
if (bl_res->macro_f1 >= std_res->macro_f1) {
    fprintf(f, "DECISAO: ... ADOTADO (...)\n");
} else {
    fprintf(f, "DECISAO: ... REJEITADO (...)\n");
}
```

### `fold->n_train`-prefix-only computation (no augmented/SMOTE rows)
**Source:** `CLAUDE.md`'s Methodological Notes; exercised at `src/main.c:458` (`norm_fit(train_x_all, fold->n_train, nf_all, &norm)`)
**Apply to:** `paraconsistent_select()`'s call sites for both Master (`tr_x_v[0:fold->n_train]`) and Expert (rows of `tr_x_v[0:fold->n_train]` filtered by `train_y_all[i] != CLASS_NORMAL`)
```c
NormParams norm; norm_fit(train_x_all, fold->n_train, nf_all, &norm);   /* fit: original rows only */
norm_transform(train_x_all, n_train_aug, &norm);                        /* apply: all rows (orig+aug) */
```

### Bootstrap CI / McNemar significance testing
**Source:** `include/metrics.h:71-73,114-117`, called at `src/main.c:609,614-616,736,874`
**Apply to:** `mode_paraconsistent_ab()`'s comparison (with-selection vs without-selection arms) and `write_gap_adoption_status()`'s consolidated table
```c
void metrics_bootstrap_ci(const int *y_true, const int *y_pred, int n_samples,
                          int n_bootstrap, unsigned int seed,
                          ConfidenceInterval results[CI_N_METRICS]);
void metrics_mcnemar(const int *y_true,
                     const int *y_pred_a, const int *y_pred_b,
                     int n_samples,
                     float *chi2_out, float *p_value_out);
```

### Binary persistence with bounds-checked load
**Source:** `src/feature_select.c` (full file, `selected_save`/`selected_load`)
**Apply to:** persisting `sel_m[v]`/`sel_e[v]` per (fold, vowel, network) — reuse unchanged, no new format
```c
int selected_save(const char *path, const int *indices, int n);
int selected_load(const char *path, int *indices, int *n);   /* bounds-checks *n <= TOTAL_FEATURES on load */
```

## No Analog Found

None. Every file/insertion-point in this phase's scope has a strong, directly-reusable in-repo analog — this is explicitly called out in `03-RESEARCH.md`'s "Don't Hand-Roll" section: "every piece of reusable infrastructure this phase needs (persistence, stats, reporting scaffolding) already exists and is proven by two prior gaps — the only genuinely new code is the μ/λ/Gc/Gct computation itself and its insertion point." The μ/λ computation formulas themselves (`feature_eta_squared`, `feature_lambda`) have no in-repo precedent for the *statistic* (they are new domain math), but RESEARCH.md's Code Examples section already provides complete, ready-to-copy C99 implementations, and the *loop shape* (row-major `[n x nf]` matrix, per-feature accumulation) directly follows `normalize.c`'s established idiom.

## Metadata

**Analog search scope:** `include/*.h` (17 files), `src/*.c` (19 files) — focused reads on `feature_select.{c,h}`, `normalize.{c,h}`, `metrics.{c,h}`, `main.c` (full file, 1024 lines, read in two passes), `config.h`, `mlp.h` (signature grep), `mlp_train.h` (signature grep), `utils.h` (signature grep)
**Files scanned:** 8 read in full or near-full (`feature_select.h`, `feature_select.c`, `normalize.h`, `normalize.c`, `metrics.h`, `config.h`, `main.c` ×2 passes), 3 grepped for signatures only (`mlp.h`, `mlp_train.h`, `utils.h`)
**Pattern extraction date:** 2026-07-29
