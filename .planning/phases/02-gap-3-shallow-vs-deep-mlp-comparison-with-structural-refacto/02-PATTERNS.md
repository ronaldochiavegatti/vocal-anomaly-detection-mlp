# Phase 2: Gap 3 — Shallow vs Deep MLP Comparison (with structural refactor) - Pattern Map

**Mapped:** 2026-07-27
**Files analyzed:** 5 (all modified, 0 new)
**Analogs found:** 5 / 5 (all are self-analogs — this phase generalizes existing functions in place, so the "closest analog" for each file is almost always the current version of the same file/function, plus one true cross-file precedent: Phase 1's `ABResult`/`mode_train_ex`/`mode_smote_ab` pattern in `src/main.c`)

## File Classification

| Modified File | Role | Data Flow | Closest Analog | Match Quality |
|----------------|------|-----------|-----------------|---------------|
| `include/mlp.h` | model/config (struct definition) | transform (struct layout only, no I/O) | Itself — current `MLP`/`Layer` struct + `mlp_init_dynamic()` declaration | exact (in-place generalization) |
| `include/config.h` | config | transform (compile-time constants) | Itself — existing `MLP_NUM_LAYERS`/`MLP_HIDDEN*_SIZE`/`DROPOUT_RATE_HIDDEN*` banner block | exact |
| `src/mlp.c` | model (forward/backward/init) | transform (CRUD-like: init/train/predict lifecycle on in-memory `MLP`) | Itself — `mlp_init_dynamic()` (`mlp.c:220-241`) is the exact analog for `mlp_init_multi()`; `mlp_backward()` (`mlp.c:282-369`) is the exact analog for its own dynamic-`max_size` fix | exact |
| `src/mlp_train.c` | service (training loop) | batch (mini-batch epoch loop) | Itself — checkpoint/SWA buffer declarations (`mlp_train.c:183-209`) | exact |
| `src/main.c` | orchestration/controller (CLI mode dispatch) | request-response (CLI mode → pipeline run → report files) | Phase 1's `ABResult` struct + `mode_train_ex()`/`mode_smote_ab()`/`write_smote_ab_report()` (`src/main.c:339-769`) | exact (cross-cutting precedent, not self) |

## Pattern Assignments

### `include/mlp.h` (model struct, transform)

**Analog:** itself, current struct + declarations (`include/mlp.h:61-77`)

**Current struct** (lines 61-66):
```c
/* Rede MLP completa */
typedef struct {
    Layer layers[MLP_NUM_LAYERS];  /* 3 camadas: hidden1, hidden2, output */
    int num_layers;
    int timestep;                  /* contador para Adam */
} MLP;
```

**Required change:** replace `MLP_NUM_LAYERS` (compile-time `3`, defined in `config.h:81`) with a new `MLP_MAX_LAYERS` (recommend `5`, per RESEARCH.md Assumption A3) **only** in this array-sizing position. Do NOT touch `net->num_layers` (already a runtime field, already the only thing every loop in `mlp.c`/`mlp_train.c` iterates by — confirmed by direct read this session).

**Declaration to add** (sibling to existing `mlp_init_dynamic` declaration at line 77):
```c
void mlp_init_dynamic(MLP *net, int input_size, int output_size);
```
New declarations following this exact doc-comment house style (matches `metrics.h`'s docstring convention cited in RESEARCH.md — parameter shapes in `[n]` notation):
```c
/*
 * Inicializa MLP com arquitetura configuravel: hidden_sizes[n_hidden] define a largura
 * de cada camada oculta, dropout_rates[n_hidden] a taxa de dropout correspondente.
 * net->num_layers e definido internamente como (n_hidden + 1).
 */
void mlp_init_multi(MLP *net, int input_size, int output_size,
                     const int *hidden_sizes, int n_hidden,
                     const float *dropout_rates);

/*
 * Retorna o numero total de parametros treinaveis (pesos + biases, somados sobre
 * todas as net->num_layers camadas). BN esta desabilitado em todo o pipeline atual
 * (use_bn=0 em todas as chamadas de layer_init), portanto nao contribui.
 */
int mlp_count_params(const MLP *net);
```

**Doc-comment house style to match** (from the file's own existing declarations, e.g. line 74-77):
```c
/*
 * Inicializa MLP com tamanho de entrada e saida dinamicos.
 */
void mlp_init_dynamic(MLP *net, int input_size, int output_size);
```

---

### `include/config.h` (config constants, transform)

**Analog:** itself, `/* ========== Arquitetura MLP ========== */` banner block (lines 75-90)

**Current block** (verbatim, lines 75-90):
```c
/* ========== Arquitetura MLP ========== */
#define MLP_INPUT_SIZE        TOTAL_FEATURES
#define MLP_HIDDEN1_SIZE      128
#define MLP_HIDDEN2_SIZE      64
#define MLP_HIDDEN3_SIZE      32
#define MLP_OUTPUT_SIZE       NUM_CLASSES      /* 5 - Para o modo legacy */
#define MLP_NUM_LAYERS        3

/* Classificacao Hierarquica */
#define MLP_BINARY_OUTPUT     2                /* Normal vs Patologico */
#define MLP_EXPERT_OUTPUT     4                /* Laringite, Psicog, Funcional, Reinke */

/* ========== Dropout ========== */
#define DROPOUT_RATE_HIDDEN1  0.5f
#define DROPOUT_RATE_HIDDEN2  0.4f
#define DROPOUT_RATE_HIDDEN3  0.0f
```

**Required change:** add `#define MLP_MAX_LAYERS 5` immediately after `MLP_NUM_LAYERS` (keep `MLP_NUM_LAYERS = 3` — RESEARCH.md is explicit this stays as `mlp_init_dynamic()`'s implicit "current prod" default via its thin-wrapper role, it is NOT deleted). Follow the exact section-banner convention (`/* ========== Nome ========== */`) already used in this file — do not introduce a new banner style.

**Open item flagged by RESEARCH.md (Assumption A1):** `DROPOUT_RATE_HIDDEN3` is currently `0.0f` (dead value from the never-taken `#else` branch in `mlp_init_dynamic`); SPEC.md's Config D table implies `0.3f`. Planner must decide explicitly whether `ARCH_CONFIGS[3]` (Config D) in `main.c` uses this `config.h` constant or a locally hardcoded `0.3f` literal in the `ArchConfig` table — this is a config.h vs. main.c ownership decision, not a mechanical copy.

---

### `src/mlp.c` (model, transform — init/forward/backward lifecycle)

**Analog:** itself — `mlp_init_dynamic()` for the new `mlp_init_multi()`, `mlp_backward()` for its own dynamic-sizing fix

**Imports pattern** (lines 13-18, unchanged, no new imports needed):
```c
#include "mlp.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
```

**Core pattern — current `mlp_init_dynamic()`** (lines 220-241, to be replaced by a thin wrapper around new `mlp_init_multi()`):
```c
void mlp_init_dynamic(MLP *net, int input_size, int output_size)
{
    net->num_layers = MLP_NUM_LAYERS;
    net->timestep = 0;

#if MLP_NUM_LAYERS == 3
    int sizes[] = { input_size, MLP_HIDDEN1_SIZE, MLP_HIDDEN2_SIZE, output_size };
    float dropout_rates[] = { DROPOUT_RATE_HIDDEN1, DROPOUT_RATE_HIDDEN2, 0.0f };
    int use_bn[] = { 0, 0, 0 };
#else
    int sizes[] = { input_size, MLP_HIDDEN1_SIZE, MLP_HIDDEN2_SIZE,
                    MLP_HIDDEN3_SIZE, output_size };
    float dropout_rates[] = { DROPOUT_RATE_HIDDEN1, DROPOUT_RATE_HIDDEN2,
                              DROPOUT_RATE_HIDDEN3, 0.0f };
    int use_bn[] = { 0, 0, 0, 0 };
#endif

    for (int i = 0; i < net->num_layers; i++) {
        layer_init(&net->layers[i], sizes[i], sizes[i + 1], use_bn[i]);
        net->layers[i].dropout_rate = dropout_rates[i];
    }
}
```
Note: `use_bn[i]` is always `0` in every branch actually compiled — BN is disabled project-wide (confirmed, matches CLAUDE.md's "Removing BN on small dataset" under What Worked). `mlp_init_multi()` should hardcode `use_bn=0` in its `layer_init()` call rather than accepting a `use_bn` parameter, to avoid resurrecting dead BN wiring.

**`layer_init()` signature it must call** (lines 123-157, unchanged, reused verbatim):
```c
static void layer_init(Layer *l, int input_size, int output_size, int use_bn)
```

**Error handling / bounds pattern — `mlp_backward()`'s stale `max_size`** (lines 282-289, the exact hazard RESEARCH.md Pitfall 2 documents):
```c
void mlp_backward(MLP *net, const float *target, float class_weight)
{
    int nl = net->num_layers;

    /* Allocate delta buffers sized for largest hidden layer */
    int max_size = MLP_HIDDEN1_SIZE;
    float *delta = (float *)safe_malloc(max_size * sizeof(float));
    float *delta_next = (float *)safe_malloc(max_size * sizeof(float));
```
**Fix pattern (from RESEARCH.md Pattern 3, to implement in-place):**
```c
    int max_size = 0;
    for (int i = 0; i < net->num_layers - 1; i++) {
        if (net->layers[i].output_size > max_size) max_size = net->layers[i].output_size;
    }
    float *delta = (float *)safe_malloc(max_size * sizeof(float));
    float *delta_next = (float *)safe_malloc(max_size * sizeof(float));
```

**All other loops already iterate by `net->num_layers` (verified, zero changes needed to loop bodies):** `mlp_forward` (line 247), `mlp_zero_gradients` (line 373), `mlp_clip_gradients` (line 392/413), `mlp_adam_update` (line 437), `mlp_l2_regularization` (line 504), `mlp_save_checkpoint`/`mlp_load_checkpoint` (lines 519/529), `mlp_save`/`mlp_load` (lines 555/577), `mlp_free` (line 596) — this is the load-bearing fact that makes the fixed-oversized-array approach safe.

**New `mlp_count_params()` pattern** (to add, per RESEARCH.md's exact formula matching `layer_init`'s allocation):
```c
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

---

### `src/mlp_train.c` (training loop, batch)

**Analog:** itself — the three `MLP_NUM_LAYERS`-sized stack array declarations

**Imports pattern** (lines 1-14, unchanged):
```c
#include "mlp_train.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
```

**Exact array declarations requiring `MLP_NUM_LAYERS` → `MLP_MAX_LAYERS`** (lines 183-185, checkpoint buffers):
```c
    /* Checkpoint buffers */
    float *best_weights[MLP_NUM_LAYERS], *best_biases[MLP_NUM_LAYERS];
    float *best_bn_gamma[MLP_NUM_LAYERS], *best_bn_beta[MLP_NUM_LAYERS];
    float *best_bn_mean[MLP_NUM_LAYERS], *best_bn_var[MLP_NUM_LAYERS];
```
and (line 204, SWA buffers):
```c
    float *swa_weights[MLP_NUM_LAYERS], *swa_biases[MLP_NUM_LAYERS];
```
All six arrays here are stack-allocated `float*[N]` (pointer arrays, not the actual float buffers — the buffers themselves are heap-allocated per-layer inside the loop at lines 187-199/205-209, sized from `net->layers[i].output_size`/`input_size`, which are already runtime values). **Only the array *declaration widths* need the macro swap — the loop bodies that fill them (`for (int i = 0; i < net->num_layers; i++)`, lines 187, 205) already iterate by the runtime field, zero logic changes needed.**

**Core pattern — the fill loop these arrays feed** (lines 187-199, unchanged logic, just needs the wider backing array):
```c
    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        best_weights[i] = (float *)safe_malloc(l->output_size * l->input_size * sizeof(float));
        best_biases[i] = (float *)safe_malloc(l->output_size * sizeof(float));
        if (l->bn.enabled) {
            best_bn_gamma[i] = (float *)safe_malloc(l->bn.size * sizeof(float));
            best_bn_beta[i] = (float *)safe_malloc(l->bn.size * sizeof(float));
            best_bn_mean[i] = (float *)safe_malloc(l->bn.size * sizeof(float));
            best_bn_var[i] = (float *)safe_malloc(l->bn.size * sizeof(float));
        } else {
            best_bn_gamma[i] = best_bn_beta[i] = best_bn_mean[i] = best_bn_var[i] = NULL;
        }
    }
```

**Cleanup pattern that must still iterate over exactly the same range it filled** (lines 302-313, unchanged, `net->num_layers`-bounded, already correct):
```c
    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        if (l->bn.enabled && best_bn_gamma[i]) { /* ... restore ... */ }
        free(best_weights[i]); free(best_biases[i]);
        free(best_bn_gamma[i]); free(best_bn_beta[i]); free(best_bn_mean[i]); free(best_bn_var[i]);
        free(swa_weights[i]); free(swa_biases[i]);
    }
```

---

### `src/main.c` (orchestration, request-response — CLI mode dispatch)

**Analog:** Phase 1's `ABResult` struct + `mode_train_ex()`/`mode_smote_ab()`/`write_smote_ab_report()` (lines 339-769) — this is the single most important reusable pattern for this phase; ARCH-03 is explicitly "widen this again," not "invent a new orchestration shape."

**Imports/enum pattern to mirror** (`SmoteMode` enum, lines 205-208 — the sibling `ArchConfig` struct should follow this exact style: plain C struct/enum, Portuguese doc-comment above it, no methods):
```c
/* Modo de operacao do SMOTE: padrao (todas as amostras da classe minoritaria sao
 * elegiveis como ponto-base de sintese) ou Borderline-SMOTE1 (Han, Wang & Mao, 2005),
 * que restringe o pool de sintese as amostras classificadas como BORDERLINE. */
typedef enum { SMOTE_STANDARD = 0, SMOTE_BORDERLINE = 1 } SmoteMode;
```

**`ABResult` struct to reuse/extend** (lines 339-351, add `param_count`/`mean_time_per_epoch` fields per RESEARCH.md's ARCH-04 requirement — do not invent a second parallel result struct):
```c
/* Resultado agregado de uma execucao completa do pipeline hierarquico (um dos dois
 * bracos do A/B, SMOTE_STANDARD ou SMOTE_BORDERLINE) -- struct de dados simples, sem
 * metodos, no mesmo estilo de MetricsResult/ConfidenceInterval (include/metrics.h).
 * Usado por mode_smote_ab() para comparar os dois modos sob a mesma seed/folds. */
typedef struct {
    float accuracy;
    float macro_f1;
    float f1_per_class[NUM_CLASSES];
    ConfidenceInterval ci[CI_N_METRICS];
    int *y_true;
    int *y_pred;
    int n;
} ABResult;
```

**`mode_train_ex()` signature to widen once more** (line 360 — the exact precedent RESEARCH.md cites as "it already grew from `mode_train()` -> `mode_train_ex(base_dir, smote_mode, result)` once before"):
```c
static int mode_train_ex(const char *base_dir, SmoteMode smote_mode, ABResult *result)
```
→ add one more parameter (`const ArchConfig *arch`), replacing the two `mlp_init_dynamic(&net_master[v], nf_vowel, 2)` / `mlp_init_dynamic(&net_expert[v], nf_vowel, 4)` calls at lines 464/487 with `mlp_init_multi(..., arch->hidden_sizes, arch->n_hidden, arch->dropout_rates)`.

**Doc-comment convention above `mode_train_ex()`** (lines 353-359, must be updated to describe the new `arch` parameter, matching this exact prose style — result==NULL vs result!=NULL branching explained):
```c
/* mode_train_ex(): executa o pipeline hierarquico completo com o modo SMOTE indicado.
 * result == NULL: execucao CLI simples (modos train/full) -- nomes de arquivo de saida
 * sem sufixo, all_y_true/all_y_pred liberados ao final, comportamento identico ao
 * mode_train() original.
 * result != NULL: execucao de comparacao A/B (mode_smote_ab()) -- nomes de arquivo
 * sufixados por modo, all_y_true/all_y_pred NAO sao liberados aqui (posse transferida
 * para o chamador via *result), que deve libera-los apos o uso. */
```

**Vowel-loop `mlp_init_dynamic` call sites to change** (lines 436-488, the fragile 15+-allocation-per-vowel block RESEARCH.md's Pitfall 4 explicitly warns against restructuring with new control flow — only the two `mlp_init_dynamic` call sites change, no new `continue`/`return` added):
```c
        MLP net_master[3], net_expert[3];
        float cw_binary[] = {0.9f, 1.1f}, cw_expert[] = {1.0f, 1.2f, 1.2f, 1.4f};
        for (int v = 0; v < 3; v++) {
            /* ... tr_x_v/vl_x_v/tr_y_bin/vl_y_bin setup unchanged ... */
            mlp_init_dynamic(&net_master[v], nf_vowel, 2); TrainHistory h_m;
            mlp_train(&net_master[v], os_m_x, os_m_y, os_n_m, vl_x_v, vl_y_bin, fold->n_val, nf_vowel, 2, cw_binary, &h_m);
            /* ... expert block ... */
            mlp_init_dynamic(&net_expert[v], nf_vowel, 4); TrainHistory h_e;
            mlp_train(&net_expert[v], os_e_x, os_e_y, os_n_e, ex_vl_x, ex_vl_y, n_ex_vl, nf_vowel, 4, cw_expert, &h_e);

            free(tr_x_v); free(vl_x_v); free(tr_y_bin); free(vl_y_bin); free(os_m_x); free(os_m_y);
            free(ex_tr_x); free(ex_tr_y); free(ex_vl_x); free(ex_vl_y); free(os_e_x); free(os_e_y);
            train_history_free(&h_m); train_history_free(&h_e);
        }
        /* ... */
        for (int v = 0; v < 3; v++) { mlp_free(&net_master[v]); mlp_free(&net_expert[v]); }
```
Insert `timer_now()`-based capture (RESEARCH.md's Code Examples section) immediately around each `mlp_train(...)` call, accumulating into a per-config running mean stored in the widened `ABResult`.

**Report-writing pattern to extend from 2-arm to 4-arm** (`write_smote_ab_report()`, lines 664-750 — the exact template for the new `write_arch_compare_report()`):
```c
static void write_smote_ab_report(const ABResult *std_res, const ABResult *bl_res,
                                   const char *report_path, const char *csv_path)
{
    float chi2_ab, p_ab;
    metrics_mcnemar(std_res->y_true, bl_res->y_pred, std_res->y_pred, std_res->n, &chi2_ab, &p_ab);
    FILE *f = fopen(report_path, "w");
    /* ... fprintf table of accuracy/macro_f1/per-class-F1 with CI, one row per arm ... */
    if (bl_res->macro_f1 >= std_res->macro_f1) {
        fprintf(f, "DECISAO: Borderline-SMOTE ADOTADO (...)\n");
    } else {
        fprintf(f, "DECISAO: Borderline-SMOTE REJEITADO (...)\n");
    }
    fclose(f);
    FILE *cf = fopen(csv_path, "w");
    fprintf(cf, "metric,standard,standard_ci_lower,standard_ci_upper,borderline,borderline_ci_lower,borderline_ci_upper\n");
    /* ... one fprintf row per metric ... */
    fclose(cf);
}
```
**Generalization for `write_arch_compare_report()`:** loop over `results[4]`/`ARCH_CONFIGS[4]` instead of two named params; decision logic becomes the 1-SE-band + McNemar composition (RESEARCH.md "State of the Art" section) instead of the simple `>=` comparison shown above — but the fixed-rule-in-code principle (never manual/visual pick, per Anti-Patterns) is identical.

**Orchestrator pattern to extend from 2-call to 4-call sequential loop** (`mode_smote_ab()`, lines 756-769 — exact template for `mode_arch_compare()`):
```c
static int mode_smote_ab(const char *base_dir)
{
    log_info("=== MODO: A/B BORDERLINE-SMOTE (Gap 2) ===");
    log_info("Atencao: modo de longa duracao (~60-180 min) -- executa o pipeline hierarquico completo duas vezes (uma por modo SMOTE)");
    ABResult res_standard = {0}, res_borderline = {0};
    if (mode_train_ex(base_dir, SMOTE_STANDARD, &res_standard) != 0) return -1;
    if (mode_train_ex(base_dir, SMOTE_BORDERLINE, &res_borderline) != 0) return -1;
    write_smote_ab_report(&res_standard, &res_borderline,
                           "results/train_log_v32_gap2_smote_ab.txt",
                           "results/smote_ab_comparison.csv");
    free(res_standard.y_true); free(res_standard.y_pred);
    free(res_borderline.y_true); free(res_borderline.y_pred);
    return 0;
}
```

**CLI dispatch pattern to extend** (`main()`, lines 777-788 — add one `strcmp` line, exact same style):
```c
    if (strcmp(mode, "smote-ab") == 0) return mode_smote_ab(base_dir) == 0 ? 0 : 1;
    /* NEW: if (strcmp(mode, "arch-compare") == 0) return mode_arch_compare(base_dir) == 0 ? 0 : 1; */
    return 1;
```

**Fixed SMOTE mode to pass through unchanged (per RESEARCH.md — Gap 2's decision is locked, not re-compared):** every `mode_train_ex(base_dir, ..., &results[i])` call in the new `mode_arch_compare()` must pass whichever `SmoteMode` Phase 1/STATE.md recorded as adopted (`SMOTE_BORDERLINE`, per RESEARCH.md's repeated citation) — verify this against `.planning/STATE.md`'s actual recorded decision before hardcoding, do not assume without checking that file.

---

## Shared Patterns

### Struct/array-sizing generalization (compile-time macro → runtime-safe fixed-headroom array)
**Source:** `include/mlp.h:63` (`Layer layers[MLP_NUM_LAYERS]`) + `src/mlp_train.c:183-185,204` (6 pointer-array declarations)
**Apply to:** `include/mlp.h`, `include/config.h` (new `MLP_MAX_LAYERS`), `src/mlp_train.c`
**Rule:** Every declaration site changes from `MLP_NUM_LAYERS` to `MLP_MAX_LAYERS`. Every loop body that walks these arrays already uses `net->num_layers` and needs zero changes — verified across all of `mlp.c`/`mlp_train.c` this session (see per-file sections above for the exhaustive list of confirmed-safe loop sites).

### Dynamic-vs-macro sizing bug class (Pitfall 2)
**Source:** `src/mlp.c:287` (`int max_size = MLP_HIDDEN1_SIZE;`)
**Apply to:** `src/mlp.c` only — one isolated fix, same commit as the `MLP_MAX_LAYERS` bump per RESEARCH.md's explicit sequencing.

### Reusable multi-run comparison orchestration (`ABResult` + widen-`mode_train_ex`-in-place + report writer + CLI dispatch line)
**Source:** `src/main.c:339-769` (Phase 1's SMOTE-04 A/B, entire pattern)
**Apply to:** `src/main.c`'s new `ArchConfig`/`mode_arch_compare()`/`write_arch_compare_report()`/CLI dispatch — this is the dominant cross-cutting pattern for this entire phase's orchestration layer. Every new orchestration construct (config table, sequential-run loop, decision rule, CSV/report export, CLI dispatch line) has a direct 1:1 precedent already in this file; nothing in the orchestration layer should be invented from scratch.

### Statistical primitives (bootstrap CI, McNemar) — reuse verbatim, do not reimplement
**Source:** `include/metrics.h` (`metrics_bootstrap_ci()` line 71, `metrics_mcnemar()` line 114), already wired into `mode_train_ex()` at `src/main.c:550,555-557`
**Apply to:** `mode_arch_compare()`/`write_arch_compare_report()` — same call signatures, same seed (`RANDOM_SEED`), same `CI_N_METRICS`-indexed `ConfidenceInterval[]` array shape.

### High-resolution timing
**Source:** `include/utils.h:66` (`double timer_now(void);`), pattern already used in `feature_extract.c`
**Apply to:** wrap each `mlp_train()` call inside `mode_train_ex()`'s vowel loop (lines 465, 488) with `t0 = timer_now(); ...; dt = timer_now() - t0;`, accumulate `dt / history.num_epochs` into a per-config running mean for `ABResult.mean_time_per_epoch`.

## No Analog Found

None. Every file in scope is a targeted, in-place generalization of an existing function/struct/loop in the same file, or (for the `main.c` orchestration additions) a direct structural extension of Phase 1's already-built `ABResult`/`mode_train_ex`/`mode_smote_ab` pattern. RESEARCH.md's own "Don't Hand-Roll" table confirms every supporting primitive (bootstrap CI, McNemar, timer, structured-result struct) already exists and is reused, not newly invented.

## Metadata

**Analog search scope:** `include/mlp.h`, `include/config.h`, `src/mlp.c` (full, 600 lines), `src/mlp_train.c` (full, 349 lines), `src/main.c` (targeted reads: lines 200-360, 360-630, 629-788 — full file effectively covered non-overlapping), `include/metrics.h` (targeted grep + read for `ConfidenceInterval`/`CI_N_METRICS`/`metrics_mcnemar`/`metrics_bootstrap_ci` signatures), `include/utils.h` (targeted grep for `timer_now`)
**Files scanned:** 7 (5 modified-in-scope + 2 read-only-for-signatures: `metrics.h`, `utils.h`)
**Pattern extraction date:** 2026-07-27
