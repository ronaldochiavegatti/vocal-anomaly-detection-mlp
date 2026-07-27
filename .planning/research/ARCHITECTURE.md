# Architecture Research — Integrating SPEC.md Gaps into v29 Hierarchical Late Fusion

**Domain:** C99 offline ML pipeline (vocal anomaly detection), single-file orchestrator (`src/main.c`)
**Researched:** 2026-07-27
**Confidence:** HIGH (all claims verified against current `src/main.c`, `include/mlp.h`, `include/config.h`, `include/feature_select.h`, `src/mlp.c`, `src/mlp_train.c` — not from SPEC.md's prose alone)

## Standard Architecture (current v29, as verified)

```
mode_train() — src/main.c:235-343
│
├─ dataset_load() → FeatureMatrix (features.csv, 251 cols = 3×83 + 2 meta)
├─ kfold_split() → KFoldSplits (5 folds, patient-level, stratified)
├─ precalculate_augmentations() → aug_cache (8×/minority patient, ONCE, pre-fold-loop)
│
└─ for f in 0..4 (fold loop, main.c:255-337):
     ├─ build train_x_all/train_y_all (fold->n_train rows, 251-wide)
     ├─ collect_augmented_features() → n_train_aug rows (train_x_all realloc'd)
     ├─ norm_fit(train_x_all, fold->n_train [ORIGINAL only], 251) → NormParams
     ├─ norm_transform() on train_x_all[0..n_train_aug] AND val_x_all
     │
     └─ for v in 0..2 (vowel loop, main.c:275-317):
          ├─ slice 251-wide → nf_vowel=85-wide (83 vowel feat + 2 meta) into tr_x_v/vl_x_v
          ├─ map_to_binary_labels() → tr_y_bin/vl_y_bin
          ├─ MASTER: smote_oversample(tr_x_v, tr_y_bin, ..., 2 classes) → os_m_x/os_m_y
          │          mlp_init_dynamic(&net_master[v], nf_vowel=85, 2)   [main.c:294]
          │          mlp_train(&net_master[v], os_m_x, ...)             [main.c:295]
          ├─ filter pathological rows → ex_tr_x/ex_tr_y (4-class, label-1 remapped)
          └─ EXPERT: smote_oversample(ex_tr_x, ex_tr_y, ..., 4 classes) → os_e_x/os_e_y
                     mlp_init_dynamic(&net_expert[v], nf_vowel=85, 4)    [main.c:311]
                     mlp_train(&net_expert[v], os_e_x, ...)             [main.c:312]
     │
     └─ for i in val samples: predict_hierarchical_late_fusion(net_master[3], net_expert[3], x_i)
          [main.c:39-69: per-vowel mlp_forward on Master+Expert, average P(pathology)
           and P(expert class) across 3 vowels, threshold 0.5, argmax+1]
```

**Per full run:** 5 folds × 3 vowels × 2 networks (Master, Expert) = **30 independently trained MLPs**. This is the number that matters for every gap below — each of the three new techniques must decide whether it operates per-fold, per-vowel, per-(vowel,network), or per-(fold,vowel,network).

### Component Responsibilities (relevant to the 3 gaps)

| Component | Responsibility | File:Lines |
|-----------|----------------|------------|
| `smote_oversample()` / `find_knn()` | Standard SMOTE, called once for Master (2-class) and once for Expert (4-class) per vowel per fold | `src/main.c:189-231` |
| `mlp_init_dynamic()` | Builds a fixed-depth (2-hidden-layer) MLP for variable input/output size | `src/mlp.c:220-241` |
| `MLP` / `Layer` structs | Fixed-size `Layer layers[MLP_NUM_LAYERS]` array (compile-time bound) | `include/mlp.h:33-66` |
| `predict_hierarchical_late_fusion()` | Inference-time fusion; hard-codes the 85-wide per-vowel slice, no feature-subset awareness | `src/main.c:39-69` |
| `feature_select.h` (`selected_save`/`selected_load`) | Generic binary index persistence — **exists and matches SPEC.md's assumption exactly**, but no selection *logic* exists anywhere in the tree | `include/feature_select.h`, `src/feature_select.c` |

## Gap-by-Gap Integration Analysis

### Gap 1 — Paraconsistent Feature Selection

**Answer to "once per vowel or once per (network, vowel)?": once per (network, vowel) pair — i.e. twice per vowel, 6 times per fold, 30 times per run.**

Reasoning verified against actual data flow, not just SPEC.md's assertion: Master and Expert are trained on *different label spaces* over the *same* 85-wide feature slice — Master sees binary labels (`tr_y_bin`, main.c:289), Expert sees 4-class labels only on the pathological subset (`ex_tr_y`, main.c:301, values 0-3 after `-1` remap). Fisher one-vs-rest separability (μ) and intra-class consistency (λ) are computed against `y`, so they are mathematically different computations for Master vs Expert even on identical `x`. SPEC.md's own text agrees ("os índices selecionados para Mestra e Especialista podem e devem ser diferentes") — this part of SPEC.md is internally consistent with the actual code structure.

**Exact insertion points (current main.c, verified line numbers):**
- Master: insert `paraconsistent_select()` call **between** line 288 (`map_to_binary_labels(...)`) and line 292 (`smote_oversample(tr_x_v, tr_y_bin, ...)`) — i.e. right after `tr_x_v`/`tr_y_bin` exist and before SMOTE touches them. Selection must run on `tr_x_v`/`tr_y_bin` at their *pre-SMOTE, pre-selection* width (`nf_vowel=85`), never on `vl_x_v` (mirrors the existing `norm_fit`-on-train-only rule already documented in CLAUDE.md).
- Expert: insert between line 308 (end of `ex_vl_y` population loop) and line 309 (`smote_oversample(ex_tr_x, ex_tr_y, ...)`) — operating on `ex_tr_x`/`ex_tr_y` (4-class, already pathology-only, already remapped to 0-3).
- After each selection call, **both** the corresponding train and val arrays need column-filtering to the selected subset before they reach `smote_oversample()`/`mlp_init_dynamic()`/`mlp_train()`: `tr_x_v`+`vl_x_v` (Master) and `ex_tr_x`+`ex_vl_x` (Expert). This is a new filtering step not mentioned explicitly by SPEC.md's code sketch — it only shows the `paraconsistent_select()` call, not the subsequent "apply selection to both matrices" step, which requires an explicit helper (e.g. `apply_feature_selection(float *x, int n, int nf_in, const int *selected, int n_selected, float *x_out)`).

**Data flow requirement not covered by SPEC.md's sketch — inference-time propagation.** This is the most significant integration risk for Gap 1. `mlp_init_dynamic(&net_master[v], nf_vowel, 2)` currently always uses the fixed `nf_vowel=85`. Once feature selection is added, the *actual* input width becomes `ns_m` (Master) / `ns_e` (Expert), which varies **per fold per vowel per network** (up to 30 distinct values in one run). Everywhere downstream that reconstructs an 85-wide `x_v`/`xv` vector from the full 251-wide sample and feeds it to `mlp_forward()` must instead apply the *same* selected-index list used at training time for that specific (fold, vowel, network):
- `predict_hierarchical_late_fusion()` (`main.c:39-69`) currently takes only `(MLP master[3], MLP expert[3], const float *x_all)` and internally `memcpy`s the full 85-wide slice (line 47-49). Its signature **must grow** to accept per-vowel selected-index arrays and counts, e.g. `predict_hierarchical_late_fusion(MLP master[3], MLP expert[3], const float *x_all, const int *sel_master[3], const int ns_master[3], const int *sel_expert[3], const int ns_expert[3])`, then build `x_v` by gathering only the selected columns instead of a flat `memcpy`.
- The validation-loop's own **duplicate** inline slicing logic at `main.c:319-332` (which does NOT call `predict_hierarchical_late_fusion` for the probability-averaging part — it independently rebuilds `xv[251]` and calls `mlp_forward` a second time to get `p_norm`/`p_exp` for `all_y_prob`) has the exact same problem and must be updated in parallel, or refactored to reuse a single fusion+probability function instead of duplicating the logic (this duplication already exists today for reasons unrelated to Gap 1, but Gap 1 makes the duplication a correctness hazard: two independent per-vowel slicing sites must now both apply feature selection consistently, or the recorded probabilities in `results/metrics_global.csv` will silently disagree with the discrete predictions used for the fold's confusion matrix).
- Practical consequence: the per-(fold,vowel,network) selected-index arrays (`sel_master[fold][v]`, `sel_expert[fold][v]`) must be kept alive in memory (or persisted to `models/selected_{master,expert}_fold{k}_v{vowel}.bin` via the *already-existing* `selected_save`/`selected_load` from `feature_select.h`) from the moment they're computed in the training sub-loop until the fold's validation loop runs, since both live inside the same fold iteration today — no cross-fold persistence is strictly required for correctness (only if `mode_validate_external` is ever implemented, which it currently is not — it is a no-op stub, `main.c:354-358`).

**Struct/header changes required for Gap 1:**
- New `src/feature_select_paraconsistent.c` + `include/feature_select_paraconsistent.h` exactly as SPEC.md specifies — no conflict with existing `feature_select.h` (that header only has persistence functions, `selected_save`/`selected_load`, confirmed by direct read; there is no naming collision).
- No changes needed to `FeatureMatrix`, `KFoldSplit`, or `NormParams` — feature selection operates strictly downstream of normalization (norm_fit/norm_transform already happen on the full 251-wide vector before the per-vowel slicing loop, `main.c:267-271`), so it is a pure post-normalization column filter, consistent with SPEC.md's requirement that μ/λ be computed only on already-fold-scoped training data.
- `MLP`/`Layer` structs are untouched by Gap 1 — only the `input_size` argument passed to `mlp_init_dynamic()` changes from a constant (`nf_vowel=85`) to a per-instance variable (`ns_m`/`ns_e`), which the existing `mlp_init_dynamic(net, input_size, output_size)` signature already supports without modification.

### Gap 2 — Borderline-SMOTE

**No drift from SPEC.md here** — verified the current `smote_oversample()` (`main.c:205-231`) and `find_knn()` (`main.c:189-203`) signatures character-for-character against SPEC.md's proposed `smote_oversample_ex()` sketch, and they match SPEC.md's assumptions about parameter order/types exactly. This is the lowest-risk gap to implement first, confirming SPEC.md's own prioritization.

**Exact insertion points:**
- Add `SmoteMode` enum and `find_knn_global()` (searches across ALL training samples in the current vowel/network's `x_in`, not just same-class `class_idx[c]`) alongside the existing `find_knn()` at `main.c:189-203`.
- Modify `smote_oversample()` body (`main.c:205-231`) to accept a `SmoteMode` parameter and, when `SMOTE_BORDERLINE`, replace the base-sample sampling pool at line 223 (`int base_idx = class_idx[c][rng_int(n_class)];`) with a `class_idx_borderline[c]` array built via one `find_knn_global()` call per minority-class sample before entering the per-class synthesis loop.
- Two call sites needing the new mode parameter, unchanged in count from today: `main.c:293` (Master, 2-class) and `main.c:310` (Expert, 4-class) — both already pass `num_classes` explicitly, so adding a `SmoteMode mode` argument is a pure signature-widening change with two call-site edits, no structural changes to the fold/vowel loop nesting.
- **Interaction with Gap 1 ordering matters**: if Gap 1 lands first, `smote_oversample()`'s `nf` parameter at these two call sites becomes the *post-selection* width (`ns_m`/`ns_e`) rather than the constant `nf_vowel`. Borderline-SMOTE's `find_knn_global()` distance computation is therefore sensitive to whichever features survived selection — this is a soft dependency (Gap 2 works standalone, but its k-NN neighbor sets change if run after Gap 1 vs. before), not a hard blocker, but should be noted in the A/B log naming (`results/train_log_vXX_<gap-name>.txt`) so comparisons aren't accidentally made across incompatible feature-selection states.

### Gap 3 — Shallow vs. Deep MLP Comparison

**This is where SPEC.md's technical description of the *current* architecture is factually wrong, not just stale-line-numbers wrong — flagged as the most important drift in this research.**

SPEC.md states (SPEC.md:210-211): *"`mlp_init_dynamic()` hoje só suporta 1 camada oculta (`num_layers = (hidden_size > 0) ? 3 : 2`, ou seja, Input→Hidden(128)→Output)."*

Verified against `src/mlp.c:220-241` (current, actual code):
```c
void mlp_init_dynamic(MLP *net, int input_size, int output_size)
{
    net->num_layers = MLP_NUM_LAYERS;   /* always the compile-time macro, currently 3 */
    ...
#if MLP_NUM_LAYERS == 3
    int sizes[] = { input_size, MLP_HIDDEN1_SIZE, MLP_HIDDEN2_SIZE, output_size };
    float dropout_rates[] = { DROPOUT_RATE_HIDDEN1, DROPOUT_RATE_HIDDEN2, 0.0f };
```
There is **no runtime ternary on `hidden_size`** anywhere in the function (the function doesn't even take a `hidden_size` parameter) — the "1 hidden layer" branch SPEC.md describes does not exist in the current signature at all. The actual current architecture already has **2 hidden layers** (`MLP_HIDDEN1_SIZE=128`, `MLP_HIDDEN2_SIZE=64`, from `config.h:77-78`), i.e. `Input → Dense(128) → Dense(64) → Output`. This matches `.planning/codebase/ARCHITECTURE.md`'s independently-verified description ("Input→Hidden1→Hidden2→Output") and `CLAUDE.md`'s architecture table — both agree with the actual code, and both **disagree with SPEC.md's Gap 3 framing**.

**Consequence for the roadmap:** SPEC.md's comparison table (SPEC.md:229-237) mislabels its baseline. Its "Config A (atual, rasa) — [128]" is **not** what `mode_train()` currently runs — the actual current production config is `[128, 64]`, which is SPEC.md's own **"Config C (profunda 2 camadas)"**. Two corrections needed before building Gap 3:
1. Relabel: current baseline = SPEC.md's Config C. SPEC.md's Config A ([128] only, 1 hidden layer) and Config B ([64] only) are both *new, shallower* configs never previously run, not simplifications of an existing baseline.
2. Only SPEC.md's Config D ([128, 64, 32], 3 hidden layers) is actually deeper than what already runs today — the "comparison" is therefore really 3 new configs (A, B, D) vs. the existing, already-deployed C, not 4 new configs vs. some simpler status quo.

**Structural blocker not fully resolved by SPEC.md's sketch:** `MLP.layers` is a **fixed-size compile-time array** (`Layer layers[MLP_NUM_LAYERS]`, `include/mlp.h:63`), where `MLP_NUM_LAYERS` is a `config.h` preprocessor constant currently `3`. SPEC.md acknowledges this ("verificar que `net->layers[4]`... comporta `n_hidden+1` camadas") but underestimates the blast radius: `mlp_train.c` **also** declares three more fixed-size stack arrays sized directly off the same macro — `float *best_weights[MLP_NUM_LAYERS]`, `best_bn_gamma/beta/mean/var[MLP_NUM_LAYERS]`, and `swa_weights/swa_biases[MLP_NUM_LAYERS]` (`src/mlp_train.c:183-205`). Good news (verified, not assumed): every loop in `mlp.c` and `mlp_train.c` that walks layers already iterates `for (i = 0; i < net->num_layers; i++)` — i.e. the **runtime** logic is already parameterized by the struct field, not the macro. Only the **array declarations** are macro-bound. This means the lowest-risk fix is:
- Bump the macro used for *array sizing only* to a safe maximum (e.g. rename/introduce `MLP_MAX_LAYERS 5` in `config.h`, used solely for `Layer layers[MLP_MAX_LAYERS]` and the `mlp_train.c` stack-array declarations), while `net->num_layers` (already a runtime struct field, already used everywhere else) is set per-instance by the new `mlp_init_multi()` to `n_hidden + 1` (1 to 4, config D needs 4).
- No malloc-based dynamic array is required — the existing code's discipline of always indexing by `net->num_layers` rather than the macro makes the fixed-max-size approach safe and low-diff.
- `mlp_init_dynamic()` should become a thin wrapper calling `mlp_init_multi(net, input_size, output_size, (int[]){MLP_HIDDEN1_SIZE, MLP_HIDDEN2_SIZE}, 2, (float[]){DROPOUT_RATE_HIDDEN1, DROPOUT_RATE_HIDDEN2})` to preserve exact current behavior for any code that still calls it directly (backward compatibility, zero regression risk for Gaps 1/2 which don't touch architecture depth).

**Exact insertion points in main.c for the comparison protocol itself:** SPEC.md's nested-CV protocol ("outer 5-fold × 4 configs, mesma partição em todas") requires re-running the **entire** existing fold+vowel loop body (`main.c:255-337`, currently ~83 lines) once per architecture config, with results kept separate per config. There is no way to satisfy "same partition across all configs" by nesting configs *inside* the fold loop — the configs must be the **outer** loop, folds nested inside (or the fold loop body extracted into a function parameterized by `(hidden_sizes, n_hidden, dropout_rates)` for Master and Expert independently, called once per config from a new `for (config in {A,B,C,D})` wrapper). This is the largest single-file structural change of the three gaps — see Build Order below for why this argues for doing it only after `main.c` is refactored.

## Data Flow Summary (new parameters, explicit direction)

| New data | Produced by | Scope | Consumed by | Struct/param change needed |
|---|---|---|---|---|
| `selected_master[fold][v]` (indices + count) | `paraconsistent_select()` call before Master's `smote_oversample` | Per (fold, vowel) — Master only | `mlp_init_dynamic`'s `input_size` arg, `predict_hierarchical_late_fusion`'s slicing, validation-loop's duplicate slicing (`main.c:319-332`) | `predict_hierarchical_late_fusion()` signature must grow (see Gap 1); no header struct changes |
| `selected_expert[fold][v]` (indices + count) | `paraconsistent_select()` call before Expert's `smote_oversample` | Per (fold, vowel) — Expert only, **independent from Master's set** | same as above, Expert side | same |
| `SmoteMode mode` | Caller (main.c fold/vowel loop) | Global run-level choice (A/B flag), not per-fold/vowel | `smote_oversample()`/`smote_oversample_ex()` | New `typedef enum` in main.c or a shared header; two call-site edits |
| `hidden_sizes[]`, `n_hidden`, `dropout_rates[]` (per network: Master, Expert) | Outer config loop (A/B/C/D) | Global per config-run, applied identically to all 5 folds × 3 vowels for that config | `mlp_init_multi()` replacing both `mlp_init_dynamic()` call sites | `MLP.layers` array bound (`mlp.h`), `mlp_train.c`'s 3 stack-array declarations — both change from `MLP_NUM_LAYERS` to a new `MLP_MAX_LAYERS` |

All three new data items are **per-(fold, vowel[, network])** scoped, matching the granularity at which `net_master[3]`/`net_expert[3]` already exist today (`main.c:273`) — no change to `FeatureMatrix`, `KFoldSplit`, or `NormParams` is required for any of the three gaps; those structs remain at their current (full-251-width, per-patient, per-fold) scope, and all three techniques hook in strictly *after* normalization and *before or during* the per-vowel/per-network MLP instantiation.

## SPEC.md Drift Summary (explicit call-outs)

| SPEC.md claim | Verified reality | Severity |
|---|---|---|
| Gap 1: "reaproveitar padrão já usado em `select_features_variance`, ver WIP `src/main.c` linhas ~20-47" | `select_features_variance` does not exist anywhere in the current tree (`grep -rn "select_features"` returns nothing in `src/`/`include/`). Lines 20-47 of the actual current `main.c` are includes + `map_to_binary_labels` + `predict_hierarchical_late_fusion` — no per-class mean/variance computation exists there or elsewhere to copy a style from. | Medium — implementers must write Gap 1's per-class mean/variance from scratch; the closest actual style precedent is `smote_oversample()`'s `class_idx[]` grouping-by-class pattern (`main.c:213-216`), not the named (nonexistent) function. |
| Gap 3: "`mlp_init_dynamic()` hoje só suporta 1 camada oculta (`num_layers = (hidden_size > 0) ? 3 : 2`)" | False. Current `mlp_init_dynamic()` (`src/mlp.c:220-241`) takes no `hidden_size` parameter, has no such ternary, and always builds **2 hidden layers** (`[128, 64]`) via a compile-time `#if MLP_NUM_LAYERS == 3` branch keyed off `config.h`'s `MLP_NUM_LAYERS=3`. | **High** — this mislabels the whole Gap 3 comparison table. SPEC.md's "Config A (atual, rasa)" is not the current production config; SPEC.md's "Config C (profunda 2 camadas)" *is* the current production config. Roadmap must relabel before building Gap 3's comparison. |
| Gap 3: `net->layers[4]` sizing concern | Correct in spirit (fixed-size array is a real blocker), but SPEC.md doesn't mention that `src/mlp_train.c` independently declares 3 more `MLP_NUM_LAYERS`-sized stack arrays (checkpoint weights/biases, BN stats, SWA accumulators, lines 183-205) that also need the same bound increased. | Medium — larger blast radius than SPEC.md implies, but low actual risk since all loop bodies already index by `net->num_layers`, not the macro. |
| Gap 2: `smote_oversample()`/`find_knn()` signatures | Match SPEC.md's assumptions exactly, verified line-by-line. | None — no drift. |
| `feature_select.h` (`selected_save`/`selected_load`) reuse for Gap 1 persistence | Matches SPEC.md's assumption exactly — generic binary format, no collision with the new paraconsistent module's naming. | None — no drift. |
| SPEC.md's reference baseline numbers (Macro F1 0.4435, accuracy 69.4%, commit `e63483a`) | `results/train_log_v29_late_fusion.txt` (the log that should contain these numbers) is **empty (0 lines)** in the current working tree. `git log` confirms `e63483a` is indeed the current `HEAD` commit ("feat: implement Hierarchical Late Fusion and Multi-Vowel Ensemble (v29)"), so the commit reference is accurate, but the specific 0.4435/69.4% figures cannot be independently verified from any file currently in the repo (they may exist only in the proposer's local run output). | Low-medium — doesn't block implementation, but the A/B "do not regress below this number" gate SPEC.md defines has no artifact backing it in-repo; recommend re-establishing the baseline by running `make full` once before starting Gap 2, and saving that log explicitly (e.g. `results/train_log_v29_baseline_reconfirmed.txt`) as the actual regression-gate reference. |
| CLAUDE.md's "Current Best Results" section (65.4%/0.4115, 5-class flat MLP) | Describes the **pre-v29** architecture (flat 5-class softmax), not the current Hierarchical Late Fusion pipeline SPEC.md and this research are built against. Already flagged as stale by `.planning/codebase/STRUCTURE.md`. | Low — orthogonal to the 3 gaps, but worth updating alongside SPEC.md's own "update CLAUDE.md" acceptance criterion so future readers aren't comparing Gap results against the wrong baseline architecture. |

## Suggested Build Order (revised from SPEC.md's, accounting for main.c's size and the drift found above)

SPEC.md's own recommended order (Gap 2 → Gap 3 → Gap 1) is **directionally correct and confirmed sound** by this research — the two additions below refine *why*, given `main.c` is already 369 lines with no test harness:

1. **Gap 2 (Borderline-SMOTE) first**, exactly as SPEC.md recommends. Confirmed lowest risk: two call-site edits, one new mode enum, no struct changes, no change to loop nesting, no change to `mlp_init_dynamic()` call arguments. This also validates the A/B comparison workflow (`results/train_log_vXX_<gap>.txt`, same-seed re-run) before the two harder gaps need it.

2. **Gap 3 (shallow vs. deep) second, but only after a small preparatory refactor**, for a reason SPEC.md doesn't call out: Gap 3's nested-CV protocol requires the fold+vowel loop body to become reusable across 4 configs (outer loop over configs, folds nested inside — see analysis above). Given `main.c:255-337` is already the largest single block in the file, **extract that block into a static function** (e.g. `run_fold_cv(FeatureMatrix*, KFoldSplits*, Dataset*, aug_cache, MasterConfig*, ExpertConfig*, MetricsResult *out)`) as a first, structure-only, zero-behavior-change commit — verified safe because it doesn't touch any algorithm, just moves code — before adding the config-comparison outer loop. Also apply the `MLP_MAX_LAYERS` bump (mlp.h + mlp_train.c's 3 stack arrays) and `mlp_init_multi()` in this phase, with `mlp_init_dynamic()` kept as a compatibility wrapper so Gap 1's and Gap 2's code (already merged from step 1) needs zero changes.

3. **Gap 1 (paraconsistent selection) last**, exactly as SPEC.md recommends, and for the same reason SPEC.md gives (avoid re-tuning against a moving architecture/SMOTE-mode target) — but with the added, verified-necessary step of widening `predict_hierarchical_late_fusion()`'s signature (and de-duplicating or synchronizing it with the validation loop's inline copy at `main.c:319-332`) to carry per-vowel-per-network selected-index arrays through to inference time. This signature change is the single riskiest edit of all three gaps because it touches the *only* function called for every validation sample in every fold — schedule it with its own isolated A/B run before combining with whatever SMOTE mode / architecture config won in steps 1-2.

**Do not attempt Gap 1 before Gap 3's `run_fold_cv()` extraction lands**, even though Gap 1 doesn't strictly need multi-depth networks — the reason is purely `main.c` size/maintainability: by the time Gap 1's per-(fold,vowel,network) selected-index bookkeeping is added on top of an already-4x-config-multiplied fold loop (if Gap 3 is done sloppily by copy-pasting the loop body 4 times instead of extracting a function), the file becomes unmaintainable and error-prone to keep the two duplicate slicing sites (`predict_hierarchical_late_fusion` and the inline validation-loop copy) in sync.

## Anti-Patterns to Avoid During Implementation

### Anti-Pattern 1: Silently duplicating the per-vowel slice-and-forward logic
**What people do:** Add feature-selection-aware slicing to `predict_hierarchical_late_fusion()` but forget the validation loop's separate inline copy at `main.c:319-332` (which exists today purely to populate `all_y_prob` for `results/metrics_global.csv` and does **not** call `predict_hierarchical_late_fusion()` — it reimplements the same per-vowel forward pass independently).
**Why it's wrong:** The discrete prediction (`all_y_pred`, from `predict_hierarchical_late_fusion`) and the recorded probability (`all_y_prob`, from the inline duplicate) would use different feature subsets after Gap 1, producing internally inconsistent metrics/CSV output that would silently fail McNemar-style downstream analysis without any compile or runtime error.
**Do this instead:** Before adding Gap 1, refactor these two call sites to share one function that returns both the discrete class and the probability vector, then add feature-selection parameters to that single function.

### Anti-Pattern 2: Sizing `MLP.layers[]` per-config instead of to a fixed safe maximum
**What people do:** Try to make `Layer *layers` a `malloc`'d pointer sized exactly to `n_hidden+1` per network instance, to "save memory."
**Why it's wrong:** Every other function in `mlp.c`/`mlp_train.c` (checkpoint save/restore, SWA, BN stats) already assumes array semantics keyed to a compile-time bound for its own **local** stack arrays (`mlp_train.c:183-205`); introducing heap allocation for `MLP.layers` alone creates an inconsistent mix of stack-bound and heap-bound layer counts across the two files, doubling the surface area of the change for negligible memory savings (a handful of unused `Layer` structs per shallow-config network instance, on a codebase that already allocates full per-fold/per-vowel/per-network feature matrices in the megabytes).
**Do this instead:** Bump the array-sizing constant (introduce `MLP_MAX_LAYERS`) to the largest config actually tested (5, covering SPEC.md's Config D), and rely on the already-correct `net->num_layers` runtime field everywhere else — verified above to already be the pattern every loop uses.

## Sources

- `.planning/codebase/ARCHITECTURE.md` (2026-07-27, machine-generated, cross-checked against source — treated as HIGH confidence baseline)
- `.planning/codebase/STRUCTURE.md` (2026-07-27, same)
- `SPEC.md` (2026-07-27, gap-analysis document — treated as a hypothesis to verify, not ground truth; see Drift Summary)
- `src/main.c` (read in full, 369 lines, current HEAD `e63483a`)
- `include/mlp.h`, `src/mlp.c` (lines 215-241 read directly for `mlp_init`/`mlp_init_dynamic`)
- `src/mlp_train.c` (grepped for `MLP_NUM_LAYERS`/`net->num_layers` usage, lines 183-302)
- `include/config.h` (full read — confirms `MLP_NUM_LAYERS=3`, `MLP_HIDDEN1_SIZE=128`, `MLP_HIDDEN2_SIZE=64`)
- `include/feature_select.h`, `include/kfold.h`, `include/normalize.h`, `include/feature_extract.h` (full reads)
- `results/train_log_v29_late_fusion.txt` (confirmed empty, 0 lines — cannot verify SPEC.md's cited baseline numbers from repo artifacts)
- `results/train_log_v30.txt`, `results/train_log_v31.txt` (confirmed collapsed-to-majority-class failure mode SPEC.md warns against: Macro F1 0.15-0.31 vs. SPEC.md's 0.4435 reference)
- `git log --oneline` (confirmed `e63483a` is current HEAD, matching SPEC.md's stated commit)

---
*Architecture research for: C99 vocal-anomaly MLP pipeline — SPEC.md gap integration*
*Researched: 2026-07-27*
