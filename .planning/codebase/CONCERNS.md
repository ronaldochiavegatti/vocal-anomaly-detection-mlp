# Codebase Concerns

**Analysis Date:** 2026-07-27

## Tech Debt

**CLAUDE.md / MEMORY.md describe an architecture that no longer exists in the code (critical drift):**
- Issue: `CLAUDE.md` and the auto-memory (`MEMORY.md`) describe a single flat 5-class MLP fed by 237 features (`TOTAL_FEATURES=237`, `FEATURES_PER_VOWEL=79`), a pipeline that runs `select_features()`, `inner_cv_select_thresholds()`, baselines (majority/kNN/logreg), bootstrap CI, McNemar test, and exports `learning_curves.csv`/`roc_curves.csv`/`pr_curves.csv`/`baselines.csv`/`feature_importance.csv`. The actual HEAD (`e63483a`, "Hierarchical Late Fusion", v29) does none of this.
- Files: `CLAUDE.md`, `include/config.h` (`TOTAL_FEATURES=251`, `FEATURES_PER_VOWEL=83`, `NUM_SPECTRAL_FEATURES=55` including 4 glottal features not mentioned anywhere in CLAUDE.md), `src/main.c` (`mode_train`, function `predict_hierarchical_late_fusion`)
- Impact: Anyone (human or agent) using CLAUDE.md as ground truth to plan a phase or fix a bug will reference functions/files/constants that don't exist (`select_features`, `inner_cv_select_thresholds`, `models/best_model.bin`) or that are stale (class weights, learning rate, L2 lambda all differ between doc and `config.h`). `mode_train` now trains **two separate networks per vowel** (a binary Master net Normal-vs-Pathological, output=2, and a 4-class Expert net for the pathological subtypes, output=4) combined via late fusion — not the single 5-class network CLAUDE.md documents.
- Fix approach: Regenerate `CLAUDE.md`'s "Architecture", "Data Flow", "Key Hyperparameters", and "Output Files" sections directly from `src/main.c` v29 and `include/config.h` before starting any new phase; do this as its own small task, not mixed with a feature change.

**Feature selection module is dead code:**
- Issue: `src/feature_select.c`/`include/feature_select.h` only contain `selected_save`/`selected_load` (binary I/O stubs). The actual selection logic (variance/correlation thresholding) referenced by old docs and by `SPEC.md`'s Gap 1 write-up no longer exists in the repo. `mode_train` in `src/main.c` never calls any selection function — all 83 features/vowel (+2 metadata) go straight into normalization and training.
- Files: `src/feature_select.c`, `include/feature_select.h`, `src/main.c` (no call site)
- Impact: Networks train on the full ~85-feature vowel vector with no dimensionality reduction; SPEC.md Gap 1 (paraconsistent feature selection) has no existing "selected_fold{k}.bin" infrastructure to hook into beyond the unused save/load pair.
- Fix approach: When implementing SPEC.md Gap 1, build the paraconsistent selector fresh; reuse only `selected_save`/`selected_load` for persistence as SPEC.md already proposes.

**Baseline classifiers (kNN, logistic regression, majority class) are unreachable dead code:**
- Issue: `src/knn.c`, `src/logreg.c` and their headers are compiled into the binary and `#include`d in `src/main.c` (lines 15-16), but no `knn_*`/`logreg_*` function is ever called from `mode_train`. `results/baselines.csv` is never produced.
- Files: `src/knn.c`, `src/logreg.c`, `include/knn.h`, `include/logreg.h`, `src/main.c`
- Impact: The "McNemar vs 3 baselines" comparison documented in CLAUDE.md's Current Best Results section cannot be reproduced by the current binary — those numbers came from an earlier architecture. Any academic claim citing baseline comparisons must be re-validated or the claim removed.
- Fix approach: Either wire these back into `mode_train` (compute baselines per fold on the same splits) or remove the dead includes/files and drop baseline claims from documentation until reimplemented.

**`train_history_export_csv`, ROC/PR export, and bootstrap CI/McNemar are implemented but never invoked:**
- Issue: `src/mlp_train.c:339` (`train_history_export_csv`) and `src/metrics.c:167` (`metrics_mcnemar`), `src/metrics.c:206` (`metrics_bootstrap_ci`) are fully implemented and exported via headers, but `src/main.c`'s `mode_train` calls none of them — only `metrics_export_csv` for the final aggregate confusion matrix.
- Files: `src/mlp_train.c`, `src/metrics.c`, `src/main.c`
- Impact: No per-epoch learning curves, no confidence intervals, no significance testing are produced by `make train`/`make full` today, despite CLAUDE.md listing these as standard outputs and despite the functions existing and presumably working (they were exercised by an earlier architecture version, per git history `2e5af63`).
- Fix approach: Add call sites in `mode_train`'s fold loop (`src/main.c:255-337`) for `train_history_export_csv` (per network) and `metrics_bootstrap_ci`/`metrics_mcnemar` (on the aggregated `all_y_true`/`all_y_pred`) if these academic artifacts are still required for the PIBIC deliverable.

**Hardcoded magic number instead of the vowel feature-count macro:**
- Issue: `src/main.c:325` declares `float xv[251];` inside the prediction loop, where `251` is `TOTAL_FEATURES` (the *per-patient* total), not the per-vowel input size actually used (`nf_vowel = FEATURES_PER_VOWEL + NUM_METADATA_FEATURES = 85`, per `src/main.c:253`). Only the first 85 slots are ever written/read.
- Files: `src/main.c:319-332` (`predict_hierarchical_late_fusion` inlined in the val-prediction loop)
- Impact: Wastes stack space (251 vs 85 floats, harmless today) but is a landmine: if `TOTAL_FEATURES` shrinks below 85 in a future change (e.g. after implementing SPEC.md Gap 1's paraconsistent selection, which reduces feature count), this stack buffer would still be sized off the *old* macro, masking the fact that it should instead be sized as `nf_vowel`. Conversely, if a future refactor renames/removes `TOTAL_FEATURES`'s relationship to this buffer without noticing the hardcoded literal, nothing will fail to compile — it will just silently keep working with an oversized buffer or, in the worst case, corrupt the stack if someone "fixes" the literal to a too-small value.
- Fix approach: Replace `float xv[251]` with `float xv[FEATURES_PER_VOWEL + NUM_METADATA_FEATURES];` or reuse the existing `nf_vowel` variable already computed at `src/main.c:253`.

**Stale/misleading comment blocks describing old feature counts:**
- Issue: The top-of-file comment in `src/feature_extract.c:1-11` says "Extrai 4 features temporais... 7 features espectrais... 18 features wavelet... 29 features por vogal", which matches none of the current constants (`NUM_TEMPORAL_FEATURES=10`, `NUM_SPECTRAL_FEATURES=55`, `NUM_WAVELET_FEATURES=18`, `FEATURES_PER_VOWEL=83`). Similarly `include/mlp.h:4-6` and `src/mlp.c:4-10` describe an "Input(150) -> Dense(256, BN, LeakyReLU) -> Dense(128) -> Dense(64) -> Dense(3, Softmax)" architecture that matches neither the 5-class legacy path nor the current Master(2)/Expert(4) hierarchical nets.
- Files: `src/feature_extract.c`, `include/mlp.h`, `src/mlp.c`
- Impact: Low runtime risk, but actively misleads anyone reading the code cold (or an LLM agent grepping for "how many features per vowel" and trusting the comment over `config.h`).
- Fix approach: Delete or rewrite these header comments to match `include/config.h`'s current constants; treat `config.h` as the single source of truth and have other files reference it rather than restate numbers.

**Legacy 5-class single-network code path is unused but still compiled:**
- Issue: `mlp_init()` (`src/mlp.c:215`) builds a network sized `MLP_INPUT_SIZE` x `MLP_OUTPUT_SIZE` (5-class legacy, per `include/config.h:80` comment "Para o modo legacy"), but nothing in `src/main.c` calls `mlp_init()` — only `mlp_init_dynamic()` is used, with output sizes 2 (master) and 4 (expert).
- Files: `src/mlp.c:215-218`, `include/config.h:76-80`
- Impact: Dead but harmless; increases surface area for confusion when onboarding.
- Fix approach: Remove `mlp_init()`/`MLP_INPUT_SIZE`/`MLP_OUTPUT_SIZE` or clearly mark them `deprecated` if kept for a future "flat 5-class" comparison baseline.

**Unfinished CLI mode (`external`):**
- Issue: `mode_validate_external()` (`src/main.c:354-358`) logs a message and immediately `return 0`, with the actual external-validation logic left as a comment: `/* Implementar carregando os 6 best_models se necessario */`. `argc`/`external_dir` parameter is unused (would trigger `-Wunused-parameter` if `-Wextra` enforced it strictly; currently silent because the parameter name is still referenced... actually it is not referenced, verify build warnings).
- Files: `src/main.c:354-358`
- Impact: `./build/vocal_detect external <dir>` silently "succeeds" without doing anything — anyone relying on it for generalization testing gets a false positive (exit code 0, no output, no error).
- Fix approach: Either implement external validation (load `models/best_master.bin`/`models/best_expert.bin`, run on a held-out directory) or make the stub fail loudly (`return -1` with a clear "not implemented" log) so silent no-ops aren't mistaken for successful runs.

## Known Bugs

**Data race on the global RNG state during parallel audio augmentation (breaks reproducibility guarantee):**
- Symptoms: `RANDOM_SEED=42` is documented (CLAUDE.md, "Methodological Notes") as making "all randomness reproducible", but `rng_state` (`src/utils.c:52`, a single `static unsigned int`) is mutated by `xorshift32()` without any synchronization, while `precalculate_augmentations()` in `src/main.c:134-166` runs `#pragma omp parallel for schedule(dynamic, 1)` over patients and calls `wav_aug_noise()` (`src/wav_augment.c:7-26`, which calls `rng_normal()` → `rng_uniform()` → `xorshift32()`) concurrently for augmentation cases 0 and 7.
- Files: `src/utils.c:52-69` (RNG state + `xorshift32`), `src/main.c:137,149,156` (parallel loop calling `wav_aug_noise`), `src/wav_augment.c:7-26`
- Trigger: Any `make train`/`make full` run — the augmentation pre-calculation step always parallelizes over all pathological patients (line 137), and the noise-augmentation branches (`case 0`, `case 7`) always execute for every one of them.
- Workaround: None currently. The bug does not crash (x86 doesn't fault on unsynchronized 32-bit read/modify/write in practice) but it makes the *exact* numeric content of the noise-augmented samples non-deterministic and dependent on OpenMP thread scheduling/`OMP_NUM_THREADS`, undermining the "reproducible with seed 42" claim for any run that includes the (currently always-executed) audio-domain augmentation path. Re-running `make clean && make full` twice on the same machine may already produce numerically different results even though `results/features.csv` (pure feature extraction, correctly not touching the RNG — see `src/feature_extract.c:113`) is deterministic.
- Fix approach: Either (a) drop `#pragma omp parallel for` from `precalculate_augmentations` (simplest, correctness over speed — this loop was ~1-2 min per the v31 log timestamps, extraction itself is already fast at ~40-55s), or (b) give each OpenMP thread its own RNG stream (e.g., seed a thread-local xorshift state from `omp_get_thread_num()` + `RANDOM_SEED`, applied deterministically per patient index rather than per thread to stay reproducible under different thread counts).

**`results/train_log_v30.txt` / `train_log_v31*.txt`: collapsed "nested stacked hierarchy" experiment — confirmed root-cause pattern, do not resume from these:**
- Symptoms: Both logs show `val_loss=-nan val_acc=-nan val_f1=0.000` for every logged epoch of every sub-network, across all folds. The final aggregated result in `train_log_v31.txt` collapses to predicting almost exclusively the majority class (Normal): confusion matrix shows 685/687 Normal correctly classified but Laringite/Disfonia Psicogenica/Reinke rows are entirely zero predictions, giving Macro F1 ≈ 0.157 (v31) / ≈ 0.28 (v30) — far below the v29 HEAD baseline (Macro F1 0.4423, Acc 69.4%).
- Files: `results/train_log_v30.txt`, `results/train_log_v31.txt`, `results/train_log_v31_intel.txt` (all untracked/WIP, not part of any commit)
- Trigger: A "Stacked Hierarchy" / "Nested Stacked Hierarchy" architecture variant (per the logs' own banner text, "Classificador Vocals - Stacked Hierarchy (v31)" and "Vocal Stacked v31.2") that expanded `TOTAL_FEATURES` to 417/447 (vs. current 251) — i.e., a different, larger feature set than v29 HEAD, and (based on the NaN pattern) very likely fed a zero-sample validation split into `mlp_evaluate()`/`compute_val_macro_f1()` for at least one nested/inner classifier.
- Root cause candidate confirmed in current code too: `mlp_evaluate()` (`src/mlp_train.c:318-335`) does `*loss_out = total_loss / n;` and returns `(float)correct / n` with **no guard for `n == 0`**. If any nested split during that experiment produced an empty validation subset for a sub-network (plausible in a multi-level "stacked" design with small minority classes), this would produce exactly the observed `NaN` propagating into `val_macro_f1`, which is the early-stopping/checkpoint-selection metric (`src/mlp_train.c:284`, `if (val_macro_f1 > best_val_macro_f1)`) — `NaN > best_val_macro_f1` is always false, so the checkpoint would freeze at whatever state existed at epoch 0 (near-random init) or patience would exhaust immediately, both consistent with a majority-class-collapsed model.
- Workaround: SPEC.md already correctly instructs "Todo trabalho deve partir do HEAD atual (v29, íntegro) — não do WIP quebrado". These three log files should not be used as a basis for future work and could be deleted or moved to an `archive/` subfolder to avoid confusing future contributors who might otherwise assume they represent a viable checkpoint to resume from.
- Fix approach (if the "nested stacked hierarchy" direction is ever revisited): Guard `mlp_evaluate()`/`compute_val_macro_f1()` against `n == 0` (return a sentinel that never wins early-stopping comparisons, e.g. `-1.0f`, and log a warning) so this failure mode fails loud instead of silently corrupting model selection.

**Unvalidated `num_channels` in WAV parsing can divide by zero:**
- Symptoms: `wav_read()` (`src/wav_io.c:118-119`) computes `wav->num_samples = (int)(data_size / (bytes_per_sample * wav->num_channels));` without checking that `wav->num_channels > 0`. A malformed/truncated `fmt ` chunk (e.g., `num_channels` read as 0 due to a corrupted or non-standard WAV in the SVD dataset) triggers an integer division by zero (undefined behavior, typically SIGFPE crash on x86 Linux).
- Files: `src/wav_io.c:74-119`
- Trigger: Any WAV file in `saudavel/`, `laringite/`, `disfonia_psicogênica/`, `disfonia_funcional/`, or `edema_de_reinke/` with a malformed `fmt ` chunk reporting 0 channels. Not currently observed (extraction logs report "0 erros" across 1098 patients), but not defensively handled either.
- Workaround: None needed today since the dataset is clean, but this is a crash-on-bad-input risk if the dataset is ever refreshed/re-exported from a different tool.
- Fix approach: Validate `wav->num_channels > 0` and `wav->bits_per_sample > 0` right after reading the `fmt ` chunk, `log_error` and `return -1` otherwise (matching the existing pattern for `audio_format != 1` two lines above).

## Security Considerations

**Not applicable / low relevance:** This is an offline, single-user academic C pipeline with no network exposure, no external service calls, and no user-supplied input beyond local file paths passed on the command line. No secrets, credentials, or PII-adjacent handling beyond patient age/sex stored in `overview_merged.csv` (a local research dataset, not distributed).

**Path handling has no traversal protection, but exposure is nil:**
- Risk: `snprintf(csv_path, 1024, "%s/%s", base_dir, CSV_METADATA)` (`src/main.c:238`) and similar path-building in `dataset.c` trust `argv[2]` (`base_dir`) without sanitization.
- Files: `src/main.c:238,240,347,350`
- Current mitigation: None; not needed — this is a local CLI tool run by its own author on trusted input.
- Recommendations: No action needed unless this tool is ever wrapped in a service that accepts untrusted `base_dir` values.

## Performance Bottlenecks

**`find_knn()` in SMOTE is O(n²) per synthetic sample, recomputed from scratch every call:**
- Problem: `find_knn()` (`src/main.c:189-203`) recomputes squared distances from `base` to every member of `class_indices` (size `n_class`) on every call, and `smote_oversample()` (`src/main.c:205-231`) calls it once per synthetic sample needed (`max_count - n_class` times per class). For the largest class imbalance (Normal ~687 vs Reinke ~68 per full dataset, worse inside a single fold/vowel/expert split), this is `O(n_synthetic * n_class * nf)` per fold/vowel/network (4 SMOTE calls per fold: master + expert, ×5 folds ×3 vowels = 30 calls total).
- Files: `src/main.c:189-231`
- Cause: No caching of pairwise distances across synthetic-sample generations for the same class; k-NN is recomputed independently for each of the potentially hundreds of synthetic samples needed to balance a class.
- Improvement path: Precompute a distance matrix or a k-NN adjacency list once per class before the synthesis loop, then reuse it across all `n_synthetic` iterations for that class. Given `nf_vowel≈85` and class sizes in the low hundreds, this is not currently a bottleneck at the whole-pipeline level (feature extraction and MLP training dominate wall-clock time) — flagged for awareness if class sizes or `nf_vowel` grow substantially (e.g. after adding more feature families).

**`update_bn_stats_from_data()` recomputes forward activations from scratch every epoch even though BatchNorm is currently disabled everywhere:**
- Problem: `mlp_train()` calls `update_bn_stats_from_data(net, train_x, n_train, num_features)` unconditionally at the top of every epoch (`src/mlp_train.c:212`). Internally this function does check `if (!l->bn.enabled) continue;` per layer (`src/mlp_train.c:78`), so with `use_bn[] = {0,0,0}` (`src/mlp.c:228`, the current default) the inner sampling/forward-recompute loops never execute their expensive body — but the function call and its `n_sample = min(n_train, 256)` bookkeeping still runs on every epoch for every one of the 30 (master+expert × 5 folds × 3 vowels) networks trained per `make full` run.
- Files: `src/mlp_train.c:70-158`, `src/mlp.c:228,234`
- Cause: The BN-disabled fast path still pays the function-call and loop-setup overhead every epoch; harmless in practice (early `continue` before any real work) but a wasted call given BN has been confirmed (CLAUDE.md "What doesn't work") not to help on this dataset size.
- Improvement path: Skip calling `update_bn_stats_from_data()` entirely when no layer has BN enabled (check once outside the epoch loop), or remove the BN scaffolding altogether if it will not be revisited — it currently adds ~140 lines of dead-weight complexity to `mlp_train.c`/`mlp.c` for a technique the project has already decided against.

## Fragile Areas

**`src/main.c`'s `mode_train` fold loop: dense, single-letter-heavy, manually-managed buffers:**
- Files: `src/main.c:235-343`
- Why fragile: The entire hierarchical late-fusion training loop (feature slicing per vowel, binary/expert label remapping, SMOTE, per-vowel train/val splits) is implemented in one ~110-line function with 15+ manually paired `malloc`/`free` calls per vowel iteration (`src/main.c:276-316`) and terse variable names (`tr_x_v`, `vl_y_bin`, `ex_tr_x`, `os_e_y`, etc.) that make it easy to mismatch a buffer's intended size (see the `xv[251]` issue above) or forget a free path on an early-return (there currently are no early returns inside the fold loop, but adding one — e.g., for the SPEC.md Gap 1 "if zero features survive, relax and retry" logic — without auditing all 15+ live allocations would be easy to get wrong).
- Safe modification: Before adding new logic inside the fold loop (e.g., SPEC.md's paraconsistent feature selection or Borderline-SMOTE), extract the per-vowel block (`src/main.c:275-317`) into its own named function (`train_vowel_networks(...)`) taking/returning explicit structs, so allocation lifetimes are scoped to one call frame instead of interleaved with 3 other vowel iterations' buffers.
- Test coverage: None — see Test Coverage Gaps below. There is no automated way to detect a leaked or double-freed buffer here short of manual review or running under Valgrind/ASan, neither of which is wired into the build.

**Manual memory management with no sanitizer/Valgrind run in the toolchain — resource leaks confirmed present but currently harmless:**
- Files: `src/main.c` (`mode_train`, `Makefile`)
- Why fragile: `mode_train()` (`src/main.c:235-343`) never calls `dataset_free(&ds)` or `features_free(&fm)` before returning — both `Dataset ds` (loaded at `src/main.c:239`) and `FeatureMatrix fm` (loaded/extracted at `src/main.c:241-244`) leak for the lifetime of the process. Contrast with `mode_extract()` (`src/main.c:345-352`), which correctly calls both `features_free(&fm)` and `dataset_free(&ds)`. Since `mode_train` is only ever invoked once per process (`main()` exits immediately after), this leak is currently harmless (reclaimed by OS on exit) but is inconsistent with the rest of the codebase's discipline and would become a real leak if `mode_train` were ever called in a loop (e.g., a future hyperparameter-sweep driver or the SPEC.md Gap 3 "train 4 configs × 5 folds" comparison harness, which is exactly the kind of change likely to turn this into a real, cumulative leak).
- Safe modification: Add `features_free(&fm); dataset_free(&ds);` before each `return` in `mode_train` (mirroring `mode_extract`'s pattern) — cheap, safe, no behavior change today, and removes a landmine for future multi-run drivers.
- Test coverage: No sanitizer build target exists in the `Makefile` (`CFLAGS` has no `-fsanitize=address,undefined`). Recommend adding a `make debug`/`make asan` target using `gcc ... -fsanitize=address,undefined -g -O0` for periodic manual leak/UB audits, especially before/after SPEC.md Gap 1-3 changes touch the fold loop's allocation patterns.

**`smote_oversample()` degenerates when a class has very few training samples in a given fold/vowel split:**
- Files: `src/main.c:205-231`, `find_knn` at `src/main.c:189-203`
- Why fragile: When `n_class == 1` for some class in a given fold (plausible for the 4-class Expert net on the smallest classes — Disfonia Psicogenica n≈91 and Edema de Reinke n≈68 total, further split 5 ways for CV and reduced further by the train/val split), `find_knn`'s only candidate neighbor is the sample itself (guarded to `dist=1e30f` but still selected as the single nearest "neighbor" since it's the only entry), so the "synthetic" sample generated is an exact duplicate of the base sample (`alpha` interpolation between a point and itself). This silently degrades SMOTE to plain oversampling-by-duplication for that class/fold without any log/warning.
- Safe modification: Add a guard in `smote_oversample` (`src/main.c:218-229`) to detect `n_class <= 1` and either skip synthesis (log a warning, fall back to duplication explicitly and label it as such) or borrow from a neighboring class prototype — visible behavior either way, current silent duplication should at least be logged so it's visible in `make full` output.
- Test coverage: No unit tests exist for `smote_oversample`/`find_knn` in isolation (see Test Coverage Gaps).

## Scaling Limits

**Dataset size (1098 patients, severe class imbalance) is a hard ceiling documented and accepted, not a code defect:**
- Current capacity: 687/140/91/112/68 per class; smallest two classes (Disfonia Psicogenica=91, Edema de Reinke=68) limit how much any architecture change can improve minority-class F1.
- Limit: Per `CLAUDE.md`'s "Fundamental Bottleneck" section, Disfonia Psicogenica vs Disfonia Funcional show one-vs-rest AUC ≈ 0.62–0.64 — acoustically near-indistinguishable given both are functional dysphonias without structural lesion. This ceiling is corroborated by the current v29 results (`results/metrics_global.csv`): Disfonia Psicogenica F1=0.286, Disfonia Funcional F1=0.251, both far below Normal (F1=0.869) and even below Laringite (F1=0.382)/Reinke (F1=0.423), which have structural/inflammatory correlates.
- Scaling path: Not solvable by architecture/hyperparameter tuning alone (already explored extensively per CLAUDE.md's "What doesn't work" list). Genuine improvement requires either (a) more labeled data for the two functional-dysphonia classes, (b) non-acoustic features (e.g., laryngoscopic/videostroboscopic data, patient-reported symptom questionnaires) to break the acoustic ceiling, or (c) reframing the task (e.g., collapsing the two functional classes into one "Functional Dysphonia" super-class for a 4-class problem, trading granularity for measurable separability) — this last option is a scope/requirements decision, not an engineering one.

**In-memory feature matrix and per-fold buffers scale linearly and are fine at current size, but `find_knn`'s O(n²) behavior (see Performance Bottlenecks) would become a real bottleneck if the dataset grows 5-10×** without a corresponding refactor to precomputed neighbor lists.

## Dependencies at Risk

**None identified.** The project has zero external library dependencies beyond the C standard library, `libm`, and OpenMP (`gcc -fopenmp`) — all resolved via the system compiler toolchain (`Makefile`). This is a strength, not a risk: no version-pinning, no supply-chain surface, no dependency-update burden.

## Missing Critical Features

**SPEC.md Gap 1 — Paraconsistent feature selection (Prioridade ALTA, per SPEC.md):**
- Problem: The original PIBIC proposal specifies paraconsistent evidence-based feature selection (LPA2v, citing Avron/Arieli/Zamansky and Abe) as part of the methodology; the current pipeline applies **no feature selection at all** — all ~85 features/vowel go directly to each network.
- Blocks: Academic defensibility of the "Seleção de Características" methodology section against a review board that will likely ask about it directly (per SPEC.md's own framing: "Uma banca avaliadora tende a perguntar especificamente sobre isso").
- Status: Fully specified in `SPEC.md` (function signature `paraconsistent_select()`, integration points in `src/main.c`'s fold loop before `mlp_init_dynamic`, acceptance criteria tied to a Macro F1 no-regression rule against the current baseline of 0.4423/69.4%) but not implemented in any `.c`/`.h` file yet.

**SPEC.md Gap 2 — Borderline-SMOTE (Prioridade MÉDIA, per SPEC.md):**
- Problem: `smote_oversample()` (`src/main.c:205-231`) implements standard SMOTE (Chawla et al., 2002) with no borderline/safe/noise sample classification (Han, Wang & Mao, 2005), despite Borderline-SMOTE being named explicitly in `CLAUDE.md`'s "What Worked" section (a documentation inaccuracy in itself — CLAUDE.md's own text claims "Borderline-SMOTE (better than regular SMOTE)" was already implemented and validated, which SPEC.md's audit contradicts by inspecting the actual code).
- Blocks: The citation `HAN; WANG; MAO, 2005` cannot honestly be used in the academic report/poster until this gap is closed — SPEC.md explicitly flags this as a previously-made citation error ("erro já corrigido uma vez nesta sessão").
- Status: Fully specified in `SPEC.md` (mode enum, `find_knn_global()` variant, safe/borderline/noise classification rule) but not implemented.

**SPEC.md Gap 3 — Shallow vs. deep network comparison (Prioridade MÉDIA-BAIXA, per SPEC.md):**
- Problem: The proposal requires comparing shallow vs. deep architectures and justifying the final choice by complexity/performance trade-off; only one shallow configuration (`Dense(128)`, 1 hidden layer effectively, since `MLP_NUM_LAYERS=3` means Input→Hidden1→Hidden2→Output with `use_bn`/`dropout` per layer — see `include/config.h:81` and `src/mlp.c:225-235`) has ever been evaluated for the Master/Expert networks.
- Blocks: The "justificativa da escolha por complexidade" the proposal explicitly requires has no comparative data to cite.
- Status: Fully specified in `SPEC.md` (proposed `mlp_init_multi()` generalizing `mlp_init_dynamic()`, 4 configs A-D, nested-CV comparison protocol using existing McNemar/bootstrap infra) but not implemented. Note: implementing this safely requires first fixing the `mlp_backward()` hardcoded `max_size = MLP_HIDDEN1_SIZE` delta-buffer sizing (`src/mlp_train.c` — actually `src/mlp.c:287`, see below) if any tested config uses a hidden layer wider than 128.

**Fixed-size delta buffer in backprop assumes no hidden layer exceeds `MLP_HIDDEN1_SIZE` (128):**
- Problem: `mlp_backward()` (`src/mlp.c:282-289`) allocates `delta`/`delta_next` sized `max_size = MLP_HIDDEN1_SIZE` (128), hardcoded rather than computed as `max(layer sizes)`. This is safe today because `MLP_HIDDEN1_SIZE=128` is in fact the largest layer in both the 3-layer and (unused) 4-layer (`#else` branch, `src/mlp.c:229-234`) configurations.
- Blocks: SPEC.md Gap 3's proposed Config C/D (`[128, 64]`, `[128, 64, 32]`) happen to still fit under 128, but any future config with a first hidden layer wider than 128 (e.g., testing `[256, 128]` as CLAUDE.md's very first `mlp.h` docstring once described) would silently overflow this buffer — a stack/heap corruption bug that would not be caught by any test in this repo.
- Fix approach: Compute `max_size` from the actual layer sizes in the `MLP` struct being backpropagated, not from the `MLP_HIDDEN1_SIZE` constant.

## Test Coverage Gaps

**No automated tests exist anywhere in the repository:**
- What's not tested: Everything — there is no test framework, no `tests/` directory, no assertions beyond `main()`'s own training run. `make test` is defined in `Makefile:27-28` but simply invokes `./build/vocal_detect test`, and `main()` (`src/main.c:360-369`) has no `"test"` case, so it silently falls through to `return 1` after printing only the startup banner line.
- Files: `Makefile:27-28`, `src/main.c:360-369` (confirmed also documented as a known caveat in `CLAUDE.md`: "`test` mode is not implemented in `main.c`")
- Risk: Every one of the concrete bugs/fragile-areas documented above (RNG race, `n==0` division in `mlp_evaluate`, the `xv[251]` magic number, dataset/feature leaks in `mode_train`, SMOTE degeneration on tiny classes) would be caught immediately by a small unit-test suite around `src/utils.c` (RNG determinism under threading), `src/mlp_train.c` (`mlp_evaluate` with `n=0`), and `src/main.c`'s SMOTE/feature-slicing helpers — none of which currently exist.
- Priority: **High** for the RNG-race and `n==0`-division issues specifically (both are silent-corruption failure modes that already manifested once, in the collapsed v30/v31 experiment) — Medium for the rest, given the project's academic/single-researcher context and the fact that end-to-end validation currently happens via full `make full` runs compared against known-good Macro F1/Accuracy numbers (an acceptable, if slow, substitute for unit tests in this context).

**DSP/feature-extraction correctness (jitter, shimmer, HNR, F0, formants, MFCC, CPP, wavelet stats) has no ground-truth comparison:**
- What's not tested: None of `src/feature_temporal.c`, `src/feature_spectral.c`, `src/feature_wavelet.c` are validated against a reference implementation (e.g., Praat, librosa) or a synthetic signal with known jitter/shimmer/F0 values.
- Files: `src/feature_temporal.c`, `src/feature_spectral.c`, `src/feature_wavelet.c`
- Risk: A subtle DSP bug (off-by-one in frame windowing, incorrect mel-scale conversion, wrong quefrency range for CPP) would silently produce plausible-looking but wrong feature values — there is no way to distinguish "the acoustic ceiling is real" from "a feature extraction bug is suppressing signal" without an independent reference check. Given CLAUDE.md already documents extensive experimentation confirming the ceiling is robust across many hyperparameter/architecture changes, this is lower-probability, but it is the kind of gap that specifically undermines the "Fundamental Bottleneck" claim's credibility if ever audited.
- Priority: Medium — a one-time validation against 3-5 synthetic test tones (known F0, known injected jitter) for `feature_temporal.c`/`feature_spectral.c` would substantially de-risk the project's central acoustic-ceiling claim at modest one-time cost.

---

*Concerns audit: 2026-07-27*
