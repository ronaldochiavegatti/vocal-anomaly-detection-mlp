# Coding Conventions

**Analysis Date:** 2026-07-27

## Language & Compiler

- **Standard:** C99 (`-std=c99`)
- **Compiler:** gcc, flags `-O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -fopenmp`
- **Linker:** `-lm -fopenmp`
- Code must compile cleanly under `-Wall -Wextra` (format-truncation warnings from `snprintf` path building are the only suppressed class). Treat any new warning as a bug to fix, not to suppress.
- `_POSIX_C_SOURCE` is defined at the top of files that need POSIX APIs (e.g. `#define _POSIX_C_SOURCE 199309L` in `src/utils.c` for `clock_gettime`, `#define _POSIX_C_SOURCE 200809L` in `src/dataset.c` for `dirent.h`/`sys/stat.h`). Add this **before any `#include`** when a new file needs POSIX-only functions.
- There is a repo hook (`.claude/hooks/compile-check.sh`) that automatically runs `make` after any edit to a `.c`/`.h` file — code is expected to build after every change, not just at the end of a task.

## File Organization

**One module = one `.c` + one `.h` pair**, named after its responsibility, e.g. `src/kfold.c` / `include/kfold.h`, `src/normalize.c` / `include/normalize.h`. There is no `src/lib/` or `src/core/` subdivision — all 19 modules live flat in `src/` and `include/`.

**File header comment** (mandatory for every `.c` and `.h` file), in Portuguese, describing purpose and (for algorithm-heavy files) the method:
```c
/*
 * kfold.c - Stratified K-Fold Cross-Validation
 *
 * Separa indices por classe, embaralha cada grupo,
 * e distribui proporcionalmente nos K folds.
 */
```
Header files repeat the same block but describe the public contract rather than the implementation:
```c
/*
 * kfold.h - Stratified K-Fold Cross-Validation
 *
 * Divide o dataset em K folds estratificados por classe,
 * garantindo separacao por paciente (nao por gravacao).
 */
```

**Include guards:** `#ifndef MODULE_H` / `#define MODULE_H` / `#endif /* MODULE_H */` (module name upper-cased, matches filename). No `#pragma once`.

**Include order inside a `.c` file:** own header first, then other project headers (`utils.h`, `config.h`, etc.), then system headers (`<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<math.h>`, ...). See `src/kfold.c`, `src/dataset.c`, `src/metrics.c`.

## Naming Patterns

**Files:** `snake_case`, matching the primary struct/domain they implement (`feature_spectral.c`, `wav_augment.c`, `mlp_train.c`).

**Functions:** `snake_case`, prefixed with the module name acting as a namespace:
- `mlp_forward`, `mlp_backward`, `mlp_init_dynamic`, `mlp_save_checkpoint` (module `mlp`)
- `norm_fit`, `norm_transform`, `norm_save`, `norm_load`, `norm_free` (module `norm`)
- `kfold_split`, `kfold_free` (module `kfold`)
- `metrics_compute`, `metrics_print`, `metrics_bootstrap_ci`, `metrics_mcnemar` (module `metrics`)
- `dataset_load`, `dataset_free` (module `dataset`)
- `rng_seed`, `rng_uniform`, `rng_int`, `rng_normal`, `rng_shuffle_int` (module `rng`, lives in `utils.c`)
- `log_debug`, `log_info`, `log_warn`, `log_error` (module `log`, lives in `utils.c`)

File-local helpers that are not part of the module's public API are declared `static` and named descriptively without the module prefix, e.g. `static int is_directory(...)`, `static void associate_csv_metadata(...)` in `src/dataset.c`; `static float cosine_annealing_lr(...)`, `static int argmax(...)` in `src/mlp_train.c`. There are 80+ `static` functions across `src/` — **prefer `static` for anything not declared in the corresponding header.**

**Types:** `PascalCase` for `struct`/`typedef` (`MLP`, `Layer`, `BatchNorm`, `Dataset`, `Patient`, `FeatureMatrix`, `SpectralFeatures`, `KFoldSplits`, `FoldSplit`, `MetricsResult`, `NormParams`, `TrainHistory`, `ConfidenceInterval`, `CsvData`). Always paired with `typedef struct { ... } Name;` — no separate `struct Name` tag used elsewhere in the code.

**Constants / macros:** `UPPER_SNAKE_CASE`, defined in `include/config.h` for anything that is a tunable hyperparameter, dimension, or path (`NUM_CLASSES`, `TOTAL_FEATURES`, `MLP_HIDDEN1_SIZE`, `LEARNING_RATE`, `RANDOM_SEED`, `CLASS_WEIGHT_REINKE`). File-local magic numbers get a local `#define` at the top of the `.c` file instead (`#define MIN_STD 1e-8f` in `normalize.c`, `#define ROC_N_THRESHOLDS 101` in `metrics.c`, `#define INITIAL_CAPACITY 1024` in `dataset.c`).

**Variables:** `snake_case`, short and local (`n_train`, `n_val`, `nf_all`, `os_m_x` for "oversampled master x"). Loop indices are single letters chosen by axis: `i`/`j` generic, `c` for class, `v` for vowel, `f` for feature, `l`/`li` for layer, `s`/`b` for sample/batch.

**No camelCase** anywhere in identifiers (verified by grep) — this is a strict convention across the whole codebase. Struct field access on `PascalCase` types uses `snake_case` fields (`net->layers[i].output_size`, `patients[i].class_label`).

## Comments

- All prose comments are in **Portuguese** (matching the academic/PIBIC context of the project) — code identifiers themselves are in English/technical terms (`weights`, `dropout_mask`, `forward`).
- Comment style: `/* ... */` block comments for section headers and function documentation; `//` is essentially absent from the code (grep found none used as a marker) — do not introduce `//` line comments, stay consistent with `/* */`.
- Section dividers inside header files use a consistent banner: `/* ========== Nome da Secao ========== */` (see `config.h`, `utils.h`, `mlp_train.c`'s "Checkpoint buffers" style groupings are inline, but constants files always use this banner).
- Every non-trivial public function in a `.h` file has a `/* ... */` doc comment above its declaration describing parameters and return value semantics (see `metrics.h` for the most complete example — every function documents input array shapes with `[n]` / `[n x m]` notation and cites the algorithm's academic source, e.g. "Teste de McNemar ... (Edwards, 1948)", "Breiman, 2001" for permutation importance).
- No TODO/FIXME/HACK/XXX markers exist anywhere in `src/` or `include/` — the project's own convention is to resolve or document issues in `SPEC.md`/`CLAUDE.md` rather than leave inline markers.

## Error Handling

**No exceptions (C99).** Error handling is by return code, consistently `int` functions returning `0` on success and `-1` on failure:
```c
int dataset_load(const char *base_dir, const char *csv_path, Dataset *ds);
int norm_save(const NormParams *params, const char *path);
int metrics_export_csv(const MetricsResult *result, const char *path);
```
Callers check with `if (fn(...) != 0) { ...; return -1; }` and propagate upward — see the chain in `mode_train()`/`mode_extract()` in `src/main.c`. `main()` maps the top-level mode function's return value to a process exit code (`0` or `1`).

**Fatal allocation failures abort the process** — never handled as recoverable errors. All heap allocation goes through `safe_malloc`/`safe_calloc`/`safe_realloc` in `src/utils.c`, which print `[FATAL] ... falhou para %zu bytes` to stderr and call `exit(EXIT_FAILURE)` if the underlying `malloc`/`calloc`/`realloc` returns `NULL`. **Do not call `malloc`/`calloc`/`realloc` directly in new code** — always use the `safe_*` wrappers so allocation failures are handled uniformly.

**Recoverable I/O errors** (missing file, bad format) use the `-1` return convention and are logged via `log_error`/`log_warn` before returning, e.g. `src/dataset.c`'s `enumerate_class()`:
```c
DIR *dir = opendir(dir_path);
if (!dir) {
    log_error("Nao foi possivel abrir diretorio: %s", dir_path);
    return 0;
}
```
File-open failures in binary/CSV serialization (`norm_save`, `norm_load`, `selected_save`, `selected_load`, `metrics_export_csv`) simply `return -1` without logging — the caller is expected to decide whether the failure is fatal.

**No `assert()` usage** anywhere in the codebase — bounds/sanity checks are explicit `if` guards, not asserts (e.g. `if (t >= 0 && t < NUM_CLASSES && p >= 0 && p < NUM_CLASSES)` in `metrics_compute`).

**Cache/format validation pattern:** functions that load previously-serialized data validate structural invariants (column count, feature count) and reject silently by returning `-1` rather than crashing — see `features_load_csv()` in `src/main.c` (rejects if CSV column count != `TOTAL_FEATURES`) and `selected_load()` in `src/feature_select.c` (rejects if `n <= 0 || n > TOTAL_FEATURES`).

## Logging

**Framework:** custom minimal logger in `src/utils.c` / `include/utils.h` — no external logging library.

**Levels:** `LOG_DEBUG < LOG_INFO < LOG_WARN < LOG_ERROR`, set globally via `log_set_level()` (called once in `main()` with `LOG_INFO`). Output destination defaults to `stderr`, overridable via `log_set_file()`.

**Usage pattern:**
```c
log_info("Classe %d (%s): %d pacientes", c, class_dirs[c], added);
log_error("Nao foi possivel abrir diretorio: %s", dir_path);
log_debug("Paciente %d: WAVs faltando, pulando", patient_id);
```
Every log line is automatically timestamped (`[YYYY-MM-DD HH:MM:SS] [LEVEL] message`) and flushed immediately (`fflush`). Log messages are in Portuguese, matching comments.

`log_info` is also used for progress/结果 reporting during training (epoch summaries every 10 epochs, fold results, aggregated results) — this doubles as the project's "test output" since there is no separate reporting mechanism (see TESTING.md).

## Function Design

**Size:** Ranges widely. Small utility functions (`argmax`, `is_directory`) are 3-10 lines. Orchestration functions in `main.c` (`mode_train`, `precalculate_augmentations`) run 60-100+ lines because they inline the full per-fold pipeline (SMOTE, normalization, per-vowel MLP splitting) rather than being split into many tiny helpers — this is accepted practice in this codebase for the top-level training loop, but new *reusable* algorithms (feature extraction, metrics, DSP) should still be factored into small single-purpose functions in their own module, matching the rest of `src/`.

**Parameters:** Explicit array + length pairs everywhere (`const float *x, int n`), never hidden globals for data. Output is returned either via return value (for scalars: `float mlp_evaluate(...)`) or via `_out` suffixed pointer parameters (for computed aggregates: `float *chi2_out, float *p_value_out` in `metrics_mcnemar`; `float *importance_acc_out` in `metrics_permutation_importance`). Row-major flattened matrices are the norm for 2D data (`features[i * num_features + j]`), always documented with a `[n x m]` shape comment.

**Return values:** `int` 0/-1 for fallible operations; `float`/computed value directly for pure calculations with no failure mode (`mlp_loss`, `cosine_annealing_lr`); `void` for in-place mutation or side-effect-only functions (`mlp_forward`, `norm_transform`, `log_info`).

## Module Design

**Exports:** Each `.h` declares only the functions and types meant for cross-module use; internal helpers stay `static` in the `.c` file (see Naming Patterns above). There are no "internal" headers (e.g. `*_internal.h`) — if a function isn't in the header, it is private by convention.

**No barrel files / no single "API" header** — `main.c` includes every module header it needs directly (18 `#include`s at the top of `src/main.c`). There is no facade or re-export layer.

**Global/static state is minimal and explicit:** the RNG state (`static unsigned int rng_state` in `utils.c`), the log level/file (`static LogLevel current_log_level`, `static FILE *log_file`), and class name lookup tables (`static const char *class_names[NUM_CLASSES]` in `metrics.c`, `static const char *vowel_names[NUM_VOWELS]` in `dataset.c`) are the only module-level statics. All per-run/per-fold data (datasets, MLP weights, fold splits) is passed explicitly through function parameters — no hidden singletons for pipeline state.

**Config centralization:** All tunable constants (feature counts, hyperparameters, class weights, paths) live in `include/config.h` and are referenced by name everywhere — never re-declared or hardcoded in `.c` files. When adding a new hyperparameter, add it to `config.h` under the matching `/* ========== Section ========== */` banner rather than defining it locally.

---

*Convention analysis: 2026-07-27*
