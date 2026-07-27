# Testing Patterns

**Analysis Date:** 2026-07-27

## No Unit Test Framework

**There is no automated test suite in this codebase.** There is no CUnit/Unity/Check/CMocka, no `tests/` directory, no `*.test.c`/`*_test.c` files, and no assertions library. This is a deliberate characteristic of the project, not an oversight to silently "fix" by bolting on a generic test framework — read this document fully before assuming test infrastructure is missing/needed.

**The Makefile has a vestigial `test` target that does not work:**
```makefile
test: $(TARGET)
	./$(TARGET) test
```
`main()` in `src/main.c` only recognizes the modes `extract`, `train`, `full`, and `external`:
```c
if (strcmp(mode, "extract") == 0) return mode_extract(base_dir) == 0 ? 0 : 1;
if (strcmp(mode, "train") == 0 || strcmp(mode, "full") == 0) return mode_train(base_dir) == 0 ? 0 : 1;
if (strcmp(mode, "external") == 0) return mode_validate_external(base_dir) == 0 ? 0 : 1;
return 1;
```
`mode_validate_external()` (the `external` mode) is also a stub — it logs a message and returns `0` without doing anything (`src/main.c:354-358`). **Do not attempt to implement `make test` by guessing what it "should" do** — if a task requires it, treat it as a new feature to be scoped explicitly, not a bug fix.

**Why there's no unit test suite (project reality, not a gap to silently patch):**
- This is a from-scratch numerical/DSP/ML pipeline in C99 with no dependency-injection seams — nearly every function operates on large float arrays (feature matrices, MLP weights) derived from real WAV audio and a real patient metadata CSV.
- Correctness for this class of project (signal processing + ML) is fundamentally **statistical**, not exact-value: a jitter/shimmer/MFCC extractor can only be validated against reference values (e.g. from Praat) or via its downstream effect on classification metrics — not via `assertEqual`.
- The project's actual quality gate is **end-to-end 5-fold cross-validation metrics compared across versions (A/B)**, described below, which is how every architecture/hyperparameter change in the git history (`train_log_v2.txt` through `train_log_v31_intel.txt` in `results/`) has been validated.

## How Correctness Is Actually Validated

### 1. Compile-time check (fastest feedback loop)

`.claude/hooks/compile-check.sh` is a PostToolUse hook that runs `make` automatically after every edit to a `.c`/`.h` file. This is the closest thing to a "test" that runs on every change: **the code must compile cleanly under `-Wall -Wextra -std=c99`** before any other validation is meaningful. Treat a new compiler warning as a blocking regression.

### 2. Feature extraction determinism via cached CSV

`results/features.csv` is a cached `~1098 × TOTAL_FEATURES` matrix produced by `make extract` (`mode_extract()` in `src/main.c`, backed by `features_extract_all()` in `feature_extract.c`). Its header row's column count is validated against `TOTAL_FEATURES` at load time (`features_load_csv()`, `src/main.c:73-99`):
```c
int n_cols = 1;
for (const char *p = buf; *p && *p != '\n' && *p != '\r'; p++) if (*p == ',') n_cols++;
if (n_cols - 1 != TOTAL_FEATURES) { fclose(f); return -1; }
```
This means **any change to `config.h`'s feature-count constants (`NUM_TEMPORAL_FEATURES`, `NUM_SPECTRAL_FEATURES`, `NUM_WAVELET_FEATURES`, `NUM_METADATA_FEATURES`, `TOTAL_FEATURES`) automatically invalidates the stale cache** and forces re-extraction — this is the project's substitute for a "schema migration test". After changing feature extraction code, always delete `results/features.csv` (or `make clean`, which also does this) and re-run `make extract`/`make full` to confirm the new column count matches and extraction completes without `[FATAL]` aborts.

### 3. Stratified 5-fold cross-validation (the real correctness signal)

`kfold_split()` (`src/kfold.c`) partitions the ~1098 patients into `K_FOLDS=5` stratified folds using `RANDOM_SEED=42` (fixed for reproducibility — see `rng_seed(RANDOM_SEED)` before `kfold_split()` in `mode_train()`). Each fold is trained and evaluated independently (`mode_train()`, `src/main.c:235-343`); per-fold and aggregate (out-of-fold, concatenated) metrics are what determines whether a change is an improvement or a regression. There is no held-out test set beyond the 5-fold rotation — the concatenated out-of-fold predictions across all 5 folds serve as the full-dataset "test" result.

**Run it:**
```bash
mkdir -p results models     # once, before first run
make clean                  # invalidates results/*.csv cache — use when TOTAL_FEATURES changed
make full                   # or `make train` — trains + evaluates all 5 folds (~60-90 min)
```

### 4. Statistical validation layer (`src/metrics.c`)

Beyond raw accuracy/F1, the project implements formal statistical tests to avoid over-interpreting noisy small-sample results:

- **Bootstrap confidence intervals** — `metrics_bootstrap_ci()` resamples the concatenated out-of-fold predictions with replacement `N=1000` times (`seed=42`) and reports the 2.5/97.5 percentile band for accuracy, macro F1, and per-class F1 (`CI_ACCURACY` ... `CI_F1_REINKE` in `include/metrics.h`). A new result is only considered a real improvement if its point estimate and CI don't overlap the previous baseline's CI.
- **McNemar's test** — `metrics_mcnemar()` compares the MLP's predictions against baseline classifiers (majority class, kNN, logistic regression) on the same samples, using Edwards' continuity-corrected chi-square (`(|b-c|-1)^2/(b+c)`) and reports a p-value via `erfc`. This is how the project establishes that the MLP is *statistically* better than a trivial baseline, not just numerically higher on one run.
- **ROC/AUC and Precision-Recall curves** — `metrics_roc_auc()` / `metrics_pr_curve()` compute one-vs-rest curves via the trapezoid rule over 101 thresholds, exported to `results/roc_curves.csv` / `results/pr_curves.csv`.
- **Permutation feature importance** (Breiman, 2001) — `metrics_permutation_importance()` shuffles one selected feature column at a time and measures the drop in accuracy/macro-F1, exported to `results/feature_importance.csv`. Used to sanity-check that "important" features make acoustic sense (not to gate builds).

### 5. Manual A/B comparison of train logs (the project's real regression test)

Every experimental run's full stdout/stderr is preserved as `results/train_log_v{N}[_tag].txt` (30+ versioned logs currently checked in, from `train_log_v2.txt` to `train_log_v31_intel.txt`). **This is the project's de facto regression suite**: a change is accepted only if the new run's aggregate Macro F1 / accuracy / per-class F1 in `results/metrics_global.csv` (and the printed confusion matrix from `metrics_print()`) is compared side-by-side against the most recent known-good log, using the *same* `RANDOM_SEED=42` folds so the comparison is apples-to-apples.

`SPEC.md` codifies this as an explicit acceptance rule for any future change:
> nenhuma mudança deste spec deve ser incorporada em definitivo se piorar o Macro F1 global de referência (0,4435) ou a acurácia (69,4%) obtidos no 5-fold CV atual. Toda mudança é validada por comparação A/B com os mesmos folds (RANDOM_SEED=42), nunca por inspeção visual dos resultados.

("no change from this spec should be permanently incorporated if it worsens the reference global Macro F1 (0.4435) or accuracy (69.4%) obtained in the current 5-fold CV. Every change is validated by A/B comparison on the same folds, never by visual inspection of the results.")

**In practice, when modifying anything in the extraction/training/evaluation pipeline:**
1. Run `make clean && mkdir -p results models` if feature counts changed, otherwise just re-run.
2. Run `make full 2>&1 | tee results/train_log_vNEXT_<short-description>.txt` to capture a new versioned log (matches the existing naming convention).
3. Compare the new `results/metrics_global.csv` (accuracy, macro F1, per-class F1, bootstrap CI) against the most recent prior log/CSV.
4. Only keep the change if it does not regress Macro F1 / accuracy below the last accepted baseline (see current baseline figures in `CLAUDE.md` and `SPEC.md`).
5. `.claude/hooks/metrics-summary.sh` automatically prints `results/metrics_global.csv` back to the terminal after any `make full`/`make train` invocation, and `.claude/hooks/validate-dirs.sh` warns before that if `results/`/`models/` don't exist yet — both are convenience hooks, not a replacement for step 3-4's manual comparison.

### 6. Learning curves for overfitting diagnosis

`train_history_export_csv()` (`src/mlp_train.c`) writes per-fold, per-epoch train/val loss and accuracy/F1 to `results/learning_curves.csv`. This is inspected manually (not asserted programmatically) to check for overfitting (train/val divergence) or early-stopping-too-early symptoms when a change unexpectedly regresses macro F1.

## What "Adding a Test" Means in This Codebase

Given the absence of a unit-test framework, if a future task explicitly requests test coverage, prefer one of these approaches consistent with the existing patterns rather than introducing a new testing framework wholesale:

- **DSP/feature-level checks**: a small standalone `.c` file (compiled ad hoc, not wired into the `Makefile`'s object graph) that feeds a synthetic/reference WAV or hand-computed signal into a single extractor function (e.g. `temporal_extract`, `spectral_extract`) and prints the result for manual comparison against a known reference (e.g. Praat output for jitter/shimmer/HNR). This matches how the codebase already validates DSP correctness informally.
- **Pipeline-level checks**: rely on the existing `extract` → `train`/`full` → `results/metrics_global.csv` flow; do not fake this with mocked data, since the entire point of the pipeline is behavior on the real 1098-patient dataset with its real class imbalance.
- **Statistical validation**: extend `src/metrics.c` (bootstrap CI / McNemar / ROC) rather than writing new one-off validation scripts, since `metrics.h` is the established place for evaluation logic.

## Mocking, Fixtures, Coverage

**Mocking:** Not applicable — there is no dependency-injection layer or interfaces to mock. All modules operate on concrete arrays/structs; "swapping" data sources happens only in the `extract` vs. `train` mode split (whether features come from cached CSV or fresh WAV extraction), governed by `features_load_csv()`'s cache-hit/miss logic in `mode_train()`.

**Fixtures / synthetic data:** None checked in. All validation runs against the real SVD-derived dataset (`saudavel/`, `laringite/`, `disfonia_psicogênica/`, `disfonia_funcional/`, `edema_de_reinke/` directories + `overview_merged.csv`), which must be present at the working directory as documented in `CLAUDE.md`.

**Coverage:** No coverage tooling (`gcov`/`lcov`) is configured. Given there is no unit-test suite, code coverage is not a meaningful metric for this project; the parallel concept here is CV coverage — does the 5-fold split expose the model to all classes with representative confusion matrices — which is inspected via `metrics_print()`'s confusion matrix output on every run.

---

*Testing analysis: 2026-07-27*
