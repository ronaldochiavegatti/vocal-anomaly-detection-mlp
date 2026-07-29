---
phase: 02-gap-3-shallow-vs-deep-mlp-comparison-with-structural-refacto
verified: 2026-07-29T00:49:50Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Phase 2: Gap 3 — Shallow vs Deep MLP Comparison (with structural refactor) Verification Report

**Phase Goal:** The network's hidden-layer configuration is chosen by reproducible statistical comparison across 4 candidate depths, backed by a codebase that can represent variable-depth networks safely, not by inspection or accident.
**Verified:** 2026-07-29T00:49:50Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Documentation labels current production network as Config C [128,64], not Config A [128] | ✓ VERIFIED | `CLAUDE.md:80` states production is Config C `[128,64]`, explicitly "NAO a 'Config A [128]'"; `src/main.c:361-371` `ARCH_CONFIGS[2]` is named `"C"` with `{128,64,0}` and a comment reiterating the correction |
| 2 | `mlp_init_multi()` accepts configurable hidden_sizes/dropout_rates; `MLP_MAX_LAYERS` widens fixed buffers to hold Config D (3 hidden layers) with no OOB writes | ✓ VERIFIED | `include/mlp.h:63,85-87` (`Layer layers[MLP_MAX_LAYERS]`, `mlp_init_multi()` decl); `src/mlp.c:220-236` (implementation, dynamic loop by `n_hidden+1`); `src/mlp.c:296-301` (`mlp_backward()`'s delta buffer now computed from `net->layers[i].output_size`, not `MLP_HIDDEN1_SIZE`); `src/mlp_train.c:184-186,205` (6 checkpoint/SWA arrays widened to `MLP_MAX_LAYERS`). Independently re-verified via a fresh ad hoc ASan run this session (see Behavioral Spot-Checks) — exit 0, zero AddressSanitizer errors, `num_layers=4`, `param_count=21476` (matches Config D Expert net exactly) |
| 3 | A single reusable fold+vowel training function trains all 4 configs against identical 5-fold partitions — no duplicated per-config loop | ✓ VERIFIED | `src/main.c:392` — `mode_train_ex(base_dir, smote_mode, arch, reg, result)` is the sole function containing the fold+vowel loop; `mode_train()` (`:690`), `mode_smote_ab()` (`:819`), and `mode_arch_compare()` (`:955`, looping `a=0..3`,`r=0..2`) all call this one function — grep confirms no second `for` fold-loop exists elsewhere in `main.c` |
| 4 | `results/` contains a 4-config comparison table (accuracy, Macro F1, per-class F1, param count, time/epoch) with McNemar/bootstrap CI between best config and every simpler config | ✓ VERIFIED | `results/arch_compare_comparison.csv` (13 lines: header + 12 real arm rows, all fields populated with real, non-degenerate, non-placeholder numbers) and `results/train_log_v33_gap3_arch_compare.txt` (full 12-row table with bootstrap CI columns + McNemar chi2/p vs `ao` for A/B/C) — verified by direct file read, not SUMMARY paraphrase |
| 5 | 1-SE-rule + McNemar decision rule applied, adopted config stated explicitly with justification | ✓ VERIFIED | `src/main.c:839-948` (`write_arch_compare_report()`) implements the fixed 5-step procedure in code (best-reg-per-arch → best-arch-overall → 1-SE band from bootstrap CI → McNemar gate → fewest-params tiebreak); the generated report's `DECISAO:` line (`results/train_log_v33_gap3_arch_compare.txt:38`) names Config C, baseline reg, 38918 params, with the exact numeric justification. Hand-reproduced the SE/band/gate arithmetic from the raw CSV this session and confirmed internal consistency (band=[0.4554,0.4741], C=0.4587 passes, McNemar C-vs-D p=0.7463) |
| 6 | `CLAUDE.md` updated with Gap 3 outcome regardless of adopted config | ✓ VERIFIED | `CLAUDE.md:167-204` — new "Gap 3 Outcome" subsection with full 12-row table, best-per-arch summary, 1-SE band, McNemar results, verbatim `DECISAO:` quote, explicit "No production config.h change required" statement, and the Pitfall 3 caveat (Config D/strong collapse) carried forward without softening |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `CLAUDE.md` | ARCH-01 correction + Gap 3 Outcome subsection + L2 lambda fix | ✓ VERIFIED | Line 80 (Config C correction), line 101 (`L2 lambda \| 0.001`, stale `0.003` gone), lines 167-204 (Gap 3 Outcome) |
| `include/config.h` | `MLP_MAX_LAYERS` constant | ✓ VERIFIED | Line 82: `#define MLP_MAX_LAYERS 5` |
| `include/mlp.h` | `Layer layers[MLP_MAX_LAYERS]`, `mlp_init_multi()`, `mlp_count_params()` decls | ✓ VERIFIED | Lines 63, 85-87, 94 |
| `src/mlp.c` | `mlp_init_multi()` impl, thin `mlp_init_dynamic()` wrapper, dynamic `mlp_backward()` sizing, `mlp_count_params()` | ✓ VERIFIED | Lines 220-248, 296-301, 613-621 |
| `include/mlp_train.h` + `src/mlp_train.c` | `l2_lambda` runtime param, `MLP_MAX_LAYERS`-sized buffers | ✓ VERIFIED | `mlp_train.h:52`, `mlp_train.c:164,184-186,205,256` |
| `src/main.c` | `ArchConfig`/`RegSetting`/`ARCH_CONFIGS`/`REG_MULTIPLIER`, widened `mode_train_ex()`, `mode_arch_compare()`, `write_arch_compare_report()`, CLI dispatch | ✓ VERIFIED | Lines 355-392, 690, 839-1021 (`"arch-compare"` dispatch at 1021) |
| `results/arch_compare_comparison.csv` | 13-line (1 header + 12 rows) machine-readable comparison | ✓ VERIFIED | Confirmed 13 lines, real non-degenerate values (accuracy 0.61-0.70, macro_f1 0.28-0.47 range, param counts distinct per arch) |
| `results/train_log_v33_gap3_arch_compare.txt` | Full report with `DECISAO:` | ✓ VERIFIED | Exactly 1 `DECISAO:` line, internally consistent with CSV |
| 48 per-arm diagnostic files (`metrics_global`/`bootstrap_ci`/`mcnemar_vs_baselines`/`smote_borderline_counts` × 4 arch × 3 reg) | Non-empty | ✓ VERIFIED | Loop check over all 48 expected filenames — zero `MISSING:` lines emitted |
| `results/metrics_global.csv`, `bootstrap_ci.csv`, `mcnemar_vs_baselines.csv` (plain CLI regression) | Byte-identical to pre-Phase-2 committed baseline | ✓ VERIFIED | `git log` shows last content-changing commit for these 3 files predates Phase 1/2 (`fdb51ce`); Plan 02-03's regression-check commit (`c2255f4`) touched only the console-log file, confirming zero diff to the three regenerated files; current working tree shows no uncommitted diff either |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `mlp_init_multi()` | `Layer layers[MLP_MAX_LAYERS]` | loop `i<net->num_layers` calling `layer_init` | ✓ WIRED | `src/mlp.c:232-235` |
| `mode_arch_compare()` | `mode_train_ex(..., &ARCH_CONFIGS[a], (RegSetting)r, &results[a][r])` | 12 sequential calls | ✓ WIRED | `src/main.c:955-1000` range (confirmed via grep, 12-iteration loop `a=0..3`,`r=0..2`) |
| `write_arch_compare_report()` | `results/train_log_v33_gap3_arch_compare.txt` | 5-step decision procedure → `DECISAO:` | ✓ WIRED | Report file exists, contains 1 `DECISAO:` line, numbers hand-verified consistent |
| `mode_train()` / `mode_smote_ab()` | `&ARCH_CONFIGS[2]`, `REG_BASELINE` | preserves production behavior | ✓ WIRED | `src/main.c:690` (`mode_train`), `mode_smote_ab()`'s 2 call sites also pass `&ARCH_CONFIGS[2], REG_BASELINE` |
| `CLAUDE.md Gap 3 Outcome` | `results/train_log_v33_gap3_arch_compare.txt` | file-path reference + verbatim quote | ✓ WIRED | `CLAUDE.md:196-198,204` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| `results/arch_compare_comparison.csv` | 12 arms' accuracy/macro_f1/param counts | Real ~5h30min execution of `./build/vocal_detect arch-compare .` (per 02-03-SUMMARY, cross-verified: byte-identical regression-check + C/baseline row matches Phase 1's independently-obtained v32 Borderline-SMOTE macro_f1 of 0.4587 exactly) | Yes | ✓ FLOWING |
| `CLAUDE.md` Gap 3 table | Same 12-arm numbers | Copied verbatim from `results/train_log_v33_gap3_arch_compare.txt` | Yes (cross-checked byte-for-byte against source file this session) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `make` compiles cleanly (0 errors) | `make clean && make` | 0 errors, 3 pre-existing warnings (all predate this phase, documented in 02-01-SUMMARY) | ✓ PASS |
| Config D's array-sizing safety (independent re-verification, not trusting 02-01-SUMMARY's ASan claim) | Ad hoc `-fsanitize=address` build calling `mlp_init_multi(nf=85, nc=4, hidden=[128,64,32], dropout=[0.70,0.56,0.42])` + `mlp_train(..., l2=0.0014, ...)` + `train_history_free` + `mlp_free` | Exit 0, zero AddressSanitizer errors, `num_layers=4`, `param_count=21476` (matches SUMMARY's claimed value for Config D's Expert net exactly) | ✓ PASS |
| 48 per-arm diagnostic files present and non-empty | Loop `test -s` over all 48 expected filenames | Zero `MISSING:` output | ✓ PASS |
| Plain CLI regression: `results/metrics_global.csv` etc. unchanged from pre-Phase-2 baseline | `git log`/`git status --porcelain` on the 3 files | No content-changing commits since `fdb51ce` (pre-Phase-1); zero uncommitted diff | ✓ PASS |
| `DECISAO:` internal consistency | Hand-recomputed SE/band/McNemar-gate/param-tiebreak from raw CSV numbers | Band=[0.4554,0.4741] matches report; C(0.4587) passes band+McNemar(p=0.7463); adopted=C (38918 params) < D (42886) among passing candidates | ✓ PASS |

*Note:* Running `make clean` during this verification transiently deleted the committed `results/*.csv` files (per the Makefile's documented `make clean` behavior). This was immediately caught and restored via `git checkout -- results/` before concluding verification; `git status --porcelain` confirms the working tree matches its pre-verification state (only the 7 pre-existing untracked items remain).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ARCH-01 | 02-01 | Config label correction before any comparison exists | ✓ SATISFIED | `CLAUDE.md:80`, `src/main.c:361-371` |
| ARCH-02 | 02-01 | `mlp_init_multi()`, `MLP_MAX_LAYERS` widening incl. mlp_train.c buffers | ✓ SATISFIED | `include/mlp.h`, `src/mlp.c`, `src/mlp_train.c` (see Artifacts table) |
| ARCH-03 | 02-02 | Single reusable fold+vowel training function for all 4 configs | ✓ SATISFIED | `src/main.c:392` `mode_train_ex()` |
| ARCH-04 | 02-02, 02-03 | Comparison table with param count/time-per-epoch + McNemar/bootstrap CI | ✓ SATISFIED | `results/arch_compare_comparison.csv`, `results/train_log_v33_gap3_arch_compare.txt` |
| ARCH-05 | 02-02, 02-03 | 1-SE + McNemar decision rule, never visual inspection | ✓ SATISFIED | `write_arch_compare_report()`, `DECISAO:` sentence, hand-verified |
| ARCH-06 | 02-04 | `CLAUDE.md` updated regardless of outcome | ✓ SATISFIED | `CLAUDE.md:167-204` |

No orphaned requirements: all 6 ARCH-01..06 IDs are declared across the 4 plans' frontmatter and match `.planning/REQUIREMENTS.md`'s Phase 2 mapping exactly (lines 24-29, 82-87).

### Anti-Patterns Found

None. Grep for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER|not yet implemented|coming soon` across all 7 phase-modified files (`include/config.h`, `include/mlp.h`, `src/mlp.c`, `include/mlp_train.h`, `src/mlp_train.c`, `src/main.c`, `CLAUDE.md`) returned zero matches (the only hit was `CLAUDE.md`'s own pre-existing prose describing the project's "no TODO markers" convention, not a marker itself).

Pre-existing compiler warnings (3, all predating this phase and documented in 02-01-SUMMARY.md): `mode_validate_external` unused parameter, `fgets` ignoring return value in `features_load_csv`, `best_val_acc` unused variable in `mlp_train.c`. None introduced by this phase's changes; none block the goal.

### Human Verification Required

None. All must-haves are mechanically/directly verifiable from source code, compiled binary behavior, and generated result files — no visual/UX/external-service verification applies to this phase's scope (pure statistical/structural C refactor + documentation).

### Gaps Summary

No gaps. All 6 ROADMAP success criteria are independently verified against the actual codebase (not just SUMMARY claims):

1. Documentation correction (Config C vs Config A) — present in both `CLAUDE.md` and the `ARCH_CONFIGS` naming in code, before any comparison result existed (git commit order confirms `9d5611c` docs precedes `d8211e7`/`fe1cd3b` feat commits).
2. `mlp_init_multi()` + `MLP_MAX_LAYERS` widening — present, and independently re-verified safe under a fresh ASan run in this verification session (not merely trusting the SUMMARY's prior ASan claim).
3. Single reusable `mode_train_ex()` — confirmed as the sole fold+vowel loop, reused by all 3 call paths (`train`, `smote-ab`, `arch-compare`).
4. Real 4-config (12-arm) comparison table in `results/` with McNemar/bootstrap CI — confirmed via direct file read of both the CSV and the txt report, cross-checked against 02-03-SUMMARY's claimed numbers (exact match).
5. 1-SE + McNemar decision rule — confirmed in code (`write_arch_compare_report()`) and hand-reproduced from raw numbers independently in this session.
6. `CLAUDE.md` Gap 3 Outcome — present, accurate, includes the "no production change required" statement and the Pitfall 3 caveat without omission.

The one operational deviation during this verification (`make clean` transiently deleting `results/*.csv`) was self-inflicted by the verifier's own build-check command, immediately detected via `git status`, and fully restored via `git checkout -- results/` before this report was finalized — it does not reflect any defect in the phase's deliverables.

---

*Verified: 2026-07-29T00:49:50Z*
*Verifier: Claude (gsd-verifier)*
