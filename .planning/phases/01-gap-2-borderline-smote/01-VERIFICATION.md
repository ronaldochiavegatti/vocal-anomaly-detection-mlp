---
phase: 01-gap-2-borderline-smote
verified: 2026-07-28T01:39:25Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 1: Gap 2 — Borderline-SMOTE Verification Report

**Phase Goal:** Borderline-SMOTE (Han, Wang & Mao, 2005) is implemented as a selectable oversampling mode alongside standard SMOTE, and its adoption or rejection is decided by reproducible A/B comparison rather than assumption.
**Verified:** 2026-07-28T01:39:25Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `smote_oversample()` accepts a `SmoteMode` param; borderline mode classifies samples via `find_knn_global()` with untruncated `2*m >= k`, `m==k` checked first | ✓ VERIFIED | `src/main.c:208` enum; `:234-250` `find_knn_global()` (RNG-free, confirmed no `rng_` calls in body); `:257-262` `classify_borderline()` — `if (m == k) return 2;` precedes `if (2 * m >= k) return 1;` (line 259 before 260) |
| 2 | Interpolation always uses `find_knn()` restricted to same-class neighbors; never reuses the global k-NN list | ✓ VERIFIED | `src/main.c:325` — `find_knn(x_in, base_idx, class_idx[c], n_class, nf, knn, neighbors)` unchanged/textually distinct from `find_knn_global()`; only `base_idx` is drawn from the mode-dependent `pool` (line 321-322, 324), the interpolation candidate set (`class_idx[c]`/`n_class`) never varies |
| 3 | Empty borderline pool and `n_class<=1` both log an explicit warning, no crash, no degenerate synthetic sample; `n_class<=1` SKIPS synthesis entirely (not log-and-duplicate) | ✓ VERIFIED | `src/main.c:286-291` sets `n_synthetic = 0` when `n_class <= 1` (skip, not duplicate) with `log_warn(...)` guarded by `n_synthetic > 0`; `:313-315` logs `"sem amostras borderline"` when `n_borderline == 0`; `*n_out = out_idx;` present at `:334` to correct row-count for skipped classes; no "log-and-duplicate" text anywhere in file (`grep -c "sera duplicata"` = 0) |
| 4 | `results/` contains a real A/B report: Macro F1 + per-class F1 table, McNemar/bootstrap CI, safe/borderline/noise count table, explicit adopt/reject sentence | ✓ VERIFIED | `results/train_log_v32_gap2_smote_ab.txt` contains full comparison table + `McNemar direto ... chi2=0.1928 p=0.6606` + `DECISAO: Borderline-SMOTE ADOTADO ...`; `results/smote_ab_comparison.csv` (8 lines, 7 metrics × 2 arms + CI); `results/smote_borderline_counts.csv` (91 lines: header + 90 rows, `fold,vowel,network,class,safe,borderline,noise`) |
| 5 | `CLAUDE.md` updated (Optimization History / What Worked / What Didn't Work) with Gap 2 outcome, regardless of adopted mode | ✓ VERIFIED | `CLAUDE.md:134` "What works" bullet replaced with citation-precise, evidence-backed claim (`Han/Wang/Mao 2005`, file pointer, delta, McNemar caveat); `CLAUDE.md:142-164` new `## Gap 2 Outcome — Borderline-SMOTE A/B (v32)` subsection with verbatim DECISAO sentence, comparison table, significance caveat, fallback frequency, file pointers |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/main.c` — `SmoteMode`/`SmoteBorderlineCounts`/`find_knn_global()`/`classify_borderline()` | Core Borderline-SMOTE1 algorithm | ✓ VERIFIED | All present, correct branch order, RNG-free, Portuguese comments only (`grep -c "/\* NOISE \*/\|/\* SAFE \*/\|(danger)"` = 0) |
| `src/main.c` — `smote_oversample()` extended signature | `SmoteMode smote_mode, SmoteBorderlineCounts *bcounts` params, `bcounts` never collides with pre-existing `counts` | ✓ VERIFIED | Signature at `:264-266`; `counts` (pre-existing per-class array) and `bcounts` (new param) coexist without collision; `*n_out = out_idx;` present at `:334` |
| `src/main.c` — `mode_train_ex()`/`mode_train()`/`mode_smote_ab()`/`write_smote_ab_report()` | Orchestration layer, `smote-ab` CLI mode | ✓ VERIFIED | `mode_train_ex()` at `:360`; thin wrapper `mode_train()` at `:627`; `write_smote_ab_report()` at `:664`; `mode_smote_ab()` at `:756`; dispatch line `:786` |
| `results/smote_borderline_counts.csv` | Real per-class/fold/vowel/network safe/borderline/noise counts | ✓ VERIFIED | 90 data rows, schema exact match, verified real data (494/51/4 etc., not all-zero placeholders except the 2 structurally-unused class slots) |
| `results/train_log_v32_gap2_smote_ab.txt`, `results/smote_ab_comparison.csv` | Full A/B report + machine-readable comparison | ✓ VERIFIED | Both present, content cross-checked against 01-03-SUMMARY.md's recorded verdict — numbers match exactly (macro_f1 standard=0.4338/0.433753, borderline=0.4565/0.456541, McNemar chi2=0.1928 p=0.6606) |
| `CLAUDE.md` Gap 2 Outcome section | Dated outcome documentation | ✓ VERIFIED | Present, accurate, matches results/ evidence verbatim |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `smote_oversample()` classification block | `find_knn_global()` | direct call over all-class candidate set | ✓ WIRED | `src/main.c:303` `find_knn_global(x_in, class_idx[c][i], n_in, nf, BORDERLINE_M_NEIGHBORS, neighbors_g)` |
| `smote_oversample()` synthesis loop | `find_knn()` (same-class only) | interpolation restricted to `class_idx[c]`/`n_class` | ✓ WIRED | `src/main.c:325` — textually unchanged from pre-phase, never references `class_idx_borderline` |
| `mode_smote_ab()` | `mode_train_ex(base_dir, SMOTE_STANDARD/BORDERLINE, &res_*)` | two sequential calls, shared RNG reseed via `kfold_split()` | ✓ WIRED | `src/main.c:761-762` |
| `write_smote_ab_report()` | `results/train_log_v32_gap2_smote_ab.txt` + `results/smote_ab_comparison.csv` | fopen/fprintf, code-computed decision | ✓ WIRED | Verified file contents match code-generated format exactly; decision rule (`bl_res->macro_f1 >= std_res->macro_f1`) at `src/main.c:731-737` confirmed to match the actual `ADOTADO` output (0.4587 >= 0.4351) |
| CLI `train`/`full` (production path) | `mode_train_ex(base_dir, SMOTE_STANDARD, NULL)` | thin wrapper, unsuffixed filenames | ✓ WIRED | `src/main.c:627`; regenerated `results/metrics_global.csv` confirmed byte-identical to Phase 0 baseline (macro_f1=0.451440, accuracy=0.697632) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| `results/smote_borderline_counts.csv` | `bcounts->safe/borderline/noise` | `classify_borderline()` output inside real fold/vowel/network loop of an actual `smote-ab` run | Yes — verified real, non-degenerate counts (e.g., row 1: `494,51,4`), confirmed only the 2 structurally-unused class slots (Master class=1, Expert class=0) are all-zero by construction | ✓ FLOWING |
| `results/train_log_v32_gap2_smote_ab.txt` DECISAO sentence | `ABResult.macro_f1` for both arms | `mode_train_ex()` → `metrics_compute()`/`metrics_bootstrap_ci()` on real out-of-fold predictions from an actual training run | Yes — cross-checked against `results/smote_ab_comparison.csv` and `01-03-SUMMARY.md`; numbers agree across all three independent sources | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Clean compile, zero errors | `make clean && make` (full rebuild) | 0 errors, 3 warnings (2 documented pre-existing in `src/main.c` + 1 pre-existing in `src/mlp_train.c`, `best_val_acc` unused, confirmed present in Phase 0's own final commit `96fa69c`, not a phase 1 regression — invisible during phase 1's own incremental-build acceptance checks since `mlp_train.c` was never recompiled during this phase's edits) | ✓ PASS |
| `m == k` branch precedes `2 * m >= k` branch (untruncated boundary, no `m >= k/2` anywhere) | `grep -n "m == k"` / `grep -n "2 \* m >= k"` line order | Line 259 before 260; no `m >= k / 2` pattern found in file | ✓ PASS |
| `find_knn_global()` is RNG-free | `awk` over function body, `grep -c "rng_"` | 0 | ✓ PASS |
| No English classifier comments (`/* NOISE */`, `/* SAFE */`, `(danger)`) | grep | 0 matches | ✓ PASS |
| Production `train`/`full` CLI path unaffected by refactor | Regenerated `results/metrics_global.csv` compared to Phase 0 baseline | Byte-identical (macro_f1=0.451440, accuracy=0.697632, matches `01-03-SUMMARY.md`'s claim exactly) | ✓ PASS |
| `smote_borderline_counts.csv` real-data sanity check (fallback frequency) | `awk` count of `borderline==0` rows, cross-checked against safe/noise columns | 30/90 rows have `borderline==0`; all 30 are exactly the structurally-unused class slots (`(0,0,0)`); 0/60 real classification rows hit the fallback — matches SUMMARY's claim exactly | ✓ PASS |
| All commits referenced in SUMMARYs exist | `git cat-file -e <hash>` for 40f3efe, 6b371a8, a3aef7b, 60d5a10, 09d95f0, 830fbcd, e00943b | All present | ✓ PASS |

### Probe Execution

No dedicated probe scripts (`scripts/*/tests/probe-*.sh`) exist in this repository/phase — this project has no `scripts/` test-probe convention. Not applicable; behavioral spot-checks (above) substitute for this phase's compiled/runtime verification, which is the project's own established pattern (confirmed against `make`/CLI invocation, consistent with Phase 0's precedent).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|--------------|-------------|--------------|--------|----------|
| SMOTE-01 | 01-01, 01-02 | `SmoteMode` param + borderline classification via `find_knn_global()`, untruncated `2*m>=k` | ✓ SATISFIED | `src/main.c:208,234-262` |
| SMOTE-02 | 01-01 | Interpolation always same-class via `find_knn()`, never reuses global k-NN list | ✓ SATISFIED | `src/main.c:325` unchanged call site |
| SMOTE-03 | 01-01 | Empty pool + `n_class<=1` fallback: explicit warning, no crash, no degenerate sample (skip, not duplicate) | ✓ SATISFIED | `src/main.c:286-291, 313-315` |
| SMOTE-04 | 01-02, 01-03 | A/B report: Macro F1 + per-class F1, McNemar/bootstrap CI, safe/borderline/noise table, adopt/reject decision | ✓ SATISFIED | `results/train_log_v32_gap2_smote_ab.txt`, `results/smote_ab_comparison.csv`, `results/smote_borderline_counts.csv` (real run, not just code) |
| SMOTE-05 | 01-04 | `CLAUDE.md` updated regardless of outcome | ✓ SATISFIED | `CLAUDE.md:134, 142-164` |

All 5 requirement IDs (SMOTE-01 through SMOTE-05) declared across the 4 plans' frontmatter are accounted for in `.planning/REQUIREMENTS.md` (lines 16-20, all marked `[x]` complete, all mapped to Phase 1 in the traceability table lines 77-81). No orphaned requirements found.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `CLAUDE.md` | 66 (Data Flow diagram, pre-existing) | `smote_oversample(): Borderline-SMOTE (k=5) to balance minority classes` — implies Borderline-SMOTE is the production default; the actual production `train`/`full` CLI path still defaults to `SMOTE_STANDARD` (`mode_train()` thin wrapper at `src/main.c:627` passes `SMOTE_STANDARD, NULL`) | ℹ️ Info | Pre-existing documentation drift, present since before Phase 1 (confirmed via `git log -p -- CLAUDE.md`, unchanged since commit `2e5af63`, predating this phase). Plan 01-04 explicitly scoped its edit to "Important Constraints" + new subsection only, and this line lives in the Architecture/Data-Flow section which was out of scope. Not a phase-1-introduced regression, and none of ROADMAP.md's 5 success criteria nor REQUIREMENTS.md's SMOTE-01..05 require flipping the production default to Borderline-SMOTE — the roadmap goal is "selectable... decided by reproducible A/B comparison," not "swap the default." Flagged for awareness only, not scored as a gap. |
| `src/mlp_train.c` | 170 | unused variable `best_val_acc` | ℹ️ Info | Pre-existing (present in Phase 0's own final commit `96fa69c`, file untouched by Phase 1) — not introduced by this phase. |

No `TODO`/`FIXME`/`HACK`/`TBD`/`XXX`/placeholder markers found in any file touched by this phase (`src/main.c`, `CLAUDE.md`).

### Human Verification Required

None. All must-haves are verifiable by direct code/data inspection (grep, file content, arithmetic cross-checks), consistent with this phase's own scope (algorithmic correctness by construction, CLI orchestration, and a real executed pipeline run producing durable evidence — no UI/UX/visual/real-time component exists in this offline C research pipeline).

### Gaps Summary

No blocking gaps found. All 5 roadmap success criteria and all 5 requirement IDs (SMOTE-01 through SMOTE-05) are verified against the actual codebase, not just SUMMARY.md claims:

- The `bcounts`/`counts` naming collision that was caught during plan-checking was confirmed correctly resolved in the executed code (`counts` at `src/main.c:268` unchanged, `bcounts` as the new parameter name, no redeclaration conflict).
- `*n_out = out_idx;` is present and correctly placed before the function's final frees (`src/main.c:334`), confirming the skip-synthesis row-count correction actually landed.
- The `n_class<=1` fallback was independently confirmed to be a true SKIP (via `n_synthetic = 0`), not the earlier "log-and-duplicate" behavior — no such text exists anywhere in `src/main.c`.
- The A/B report in `results/` was independently re-derived from raw CSV numbers (not merely trusted from SUMMARY prose) and found internally consistent: the `ADOTADO` decision matches the sign of `(borderline_macro_f1 - standard_macro_f1)` under both point-estimate and bootstrap-mean readings.
- The plain `train`/`full` CLI regression path was confirmed non-regressed — `results/metrics_global.csv` (currently on disk) is byte-identical to the documented Phase 0 baseline.
- One minor, pre-existing (not phase-1-introduced) documentation inconsistency was noted in `CLAUDE.md`'s Data Flow section (Info-level, not scored against this phase's success criteria).

**Verification note:** During this verification, `make clean` was run to perform a full clean rebuild sanity check; this transiently deleted 11 tracked `results/*.csv` files (`bootstrap_ci.csv`, `bootstrap_ci_{standard,borderline}.csv`, `mcnemar_vs_baselines.csv`, `mcnemar_vs_baselines_{standard,borderline}.csv`, `metrics_global.csv`, `metrics_global_{standard,borderline}.csv`, `smote_ab_comparison.csv`, `smote_borderline_counts.csv`). These were immediately restored via `git checkout -- results/...` before any further inspection; `git status` confirms no diff remains against the committed state. This is noted here for transparency, not as a phase defect.

---

*Verified: 2026-07-28T01:39:25Z*
*Verifier: Claude (gsd-verifier)*
