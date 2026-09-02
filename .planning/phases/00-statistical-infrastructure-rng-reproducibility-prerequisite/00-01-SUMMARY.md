---
phase: 00-statistical-infrastructure-rng-reproducibility-prerequisite
plan: 01
subsystem: infra
tags: [c99, openmp, rng, reproducibility, cli]

# Dependency graph
requires: []
provides:
  - "precalculate_augmentations() runs single-threaded and is deterministic under RANDOM_SEED=42 (no more OpenMP RNG race)"
  - "verify-rng CLI mode: a fast (~20 min single-threaded, vs 60-90 min full make full) automated determinism check"
  - "results/rng_reproducibility_check.txt: durable, empirical proof (byte-identical double-run cmp) of RNG determinism"
affects: [01-gap2-borderline-smote, 02-gap3-shallow-vs-deep, 03-gap1-paraconsistent-selection]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fast verification scaffold pattern: isolate the racy/expensive component into its own CLI mode with a deterministic binary dump, instead of requiring a full 60-90 min pipeline run to check reproducibility"

key-files:
  created:
    - results/rng_reproducibility_check.txt
  modified:
    - src/main.c
    - .gitignore

key-decisions:
  - "Fixed the OpenMP RNG race by removing #pragma omp parallel for (INFRA-02's explicit sanctioned fallback) rather than adding per-thread RNG streams, since precalculate_augmentations() is not the pipeline's wall-clock bottleneck (feature extraction and MLP training dominate)"
  - "results/aug_cache_verify.bin (8.8MB generated verification cache) added to .gitignore, consistent with the existing results/features.csv pattern -- the durable evidence is the small text file, not the regeneratable binary"

patterns-established:
  - "Any future #pragma omp parallel for added around a loop must first be grepped for rng_* calls (documented inline in src/main.c and enforced by convention, not tooling, per T-00-02 in the plan's threat model)"

requirements-completed: [INFRA-02]

# Metrics
duration: 70min
completed: 2026-07-27
---

# Phase 0 Plan 1: RNG Race Fix + verify-rng Determinism Check Summary

**Removed the OpenMP data race on the global RNG in `precalculate_augmentations()` and added a `verify-rng` CLI mode that empirically proved byte-identical, deterministic output across two independent runs with `RANDOM_SEED=42`.**

## Performance

- **Duration:** 70 min (dominated by two ~20-minute single-threaded `verify-rng` runs — the augmentation loop is now sequential per the fix)
- **Started:** 2026-07-27T15:08:54-03:00
- **Completed:** 2026-07-27T16:18:38-03:00
- **Tasks:** 2
- **Files modified:** 3 (`src/main.c`, `.gitignore`, `results/rng_reproducibility_check.txt` created)

## Accomplishments
- Removed the confirmed OpenMP RNG race in `precalculate_augmentations()` (CONCERNS.md "Known Bugs"; PITFALLS.md Pitfall 13) by deleting the `#pragma omp parallel for schedule(dynamic, 1)` directive — the loop now consumes the global `rng_state` (src/utils.c) in deterministic, sequential patient-index order
- Added `mode_verify_rng()` and a `verify-rng` CLI dispatch case, giving the project a fast (~20 min single-threaded, vs. 60-90 min for a full `make full`), repeatable, automated way to verify RNG determinism at any point in the future
- Ran `./build/vocal_detect verify-rng .` twice consecutively and confirmed via `cmp` that the two runs produced byte-identical `results/aug_cache_verify.bin` (8,819,136 bytes; 1098 patients x 8 augmentations x 251 features x 4 bytes/float) — `cmp exit code: 0`
- Confirmed the only other OpenMP-parallel loop in the codebase (`features_extract_all()` in `src/feature_extract.c`) has zero `rng_*` calls, so it does not carry the same race
- Documented the full command sequence, verdict, and git commit hash in `results/rng_reproducibility_check.txt` as durable evidence for the PIBIC committee and future contributors

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove the OpenMP RNG race and add `verify-rng` mode** - `47f6070` (fix)
2. **Task 2: Empirically prove RNG determinism with a double-run byte comparison** - `3ed5dac` (test)

**Plan metadata:** (this commit) `docs(00-01): complete RNG reproducibility plan`

## Files Created/Modified
- `src/main.c` - Removed `#pragma omp parallel for` from `precalculate_augmentations()` (with an inline warning comment against reintroducing it); added `mode_verify_rng()` and the `verify-rng` CLI dispatch case
- `.gitignore` - Added `results/aug_cache_verify.bin` (regeneratable verification cache, not durable evidence)
- `results/rng_reproducibility_check.txt` - New: commands run, `cmp` verdict (`cmp exit code: 0`), git commit hash, date, and the supplementary `grep` check on `feature_extract.c`

## Decisions Made
- Fixed the race by removing OpenMP parallelism (INFRA-02's explicit fallback option) rather than introducing per-thread RNG state, because `precalculate_augmentations()` is not the pipeline's wall-clock bottleneck — trading its parallelism for correctness costs ~20 min per full run but is not on the hot path for iterative A/B experimentation (which uses cached `results/features.csv` and does not re-run augmentation precomputation unless folds/seed change)
- Kept `results/aug_cache_verify.bin` out of git (added to `.gitignore`) since it is an 8.8MB regeneratable artifact, not documentation — matching the existing `results/features.csv` cache-exclusion pattern already established in this repo

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Restored `results/metrics_global.csv` after `make clean` deleted it**
- **Found during:** Task 1 verification (`make clean && make`)
- **Issue:** The Makefile's `clean` target runs `rm -f results/*.csv`, which deleted the git-tracked `results/metrics_global.csv` (the documented v29 baseline metrics file) as a side effect of running the plan's own mandated verification command
- **Fix:** `git checkout -- results/metrics_global.csv` to restore the tracked file before committing Task 1, so the baseline reference file was not lost
- **Files affected:** `results/metrics_global.csv` (restored, not modified — no functional change)
- **Verification:** `git status --short` confirmed the file was back to its committed state with no diff
- **Committed in:** N/A (restored before staging; never part of a commit as a deletion)

**2. [Rule 2 - Missing critical] Added `results/aug_cache_verify.bin` to `.gitignore`**
- **Found during:** Task 2, after generating the verification artifact
- **Issue:** The plan's artifact list names `results/aug_cache_verify.bin` as a produced file but does not specify whether it should be committed; leaving an 8.8MB untracked binary in the working tree indefinitely (or accidentally committing it) would bloat the repo without adding documentation value
- **Fix:** Added the file to `.gitignore`, consistent with the existing `results/features.csv` pattern (large regeneratable pipeline caches are excluded; the small, durable `results/rng_reproducibility_check.txt` carries the actual evidence)
- **Files modified:** `.gitignore`
- **Verification:** `git status --short` shows the binary as ignored, not untracked
- **Committed in:** `3ed5dac` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug/regression prevention, 1 missing housekeeping convention)
**Impact on plan:** Both fixes are necessary for repo hygiene and preventing accidental loss of the documented baseline; no scope creep into Task 1/2's actual functional work.

## Issues Encountered
- The background `verify-rng` process was interrupted twice across conversation-turn boundaries during initial attempts (the detached child process did not survive being backgrounded without an explicit `setsid`/`nohup`/`disown` and without actively polling to completion within the same turn). Resolved by re-running with `setsid nohup ... &disown`, redirecting output directly to a log file (avoiding `tail`'s buffer-until-EOF behavior masking progress), and polling to completion in-turn rather than ending the turn while the process was still running. No impact on the correctness of the final result — both completed runs produced byte-identical output.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- INFRA-02 is satisfied: the RNG race is fixed (not merely documented), and a repeatable ~20-minute `verify-rng` check exists for any future re-verification need
- Every downstream gap (Gap 2 in Phase 1, Gap 3 in Phase 2, Gap 1 in Phase 3) can now rely on "same seed, same folds" being an empirically verified guarantee for the augmentation-precomputation stage of the pipeline
- No blockers identified for the next plan in Phase 0 (baseline reconfirmation, per STATE.md's open blocker on `results/train_log_v29_baseline_reconfirmed.txt` not existing yet)

---
*Phase: 00-statistical-infrastructure-rng-reproducibility-prerequisite*
*Completed: 2026-07-27*

## Self-Check: PASSED

- FOUND: src/main.c
- FOUND: results/rng_reproducibility_check.txt
- FOUND: .planning/phases/00-statistical-infrastructure-rng-reproducibility-prerequisite/00-01-SUMMARY.md
- FOUND: commit 47f6070
- FOUND: commit 3ed5dac
