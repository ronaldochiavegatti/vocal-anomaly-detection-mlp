---
phase: 02-gap-3-shallow-vs-deep-mlp-comparison-with-structural-refacto
plan: 04
subsystem: docs
tags: [documentation, claude-md, arch-compare, mcnemar, gap-3-evidence]

# Dependency graph
requires:
  - phase: 02-03
    provides: "Real, freshly-executed 12-arm arch-compare verdict (results/train_log_v33_gap3_arch_compare.txt, results/arch_compare_comparison.csv) and the mechanically re-derived adoption decision (Config C at baseline regularization, 38918 params)"
provides:
  - "CLAUDE.md's Key Hyperparameters table corrected: L2 lambda 0.003 -> 0.001 (matches real include/config.h constant)"
  - "New '## Gap 3 Outcome — Shallow vs Deep MLP Comparison (v33)' subsection in CLAUDE.md documenting the 12-arm comparison, the 1-SE + McNemar decision procedure, the verbatim DECISAO sentence, and the Pitfall 3 mean_epochs_to_stop caveat"
  - "Satisfies ARCH-06 (documentation update 'independente do resultado')"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Documentation-only plan mirroring Phase 1's 01-04 precedent exactly: correct a stale hyperparameter value + add a new dated 'Gap N Outcome' subsection sourced directly from the prior plan's SUMMARY.md verdict and the raw train_log txt file, never asserted independently of that evidence chain"

key-files:
  created: []
  modified:
    - CLAUDE.md

key-decisions:
  - "No production config.h change required: the adopted arm (Config C, baseline regularization) is today's exact compiled production configuration, confirmed and documented explicitly in the new Gap 3 Outcome subsection -- Gap 3's outcome is a validated no-op for production code, not a migration task."
  - "Pitfall 3 caveat (Config D at strong regularization collapsing, mean_epochs_to_stop=46.4 vs its own light=76.8/baseline=87.3, macro_f1 crashing to 0.2777 -- statistically indistinguishable from MajorityClass) carried forward verbatim into CLAUDE.md per the plan's explicit instruction not to omit it even though the adopted architecture (C) is not the deepest one."

requirements-completed: [ARCH-06]

# Metrics
duration: 8min
completed: 2026-07-29
---

# Phase 2 Plan 4: CLAUDE.md Gap 3 Outcome Documentation Summary

**Corrected the stale L2 lambda documentation (0.003 -> real config.h value 0.001) and added a new dated "Gap 3 Outcome" subsection to CLAUDE.md with the full 12-arm shallow-vs-deep MLP comparison, the 1-SE + McNemar adoption procedure, the verbatim DECISAO sentence (adopted = Config C at baseline regularization, no production change required), and the Pitfall 3 regularization-collapse caveat for Config D.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-07-29 (session start)
- **Completed:** 2026-07-29
- **Tasks:** 1 completed
- **Files modified:** 1 (CLAUDE.md)

## Accomplishments

- Corrected `CLAUDE.md`'s Key Hyperparameters table: `L2 lambda | 0.003` → `L2 lambda | 0.001`, matching the real `include/config.h` constant this phase's entire regularization sweep (light=0.6x/baseline=1.0x/strong=1.4x) was built on.
- Added `## Gap 3 Outcome — Shallow vs Deep MLP Comparison (v33)` immediately after the existing `## Gap 2 Outcome` subsection, containing:
  1. The ARCH-01 correction restated for permanence (Config C `[128,64]` confirmed production, not Config A).
  2. The condensed 12-row comparison table (architecture × regularization, accuracy, macro F1, param counts, mean_epochs_to_stop).
  3. The best-per-architecture summary (A→light, B→light, C→baseline, D→light).
  4. The 1-SE band `[0.4554, 0.4741]` and best-overall (`ao` = D/light, macro_f1=0.4741).
  5. McNemar p-values of A, B, C vs `ao` (D) — none significant, with the band-gate-excludes-A/B-first logic stated explicitly.
  6. The verbatim `DECISAO:` sentence copied directly from `results/train_log_v33_gap3_arch_compare.txt` (adopted = C, baseline, 38918 params).
  7. Explicit statement that no production `config.h` change is required, since the adopted arm is today's exact compiled defaults.
  8. The Pitfall 3 caveat (Config D/strong collapse: `mean_epochs_to_stop`=46.40, macro_f1=0.2777, statistically indistinguishable from MajorityClass) carried forward verbatim, framed as an open question ("depth doesn't help" vs. "depth needs different tuning") rather than a settled negative result.
  9. File pointers to `results/train_log_v33_gap3_arch_compare.txt` and `results/arch_compare_comparison.csv`.
- Verified `git diff --stat CLAUDE.md` shows exactly 1 file changed, with edits scoped precisely to the intended hyperparameter row and new subsection — no other section touched.

## Task Commits

1. **Task 1: Update CLAUDE.md with the Gap 3 (shallow vs deep MLP) A/B outcome and the L2 lambda correction** - `683f7d7` (docs)

**Plan metadata:** (this commit, docs: complete plan)

## Files Created/Modified

- `CLAUDE.md` - Corrected `L2 lambda | 0.003` → `L2 lambda | 0.001` in Key Hyperparameters table; added new `## Gap 3 Outcome — Shallow vs Deep MLP Comparison (v33)` subsection (40 lines) after `## Gap 2 Outcome`, before `## Output Files`

## Decisions Made

- Sourced every number in the new subsection directly from `02-03-SUMMARY.md`'s recorded verdict and cross-checked the verbatim `DECISAO:` sentence directly against `results/train_log_v33_gap3_arch_compare.txt` (line 38) rather than trusting the SUMMARY's paraphrase alone — confirmed byte-identical.
- Explicitly stated that no production `config.h` migration is needed as a result of Gap 3 (adopted = today's exact compiled defaults), consistent with the plan's instruction to distinguish "comparison + decision documented" (in scope) from "migrating production defaults" (explicitly out of scope for this phase).
- Carried the Pitfall 3 caveat forward without softening or omitting it, per the plan's explicit instruction, even though it does not affect the adoption decision.

## Deviations from Plan

None — plan executed exactly as written. All acceptance criteria were met on the first edit; no Rule 1-4 deviations were needed.

### Auto-fixed Issues

None required.

## Issues Encountered

None.

## User Setup Required

None — documentation-only plan, no external service configuration required.

## Next Phase Readiness

- Phase 2 (Gap 3 - Shallow vs Deep MLP Comparison) is now fully complete: all 4 plans (02-01 structural refactor, 02-02 arch-compare mode implementation, 02-03 real 12-arm execution + decision verification, 02-04 this documentation update) executed and committed.
- `CLAUDE.md` now accurately reflects the real Gap 3 outcome and satisfies ARCH-06's "independente do resultado" documentation requirement.
- No architectural change to production `config.h` is pending from Gap 3 — the adopted configuration (Config C, baseline regularization) is already what's compiled today.
- Ready to proceed to Phase 3 (Gap 1 - Paraconsistent Feature Selection), per the SPEC.md-mandated Gap 2 → Gap 3 → Gap 1 implementation order. Phase 3 should incorporate the Gap 2 (Borderline-SMOTE, adopted) and Gap 3 (Config C baseline, confirmed unchanged) decisions as its starting point, per the project's explicit "Gap 1 deve incorporar as melhores config/modo já validados nos passos anteriores" constraint.
- No blockers identified.

## Self-Check: PASSED

- `CLAUDE.md` modification verified present via `git diff --stat` (1 file changed, 40 insertions, 1 deletion).
- Commit `683f7d7` verified present in `git log --oneline` (confirmed via the commit operation's own output; hash matches `git rev-parse --short HEAD` immediately after commit).
- All 6 acceptance-criteria greps re-verified after the edit: `L2 lambda | 0.001` → 1; `L2 lambda | 0.003` → 0; `## Gap 3 Outcome` → 1; `train_log_v33_gap3_arch_compare.txt` → 2; `DECISAO:` → 2; `mean_epochs_to_stop|Pitfall 3|para tuning distinto por profundidade` → 3. No missing items.

---
*Phase: 02-gap-3-shallow-vs-deep-mlp-comparison-with-structural-refacto*
*Completed: 2026-07-29*
