---
phase: 01-gap-2-borderline-smote
plan: 04
subsystem: documentation
tags: [documentation, borderline-smote, mcnemar, academic-rigor, claude-md]

# Dependency graph
requires:
  - phase: 01-gap-2-borderline-smote (plan 03)
    provides: "Real A/B verdict: DECISAO Borderline-SMOTE ADOTADO (Macro F1 borderline=0.4587 >= padrao=0.4351, delta=+0.0235, McNemar chi2=0.1928 p=0.6606 not significant); empty-pool fallback fired 0/60 real rows"
provides:
  - "CLAUDE.md's 'What Works' Borderline-SMOTE claim corrected from bare/unbacked mention to a citation-precise (Han/Wang/Mao 2005), evidence-backed claim pointing to results/train_log_v32_gap2_smote_ab.txt"
  - "New '## Gap 2 Outcome — Borderline-SMOTE A/B (v32)' subsection in CLAUDE.md: verbatim DECISAO sentence, standard-vs-borderline comparison table, McNemar significance caveat, empty-pool-fallback frequency, file pointers"
affects: [phase-01-completion, future-gap-1-and-gap-3-documentation-pattern]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Documentation-drift-correction pattern: never leave a 'What Works' claim unbacked by a real, dated A/B artifact; when the real result carries a non-significant McNemar caveat, state that caveat explicitly rather than presenting the adopt/reject rule outcome as a proven win"

key-files:
  created:
    - .planning/phases/01-gap-2-borderline-smote/01-04-SUMMARY.md
  modified:
    - CLAUDE.md

key-decisions:
  - "Applied the ADOTADO edit path (not REJEITADO): replaced the bare 'Borderline-SMOTE' item in CLAUDE.md's What Works line with a citation-precise phrase (Han/Wang/Mao 2005, results/train_log_v32_gap2_smote_ab.txt, +0.0235 delta, McNemar p=0.66 not significant) rather than moving it to What Doesn't Work"
  - "Placed the significance caveat (McNemar p=0.6606, NOT statistically significant) in both the terse What Works bullet AND the fuller Gap 2 Outcome subsection, per the plan's explicit instruction not to gloss over it even though the bullet itself is necessarily short"
  - "New Gap 2 Outcome subsection inserted immediately after 'Important Constraints' (before 'Output Files'), matching the plan's exact placement instruction; no other CLAUDE.md section touched (confirmed via git diff --stat showing 1 file, 25 insertions/1 deletion, both hunks scoped to the intended locations)"

patterns-established:
  - "Gap outcome documentation pattern: a dated '## Gap N Outcome — <name> (vXX)' subsection with verbatim DECISAO sentence + comparison table + significance caveat + fallback/edge-case frequency + file pointers, to be reused for Gap 3 and Gap 1's own documentation-completion plans"

requirements-completed: [SMOTE-05]

# Metrics
duration: 6min
completed: 2026-07-27
---

# Phase 1 Plan 4: CLAUDE.md Gap 2 Documentation Correction Summary

**Corrected CLAUDE.md's pre-existing documentation drift — replaced the bare, unbacked "Borderline-SMOTE" claim in "What Works" with a citation-precise (Han/Wang/Mao 2005), evidence-backed statement tied to the real v32 A/B run, and added a dated "Gap 2 Outcome" subsection with the verbatim DECISAO sentence, comparison table, McNemar significance caveat (p=0.6606, not significant), and empty-pool-fallback frequency.**

## Performance

- **Duration:** ~6 min
- **Completed:** 2026-07-27
- **Tasks:** 1
- **Files modified:** 1 (CLAUDE.md)

## Accomplishments

- Read the real verdict from `01-03-SUMMARY.md`: `DECISAO: Borderline-SMOTE ADOTADO` (Macro F1 borderline=0.4587 >= padrao=0.4351, delta=+0.0235), with McNemar chi2=0.1928 p=0.6606 (NOT statistically significant)
- Since the decision was ADOTADO, replaced the bare `Borderline-SMOTE` item in the `**What works**:` line with `Borderline-SMOTE1 (Han/Wang/Mao 2005, validated by reproducible A/B in v32 — see results/train_log_v32_gap2_smote_ab.txt; Macro F1 +0.0235 point-estimate delta, McNemar p=0.66 not statistically significant)` — closing the exact documentation gap RESEARCH.md's "State of the Art" section flagged (the old claim predated any real implementation or A/B run)
- Added a new `## Gap 2 Outcome — Borderline-SMOTE A/B (v32)` subsection immediately after "Important Constraints" and before "Output Files", containing:
  - The verbatim `DECISAO:` sentence copied from `results/train_log_v32_gap2_smote_ab.txt`
  - A standard-vs-borderline comparison table (accuracy, macro_f1, per-class F1, bootstrap mean [95% CI]) sourced directly from `01-03-SUMMARY.md`'s Final Verdict table
  - The direct McNemar chi2/p between the two arms, with an explicit "NOT statistically significant" caveat and a note on which classes gained/gave back F1
  - The empty-borderline-pool fallback finding (30/90 rows fired, but all 30 are structural placeholders; 0/60 real classification rows)
  - File pointers to `results/train_log_v32_gap2_smote_ab.txt`, `results/smote_ab_comparison.csv`, `results/smote_borderline_counts.csv`
- Verified `git diff --stat CLAUDE.md` shows exactly 1 file changed (25 insertions, 1 deletion), with both hunks scoped only to the intended "Important Constraints" bullet and the new subsection — no other section (Build & Run Commands, Architecture, Current Best Results, etc.) touched

## Task Commits

1. **Task 1: Update CLAUDE.md with the Gap 2 (Borderline-SMOTE) A/B outcome** - `e00943b` (docs)

**Plan metadata:** (this commit, following) `docs(01-04): complete CLAUDE.md Gap 2 outcome plan`

## Files Created/Modified

- `CLAUDE.md` - "What works" Borderline-SMOTE claim made citation-precise and evidence-backed; new "Gap 2 Outcome — Borderline-SMOTE A/B (v32)" subsection added with DECISAO sentence, comparison table, McNemar caveat, fallback frequency, and file pointers

## Decisions Made

See `key-decisions` in frontmatter above (ADOTADO edit path selected per the real verdict; significance caveat placed in both the terse bullet and the fuller subsection; new subsection placement matches plan instructions exactly).

## Deviations from Plan

None - plan executed exactly as written. The verdict from `01-03-SUMMARY.md` was unambiguous (ADOTADO, mechanically confirmed consistent under both point-estimate and bootstrap-mean readings), so no interpretation judgment call was needed beyond following the plan's ADOTADO branch instructions verbatim.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SMOTE-05 is satisfied: `CLAUDE.md` now accurately reflects the real, tested Gap 2 outcome with concrete numbers, a significance caveat, and file pointers, regardless of the (positive) result.
- This is the last plan in Phase 1 (`01-gap-2-borderline-smote`) — all 4 plans (01-01 through 01-04) are now complete. Phase 1 is ready to be marked complete in STATE.md/ROADMAP.md.
- Per the project's own documented implementation order (`SPEC.md`/CLAUDE.md's Constraints: "Gap 2 → Gap 3 → Gap 1"), Gap 3 is the next phase to plan/execute.
- No blockers identified.

---
*Phase: 01-gap-2-borderline-smote*
*Completed: 2026-07-27*

## Self-Check: PASSED

- FOUND: CLAUDE.md
- FOUND: .planning/phases/01-gap-2-borderline-smote/01-04-SUMMARY.md
- FOUND: commit e00943b
</content>
