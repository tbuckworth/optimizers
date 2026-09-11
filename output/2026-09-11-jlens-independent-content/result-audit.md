# Independent saved-scalar arithmetic check

Codex — Spectral Optimizer Investigation · 11 September 2026 BST

**PASS — one audit invocation, 21,082 checks, no discrepancy.** The check independently recalculated all 288 item truths, chosen prefix IDs, credits, signed/absolute gaps and every saved arm/axis/reader/paired/position/agreement summary. No grading function was imported or called. The audit author also authored the grader: this is independent arithmetic, **not an independent-implementation-author review**.

## Verified results

A = signed-direction J-Lens token lists; B = selected fit-example J-Lens token lists; C = those fit prefixes. Each cell below is correct choices out of 24, with the two readers' counts out of 12 in parentheses. Agreement means choosing the same underlying prefix despite reversed presentation, not choosing the same FIRST/SECOND label.

| Axis | A correct | B correct | C correct | Agreement A / B / C, each out of 12 |
|---|---:|---:|---:|---:|
| PC1 | 10 (4+6) | 11 (6+5) | 12 (7+5) | 8 / 9 / 10 |
| PC2 | 9 (5+4) | 13 (7+6) | 12 (7+5) | 9 / 9 / 8 |
| PC3 | 12 (5+7) | 13 (7+6) | 12 (5+7) | 10 / 9 / 10 |
| PC4 | 19 (9+10) | 13 (7+6) | 12 (6+6) | 11 / 11 / 10 |
| All axes, out of 96 | 50 | 50 | 48 | 38 / 38 / 38, each out of 48 |

All 48 physical axis/pair targets are retained, each with six arm/reader choices. There are 12 unique physical text pairs, not 288 independent observations. There are no exact score ties. Always-FIRST and always-SECOND each receive 12/24 per pooled axis/arm and 48/96 per arm by the complementary presentation design; individual readers need not have balanced position controls. Reader totals rater1–rater6 are 25, 25, 25, 29, 23, 21 out of 48.

Paired A-minus-B credits by PC1–PC4 are −1, −4, −1, +6 (total 0). A-minus-C: −2, −3, 0, +7 (total +2). B-minus-C: −1, +1, +1, +1 (total +2). All item-level values and underlying choices match the saved [grade](graded/grades.json) exactly, including binary64 representations rather than a numerical tolerance.

## Positive and adverse pair-level support

**Useful positive:** PC4's A readers both correctly order P01–P05, P07, P08, P11 and P12: nine agreed-correct pairs. Both are wrong on P06 and P09; P10 is mixed. Thus 11/12 agreement is nine agreed-correct plus two agreed-wrong, not 11 correct. PC4 agreement is equally high for B, so A is not shown to be more repeatable than B.

For PC4, A's credit gains over B are P01:+2, P02:+2, P05:+1, P08:+2, P10:+1, offset by P09:−2. The other six pairs tie, giving +6 overall. Against C the gains are P01:+2, P02:+1, P03:+1, P04:+2, P10:+1; the other seven pairs tie, giving +7. This supports a concrete direction-specific positive on the retained panel, not a general J-Lens advantage or four distinct semantic clusters.

**Important adverse result:** the motivating PC3 transfer result is 12/24 for A, at the pooled constant-position controls. A's readers both correctly order P01, P02, P06, P07 and P10, both wrongly order P03, P08, P09, P11 and P12, and disagree on P04/P05. Its 10/12 agreement therefore contains five shared successes and five shared failures. The earlier favorable PC3 result did not reproduce on this panel; reader agreement alone does not establish usefulness. PC4 is an observed positive among four retained axes, not a replacement primary target selected after seeing outcomes.

These are limited descriptive findings on externally authored Wikipedia prefixes grouped by source category, with two readers per description from one inherited model configuration. They do not establish unseen pretraining data, human validity, significance, general superiority, safety effects, optimizer effects, or the cause of the source-distribution shift. Disagreement combines reader variation and presentation-order sensitivity. Full all-axis pair support is retained in the [audit receipt](result-audit-checks.json).

## Scope and provenance

The accepted [audit source](audit_saved.py), frozen at commit `73092a7ce5cde216ff871f6a1cf9c9d77de488e6`, was exercised first on fabricated data only: 9,741 checks, including ten rejected malformed/mutated cases and hand-derived tie, tiny-gap, signed-zero and prefix-agreement expectations. Main read the complete source before releasing the single actual call.

Actual audit: `2026-09-11T00:15:37.676905+00:00` to `00:15:37.713315+00:00`; exit 0. Execution used a 30-second timeout, CUDA hidden, and OMP/MKL/OpenBLAS thread counts of one. The exclusive receipt was claimed before input reads and was neither overwritten nor rerun.

Checked pinned input bytes, the unchanged 24-text/12-pair roster, unedited public prefixes/references, all anonymous item/block joins, exact allocation, common first-cohort orientation and opposite second-cohort orientation. All six raw responses match their sealed copies. The seven lock/response blobs match commit `d1abdde61a5448338d1d75788fcfc29f9197851f`; that commit is an ancestor of the frozen grade commit `cbe9c407c860d7b8e1d3a3816f03a3a061ff4b45`. The saved grade-start time `00:03:13.024964+00:00` follows the response-seal time `00:02:01.646084+00:00`. These are file/commit provenance checks, not an independent reconstruction of reader transport.

Fresh-reader isolation, history-free dispatch, no observed tool use, one response each, and lossless compact-JSON transport are attributed to the collection team's reader-delivery record (artifact not distributed in this public snapshot), not independently certified by this scalar audit. This audit does not regenerate packets or validate the RNG implementation independently; their saved structural invariants and source/input bindings were checked.

| Artifact | SHA256 |
|---|---|
| [Audit source](audit_saved.py) | `0bc6b498837cfd7095e2574444b45a69548c551e55f9fdef1bf52f39dca0e738` |
| [Audit receipt](result-audit-checks.json) | `23747f2d818bbfb4af92a77cd236afca9594816223e0a6bbbb6aea117fc97bb0` |
| [Completed grade](graded/grades.json) | `5495950b0666d4365d146af2860c757dd6df87e6409b57d6861d3640d3099231` |
| Packet manifest (artifact not distributed in this public snapshot) | `e2f48664da9dd8945ffa1576873719fa87f6cd364f93235189915d8dae5d611e` |
| Response lock (artifact not distributed in this public snapshot) | `6851e689784ac2b752b195e683a553ae8e41f6cc29d87015821e0ff876fd107e` |
| Worker `output/2026-09-11-j-lens-independent-content/forwards/scores.json` | `e03efc7d22d480b2b7f12db89021e20653db2898a7317b63b6e2b6e9a3e5e704` |

The scalar export was consumed as the frozen canonical U32/FP64 score measurement. No model, NPZ, activation/projection reconstruction, PCA, refit, forward pass, judge invocation, grading-stage rerun, or new scientific measurement occurred. Main's separate projection/serialization checks are outside this audit's independently verified scope. No next experiment is selected here.
