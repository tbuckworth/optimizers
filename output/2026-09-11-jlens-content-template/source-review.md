# Content × template diagnostic: design and source review

**PASS for the reviewed source and fixed design.** No remaining concrete source blocker identified. This is a pre-measurement review, not evidence that the semantic hypothesis is true or a substitute for the launcher's live resource checks.

## Scope and fixes

Read the complete proposed design, decision, protocol, dataset and pairs, worker `forward.py` and fabricated tests, and main `audit_saved.py`. No real tokenizer, model, scientific array or measurement output was opened or executed in this review.

Two concrete defects were corrected before their affected stages ran:

- The original producer passed the wrapped dataset/pair JSON objects to a list validator; its stage fixtures incorrectly used bare lists. The corrected `load_panel()` accepts exactly `{"rows": ...}` and `{"pairs": ...}` in both stages. Fixtures now use the frozen schema and reject bare lists/extra keys.
- The saved-array audit now requires finite float64 score/gap arrays with exact shapes and exact equality of saved gaps to saved-score observation-minus-provision differences, before its independent `math.fsum` projection checks. This preserves the registered scalar subtraction and strict-zero convention, including last-bit cases.

Neither correction changed texts, pairings, directions, prediction or acceptance criteria. Worker source/tests are frozen at `437359ac0295dca615ea236f6d6a2338bfd51338`; the audit correction is at `427c678`.

## Design and mathematical checks

All 16 strings match the proposed table and frozen inputs at `81bf8ae`: four content pairs × two templates × two poles, with eight fixed O−P contrasts. Within each pair/template only the verb changes; across templates the actor, object, action and time are retained. The common ending controls terminal-token identity, not all context, position or tokenization effects.

The primary prediction is all eight PC4 gaps strictly positive; zero is not a success. All signed magnitudes and all four fixed axes remain visible. Template sign changes use three-way positive/zero/negative signs and need not mean positive-to-negative reversals. Four content pairs are not eight independent replications.

This is a useful, small prospective test of the explicitly outcome-informed observation/provision interpretation. A positive result would support that interpretation on these controlled texts; a negative result would challenge it without undoing the earlier Wikipedia 19/24 finding. Verb identity, frequency and meaning remain confounded. The design cannot establish a general J-Lens advantage, semantic causality, or that syntax is the only source of transfer differences. No additional arm is necessary to answer this bounded question.

## Producer and audit checks

Source inspection supports the specified path: pinned offline tokenizer/model; whole-panel token preflight without truncation; one frozen-model forward per text; an observational hook at block 11 capturing the final token; unchanged canonical float32 directions cast to float64 with the original float64 mean; saved scores followed by O−P subtraction. The upstream wrapper's actual forward path uses `use_cache=False`; no new PCA, decoder or reader is invoked.

Exclusive attempt/output creation, input/source hashes, preflight receipt binding and preserved failure outputs prevent silent replacement or automatic retries. Parameter/module checks establish frozen modes, absent gradients and unchanged tracked parameter versions—not a cryptographic proof of all weight bytes. The allocator bound is not a total GPU-process memory bound; service/cgroup limits, deadline, device headroom and other GPU clients remain the main launcher's responsibility.

The corrected worker suite passed independently: **13 fabricated CPU tests, 2.624 seconds**, with CUDA hidden, offline flags and one-thread environment limits. It covers the exact wrappers, invalid rosters/tokens, final-block capture, tracked parameter mutation, canonical scoring, sign/zero cases, receipt tampering and consumed attempts. Main reports its corrected audit fixtures pass; I reviewed that correction without running an actual array audit. These fixtures do not validate the semantic hypothesis or real GPU execution.

## Reviewed SHA-256 pins

| File | SHA-256 |
|---|---|
| `protocol.md` | `620565ecc6c78c64ac395efee947e20a97772bcb6ccc804c645ccfc198d350f3` |
| `dataset.json` | `841d86753f857d686165e53d76278bc1fa40840f97e09ffad61f0c89d57fcc7d` |
| `pairs.json` | `d47d27b93e48b38deb5e4871596c906057ba46e18305e3849b02ad1b0e337ac0` |
| Worker `forward.py` | `16d53d9da76269537d8743ad92387bc8fd36f6d89c7d7dba28b86a818c877b7e` |
| Worker `test_forward.py` | `c623845c6649a1c9347ff0a8a2edd1289cf05a847ab848214047c25d6d1bf51e` |
| Main `audit_saved.py` | `9a218f074ba1c42bb390413395c68081796ed2416941a6084ac71794a9dc7c10` |

Worker files are in `/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-content-template/`. This reviewer authored earlier grading components, but not this diagnostic's producer or audit helper. No additional measurement or follow-on experiment is selected by this review.
