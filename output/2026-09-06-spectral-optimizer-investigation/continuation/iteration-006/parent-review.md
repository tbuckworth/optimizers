# Iteration 006 parent review

Reviewer: Codex parent, Spectral Optimizer Investigation. Date: 6 September 2026.

## Decision

The primary source freeze is `1221d70`; primary execution used `bdeb407` after
the separately recorded pilot/full GO. Fresh bundle 60006 was predeclared before
primary outcomes and executed after reviewed source/GO commit `1265706`.
Automatic artifact commit `c9bb05d` did not itself constitute report approval.

## Verification accepted

- Independent [raw audit](audit-results.md): 15 primary scientific sources,
  111 primary bulk artifacts, all 72,000 primary step records, 171 metrics per
  cell, 2,052 arm groups and 5,130 paired groups; separate fresh-source, 21 bulk
  artifact and 12,000 step-record checks. No producer/summarizer arithmetic is
  imported. Parent comparison confirms exact seed/arm/pair summary agreement.
- Parent read the complete independent auditor and its tests. Three additional
  [pairing regression tests](test_parent_aggregation.py) exercise the actual
  aggregator with different null masks, matching masks and a missing seed
  metric; incomplete pairs cannot become available-case means. This strengthens
  test coverage; no observed raw-aggregation defect was found.
- Independent-source [CPU verifier](verify_checkpoints_cpu.py): all 360 primary
  and 60 fresh test/validation/final-training accuracy evaluations reproduce
  exactly. Maximum CE errors are 7.9155e-8 and 9.6309e-8, below the prospectively
  fixed 5e-5 tolerance. This checks saved states and metrics, not every unsaved
  gradient or Adam state. The separate fresh training run has a different role.
- Portable archive (artifact not distributed in this public snapshot): all 84 pretest/final JSON files,
  494,973,615 original bytes compressed to 69,484,748 bytes, roundtrip SHA checks
  passing. Original files remain unchanged. Checkpoint tensors and dataset plans
  are hash-bound local large-volume artifacts, not contained in a Git clone.
- Final CPU tests: 94 iteration-006 tests pass (10.830 seconds), 17 repository
  tests pass (1.338 seconds), and six recovery-reminder tests pass (2.471 seconds).
  The canonical optimizer and original delivered PDF hashes remain unchanged.

## Interpretation corrections and limits

The audit's `HONEST-NEGATIVE` disposition and `TRUE-NULL` finding label mean no
consistent lagging benefit was established. They do **not** establish statistical
zero effect or equivalence: no powered null test or equivalence margin was
predeclared. Its shorthand about reused validation/test data is qualified here:
bundles generate their own splits from the same MNIST population and share the
official test set. The fresh bundle is not pooled into the primary estimate.

The final report retains all four primary contrasts, the fresh noisy sign
reversals, warmup-selected zero-exposure checkpoints, both validation selectors,
fixed endpoints and the conflicting earlier selected-AdamW comparison. Restored
norm means the current projection on the restored arm's own state, not another
arm's gradient or actual AdamW displacement. Retention compares the same gradient
through two bases. Calendar phases are not repair/no-repair partitions. No causal
denoising or universally superior optimizer claim follows.

The researcher workflow required the separate rerun and independent raw audit;
its report stage retains unfavorable evidence instead of requesting a tuned
retry. Ranked future tests are in next-steps.md (artifact not distributed in this public snapshot), each requiring
its own prospective design and recorded launch decision.

## Evidence identities

- Primary execution: `b44ac3901b286cfd202dcb956979facf5cbfb7316f5dec8f7d3e3182ac7f998e`.
- Fresh execution: `77c63fa4741eccbc884e63472d08932beb69075462297e5213526d73334f0799`.
- Independent audit JSON: `5a8a2f95bb4a4db148aa6b2bceb141d5be04b7093703a2a1e6b2f7e2fe22a048`.
- Independent auditor: `7ab061e1514655ff583a7af4022d2581fd3211a536c17edf5f37dbbe51ea02aa`.
- CPU verifier: `1d3fdb255c1be00668156d715723cec541afe691acaced7f6d0f4d2cb1fe3d19`.
- Original PDF: `8aabbb785e9b0c6eaa90ccd1a026caa056653080cb02a528f482892dfea0fecb`.
