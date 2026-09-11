# Endpoint/location source review

**PASS.** Reviewed the complete frozen protocol/input roster, worker `forward.py` and `test_forward.py`, and main `analyze.py`. No remaining concrete implementation or mathematical blocker identified. This review did not execute a real tokenizer, model, scientific-array calculation or prior stage.

The producer preserves all 16 shortened texts and eight pairings. Its preflight binds the old token receipt and exact text-plus-suffix identity, requires strictly shorter token IDs matching the old ID prefix, and validates contiguous verb coverage ending exactly at the verb boundary. Leading whitespace is allowed; trailing characters are not. The last token must cover the final period and end at string length. Invalid offsets or a late-row failure stop the whole panel; no endpoint substitution or dropped row is permitted.

Source inspection confirms two ordered post-block-11 captures, `verb` then `sentence_end`, in each of 16 forwards. The hook observes without replacing output. Both h32 positions use the unchanged U32 directions/source mean with float64 scoring; every O−P gap subtracts the saved score rows. No PCA, lens decoding, reader, training or old full-prefix forward is introduced. Pinned offline loading, frozen modes/gradients and tracked parameter-version checks retain the prior contract; the latter are not cryptographic weight-byte verification.

The only concrete analysis gap raised during review—array positions could be labelled without checking saved row/location metadata—was fixed at `5232cc1`. The analyzer now checks receipt/NPZ location order, preflight/input record identity and endpoint metadata, scalar JSON exports, and pair/axis/orientation metadata. Independent `math.fsum` projections are checked with absolute tolerance 1e−12; gap differences with 2e−12; saved-score gap subtraction and signs must agree exactly. This corroborates 128 new scores and 64 new gaps, reuses pinned old final scores, and retains all three locations/all four axes. It does not establish semantic validity.

Primary success remains all eight verb-position PC4 O−P gaps strictly positive; zero fails. Sentence-end values and shifts versus the old endpoint are descriptive, not alternative success criteria. A positive result would support this local correspondence, not isolate position from available causal context or lexical identity. Four contents in two templates are not eight independent contents, and the earlier full-end failure remains unchanged.

Independent fabricated checks passed: **13 worker tests in 2.674 seconds**, with CUDA hidden, offline settings and one-thread limits; the final analyzer's fabricated projection/gap/strict-zero fixture also passed. Tests use fake tokenizers, CPU models and fabricated arrays, not study data. The new receipt/input checks were source-reviewed; the small analyzer fixture is not an end-to-end test of those checks. Exclusive attempt creation and source/input receipts prevent silent replacement or automatic retries. Live service/cgroup, time, host-memory and GPU-headroom enforcement remains the main launcher's responsibility; the 8 GiB allocator ceiling is not a total GPU-process cap.

## Reviewed SHA-256 pins

| File | SHA-256 |
|---|---|
| Main `protocol.md` | `e426578248cd27f915f9e97862b48a1ebe4b978841b109dfee61f3699d8c39b3` |
| Main `dataset.json` | `bda6dbefe07ebfa62cbf2efc483b6164cc819debfd1d124460e5153bdd2b7ee8` |
| Main `pairs.json` | `d47d27b93e48b38deb5e4871596c906057ba46e18305e3849b02ad1b0e337ac0` |
| Worker `forward.py` | `bd5a997f1b284833e9edb1d099264ff81f1f9132c01fef7574920aeff0b6d747` |
| Worker `test_forward.py` | `37517156135bf668545877f6ccaa9095bb8dbaf99f6b65337f1d5c11db76d0bc` |
| Main `analyze.py` | `8e71bdd65af0b84d2536c95c532b188f6776657f1831d6374cbdafc6035ba2e5` |

Worker files are under `/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-endpoint-location/`. This reviewer did not author these producer or analyzer sources. No additional experiment or real-stage release is selected by this note.
