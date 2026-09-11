# Source acceptance — fresh authored comparison

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

## Producer and tokenizer preflight

Main read the entire new worker `forward.py` and `test_forward.py`, after
reading the complete prior producer/checker used as implementation sources.
The new producer captures only the final distinguishing-verb subtoken in
each of the32 frozen full sentences. It preserves the original model,
post-block11 state, source mean and four U32 directions. No refit, training,
decoder change, extra shortened input, or sentence-final capture is introduced.

- Producer SHA256 `bdf25ab173c32c92fe47472de2f934b16ec898024f71c04b746b616229985873`.
- Test SHA256 `75c4b51e6d8f4753d105cba23821d4371f0e7c52fa238b7633b5939a3ae17b35`.
- Main independent synthetic run: all13 tests PASS in2.607seconds,4.219seconds
  command wall time. CUDA hidden, offline flags, one thread,60second timeout.
  Both source hashes match before and after tests.

Tests cover exact32-row/16-pair identities, split-verb offsets, unpadded masks,
invalid/bool/overlong tokens, malformed JSON, incomplete final-row preflight,
frozen-token tampering, consumed-stage refusal, missing-preflight blocking,
synthetic post-block capture at exactly one position and one call per row,
no mutation/hook removal, singleton output shapes, and saved scalar/gap order.
The fake end-to-end test loads only a temporary fabricated identity-direction
archive and a tiny fake CPU model. It is not a new scientific experiment.

Main source review passes: exact input pins and whole-roster checks precede
any tokenizer/model use; preflight loads neither model nor scientific arrays;
forward uses cached-only loading, frozen eval/no_grad parameters, unchanged
adapter, and an observer hook returning no replacement. All-one masks are
implicit in the unchanged no-padding adapter call. Shapes are32×1×1024
activations,32×1×4 scores and16×1×4 O−P gaps. Float64 centering/projection is
unchanged. Stage directories are exclusive and failures retained.

The declared8GiB limit is a PyTorch allocator bound, not a hard total GPU
process limit. Main enforces separate systemd CPU/host-memory/runtime limits
and checks live resources before invocation. The preflight source hashes a
scientific archive for identity but does not deserialize its arrays.

The implementer's initial synthetic run caught one stale schema name in a
test assertion, corrected before these accepted bytes and before any actual
stage. It did not change the hypothesis, stimuli or scientific observations.

This section admits ONE whole-roster tokenizer preflight only after these
exact source bytes are committed. Neural release additionally requires the
prospective reader/analysis implementation review and a frozen successful
preflight. No actual new stage has run at this record's creation.

## Reader package, response lock and grader — accepted

Main fully read the final new `judging.py` and complete adapted
`test_judging.py`. Source SHA256
`50de1fff65d4fc50f0aaea0bf901e9096a01fcc62d0c388aba774017d8f942bf`;
tests SHA256 `8c7d1eb8ed3e99c52ef0d10e6503ef259bfeaf03ebafc1d0c26e5eab0cc1e6bf`.
Main independent21 fabricated tests PASS20.120seconds with CUDA hidden,
offline flags, single threads and60second timeout. Source and test hashes
match before/after. Implementer's21tests separately PASS18.070seconds.

The implementation follows the predetermined seed20260917, complete fa roster,
four64-choice readers, homogeneous A/C arms, reversed second-cohort positions,
unchanged references and fixed PC4 primary comparison. Exact ties receive0.5;
tiny gaps and every secondary axis/form/reader remain. All reductions and
text-level agreement/fixed-position controls are retained.

The new metadata release contains only bounded path/size/SHA256 receipts,
with a git-committed byte check before packaging. It binds producer, design,
tokenizer, forward-receipt and future score hashes without changing source
after neural outcomes. Package never opens producer, forward-receipt or score
contents; it can run with those three files absent in a synthetic fixture.
Only input metadata/text/token records and original references enter packets.

The public-only response verifier validates all256 choices and all five
committed lock/response blobs before any scientific key or private map I/O.
Grade then verifies the original committed release, reconstructs identical
public/private packets, checks forward scope/output binding, and only then
opens the exact score file. Strict JSON/byte caps, invalid release sizes and
paths, altered keys, incomplete locks and consumed stages are tested.
Source imports do not load models, arrays, tokenizer, network or old producers.

Main acceptance PASS for the prospective analysis and reader implementation.
Commit these exact bytes before the one neural run, freeze its metadata
release before packaging, then commit packets before readers. This source
acceptance does not authorize any look at new scalar contents before the lock.

## Saved-data arithmetic checker — accepted, not yet executed

Main fully read the new checker and tests, then independently ran the combined
23 producer/checker synthetic tests: PASS2.887seconds (4.522wall). All source
hashes match before/after. Checker SHA256
`7563823ce2fc906af2e93001383cfd5712924e95004200f284183d22e34ac715`;
checker-test SHA256 `a7e19926bb72088d183f3ce516b28b4971faf0346cbff59d817516ae8b83aacd`.
Frozen worker `858f5fd6734c4603b4c71145f1d36cd713c2b7f9`; implementer23tests
separately PASS2.880seconds. The exact judging helper is hash-pinned.

The checker first validates all256 public choices and five committed response
blobs through the reviewed public-only gate, before producer-helper import or
measurement-file access. It binds the judged dataset/references/protocol and
exact saved tokens/preflight to the measured receipt, then checks every saved
array/JSON identity. Scalar projections are independently recomputed using
`math.fsum`; all128 scalar values and64 O−P gaps are checked, with every
contrary/zero result retained. No model, tokenizer or PCA fit runs.
Synthetic tests include missing lock before helper/array I/O, helper hash
mismatch, wrong judged tokens, artifact tampering, repeated-stage refusal,
shape/dtype/nonfinite/JSON joins and exact-zero/negative-compatible summaries.

Main acceptance PASS. All256 reader choices must still be committed and
locked before the actual checker. This check cannot select the panel or
replace the primary reader comparison with the descriptive O−P forecast.
