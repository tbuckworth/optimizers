# Committed-response lock and JSON grader — source-only

[responses.py](responses.py), SHA256 `f59735af53f90f9afbdbb5419cdf3f6ac6695db7cd1da43771a6ceb43239e013`, and [test_responses.py](test_responses.py), SHA256 `a7d31d0f9d6efc90352735285963109ef72b1d9fdfe34c84e8786336dba696c3`, implement no acquisition, model, tokenizer, array loader or reader invocation. No actual response seal, public lock gate or grade has run in this source task. Main owns admission and exact metadata release.

The full old natural-add-on judging source was read for its commit/credit design, but is not imported or invoked. The sole runtime helper is the fully read, exact-pinned main `reader_packets.py`, SHA256 `daf506324f9da867a75d10046a3a11d8ec17b960f61bec585d3371d4cc0413a3`: strict JSON/public/response validation and deterministic packet reconstruction. The unchanged evaluator is pinned to `0393c1d0683dd43917282e65ead47f2b532efb62e0f4b2d7f7c6928132252e30`.

## Frozen release contract

Every artifact descriptor is `{path: absolute_path, size_bytes: positive_integer, sha256: lowercase_SHA256}`. The release is JSON with exactly:

```text
schema: jlens_pattern_calibration_judging_release_v1
packet_commit: exact 40-character commit
reference_commit: ef8168b6ba441c5e3a594df66d872b159387f2f1
packet_manifest: artifact descriptor
evaluation:
  source: artifact descriptor
  receipt: artifact descriptor
  scores: artifact descriptor
```

The source-pinned packet manifest binds all five public/private artifacts and original protocol/references/preflight/data; the release does not duplicate that inventory. Main freezes this release and its own commit before readers. All APIs require `release_path`, `release_sha`, `release_commit`:

- `seal(..., raw_commit, raw_paths, target)`, where `raw_paths` maps rater1..4 to artifact descriptors; explicit `JLENS_PATTERN_SEAL_RELEASE=1`.
- `verify_response_lock(..., responses, lock_sha, lock_commit)` is read-only/public-only; no release flag or scientific key reads.
- `grade(..., responses, lock_sha, lock_commit, target)`; explicit `JLENS_PATTERN_GRADE_RELEASE=1`.

Seal verifies the committed release, committed manifest/four public packets, all four raw-response commit blobs and exact 64-choice schemas. Release and packet commits must precede the raw commit. Four copies preserve raw bytes exactly; the lock binds source/release/manifest/private-map metadata/raw commit/copy digests. The private map, measured receipt, source and scores are not opened in seal or the public gate.

The gate verifies all five committed lock/copy blobs, all four raw blobs, byte equality, exact public choice IDs, source pins and raw-before-lock ancestry. Only then may grade reconstruct the exact public/private mapping from frozen references/data, validate its committed private blob, open the evaluation receipt and check source, counts, runtime, capture scope, input pins and the exact score output path/hash/size. Only after those checks does it open score JSON. No NumPy is imported. This proves recorded artifact ordering/identity, not that a raw file is the true first final; root's transport record must establish that separately.

All stages use exclusive fresh directories, private file permissions, bounded regular-file reads, no symlink leaf following, SHA/size checks, fsync, output budget and retained failure records. No automatic retries. The actual main release/response/grade directories are not created by fixtures. Source flags are not substitute resource enforcement: root uses a bounded one-thread CPU process for these JSON stages.

## Fixed numerical interpretation

All four axes and 16 pairs, four readers: 256 individual choices, 128 credits per P/U arm, 64 per cohort/arm. Exact float64-export score subtraction determines FIRST/SECOND; exact zero gives half credit. No small-gap cutoff, p-value, equivalence or favorable-axis replacement. The all-axis pilot criterion requires positive P−U in both cohorts and each P reader strictly above its own stronger realized always-FIRST/SECOND control. Both constants remain in every summary; pooled opposite orientations must equal 50%.

Outputs retain every item, axis/topic/pair/cohort/reader summary, every 128 matched gain/harm/unchanged cell and all 128 within-arm cross-cohort agreement cells. Agreement compares chosen canonical text IDs, not screen-side labels, and is not evidence of accuracy. Source reconstruction fixes the arm/axis/target mapping before key access.

Twelve fabricated tests PASS, 1.285 seconds, `/usr/bin/python3`, CPU-only/CUDA hidden, one numerical thread, offline and `timeout 60s`. Tests include full fabricated seal→committed gate→reconstruction→receipt→grade, missing key files during the public gate, altered/uncommitted raw copies, invalid first replies without repair, wrong receipt/source/count/input/output/runtime, half-credit ties, both-cohort requirement, stronger positional controls, canonical agreement, all adverse cells, malformed scores, duplicate cells, strict JSON, bounded/symlink reads and consumed-stage failures. No actual packet, response or measured key was read.

Best-practices validation used the current [Python JSON documentation](https://docs.python.org/3.12/library/json.html): bounded input is necessary for untrusted JSON, and explicit duplicate-key/nonfinite rejection is supplied by the pinned helper rather than assumed from decoder defaults. This preserves the established strict-data contract without adding a scientific criterion.
