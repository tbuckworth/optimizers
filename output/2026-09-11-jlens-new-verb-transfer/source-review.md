# Main source review — new action-word transfer

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

Main read the complete new worker `forward.py`, `check.py`, `test_forward.py`
and `test_check.py`. The producer is a standalone adaptation, not an import
or invocation of an old consumed experiment. Main also read the previous
producer and the complete upstream `jlens/hf.py` and `jlens/__init__.py` to
verify the retained adapter behavior. No new scientific data was loaded by
this review.

PASS for the bounded implementation:

- Exact 32-row namespaced roster and 16 pair order, fixed SHA-pinned strings,
  verb spans and protocol. No old 16-row or old-token-prefix assumption.
- Fast tokenizer offsets, no added special tokens, padding or truncation;
  contiguous full verb coverage, final subtoken ending exactly at the verb
  boundary, and complete terminal-period coverage. Whole-panel failure
  stops measurement rather than dropping or replacing rows.
- Two ordered post-block11 captures per new forward. The hook returns no
  replacement. The frozen adapter directly calls its text module with
  `use_cache=False`; no decoding, PCA, gradients, training or new mean.
- Canonical U32/mean64 scoring, all 256 scores and 128 O−P gaps retained.
  Independent scalar `math.fsum` checks every projection to 1e−12 and exact
  subtraction binds each gap to saved scores. JSON/NPZ locations, IDs, axes,
  pairings and preflight records are checked together.
- Primary all-16 PC4 verb gaps strictly positive; zero fails. All per-axis
  positive/zero/negative counts, paired-template successes and sign changes
  survive. No new comparator or semantic-validity claim from arithmetic.
- Exclusive attempt directories, hashed inputs/outputs, explicit parent
  release and no automatic replay. A failed actual stage remains consumed.

Main independently ran all **18 fabricated tests: PASS, 3.105 seconds**, with
CUDA hidden, offline settings, one-thread limits and a 60-second timeout.
These include fake tokenizer/model capture, exact-zero behavior, bad offsets,
whole-panel late failure, mismatched JSON/arrays, tampering and repeated-stage
refusal. They do not measure the actual new texts or certify semantics.

The current official [tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)
confirms the fast-only character-offset interface and explicit tokenization
options used here. Installed versions and local cached model/adapter hashes
remain pinned; no dependency changes or new integration are introduced.
Resource enforcement remains the launcher's responsibility: one CPU quota,
4 GiB/no-swap preflight, 8 GiB/no-swap forward, time limits and headroom checks.
The 8 GiB PyTorch allocator bound is not a total GPU-process memory cap.
Parameter ID/version checks are not cryptographic post-run weight hashing.

Reviewed worker SHA-256 values:

| Source | SHA-256 |
|---|---|
| `forward.py` | `f59d6a0da7bd6280bd9a58f4688628ab1699389fa1b9d2e7fd73422a6808fb49` |
| `check.py` | `7f74d36d9549b1ad76a02d1ce09c4aed6407c911ad580fd037eb8b7ba4f1924f` |
| `test_forward.py` | `de7b119737f3a50e56946212d3cb9b8daf2e00b400ef218eec2ef0609e784bc3` |
| `test_check.py` | `dc142b42f799e7d7d2fce29b1591958aa9ae761963dbbdf4fb4e1e26344318e3` |

Worker root is `/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree`.
Main fixed inputs/protocol are committed `d13929d`. This note records main's
review; actual preflight/forward/check receipts do not yet exist.

## Final freeze and independent check

Worker source freeze `20be33538f727a505f2812fa3e4b6c5b0135ebad` is clean.
Before freeze, Astra added a fabricated pin-before-import test and renamed
one test method; executable producer/checker bytes did not change. Main read
the final test addition and reran the final suite: **19 PASS, 3.084 seconds**.
The earlier 18-test run above was an interim suite, not the final count.
The independent reviewer also read all four files and passed all 19 tests
in 3.107 seconds, with no concrete blocker. Final hashes in the table match
the committed files. Neither review accessed actual scientific arrays.