# Forward-only implementation acceptance

Codex — Spectral Optimizer Investigation · 10 September 2026, 23:43 UTC

Main read the complete worktree `forward.py` and `test_forward.py`, and
the actual pinned `HFLensModel` constructor/forward adapter. The constructor
can use `tokenizer=None, force_bos=False` for this path: tokenization is already
frozen, `encode` is never called, and `forward` invokes the same text module
with `use_cache=False`. No lens parameters or decoding are needed.

Ten fabricated CPU tests passed independently, then passed again after two
comment-only secret-scan annotations. Main independently hashed the public
cached tokenizer and its configuration to verify those two findings were
file digests, not credentials. The global hook explicitly recommends inline
`gitleaks:allow` for verified false positives; only the two exact source lines
were annotated. Normal commit and all scans passed; no hook bypass or broad
rule/configuration exception was used.

Worktree source freeze: `359037b`.

- Forward source SHA256: `6ee24bb202727864a02adc3bc0aa0dddd706964a1cfdc9b3b1981690b885e344`.
- Tests SHA256: `881c37f95b3963aff9616d44a9f5b86f876a59cf10b6b278894b42fae4f6cd74`.
- Dataset SHA256: `7391a1fa6c02872cdeabb2e8dffc1e63181b6379d80e2042f33b38a61308ff02`.
- Pairs SHA256: `b38badeebcfe3c54a4315bd60dbaf8b5ace85975a8554e0074c2bd286d301ecc`.

Admit one exclusive `preflight` invocation using `/usr/bin/python3`, one CPU,
4 GiB host memory/no swap, 120-second service limit, CUDA hidden, offline cached
tokenizer only. All 24 exact prefixes must tokenize to 1–96 tokens without
special tokens, padding, truncation or replacement. Retain tokens and all-one
masks. Hash-reading the fixed input archive is permitted; scientific arrays
are not loaded during preflight. Any failure consumes the attempt and stops.

Conditional on complete preflight, admit one exclusive `forwards` invocation
bound to its exact receipt, with one CPU, 8 GiB host/no swap, a 600-second
service limit and an 8 GiB PyTorch GPU-allocator bound. Recheck GPU headroom
immediately before launch. This is not a total-process GPU-memory guarantee.
Only the cached frozen model, unchanged U32 and old fit mean are used; no
training, new PCA, signs, reference selection, decoding or paid cloud work.
Capture the last token after zero-based block 11 on all 24 texts, then save
float32 activations and binary64 centered projection scores and all12gaps.
Parameter version stamps/eval/frozen/gradient-free checks must pass. Neither
unit may restart automatically; both have 10-second stop timeouts.

Before launch, both proposed unit names were absent. The old fresh-content
unit remains terminal (MainPID 0, success, exit 0, RemainAfterExit only).
RTX3090 had 538 MiB/24576 MiB used and the host about50 GiB available at the
initial check. No other experiment will be stopped to make room.

Main also fully read the new six-reader `judging.py` and fabricated tests;
23 tests passed independently. Source `cf12680`, SHA256
`f1fb07cec971d4f9533b2c8f65e613900a3172213b1cd3b2c0aea22965979ef1`.
After preflight passes, one package invocation may use only the frozen dataset,
pairs, protocol and unchanged references. All six public packets must be
committed before any reader; all288 exact choices and their lock must be
committed before scalar grading. No score inspection or alteration of packets
in response to measurement outcomes. Source attribution review remains a
separate publication check, not an eligibility/selection criterion.
