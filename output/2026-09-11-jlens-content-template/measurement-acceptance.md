# Main source acceptance and bounded admission

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

Main read the complete new worker forward.py, all fabricated tests, and the
entire unchanged pinned jlens/hf.py adapter. Independent review caught a
source-only dictionary/list container mismatch before any actual preflight or
measurement. Both stage loaders and fixtures now enforce the exact new JSON
wrappers. Main read that correction; all 13 fabricated tests independently pass.
No scientific outcome was obtained, changed or discarded during this fix.

Reviewed worker source SHA256:
16d53d9da76269537d8743ad92387bc8fd36f6d89c7d7dba28b86a818c877b7e.
Reviewed tests SHA256:
c623845c6649a1c9347ff0a8a2edd1289cf05a847ab848214047c25d6d1bf51e.
The exact main input/protocol freeze is 81bf8ae. Source must be committed
unchanged before either actual stage starts.

The tokenizer explicitly disables special tokens/padding/truncation; both
model eval and no_grad are used. The adapter forwards only the text module
with use_cache=False. Its encode/unembed/lens methods are never called.
Public documentation checks are linked in protocol.md; installed source is
authoritative for this pinned path. The covariance archive is reused without
refitting or sign changes. O−P arithmetic and zero cases pass fabricated checks.

Admit one new CPU preflight followed, only if all 16 exact texts pass unchanged,
by one frozen-model run. Use fresh units:
j-lens-content-template-preflight-20260911-MMxZIs.service and
j-lens-content-template-forward-20260911-MMxZIs.service.
The two corresponding exclusive output directories must not exist before
launch. Bind model admission to the committed exact token-preflight receipt.
No restart, source relaxation, dataset replacement or retry on scientific failure.

Runtime: /usr/bin/python3, cached local model only; CPU quota one core,
preflight 4 GiB host/no swap/TimeoutStartSec120/CUDA hidden; model 8 GiB host,
no swap, RuntimeMaxSec600, 8 GiB PyTorch allocator cap, Restart=no,
TimeoutStopSec10. Current checks found ~23.6 GiB GPU free and ~50 GiB host
available. Recheck immediately before GPU launch; do not stop other work.
Old independent/fresh units are terminal, not live despite RemainAfterExit.

Main's audit_saved.py uses compensated scalar projections and exact saved-score
subtractions, then verifies JSON/input identity and every sign. Independent
review requested the exact subtraction assertion, now present in 427c678;
its fabricated orientation/zero/sign fixtures pass. This is corroboration of
new saved data, not another model run or semantic validation.