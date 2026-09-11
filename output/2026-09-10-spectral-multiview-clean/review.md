# Independent design and source review

Codex — Spectral Optimizer Investigation · 10 September 2026

**Status: source review PASS; no material design or implementation defect found.**
This is a source review, not a scientific result or launch authorization.
No scientific data, model, checkpoint, gradient or logit archive was opened,
and no experiment or audit was run by this reviewer.

Fully read the selected four-view observer design and scope decision, exact
protocol and implementation note, `spectral_multiview.py`, all runner/core/data
fixture sources, the new `_data.py` and `_core.py` helpers, and the independent
`_audit.py` and its 11 fabricated archive/tamper fixtures. Relevant inert helpers
in the unchanged accepted NumPy auditor were also checked. No fixture or
scientific job was executed by this reviewer; reported fixture passes belong
to the implementers and main agent.

The comparison asks a useful whole-policy question. Observer4 changes the
observation stream while retaining designated first-view delivery; raw4 tests
using the same additional views directly in Adam. It is not a complete
observation-by-delivery factorial. Observation scale, current-view self-inclusion
and later model/Adam/observer trajectories change together, as disclosed.

The helper preserves `g1` separately, sequentially averages four FP32 gradients
at fixed parameters, observes that average once, and projects `g1` only after
warmup. Native1 uses the unchanged canonical dispatcher. All parameter Adam
counters advance once. Raw observer metadata is zero. The data helper reuses
streams 0–2 and independently adds extra-view and secondary-readout streams.

The runner checks common initialization, raw1/native1/observer4 warmup
model/Adam identities and their exact predictions. Raw4 warmup is correctly
treated as a different trajectory. Original held-out endpoint CE and accuracy
remain primary; all 22 states, secondary translated outcomes and within-policy
warmup progress remain visible. A relative improvement caused only by raw
deterioration is not defined as recovered learning.

The protocol correctly records 120,000 total batch-gradient evaluations,
72,000 extra versus four single-view arms, with 256,000 underlying-example
occurrences per arm. Counts are not a wall-time claim. Archive inventory,
exclusive attempt, source/data pins and inherited resource guards are explicit.

This clean panel cannot certify retained wrong-label protection or a safety
intervention. The completed result that ordinary augmentation outperformed the
tested spectral recipe under wrong labels remains unchanged. The new study is
an unproven method-development proposal, not an explanation that overturns it.

The core fixtures exercise exact native-path equivalence, once-only observer
and Adam updates, paired warmup, immutable inputs and RNG, and an FP32 summation
order case that would distinguish a changed averaging implementation. The data
fixtures exercise unchanged streams 0–2, independent new streams, exact labels
and shifts, and the existing translation operator.

The auditor independently reconstructs all eight plan arrays, all saved logit
metrics and the five fixed endpoint contrasts. It checks the exact 58-artifact
inventory, source/data pins, guarded attempt/provenance binding, complete ordered
roster, counters, norm identities and common initial/warmup predictions. CE
benefit signs and warmup differences are correct. It retains all three paired
seed values. Strict JSON, NPZ headers/dtypes/shapes, finite values, direct-file
confinement and byte caps are enforced through pinned independent helpers.
Opaque checkpoint bytes are hashed, not deserialized: model/Adam digest pairing
remains a producer assertion, explicitly stated in both protocol and audit
output. This is not full independent training or input-use verification.

Reviewed SHA-256 identifiers:

| File | SHA-256 |
|---|---|
| `experiments/spectral_multiview.py` | `25ef96d7751220fde764ee5c9d7861975ce897c8274711d848dad50b6fd5bd34` |
| `experiments/spectral_multiview_core.py` | `8c226c392d51f3c8679c38b7996fc0843ce20c3097e08a7fb0b6eab6753b2d0c` |
| `experiments/spectral_multiview_data.py` | `781c6dc69e63e6db0accc6167b9b1476158265a18b20c4e8db91b7920c911efe` |
| `experiments/spectral_multiview_audit.py` | `0873c9da9162f4091a6a74a55479f1590211b259e6ff5e33d3aaaa4786e27f8a` |
| `tests/test_spectral_multiview.py` | `6d1c2da66cdc42662210d66574454aa969c19c4c090d43051eba6ae6aaaf232d` |
| `tests/test_spectral_multiview_core.py` | `767c4947564257aeddbfab484e3c6f16fa8b552d4f0e766f12f1b9b1ce49285f` |
| `tests/test_spectral_multiview_data.py` | `cf72a8d1322381cabf6584ce53295cc1e5a41a1ddfdea94ee50ac4ae375eb9b4` |
| `tests/test_spectral_multiview_audit.py` | `ff6f24a2225f579e036cf7dc21811a5372f081bc8e9f6dc903ced128a000b0c8` |
| `protocol.md` | `fe90274ce7eb1ff6e688c53782d68264f6b1bbd9320b41eab0465ca1c93acdb6` |
| `implementation-check.md` | `7174da8de32da900686f08a6ac2de881d4d3bddea28b519182aababe05d06ccf` |

Main owns user correspondence, final source freeze, resource admission and any
authorized launch; this review adds no approval gate or outcome-based stopping
rule.
