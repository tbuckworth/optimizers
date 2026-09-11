# Natural add-on measurement source preparation

Source and fabricated fixtures only. No actual new token preflight, model forward, saved-array checker, readers or grading was executed by this worker. Main owns release and resource admission. All older stages remain consumed.

## Producer ready for review

`forward.py` retains the fixed model/cache/revision, BF16 eager/eval/frozen parameters, `torch.no_grad`, unmodified post-block 11 observation and canonical float64 centered score rule. It has new exclusive `token-preflight` and `forwards` entrypoints with new schemas and release variables, and never calls an older entrypoint or decodes references. Before/after pins bind the completed collection receipt, exact dataset/pairs, collector/protocol, original directions/mean, references and cached model/tokenizer files. New collection hashes supplied by main were independently checked byte-only, without parsing article text or scientific arrays.

Exact input contract: the 32 bare-list rows `N01-L`, `N01-R`, …, `N16-R`, each exactly 16 normalized whitespace words; sixteen corresponding same-topic pairs with preserved increasing candidate IDs. No authored verb/template/old 24-row assumptions remain. Whole-panel tokenizer preflight forbids added specials, truncation or padding and requires 1–96 valid token IDs, all-one masks and exact retained character offsets. Unicode byte subtokens may share monotone character spans; all non-whitespace text must remain covered, and the last actual token's span must reach the prefix end. Captured position is always `len(input_ids)-1`, never an earlier character-selected token.

Outputs preserve singleton role `prefix_end`: `activation_11` float32 shape `(32,1,1024)`, `scores64` float64 `(32,1,4)`, `gaps64` float64 `(16,1,4)`, plus `locations=['prefix_end']`. Gaps are subtraction of saved scores, original left minus right. JSON scores use `jlens_natural_addon_scores_v1` with `locations.prefix_end` holding all four axis/value dictionaries; inputs/tokens preserve full IDs, masks, offsets, captured positions and selected substrings. Receipt scope includes all 32 forwards, unchanged parameter versions, zero reference decodes/refits, environment/cache provenance and allocated/reserved GPU peaks. The 8 GiB allocator bound is not a whole-process GPU cap.

## Checker intentionally not released

`check.py` is a source-only scaffold. Its `FORWARD_SHA` and `JUDGING_SHA` are explicitly `None`; it fails closed until main-reviewed producer and public-only response-lock verifier bytes are frozen. The intended interface is `judging.verify_response_lock(*, packets, manifest_sha256, responses, lock_sha256, lock_commit)`. Real execution must establish all 256 committed choices before even importing producer helpers or reading measurement/array files. The scaffold currently assumes the prior verifier's returned manifest `inputs` and `FROZEN` mapping; that interface must be checked against the eventual new main judging source, not assumed released.

After that gate, the checker validates input/token/receipt/export identities, corroborates all 128 scalar projections with independent `math.fsum`, and all 64 gaps by exact saved-score subtraction. It contains no reader grading or authored semantic success criterion. All axis sign counts are descriptive only. Tests mock the gate and construct only temporary synthetic archives.

## Reads and fabricated checks

The old external producer and later fresh-authored producer/checker/tests were read fully before adapting their source. No old producer is imported by the new producer. The best-practices validator skill informed local installed documentation checks (`torch.no_grad` and Transformers tokenizer defaults/offset arguments); no network was used. `no_grad` is combined with explicit eval/frozen parameters, not treated as an eval substitute. All local edits used `apply_patch`.

Commands ran with CUDA hidden, OMP/OpenBLAS threads 1, bytecode disabled and a 60-second timeout:

```sh
/usr/bin/python3 -m unittest discover -s output/2026-09-11-j-lens-natural-addon -p test_forward.py -v
/usr/bin/python3 -m unittest discover -s output/2026-09-11-j-lens-natural-addon -p test_check.py -v
```

Producer: **14 PASS, 2.612 s**. Checker scaffold: **10 PASS, 2.923 s**. Coverage includes inert import, roster/collection binding, Unicode shared offsets, bad/late offsets and whole-panel failure, last-position hook/no mutation, dtype/finite/norm checks, exact score/gap orientation, consumed/dangling stages, missing preflight before arrays/model, locked judged-token binding, missing lock before producer import, JSON tampering and exclusive checker outputs. `git diff --check` passed. Tests loaded no actual tokenizer/model or scientific archive.

| New source | SHA256 |
|---|---|
| forward.py | `23c9c7cf77f2239bd16b219c440fbf056628730cd2bdf51f66b812ab0e8f580c` |
| test_forward.py | `c817d86f044323266530da5249b1d1ab6cc66857b0485002a03742927d480a1d` |
| check.py, disabled scaffold | `f6043d8884c82d9dcdd9c72ffce14264928dd84f315e0c5889a1211e7c234a0f` |
| test_check.py | `403743bcb5c641066f8c34c72d2e50da386ac4c649cf0d1f9b091b684b038379` |

No paid compute or new reservation. No source, result, inventory or collection artifact from a completed stage was changed.

## Main's completed tokenizer preflight

Main accepted the full producer and independently passed 14 producer fixtures (3.236 s) and 10 checker-scaffold fixtures (3.367 s), with unchanged source hashes; acceptance recorded in main commit `95ee9c5`. Main then ran exactly one actual preflight: service `j-lens-natural-addon-preflight-20260911-MMxZIs`, invocation `93284f08ca9e41fcb5773b61473a46f5`. Completed `2026-09-11T05:13:50.923283+00:00`, elapsed 4.971718 s, terminal MainPID 0/success. CPU quota 1, 4 GiB host/no swap, 120-second ceiling, CUDA hidden/offline; no model or scientific arrays loaded. Main verified all 32 exact text/ID/mask/offset/final-position records, 16–43 tokens each.

Worker bookkeeping only: confirmed the receipt's source, scope, invocation and output digest; independently rehashed the three generated files and unchanged producer. Preserved bytes:

- `token-preflight/attempt.json`: `00bacf6148e691de024d606e74ce576cd268c74f96f4a64fb2f1f74f056ec0ec`
- `token-preflight/receipt.json`: `675411e8428e5a8797101b0bed85a4604a4e59d21fe41b40c498a3e330c492fb`
- `token-preflight/tokens.json`: `e87f996149ea1eb80b3fb3a6225257859f047aefaf7727beb1e792b863e0ef43`

The preflight is consumed and must not repeat. No executable source changed. Neural forwards are not released at this checkpoint; the new judging/public-lock implementation and checker pin review remain outstanding.