# Proposed endpoint/location diagnostic

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

## One primary criterion, fixed now

For each of the eight unchanged content/template cells, let
Δ₄(location) = z₄(O, location) − z₄(P, location).

**Primary success requires Δ₄(verb) > 0 in all eight cells.** Zero fails.
The verb location is chosen before any new outcomes; sentence-end improvement
cannot substitute for this criterion. If it passes, the seven previously
negative full-end cells necessarily change sign at the verb. This supports
a *verb-position-specific* version of the gloss on these four contrasts,
not sentence-level invariance or a causal explanation of the earlier result.

Report every gap at all three locations, positive/zero/negative counts and
active/passive sign consistency. Also display each cell's signed
Δ₄(verb) − Δ₄(old-full-end) and Δ₄(sentence-end) − Δ₄(old-full-end);
these are descriptive shifts, not alternative success tests. Keep PC1–PC3
at all locations descriptively. No best-location/axis selection, sign reversal,
exclusion, p-value or changed criterion after measurement.

## Exact sixteen new inputs and verb spans

Derive each string from the frozen dataset (artifact not distributed in this public snapshot)
by requiring and removing exactly the terminal suffix
` This happened yesterday.` once. Preserve every other character, ID,
punctuation and row order; retain the existing eight pairs (artifact not distributed in this public snapshot).
Spans below are zero-based, half-open character offsets in the shortened ASCII
string; they exclude surrounding spaces.

| Prior row ID | New first-sentence-only text | Verb | Character span |
|---|---|---|---|
| 01-active-O | `The technician examined the samples.` | `examined` | 15, 23) |
| 01-active-P | `The technician readied the samples.` | `readied` | [15, 22) |
| 01-passive-O | `The samples were examined by the technician.` | `examined` | [17, 25) |
| 01-passive-P | `The samples were readied by the technician.` | `readied` | [17, 24) |
| 02-active-O | `The clerk logged the requests.` | `logged` | [10, 16) |
| 02-active-P | `The clerk fulfilled the requests.` | `fulfilled` | [10, 19) |
| 02-passive-O | `The requests were logged by the clerk.` | `logged` | [18, 24) |
| 02-passive-P | `The requests were fulfilled by the clerk.` | `fulfilled` | [18, 27) |
| 03-active-O | `The attendant inventoried the materials.` | `inventoried` | [14, 25) |
| 03-active-P | `The attendant dispensed the materials.` | `dispensed` | [14, 23) |
| 03-passive-O | `The materials were inventoried by the attendant.` | `inventoried` | [19, 30) |
| 03-passive-P | `The materials were dispensed by the attendant.` | `dispensed` | [19, 28) |
| 04-active-O | `The curator catalogued the artifacts.` | `catalogued` | [12, 22) |
| 04-active-P | `The curator packaged the artifacts.` | `packaged` | [12, 20) |
| 04-passive-O | `The artifacts were catalogued by the curator.` | `catalogued` | [19, 29) |
| 04-passive-P | `The artifacts were packaged by the curator.` | `packaged` | [19, 27) |

Source dataset SHA256:
`841d86753f857d686165e53d76278bc1fa40840f97e09ffad61f0c89d57fcc7d`;
pairs SHA256:
`d47d27b93e48b38deb5e4871596c906057ba46e18305e3849b02ad1b0e337ac0`.

## Minimal proposed measurement

One frozen cached model load, sixteen new shortened-input forwards, observing
two positions from the same unmodified post-block-11 output per text:

- **verb:** final subtoken intersecting the fixed verb span;
- **sentence-end:** final input token, including the sentence's terminal period.

A separate CPU preflight would use the unchanged pinned fast tokenizer with
offset mappings, no added special tokens/padding/truncation, and the whole-panel
1–96-token limit. Require each character slice to equal its listed verb,
nonempty contiguous overlapping token coverage of that span, and the final
overlapping token's character end to equal the verb-span end. A leading-space
token is allowed. If mapping fails or merges past the verb, stop the whole
panel as a technical failure; do not choose a nearby position. Save full IDs,
masks, offsets, selected indices and exact covered substrings before model work.
Require the last token's offset to cover the final period and end at the exact
string length. Require all shortened-text IDs to equal the corresponding prefix
of the old pinned full-text token IDs, so end-of-string tokenization cannot
silently alter the shared first sentence. On mismatch stop, without replacement.
The old [token file (artifact not distributed in this public snapshot)
SHA256 is `c276357f0b89efa3c8a2f54d836d852e24535053401c1af516338a9b2207f86f`;
its preflight receipt SHA256 is
`ff398d5a0816cedb6566a867ea4d036666db3b60777a57a23c9aea5c5bf366ab`.

Keep cached Qwen/Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, BF16/eager, eval/frozen/no_grad,
unchanged text adapter, no cache/generation. Convert both captured positions to
h32. Reuse canonical U32 and source_mean64 from the archive pinned by the
[previous protocol](../2026-09-11-jlens-content-template/protocol.md), without
renormalization or PCA. At each position compute
z = (float64(h32) − source_mean64) @ float64(U32);
compute O−P by subtracting those saved score64 rows. Retain 128 new scalar
scores and 64 new gaps across both locations/all four axes.

The third location uses only the already saved full-prefix final-position
scores (artifact not distributed in this public snapshot)
(SHA256 `3d363e6a9ba927e79b7698d50a31115fe128762be8e1809be799b6025422c8eb`);
its receipt (artifact not distributed in this public snapshot)
is pinned to `6308dd68c456b1cb0fc75df576c58d17fcee98834e23642f55b1e2c0dba21f02`.
No old full-prefix forward is repeated. All stages would use new exclusive
paths and receipts, without automatic retry, decoding, readers, training,
new verbs, paid compute or extra controls.

## What this can and cannot distinguish

This cheaply distinguishes a gloss visible consistently at the verb from one
that also fails there, and shows whether the observed ordering changes between
verb, sentence end and the already measured neutral ending. It tests the
strongest concrete local prediction, not an unconstrained search for any change.

In a causal decoder the verb-position state sees only the prefix through that
subtoken. Active syntax has exposed the actor but not the object; passive
syntax has exposed the object but not the actor. The sentence-end state has
both. Thus this is **not fixed meaning with only measurement location changed**:
available causal context, token identity and position change together. The
removed ending also removes an assertion about time, not merely empty padding.
Even if causal-prefix invariance holds mathematically for fixed token IDs,
different sequence lengths/BF16 execution paths do not guarantee bit-identical
states across runs; no such equivalence is asserted or tested by rerunning.