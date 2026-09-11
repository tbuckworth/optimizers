# Proposed PC4 content × template diagnostic

Codex — Spectral Optimizer Investigation · 11 September 2026

## Fixed prediction and stimulus logic

Exactly four content pairs share their actor and object. Each contrasts a
synonym for examining/recording with a synonym for readying/supplying; none of
the distinguishing verbs copies a prominent token from the PC4 readout.
Cross each pair with two fixed templates:

- **Active:** `The {actor} {verb} the {objects}. This happened yesterday.`
- **Passive:** `The {objects} were {verb} by the {actor}. This happened yesterday.`

All 16 end with the identical phrase `This happened yesterday.` including the
period. Within each pair/template, only the verb changes. The active/passive
versions retain actor, object, action and stated time, but change syntax and
length. A common terminal string does not make the contextual last-position
representation identical.

**Advance prediction:** for each of the four pairs in each template,
`PC4(observation/recording) − PC4(provision/preparation) > 0`.
These pole names express the author's proposed manipulation, not measured
ground truth or independently validated labels.

## Exact candidate prefixes

Each table row is two prefixes; IDs are `<pair>-<template>-O` and
`<pair>-<template>-P`. O is the predicted-higher side, not a label passed to
the model. Preserve the strings exactly; no wrappers or further endings.

| Pair / template | O: observation or recording | P: provision or preparation |
|---|---|---|
| 01-active | The technician examined the samples. This happened yesterday. | The technician readied the samples. This happened yesterday. |
| 01-passive | The samples were examined by the technician. This happened yesterday. | The samples were readied by the technician. This happened yesterday. |
| 02-active | The clerk logged the requests. This happened yesterday. | The clerk fulfilled the requests. This happened yesterday. |
| 02-passive | The requests were logged by the clerk. This happened yesterday. | The requests were fulfilled by the clerk. This happened yesterday. |
| 03-active | The attendant inventoried the materials. This happened yesterday. | The attendant dispensed the materials. This happened yesterday. |
| 03-passive | The materials were inventoried by the attendant. This happened yesterday. | The materials were dispensed by the attendant. This happened yesterday. |
| 04-active | The curator catalogued the artifacts. This happened yesterday. | The curator packaged the artifacts. This happened yesterday. |
| 04-passive | The artifacts were catalogued by the curator. This happened yesterday. | The artifacts were packaged by the curator. This happened yesterday. |

## Measurement specification, if separately admitted

Freeze the final text JSON, pair/template map, source and protocol hashes before
tokenization/measurement. Fixed order: pair 01–04, active then passive, O then P.
No text changes, replacements, template selection, sign changes or exclusions
after scores. If the whole-panel tokenizer preflight fails, stop; do not drop
or truncate a stimulus.

Use the existing cached `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`, same tokenizer with no added special
tokens/chat wrapper/truncation, and the unchanged text-forward contract:
eval, frozen parameters, no gradients/cache, bfloat16 model, unmodified
post-block **layer 11, last input position** converted to h32. Record exact
input IDs/all-valid masks; require 1–96 tokens for every prefix. These are new
8-/10-word stimuli, not inputs to an old 16-word-roster CLI.

Reuse actual canonical U32 and original `source_mean64` from
preparation/directions.npz (artifact not distributed in this public snapshot),
SHA256 `47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa`.
For every prefix save all four original scores
`z = (float64(h32) − source_mean64) @ float64(U32)`.
No PCA, new mean, direction normalization, reference decoding or new judging.
Keep the existing A/B/C descriptions untouched; they motivate the diagnostic
but are not additional measured arms here.

Any admitted run would need a new guarded forwards-only adapter and fresh
exclusive preflight/measurement outputs, not a restart of a consumed handle.
Sixteen new prefix forwards, one cached model load, local bounded compute;
record immutable model/input/source provenance and preserve failures. This
document does not authorize that code or execution.

## Limits that the design does not remove

The author has seen the readout and prior outcomes. The proposed pairs are
expectation-informed, not independently authored validation. Verbs inevitably
change lexical identity, tokenization, frequency and action meaning together;
four synonyms cannot isolate an abstract semantic feature. Active/passive
syntax also changes length, word order and distance from the final verb to the
readout position. Shared endings remove terminal-string differences, not all
endpoint/context effects. Examining can overlap PC3's evaluation theme, while
packaging can overlap its physical-wrapping theme; retaining all four axes
helps display such overlap but does not make them causal controls.