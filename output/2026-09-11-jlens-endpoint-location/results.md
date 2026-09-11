# Endpoint result: a useful local PC4 contrast, not an endpoint-invariant label

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

**The prospective primary prediction passes: all 8 PC4 O−P gaps are positive
at the verb.** All8 are also positive at the first sentence's end. The already
saved later common-ending result remains1positive/7negative. No words, signs,
directions or outcome criterion were changed after these new measurements.

This is evidence for the useful local version of the observation/provision
interpretation on these contrasts. It is not a generally reliable semantic
label that can be read at arbitrary later endpoints, a causal feature, or a new
claim that spectral summaries outperform ordinary example-based interpretation.

## Exact comparison

The new texts remove only ` This happened yesterday.` from each prior input.
One new forward per shortened text captures the final subtoken of the action
verb and the final period. Prior full-prefix final scores are reused unchanged.
For example, `The technician examined the samples.` is compared with
`The technician readied the samples.`; the corresponding passive pair is kept.
Each new token sequence exactly matches the original token sequence's prefix.

The unchanged PC4 direction predicts O(observing/recording) > P(providing/
preparing). These names express the proposed semantic manipulation, not
independently certified labels. The [protocol](protocol.md) fixed the
verb-final-subtoken location and all-eight-positive criterion in advance.
Sentence end was descriptive, not a second opportunity to pass.

| Contrast / template | Verb: NEW, primary | Sentence end: NEW | Later ending: SAVED |
|---|---:|---:|---:|
| Examined − readied / active | +0.187247 | +0.085104 | −0.021032 |
| Examined − readied / passive | +0.182327 | +0.142004 | −0.003638 |
| Logged − fulfilled / active | +0.297515 | +0.089682 | −0.014417 |
| Logged − fulfilled / passive | +0.233971 | +0.038624 | +0.008168 |
| Inventoried − dispensed / active | +0.475912 | +0.047241 | −0.017616 |
| Inventoried − dispensed / passive | +0.443441 | +0.067740 | −0.010234 |
| Catalogued − packaged / active | +0.256843 | +0.016988 | −0.025598 |
| Catalogued − packaged / passive | +0.053343 | +0.009029 | −0.022520 |

Both new locations have4/4positive in each template, no exact zeros and no
active/passive sign changes. Every verb gap exceeds its first-sentence-end
gap; each in turn exceeds the corresponding saved later-ending gap. All
signed shifts are retained in [the complete comparison](analysis/results.json).
This is an observed ordering across the three measured roles, not evidence
that the score changes monotonically at every intervening token.

## All four directions, without selecting a new winner

Positive O−P gaps out of8; no zeros in any cell:

| Fixed direction | Verb | Sentence end | Later saved ending |
|---|---:|---:|---:|
| PC1 | 8 | 8 | 3 |
| PC2 | 3 | 4 | 6 |
| PC3 | 6 | 8 | 2 |
| PC4 (primary) | 8 | 8 | 1 |

The same contrasts also project positively on PC1 at both new locations and
on PC3 at sentence end. This is not a concept uniquely isolated by PC4 or
evidence of four distinct clusters. Orthogonal fitted directions need not
correspond to independent semantic factors. PC2 remains mixed rather than
being omitted from the account.

## Updated interpretation

The earlier failed gloss was not simply meaningless: its precise verb-local
prediction now holds across all four selected contrasts and both templates.
It also holds at the first sentence's period, when the full action proposition
has been supplied. It fails at the end of a later shared sentence. Thus a
context/endpoint-specific readout is a better supported working account than
either a position-invariant semantic label or universal uselessness.

This does **not** show that the model forgot the action, that elapsed token
distance caused the change, or that one specific mechanism rotates the feature.
As the mathematical note (artifact not distributed in this public snapshot) explains, a fixed scalar
projection can vanish or reverse while a multidimensional contrast remains.
Context, token identity and position differ across roles. The added sentence
also supplies time information. Causal verb prefixes lack the actor or object
that arrives later, depending on active/passive form. BF16 execution over
different sequence lengths is not asserted to be bit-identical.

The external-text [19/24 PC4 reader result](../2026-09-11-jlens-independent-content/results.md)
and prior [1/8 later-endpoint test](../2026-09-11-jlens-content-template/results.md)
both remain unchanged. They answer different questions and must not be pooled
with this count as one accuracy estimate. These four authored, reused content
contrasts expressed twice are not eight independent content replications.
Their expectation-informed choice, lexical/tokenization/frequency differences,
single small model/layer and two templates limit generalization.

Next use the already saved activation differences to distinguish reduced
contrast magnitude from a changed orientation relative to the fixed directions,
or a combination of both.
Specify that small calculation before execution. It would be descriptive
geometry, not semantic-information loss or a causal explanation; no new model,
judges, training or broader optimizer test is selected by this result.

## Direct evidence and execution

Inputs/protocol563bb3d, model adapter source1e50dcc. Main and independent review
passed13fabricatedtests; analyzer fixtures and location/input bindings checked
before real data. Final source SHA
f6b6289d484d2e832982b7a0f78cd60d6010b9d45ef80bc5724ab5cac36dda1e.
A non-executable scientific-digest scan annotation changed the pre-review hash;
main verified the entire source difference and corrected the provenance note.
No executable repair, real credential or hook bypass.

Single preflight: all16texts6–9tokens, complete verb spans, exact old-ID prefixes,
all-one masks and terminal-period coverage. No replacement/truncation. Receipt
006f52fac16dec76b93f6321b697dea781b1deaff8f9adf942e06d39fe00bcc6,
frozen176a11f before GPU execution. Captures for split verbs occur at their final
subtoken, not at a guessed common index; full contextual hidden states are used.

Single forward: 01:26:23–01:26:34UTC,10.61seconds,16newinputs/twocaptures each.
Unit j-lens-endpoint-location-forward-20260911-MMxZIs.service,
invocatione37ab6fdc9134ccc94b519f560ce864c, MainPID0/success/exit0. Cached
Qwen/Qwen3.5-0.8B, unchanged block11/U32/source mean, BF16/eager/eval/no_grad;
tracked parameter state unchanged. Peak PyTorch allocated1,548,936,704bytes,
reserved1,614,807,040bytes; not total GPU-process memory. Paid spent/reserved$0/$100.

- Producer receipt (artifact not distributed in this public snapshot): SHA4d76cce26a352fc64cace96a01cf2f8db29e2761368962ba0cae875396c51103.
- 128 new scalar scores (artifact not distributed in this public snapshot): SHA3a6fe901b23f5a3aa9ea197613af9598e117b0dc49a5034bb099aee357509c80.
- 64 new gaps (artifact not distributed in this public snapshot): SHA91630c41593d806f9760706712c7576bb858b2af799aea20229efc69d7cd9fc4.
- Both saved activation locations (artifact not distributed in this public snapshot): SHAcfd4ccb17f03498893635dbd9f6263e870f99cc9d87ee2fa1a38cad24c801539.
- [Independent scalar arithmetic and full three-location comparison](analysis/results.json): SHAe3c78d58b3e0fc0fe62ac5498a7263f8d5ee3ad3ef8a4e4d988affbafc07418e.
  All128projections/64gaps and location/input/export checks pass; maximum
  score discrepancy below 5.56e−17. This verifies arithmetic, not semantic correctness.

No new PCA, lens decode, reader, parameter update, old model replay or paid
service. All acquisition and analysis stages above are consumed.
