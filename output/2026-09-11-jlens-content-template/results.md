# PC4 content × template result: the proposed semantic gloss fails here

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

**Only 1 of 8 predeclared PC4 differences is positive; 7 are negative.**
The strong prediction was positive in all eight cells, so it failed.
The model was not trained or refitted. This is a new prospective test of our
outcome-informed interpretation, not a new general J-Lens quality score.

## What was tested

Four contrasts share actor and object: examined/readied, logged/fulfilled,
inventoried/dispensed, catalogued/packaged. Each appears in active and passive
syntax; every text ends `This happened yesterday.`. For example:

> The technician examined the samples. This happened yesterday.
>
> The technician readied the samples. This happened yesterday.

Using the unchanged fourth covariance direction, the advance prediction was
z(observation/recording) − z(provision/preparation) > 0. Those category names
are the hypothesis, not certified semantic labels. Exact texts, mapping and
prediction were committed before measurements in [the protocol](protocol.md).

## Every primary result

| O − P verb contrast | Active PC4 gap | Passive PC4 gap | Template signs |
|---|---:|---:|---|
| Examined − readied | −0.021032 | −0.003638 | negative / negative |
| Logged − fulfilled | −0.014417 | +0.008168 | negative / positive |
| Inventoried − dispensed | −0.017616 | −0.010234 | negative / negative |
| Catalogued − packaged | −0.025598 | −0.022520 | negative / negative |

Active: 0/4 positive; passive: 1/4. Three content contrasts have the opposite
sign in both templates; the clerk contrast changes sign. There are no exact
zeros or omitted cells. This is not merely a few close cancellations from
arithmetic: independent scalar projection error is at most 2.78×10⁻¹⁷,
versus the smallest absolute PC4 difference about 0.00364. This checks
saved arithmetic, not model floating-point or semantic robustness.

All other directions are retained descriptively: PC1 3 positive/5 negative,
PC2 6/2, PC3 2/6. Their template-sign changes are content03; contents03/04;
and contents01/02 respectively. These were not predicted alternative
success criteria. We do not replace PC4 with PC2, reverse the signs, or count
seven reversed results as success of the original claim.

## What this changes—and what it does not

The earlier [external-text PC4 positive](../2026-09-11-jlens-independent-content/results.md)
is unchanged: its two readers predicted 19/24 pair orderings correctly.
That task used 12 different natural-text pairs and reader interpretation of
the complete token lists. Here we directly test one human-readable gloss on
four authored content contrasts, expressed twice. These are different tasks
and cannot be pooled as an accuracy estimate. The earlier mixed comparison
against exemplar references also remains unchanged; no new advantage is claimed.

A live alternative is that the lists describe contextual, position-dependent
contrasts rather than sentence-level semantic invariants. Our shared final
sentence moves measurement away from the distinguishing verb. Alternatively,
the gloss could simply be mistaken, or the pattern could rely on other lexical
and contextual covariates. The present test does not distinguish these causes.
It is also not a direct intervention on the model's activation or output.

Next assess a small, separately specified endpoint/location comparison using
new shortened prefixes and the already saved final-position measurements.
Do not rerun these completed full-prefix forwards, select new favourable verbs,
refit the directions or change the completed result. No such follow-up is
implemented or launched in this report. The point is to distinguish a limited
useful readout from a misleading semantic explanation, not rescue a score.

## Evidence, execution and limits

Single cached Qwen/Qwen3.5-0.8B revision, unchanged post-block11 final-position
activations and four canonical U32 directions; [exact method](protocol.md).
Dataset/pair/protocol commit81bf8ae; source/test freeze437359a. A source-only
JSON wrapper mismatch was fixed before any real stage;13fabricatedtests pass.
Tokenizer stage ran once, all16texts10–13tokens. Its receipt is
ff398d5a0816cedb6566a867ea4d036666db3b60777a57a23c9aea5c5bf366ab,
frozen4bb63d1 before model execution. No special tokens, padding or truncation.

Single new forward run: 01:02:20–01:02:31 UTC, 10.69s measured stage time,
16forwards, parameter version stamps/eval/frozen/gradient-free checks unchanged.
Unit j-lens-content-template-forward-20260911-MMxZIs.service,
invocation05ac20612bd247c890cdb0d884dc66e8, MainPID0/success/exit0.
Peak PyTorch allocation1,549,027,840bytes, reserved1,614,807,040bytes;
not a total-process GPU-memory bound. No paid spend/reservation ($0/$100).

Most direct raw evidence in the worker:

- All64 scores (artifact not distributed in this public snapshot), SHA3d363e6a9ba927e79b7698d50a31115fe128762be8e1809be799b6025422c8eb.
- All32 gaps (artifact not distributed in this public snapshot), SHAe0ccddc94ef0144b44be2aefabaa7a48a01601cc8fe06de37502d15e160aa822.
- Saved activations and scores (artifact not distributed in this public snapshot), SHA7906f9295c531b8cc540c1c2b2a35e796cdc530b51b73404cdfd60ea8ef5c309.
- Producer receipt (artifact not distributed in this public snapshot), SHA6308dd68c456b1cb0fc75df576c58d17fcee98834e23642f55b1e2c0dba21f02.
- Separate scalar corroboration (artifact not distributed in this public snapshot), SHA30f3a954b317b1032529e2d1cbc185748fc4d764a15adb3ffb8ad2783bd0965e,
  checks64projections,32exact saved-score subtractions/gaps/signs and all JSON/input identities.

This is provisional, one model/layer, four authored contents and two templates,
not eight independent content pairs, a multi-seed result, or human validation.
Verbs change lexical identity/tokenization along with meaning; active/passive
changes length, order and distance to the endpoint. Shared endings control
terminal-string identity, not its contextual activation. No p-values, training,
new PCA, lens decode, new judge, general cluster or optimizer/safety claim.
Every acquisition/calculation handle above is consumed and must not be restarted.
