# Fixed-score J-Lens pattern calibration — prospective protocol

Codex — Spectral Optimizer Investigation · 11 September 2026

## Decision and hypothesis

Test whether a description calibrated on relevant activation differences helps
readers predict the **same original direction scores** on new natural text.
Control U gets the original raw examples plus original signed direction tokens.
Treatment P gets the identical raw examples plus signed calibrated-pattern
tokens. Both have12 tokens per pole and all four original axes. No PCA refit,
new score direction, optimizer training, semantic-label selection or clustering.

The [preceding synthesis](../../research/jlens_usecase_synthesis_2026-09-11.md)
motivates a population-dependent activation pattern. We select a **within-topic
pair-difference population**, because the evaluation asks which member of such
a pair has a higher score. This is not pooled raw-text covariance or separately
estimated category-specific patterns. The vocabulary description, not the
object being predicted, changes. All previous findings remain unchanged.

Hypothesis: this task-matched calibration yields positive P−U prediction credit
in both fresh reader groups and beats each calibrated reader's stronger fixed-
position control. This is an empirical pilot criterion, not a theorem implied
by calibration or a population-significance threshold. More appealing words,
positive calibration fit or input-space alignment alone do not count as success.

This document admits new preparation under existing user autonomy, not an
unreviewed acquisition/model launch. Source, fixtures, exact manifests and
current resources must be checked before their respective new stages. No
additional user approval round or issue is needed. Old stages remain consumed.

## Population, catalogue and fixed split

Use only the already saved Wikipedia namespace-zero catalogue:
`../2026-09-11-jlens-independent-content/catalogue/catalogue.json`, SHA
`c98cebf2222fc93c0cee48e4442af277cfa8743f091594a964e50a14ca86b251`.
First-root ownership order is astronomy, cooking, football, programming.
Remove every previously requested article ID, including technical skips and
copied-event metadata without double-counting an event. Do not remove by title,
semantic suitability, previous observed error or expected score.

Metadata feasibility finds173 remaining IDs:40 astronomy,26 cooking,0 football,
107 programming;86 disjoint within-topic pairs plus one unpaired programming
tail. This proves capacity, not current lead/token eligibility. Bind the full
old-request inventory and source receipts before producing the new manifest.
The frame is this captured, unbalanced remaining catalogue, not general prose,
topic balance or data unseen during model/reader pretraining.

Use SHA256 of UTF-8 strings, lowercase hexadecimal lexical ordering; decimal
page IDs without padding. Seed string `20260921`. Before any article request:

1. Within each owned/unrequested topic, sort by
   `20260921|candidate|<topic>|<pageid>`, ties by page ID.
2. Pair adjacent entries0–1,2–3,...; freeze each odd tail unused, not a spare.
3. Sort pairs globally by
   `20260921|pair|<topic>|<left_pageid>|<right_pageid>`, ties by topic rank,
   left ID, right ID. Assign candidate IDsK001 onward.
4. Assign roles by zero-based global index modulo3:0/1 calibration,2 evaluation.
   These roles never change after eligibility inspection. This gives58 candidate
   calibration pairs and28 candidate evaluation pairs before technical checks.

Target32 accepted calibration pairs (64 prefixes) and16 accepted evaluation
pairs (32 prefixes), with no document overlap anywhere. Visit the manifest in
order. If a role is already full, record a capacity skip and make no request.
Otherwise request both members, accept the whole pair only when both are
eligible, and continue within its original role if rejected. Never re-pair a
survivor, promote evaluation data into calibration or reroll the seed.
Hard cap80 actually requested pairs/160 article requests; exhausting that cap
or the manifest without both targets stops before model work. Role labels
and technical replacement rules are prospective, not chosen from outcomes.

## Source collection and observation rule

Use the same unauthenticated English Wikipedia API request and exact technical
eligibility rules as the completed [natural add-on protocol](../2026-09-11-jlens-natural-addon/protocol.md),
with these explicit new limits:160 article requests,3600seconds total,
256MiB host/no swap,32MiB new-stage output limit with failure-record reserve.
Per-request20seconds/2MiB, serial requests, fixed descriptive User-Agent,
identity encoding, no HTTP/API redirect following or automatic retries.
Retain API-advertised older-revision continuation metadata without following it;
any other unexpected continuation/schema/transport condition stops the stage.

Require namespace0, exact requested ID, nonempty title and valid source URL/
revision metadata, no missing/redirect/disambiguation marker, and at least16
Unicode whitespace words. Prefix is exactly `' '.join(extract.split()[:16])`.
No grammatical/semantic/terminal-word filters, manual edits, sentence repairs
or category quotas. Exact prefix duplicates within a candidate or against any
previously accepted calibration/evaluation prefix or old measured prefix reject
the whole candidate. Rejected strings are not reserved against later candidates.
Do not omit the second request merely because the first is technically ineligible.

Save exact requests/responses/eligibility decisions, every candidate disposition,
separate calibration/evaluation datasets and pair maps, source attribution and
a complete receipt. Label accepted pairsC01..C32 andT01..T16 in acceptance order;
each has L/R rows. Dataset fields are id,topic,prefix; pair fields are
id,left,right,topic,candidate_id,role. All96 accepted rows are fixed before
tokenizer/model work; a later failure cannot trigger replacement articles.
Root may see collected text, but no outcome may change this design.

Before redistributing selected excerpts, preserve article/history/contributor
and applicable additional attribution notices under the same source-review
discipline. No copied illustrations; excerpt-derived content and our research
code retain distinct terms. A rights concern limits redistribution, not
scientific replacement. API extracts are not certified atomic revision snapshots.

## Model, split isolation and measurement order

Fixed model: Qwen/Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`; adapter source revision
`581d398613e5602a5af361e1c34d3a92ea82ba8e`; layer11 post-block, activation
width1024. Original mean and U32 remain in directions archive SHA
`47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa`.
Use cached weights only, BF16/eager/eval/no_grad, no training or parameter updates.
At each exact prefix capture only its final actual input subtoken. No suffix,
chat wrapper, alternate position, completion, masking change or augmentation.

One whole96-prefix tokenizer preflight, no padding/truncation/added special
tokens, cap96 subword tokens each. Save IDs/masks/offsets and exact tokenizer,
Python/glibc and numerical-package metadata before model admission. All96 must
pass; no selected-row substitution. Do not run the previous preflight again.

Separate consumed stages, each once and with an explicit new path/attempt:

1. Capture the64 calibration states only; calculate/freeze four patterns from
   their32 fixed pairs; decode the8 signed normalized pattern vectors. Commit
   all new references and their input/source/receipt pins. No evaluation
   state, score, pair gap or reader outcome is allowed into this stage.
2. After that reference commit, capture the32 evaluation states once, with
   original scores `s_j=(float64(h32)−source_mean64)·float64(u32_j)` and exact
   L−R score differences. Keep the scientific key unopened by the root/readers
   until all responses are locked. No calibration updates from these states.
3. Four fresh readers, followed by the response freeze and one saved-array
   consistency check/one grade. Do not use the new key to decide whether
   judging is worth running; all256 judgments are committed to by this design.

Calibration states and reader evaluation use disjoint articles; both are frozen
before selection by score. No fit on the old error cases. Artifact separation
and controlled code paths implement the split, not merely a promise in prose.

## Exact calibrated pattern

For each fixed calibration pair i, using float64 arithmetic on saved h32 and U32:

```text
δh_i = float64(h_L,i) − float64(h_R,i)
δz_ij = δh_iᵀ float64(u_j)
r_j = Σ_i δh_i δz_ij
v_j = Σ_i (δz_ij)²
d_j = r_j / v_j
```

Every pair has unit weight; no clipping, ridge, gap threshold, norm reweighting,
topic balancing, intercept fitting or discarded axes. These are zero-intercept
least-squares patterns for pair differences, equivalently the covariance
patterns of the symmetrized empirical population{+δh_i,−δh_i}. In exact
arithmetic u_jᵀd_j=1. Independent sign reversals of whole calibration pairs
leave r and v unchanged. The pattern is not a new orthogonal PCA direction.

Under IID independent same-topic draws, the pair second moment is proportional
to a topic-weighted within-topic covariance. We do not assume that idealization
for this finite mechanically paired catalogue: the pair distribution itself
is the target. Large |δz| have more influence on this regression, whereas the
later accuracy weights each test item equally. Report that mismatch and the
per-axis largest fraction of v_j contributed by one calibration pair. Do not
change weights after observing those diagnostics.

Numerical failure policy: require all inputs/accumulators/d_j finite and every
v_j strictly positive; require finite nonzero ||d_j||. **No positive minimum
variance threshold beyond zero.** Low positive variance remains in the test,
not an axis-selection opportunity. If any axis is invalid, preserve the
whole failed stage; do not replace it with u, a smaller corpus or another fit.
Use a direct float64 batch identity in fabricated fixtures and later a saved-
array checker to verify the streaming arithmetic; never recompute model states.

Normalize each d_j in float64, cast to float32, and preserve exact ±inputs.
The scalar normalization preserves signed direction, not unit-score amplitude;
save unnormalized d/r/v as well. It matches the original unit-direction display
convention. Moment accumulation uses O(mk) extra memory/work per pair, with
m1024,k4, excluding model cost and archival storage.

## Lens and public references

Use the unchanged cached J-Lens file
`qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt`,
repository revision`0731326edff4ae730ffc5356fe1a4728c748b3a6`, SHA
`aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48`.
The pinned adapter transports float32 residuals with the float32 lens matrix,
then casts to the BF16 head dtype **before** final RMS normalization and
unembedding. Retain that actual pipeline, not the ideal linear surrogate.
Load through the existing `weights_only=True` method; no downloaded/new lens.

For each signed pattern, use exactly one `logits.float().topk(12)` call as in
the original display. Preserve selected IDs, decoded strings, scores and the
full logit vector; do not sort ties into preferable words or retry them.
PyTorch does not guarantee stable tied indices: these are captured once and
then fixed, not claimed stable under repeated runs. Report cutoff tie counts.
Keep all Unicode, whitespace, duplicates and fragments; no translation/gloss.

Control uses the original reference JSON byte-for-byte, SHA
`0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c`;
never rerun its decoder. Both arms contain the same original C example strings
and12 signed vocabulary strings per pole. Only P substitutes the newly captured
pattern lists for the original A lists. Same public field names in both arms:
`example_prefix` and `direction_tokens`. No calibration exemplar is shown.
More calibration data is intentional; a win would not prove equal-data or
equal-compute efficiency. Original/new runtime histories remain explicit.

## Reader assignment, locks and criteria

Four fresh history-free readers: rater1/3 P, rater2/4 U; cohorts(1,2) and(3,4).
One homogeneous arm per reader, four anonymous axis blocks ×16 pairs=64 choices.
Same targets/orientations/orders within a cohort; opposite FIRST/SECOND target
orientation across cohorts. Poles/score signs never flip. Exact local random
stream seed20260922; freeze its serialization/label/order algorithm and all
public transports before dispatch. Seed alone is not a complete specification.

Use the same common higher-direction-value question and FIRST/SECOND-only
responses as the previous comparison, without naming method/axis or suggesting
one list is preferable. No confidence, explanations, tools or history; maximum
two concurrent reader calls, first final only, no repairs/replacements or votes.
Record actual model/reasoning/settings and tool messages; instructional
isolation is not physical removal of tool capability. Exact new public-response
schema and transport source require review before dispatch. Commit all four raw
first responses, then all four validated response copies plus lock before
opening evaluation arrays/keys and running the new checker/grade.

Exact nonzero score gaps determine the higher member; exact ties get half credit.
No small-gap exclusion. Primary P−U summed over all four axes, out128 per arm;
also each cohort difference out64. A useful pilot requires strictly positive
gain in **both** cohorts and each P reader strictly above the maximum of its
realized always-FIRST/always-SECOND credits. Report both constants, every axis,
category/pair/reader and all gains/harms/agreement. Do not substitute a positive
axis for a failed all-axis primary. Pooled opposite-order constants are50%,
individual constants need not be. Same-family reader/arm/order effects remain.

A positive supports task-matched description calibration on this panel, not
clusters, causal interpretability or a general decoder. A negative limits this
calibration recipe, not the algebra or previous local usefulness. No automatic
new panel, alternative weighting, broader PCA fit or model training follows.

## Resource envelope and evidence handoff

Preparation is local metadata/source work. Future tokenizer: CPUQuota100%,
4GiB host/no swap,180seconds, CUDA hidden/offline. Calibration/decoder lane:
CPUQuota100%,8GiB host/no swap,900seconds, offline; evaluation lane same with
600seconds. Single numerical-library threads and8GiB PyTorch allocator limit;
the allocator limit is not a whole-process GPU cap. Fresh GPU/host headroom
and exact runtime checks are mandatory before either new lane. No paid
compute/judge/API, downloads or training; spent/reserved$0/$0 under$100.

## Source checks and interpretation boundaries

The pattern formula applies the established filter/activation-pattern relation
to a symmetrized pair population; it is not a new theorem.
[Haufe author poster](https://f1000research-files.f1000.com/posters/docs/263125503).
The nonlinear readout qualification follows the [J-Lens method](https://transformer-circuits.pub/2026/workspace/index.html)
and the exact local adapter source. Fixed historical PyTorch2.11 APIs are
checked against [topk](https://docs.pytorch.org/docs/2.11/generated/torch.topk.html)
and [safe loading options](https://docs.pytorch.org/docs/2.11/generated/torch.load.html).
No novelty, alignment, safety reliability or capabilities-speedup claim is made.
