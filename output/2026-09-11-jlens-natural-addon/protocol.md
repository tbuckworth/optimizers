# Natural-text J-Lens add-on comparison — prospective protocol

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

## Question and scope

Does adding the unchanged signed direction-token lists to the original raw
fit examples improve prediction of new natural-prefix directional scores?
Control E gets the original examples; treatment EA gets exactly those examples
plus the original direction tokens. This is a practical add-on package, not
an equal-information/token-budget comparison, PCA-vs-no-PCA test, or semantic
ground-truth benchmark. Separate-format comparisons do not answer this question.
The [reviewed assessment](../../research/jlens_incremental_utility_2026-09-11.md)
motivates this design; retain all prior positive/mixed/negative evidence.

Existing user autonomy applies. No additional approval round, issue, scheduler,
optimizer/familiarity work, training, paid judge/cloud, broader covariance fit
or old-stage replay. This protocol fixes the design before candidate selection
and new excerpts; actual code and immutable input pins require main review
before their own new stages. No new excerpt/model/reader exists at this freeze.

## Existing-source inventory and candidate pairing

Use the saved catalogue only:
`../2026-09-11-jlens-independent-content/catalogue/catalogue.json`, SHA256
`c98cebf2222fc93c0cee48e4442af277cfa8743f091594a964e50a14ca86b251`.
Verify its exact response/receipt bindings. Root ownership order is astronomy,
cooking, football, programming. Assign each namespace-0 positive integer page
ID to its first root in that order, irrespective of exclusion; retain duplicate
ownership decisions. Then remove every previously requested article ID, including
technical skips and attempts that returned data before a later local failure.
Copied request metadata are retained but do not count as another network call.

Freeze a reviewed inventory with every source path/hash/size, request page ID,
deduplicated exclusion union, exact catalogue candidates and ownership counts.
The main independently checks that inventory before selection. Titles have
been visible during metadata inspection; title knowledge is not an exclusion
or selection criterion. The frame is the remaining captured catalogue, not
representative Wikipedia/prose or pretraining-unseen text. Football has no
remaining pages; do not replace that stratum with another category.

All hashes below are SHA256 of UTF-8 strings, hexadecimal lexical order;
page IDs are decimal without padding. Seed string is exactly `20260919`.

1. Within each owned, unexcluded topic, sort by
   `20260919|candidate|<topic>|<pageid>`, breaking hash ties by page ID.
2. Pair adjacent rows0–1,2–3,... in that fixed order. Retain an odd unpaired
   final candidate as explicitly unused, not a substitute for a failed pair.
3. Sort all pairs globally by
   `20260919|pair|<topic>|<left_pageid>|<right_pageid>`, with ties broken by
   original topic rank, left page ID, right page ID. Assign candidate IDs
   `K001`, `K002`, ... after this ordering.
4. Freeze the complete selection manifest, inventory/source/protocol hashes
   and odd-tail decisions before requesting excerpts. At least16 candidate
   pairs must exist. Neither topics nor pair scores determine sampling quotas.

## Bounded source collection and exact inputs

Visit candidate pairs in manifest order. Request the left then right article
of every visited pair, even if the left is technically ineligible, unless a
transport/API/schema/resource failure stops the entire stage. Never re-pair
survivors. Accept the first16 pairs whose two articles meet all technical rules,
under a hard cap of32 visited pairs/64 article requests. If fewer16 can be
assembled, retain the failure before any model call; no new seed or category.

Each request is a serial unauthenticated GET to `https://en.wikipedia.org/w/api.php`:
action=query, format=json, formatversion=2, maxlag=5, one pageids integer,
prop=extracts|revisions|info|pageprops, explaintext=1, exintro=1, exchars=1200,
rvprop=ids|timestamp, rvlimit=1, inprop=url. No API redirect resolution or
HTTP automatic redirects. Use a descriptive project/contact User-Agent,
Accept-Encoding:identity,20second per-request timeout,2MiB response cap and
16MiB new-stage output cap with reserved failure-record space. External
collection process also has a1500second total wall limit and256MiB host cap.
Stop on HTTP/API errors, warnings, unexpected continuation/compression,
malformed/duplicate-key/nonfinite JSON or resource exhaustion. Preserve exact
request metadata/response bytes, including explicit truncated prefixes when
a response exceeds the cap. No automatic retry or identity rotation.
The sole accepted continuation is a dictionary with exactly `rvcontinue` and
`continue`, both nonempty strings: `rvlimit=1` may advertise older revisions.
Retain that marker in the raw response but never follow it; it does not imply
an incomplete single requested lead. Any other module continuation stops.

An article must return the requested page ID in namespace0, a nonempty title,
valid HTTPS en.wikipedia.org URL, positive revision ID/timestamp, and at least
16 Unicode whitespace-separated extract words; no missing/invalid/redirect/
disambiguation marker. A different returned page ID is an integrity failure,
not a skippable replacement. The service's1200character request is not an
eligibility cutoff because the returned extract can be slightly longer.
No grammatical, topical, semantic, terminal-word or quality filters.

The prefix is exactly `' '.join(extract.split()[:16])`; no other editing,
completion, chat wrapper, verb selection or punctuation repair. Reject a pair
if its prefixes duplicate each other or any previously accepted or old measured
prefix. The old32/24-row formats differ; verify the actual old24-prefix list.
Do not reserve prefix strings from rejected pairs for excluding later pairs.
Keep per-article eligibility and pair-level reasons for every visited pair.
All old requested IDs remain excluded even if their old prefixes were not used.

Accepted IDs in collection order are `N01`...`N16`; row IDs are `N01-L/R`, etc.
Dataset rows have id, topic, prefix; pairs have id,left,right,topic,candidate_id.
Attribution entries retain selected/candidate IDs, requested page ID, title,
URL/history link, reported revision metadata, exact source-response pins and
the normalization/truncation notice. Store the full32-row dataset,16pair map,
all decisions and receipt; commit before model/reader work. All rows retained
after this point; tokenizer/model failures cannot trigger replacement articles.

Before redistributing excerpts, check the fixed selected articles/history/talk
for applicable attribution notices; preserve additional credit and source terms.
API extract/revision metadata alone is not an atomic historical snapshot or
rights certification. A rights issue limits redistribution, not scientific
replacement. Do not copy illustrations. Code/research and excerpt-derived
content have distinct source terms. Follow the same attribution discipline as
the [prior source review](../2026-09-11-jlens-independent-content/attribution-review.md).

## Fixed model and measurement

Same cached Qwen/Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, adapter
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, original source_mean64 and actual
U32 archive SHA `47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa`.
BF16/eager/eval/no_grad, post-block11, only the last actual input subtoken of
each exact16-word prefix. No later suffix, shortened alternate lane, refit,
normalization/sign change, decoder recalculation or model updates.

Review source and fabricated fixtures first. One full-roster tokenizer
preflight records exact IDs/masks/offsets with no added special tokens,
truncation or padding, cap96tokens per prefix. All32 must pass. Freeze inputs,
tokenizer receipt and scientific source before one32-prefix forward. Keep
source mean, U32, feature precision and scalar rule explicit:
`s=(float64(h32)-source_mean64)·float64(u32)` for each of four axes.
Preserve all128 scalars,64 original-left-minus-right gaps and features for a
separate saved-array check after reader locking. Do not inspect the new key
to decide whether judging is worth doing; all256 choices are committed to now.

Local resource ceilings: tokenizer CPUQuota100%,4GiB host/no swap,120seconds,
CUDA hidden/offline. Forward CPUQuota100%,8GiB host/no swap,600seconds, offline,
single numerical-library threads,8GiB PyTorch allocator bound. This is not a
full-process GPU-memory cap: check actual free GPU/host headroom first. No
retrying a consumed forward. Paid spent/reserved remains $0/$0 of$100.

## Exact reference layout and readers

Original reference JSON SHA
`0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c`.
Each unchanged axis has C.positive/C.negative raw fit strings, and A.positive/
A.negative12-element token-string lists. No B arm. For each anonymous block,
represent the two poles using exactly these public field structures:

```json
{"example_prefix":"<unchanged C string>"}
{"example_prefix":"<unchanged C string>","direction_tokens":["<12 unchanged A strings>"]}
```

The first structure is E, the second EA. Preserve all Unicode, spaces,
duplicates, empty fragments and newlines, including PC4 blocked/combined fit
examples. Do not translate, paraphrase or add the retrospective semantic gloss.
Public fields positive_reference and negative_reference retain their original
signed meaning across ALL readers; do not reverse poles or score signs.
Only target FIRST/SECOND ordering reverses across cohorts. Anonymous block
labels hide axis identity. A private map records every axis/reference/target
join; no source/model/axis/arm names or scores reach readers.

Readers1/3 receive homogeneous EA; readers2/4 homogeneous E. Cohorts1=(1,2),
2=(3,4). Each reader gets four anonymous blocks ×16pairs=64choices. Use one
local `random.Random(20260920)` stream for all target orientation/label/order
choices, no rerolls. Exact algorithm/source/packets must be reviewed and frozen
before dispatch; random seed alone is not an exact transport specification.
Matched E/EA target orientations within each cohort, unchanged reference poles
everywhere and opposite target orientations across cohorts produce pooled
balanced constants. Public arms are not named to readers.

Common question: “Using only the two reference descriptions, which of the two
prefixes should have the higher value on this direction? Choose FIRST or SECOND
even if uncertain.” No confidence/explanation fields. Both packets define
that example_prefix is a reference example for that pole; when supplied,
direction_tokens are additional vocabulary-associated descriptions of that same
pole. No instruction that added tokens are more accurate or should override
examples. Targets contain the exact complete16-word prefixes, no article titles.

Use four fresh history-free reader calls, no model/reasoning overrides,
no tools requested, at most two concurrent in order1..4. One first final response
each, no follow-up/repair/replacement or majority selection. Exact schema
`jlens_natural_addon_responses_v1`, packet_id, and64 unique {item_id,choice}
records, FIRST/SECOND only. Freeze exact common wrapper and serialized public
prompts with hashes before dispatch. Record actual handles/configuration and
any observed tool messages; instruction-level isolation is not physical removal.
Commit all four raw responses, then seal exact copies and commit all five
lock/response blobs BEFORE any score/key/array inspection or grading.

## Estimand, controls and outcome handling

For each presented pair use the exact sign of s_FIRST−s_SECOND with no pole
or score-sign adjustment. Exact ties get0.5credit;
all nonzero gaps remain with no threshold or rounding before grading.
Primary is EA−E summed over all four axes, out128 per arm. Also report each
cohort difference out64. A useful incremental pilot signal requires both:

1. Both cohort EA−E differences are strictly positive.
2. Each EA reader exceeds the predeclared maximum of its own realized
   always-FIRST and always-SECOND credits over all64items.

Individual constants need not be50%; report both for every reader/axis/arm.
Opposite target orientations across cohorts and identical symmetric tie credit
give exactly half pooled constant credit. These descriptive criteria are not
population significance, a power guarantee or independent-model validation.
PC4 is a named secondary comparison, not an alternative primary selected later.

Keep all256 choices, every pair/axis/reader/cohort/category score, exact gaps,
both position controls, per-text agreement and gains/harm. Categories may have
no accepted pairs; show their counts instead of creating balanced averages.
Reader and arm effects remain mixed; multiple axes share context within reader.
There are32external texts/16pairs and one frozen small model, not256independent
samples. Opposite orders also mix reader and order effects. Extra token budget
is intentional; a positive does not prove information-efficient or causal utility.

Zero/mixed gain or harm is a result to retain, not permission to change corpus,
endpoints, readers or descriptions until the criterion passes. A positive
supports this package/population only; a negative does not erase prior local
usefulness. Neither implies clusters, safety reliability or optimizer benefits.
No near-duplicate follow-up, broader fit or training is automatically queued.
