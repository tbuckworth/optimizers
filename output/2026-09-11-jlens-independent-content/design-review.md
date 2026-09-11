# Independent-content J-Lens pilot — design review

Codex — Spectral Optimizer Investigation · 11 September 2026 BST

**PASS for source-collector implementation preparation.** This is a useful,
proportionate extension of the completed within-topic comparison. No new arm,
optimizer experiment or user-approval gate is needed. This is not acceptance
of an implementation or admission of a model run. Make the small collection
and attribution details below explicit in the implementation/next freeze.

## Scientific question and replication

The design tests whether unchanged descriptions transfer to externally
authored, mechanically selected prefixes, rather than another investigator-
written panel. Preserving all four directions and the same A/B/C references
maintains the useful positive case while retaining PC4's adverse precedent.
PC3 is honestly outcome-motivated, not retrospectively presented as an
unselected discovery. A practical positive does not require defeating C.

The roster is coherent: four captured categories × six pages gives 24 texts;
three disjoint pairs/category gives 12 pairs. Four axes give 48 targets;
three arms × two readers give 288 choices. Each arm has 96 choices overall
and 24 per axis, but still only 12 distinct pairs per axis. The allocation
provides exactly two readers per arm/axis and four blocks/48 choices per
reader, with no reader seeing competing arms for the same axis.

Categories are sampling strata, not verified semantic labels. A successful
within-category ordering is more than separating these four supplied labels,
but category membership need not make two articles close in topic. Do not
promote the new panel into a controlled test of fine-grained semantics.
Source, span length, grammatical endpoint and authoring style shift together;
weak transfer cannot identify which caused it. The protocol already preserves
this limitation and does not certify unseen pretraining data.

## Concrete collection and attribution details

1. **Treat 1200 as the requested extract length, not a strict content filter.**
   The documented `exchars` setting can return slightly more characters.
   Preserve the returned extract and derive its first 16 split words exactly;
   do not silently truncate it again or exclude a candidate solely for this
   overrun. Keep the independent 2 MiB response cap. This is a small but
   plausible API-contract failure with an easy pre-collection fix.
   [Official TextExtracts parameter documentation](https://www.mediawiki.org/wiki/Extension:TextExtracts#API).

2. **Specify how the already-promised attribution check will actually occur.**
   The listed extract/revision/info/pageprops response is not a check of the
   rendered footer, page history, talk page or an imported-content attribution
   banner. Before public redistribution, inspect attribution information for
   the fixed selected pages, preserve applicable notices and record the
   check. This must not become a semantic/quality-based selection pass: retain
   the frozen roster and report an unresolved rights issue rather than replace
   an inconvenient page. Page/history links, source-license notice and an
   explicit normalization/truncation notice are otherwise a coherent reuse
   plan. Code and analysis need not be relabeled as Wikipedia text. This is a
   publication-integrity detail, not a reason to stop collector preparation.
   [Wikimedia text reuse and modification terms, section 7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content).

3. **Preserve the distinction between an intentional bounded stop and bad sampling.**
   Categorymembers can return a continuation even with fewer than `cmlimit`
   entries after namespace filtering. Requiring no continuation is a valid
   conservative design, but a continuation is an operational stop—not proof
   that the category is too large or lacks six usable pages. Do not reinterpret
   the partial response as the whole catalogue or traverse subcategories to
   rescue the panel. The official API supports the chosen namespace/type/limit
   settings. Four catalogue calls plus at most 80 candidate calls is a modest,
   bounded source collection; any attribution inspection has a separate,
   explicit request record.
   [Official Categorymembers documentation](https://www.mediawiki.org/wiki/API:Categorymembers).

Serial GETs, a real project/contact User-Agent, maxlag, retained responses,
and stopping for API errors are consistent with the documented etiquette.
Retries must keep the same catalogue and candidate ordering; they cannot
silently generate another successful panel. No additional reachability probe
or candidate inspection was needed for this review.
[Official API etiquette](https://www.mediawiki.org/wiki/API:Etiquette).

## Model boundary and possible outcomes

The source-feasibility note identifies a coherent forward-only path: unchanged
cached model, post-block-11 last-position capture, original fit mean and exact
canonical U32 scores, with no lens decoding or PCA. The 16-word rule is not
a tokenizer-length guarantee. Before any forward, freeze the tokenizer-only
check and retain the 1–96-token operational cap; if a frozen prefix fails,
do not truncate or replace it silently. Saving exact input IDs and activations
permits subsequent saved-output checks without another model execution.
The earlier resource measurements are only an anchor, not a fresh headroom
check. Implementation acceptance remains main-owned, as already specified.

## Review scope and pins

Read the complete protocol and forward-feasibility note, with the completed
within-topic results as context. Applied pre-mortem guidance to concrete
failure modes, without inventing further experiment gates. Consulted only
the four official API/terms pages linked above; no candidate article content,
catalogue acquisition, tokenizer/model call, scientific NPZ or implementation
was opened or run for this review. The cached-model/source claims in the
feasibility note are attributed to that assessment, not re-audited here.

Reviewed protocol commit: `00e02e9`.

- Protocol SHA256: `a7361ffb80f63fc41661069dc2b98dac842051f25b03315bbc361d575a254044`.
- Feasibility SHA256: `aa69b152a8c68e72db818c6b894c108044b33702de175ad0ce9ed6122cce77ad`.

### Pre-collection clarification receipt

Main reports accepting the requested extract-length, fixed-roster attribution,
within-category interpretation and reader/order-disagreement clarifications
at `50d1e6c`, amended protocol SHA256
`78d6171dfcbd2e24ff39c4acf88138de67138003d970c43a3ef3580030c93051`.
The original review/pins above remain its historical target; this records
the disposition rather than claiming a further source or implementation
audit. Main reports no source collection or model experiment has yet run.
The design PASS stands; no additional scientific condition is requested.
