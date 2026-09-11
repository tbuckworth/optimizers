# Natural add-on implementation review

Codex — Spectral Optimizer Investigation · 11 September 2026

This is a new collection/comparison, not a replay. Existing original catalogue,
all previous article requests, original references and model geometry are
immutable inputs. User autonomy applies without another approval round.

## Official-source checks before implementation

The collector uses serial requests, a descriptive contact User-Agent, `maxlag=5`
and cacheable read-only GETs. It stops rather than retrying service errors.
These choices follow [MediaWiki API etiquette](https://www.mediawiki.org/wiki/API:Etiquette).

`exchars=1200`, `explaintext=1` and `exintro=1` are supported; the response can
exceed the requested character count, so that count is not an eligibility
cutoff. Exactly the first16 Unicode whitespace words are retained. We do not
use the discouraged sentence-count mechanism. Empty/missing extracts are
technical failures, not replacements chosen by content.
[TextExtracts documentation](https://www.mediawiki.org/wiki/Extension:TextExtracts)

Python JSON defaults accept duplicate object keys and nonfinite values. Reuse
the reviewed strict decoder (`object_pairs_hook`, `parse_constant` and a finite
float walk) and serialize with `allow_nan=False`.
[Python JSON documentation](https://docs.python.org/3.12/library/json.html)

Python's default URL opener follows HTTP redirects. The new transport overrides
`HTTPRedirectHandler.redirect_request` to raise `HTTPError`; the response body
and redirect status are retained and the stage stops. This matches the new
protocol's no-redirect rule, which the older default transport did not enforce.
[Python urllib.request documentation](https://docs.python.org/3.12/library/urllib.request.html)

## Reuse boundary

Main fully read the266-line prior `collect_text.py`, SHA256
`337e9a208f491b22ec0deb3a591b1216669eaf42db4712e43f8e1bc97e98d347`.
Its import is inert. Reuse only pinned strict JSON/byte utilities, article
eligibility and bounded exclusive stage storage. Never call
its old catalogue/excerpts entrypoints or its old-protocol completion method.
New input/receipt logic binds this study's source and protocol. Synthetic
fixtures must cover selection, whole-pair failures, non-reservation of rejected
prefixes, limits and input/source changes before actual new collection.

Before the first protocol freeze, main removed an unnecessary draft pole-flip
ambiguity: positive/negative reference meanings remain unchanged; only target
ordering reverses across cohorts. No new data or reader response existed when
this correction was made. This ensures the pooled constant-choice reference
really is50% under symmetric tie credit.

Independent Astra review caught a material draft regression before any new
selection/acquisition: the old Stage.request rejected ordinary revision-history
continuation, the same issue documented in the previous study's
collection correction (artifact not distributed in this public snapshot).
Main should have followed that correction when reviewing reuse. Main now fully
read the old resume source and implemented a new request recorder, preserving
raw bytes and accepting exactly the two-key nonempty-string revision marker.
It never follows the marker. Protocol and synthetic tests were corrected before
freeze. This follows [Revisions](https://www.mediawiki.org/wiki/API:Revisions)
(`rvlimit` enumerates one page's revisions, newest first) and
[continuation documentation](https://www.mediawiki.org/wiki/API:Continue).
Old helpers and completed attempts remain unchanged; this new draft never ran.
Review also caught that old eligibility checks technical markers before page
identity. The new wrapper now rejects any different positive returned ID before
redirect/namespace/disambiguation skips, as its stricter protocol requires;
fixtures cover each combination. Normal same-ID technical failures still skip
the whole pre-fixed pair. No candidate or data change accompanies this fix.

Main fully reviewed the independent inventory builder/tests, reran11fixtures
PASS0.065s, and released exactly one metadata builder invocation. Worker freeze
`0cebc9d871a52956ccc2369ce64833b073aedfb9`, inventory SHA256
`77e596ec66956410a90c4b7b108a86026151273a8e3ced8772939fe73bff91b9`.
Main then independently checked all101source pins, actual29article request
records/28events, copied-record flag, raw category arrays and exact exclusion
union:235unique IDs,207available (46/30/0/131), PASS. This neither reads new
article text nor computes the new selection.

Final corrected collector SHA256
`decdb21a2d76b0a2d75f7e4f0753ae76570b5c75e9aac12aa8e2002743001ee8`;
tests `5e0e6d9be5ccc60330e7043d6b63cdef45fc77ccf36b075ddd754cf9d6850568`;
protocol `a510feaf2dfdd5cf6541d974d2c305188e24d04a4e9e185ac73675970b8a6246`.
Main17synthetic tests PASS0.093s; independent Astra full re-review and all17
fixtures PASS0.090s. Both concrete defects are closed before selection or
acquisition. New collection source is accepted for one selection stage, then
one bounded excerpts stage after independent manifest inspection/freeze.
This is not a release for an old stage, tokenizer, model or readers.

Full review is retained in the worker review (artifact not distributed in this public snapshot).
Actual handles and resulting immutable pins are recorded in execution.md.
