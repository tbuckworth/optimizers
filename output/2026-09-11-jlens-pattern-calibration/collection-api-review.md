# Collection API check before acquisition

11 September 2026 · primary documentation and source inspection only

The main fully read the old natural-addon collector, its protocol and all
266 lines of the original strict-JSON/eligibility helper. No old entrypoint
was invoked. The new collector must consume the frozen manifest, not generate
another ordering, and preserve the exact technical eligibility rules.

[MediaWiki API etiquette](https://www.mediawiki.org/wiki/API:Etiquette)
supports serial GETs, a descriptive contact-bearing User-Agent, maxlag and
caching. Our fixed small-panel requests retain all of these. Although the
documentation permits backoff/retry and recommends grouping/compression, this
study deliberately uses the predeclared single-article identity-encoded calls
and stops on errors without retry; this makes per-article byte bounds and
the request inventory explicit. No rate-limit evasion or browser impersonation.

[Python's urllib documentation](https://docs.python.org/3.12/library/urllib.request.html)
defines timeout for blocking operations, not a guaranteed end-to-end elapsed
deadline. Root identified this gap before the new collector source/launch freeze
and asked the existing Astra to enforce the protocol's20-second wall deadline
around each serial fetch, retaining the ordinary socket timeout too. Fabricated
timeout and handler-restoration checks are required before admission. This is
a prospective implementation correction, not an amendment to scientific rules
or a rerun of old collection. The external3600-second stage ceiling remains.

Final collector source/fixtures, old-prefix exclusions and actual service
properties still need review. This note is not a launch or a claim that a
collector has already met these requirements.
