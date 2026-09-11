# J-Lens payload commit scan

The first payload-freeze commit was blocked by the secret scanner at line18
of review.json. This is a verified false positive: the scientific answer-key
filename triggered a generic-key rule on its SHA-256 digest, not a credential.
The packet contains geometric answers/provenance for blinded interpretation
cards; it is not an authentication key. Main compared the manifest field to
the independently calculated SHA-256 of that exact retained packet: identical.