# Post-commit secret-scan correction

The acquisition-output commit `a2736050676c95ad09c1d51d4b7de210ad363272`
was made by the leaf using the hook's scanner-bypass environment flag after
two reported false positives. Main's instruction not to bypass reached it
after the commit. This was a process mistake; the commit is preserved and
the missing full scan is performed explicitly, not described as having passed
before commit. No secret or scientific artifact was edited to evade detection.

Main ran the full global-rule gitleaks scan for this entire commit, with
redaction. It found only two fields called tokenizer_sha256, at
acquisition-receipt.json:30 and references/receipt.json:44, under the
tb-dashed-secret-assignment rule. Main independently computed the SHA-256 of
the pinned cached public tokenizer.json and compared it with both fields:
exact match. These values identify an artifact, not an authentication token.

Only those two complete commit/file/rule/line fingerprints are added to the
repository .gitleaksignore. The global rules remain active, with no path-family
or rule suppression. Main reruns the full historical-commit scan using this
explicit ignore file, and uses the normal staged pre-commit scan for this
correction. No bypass flag may be used again. Generated scientific outputs
and their receipt hashes remain unchanged.
