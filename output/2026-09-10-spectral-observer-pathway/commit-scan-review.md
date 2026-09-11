# Commit scanner: five verified image-digest false positives

Main tested the false-positive hypothesis directly: recomputed the actual
`results-geometry/access-and-usefulness.png` digest, independently hashed its
staged git blob, and required exact equality to each of the five flagged JSON
fields. All five checks passed. The values are content hashes of this plot,
not authentication material, and no credential was read or printed.

The installed global hook explicitly supports a repository `.gitleaksignore`;
its installed CLI documents this ignore path. Added only the five reported
path/rule/line fingerprints there. No global configuration, security rule,
directory-wide allowance or hook skip was introduced. A changed future value
at an excepted position must be re-reviewed; the exceptions are for these
immutable receipt fields, not permission to store credentials there.

All original scientific/delivery receipts and their hashes remain unchanged.
The normal complete staged scan must pass on the subsequent commit. The
assumption-debugger skill informed the verify-before-change sequence; the
observed failure and direct equality checks supplied the diagnostic facts.
