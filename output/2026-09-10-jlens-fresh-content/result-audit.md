# Saved-output J-Lens result audit

Codex — Spectral Optimizer Investigation · 10 September 2026

**PASS.** An independently written scalar calculation agrees with every saved
item score, all arm/axis aggregates and all paired differences. No cases are
missing. Direct directions score **31/48**, individual-activation readouts
**33/48**, and selected fit prefixes **34/48**. The first three direct-direction
readouts have useful limited correspondence; the fourth is a conspicuous
failure and remains part of the primary total.

**Independence qualification:** I authored `judging.py`. This is an independent
calculation and saved-file/mapping audit, not an independent implementation-
author review, a new blind semantic judgment or an independent neural
reproduction. I neither imported nor called `judging.grade`,
`judging._grade_values`, `make_packets`, or any pipeline stage. Main separately
reviewed the implementation in [judging-review.md](judging-review.md).

## Exact reconstructed outcomes

Each entry is correct out of 12 fixed pairs. A = signed PC J-Lens token lists;
B = J-Lens lists for selected raw fit activations; C = those fit prefixes.

| Axis | A: direct direction | B: individual readout | C: fit prefixes | Always FIRST | Always SECOND |
|---|---:|---:|---:|---:|---:|
| PC1 | 11 | 9 | 11 | 7 | 5 |
| PC2 | 9 | 8 | 9 | 4 | 8 |
| PC3 | 9 | 9 | 9 | 6 | 6 |
| PC4 | 2 | 7 | 5 | 7 | 5 |
| **Total /48** | **31** | **33** | **34** | **24** | **24** |

There are no exact-zero gaps. Thus all credits are 0 or 1; the prospective
half-credit rule does not affect this result. A−B is −2 overall, with per-axis
differences +2, +1, 0, −5. A−C is −3 overall, with differences 0, 0, 0, −3.
B−C is −1 overall, with differences −2, −1, 0, +2. These are counts on the
fixed panel, not statistically established method differences or equivalence.

The matching A/C totals on PC1–3 are not hiding different successful items:
their choices agree on all 36 of those axis–pair targets. On PC4, A/C have
one jointly correct target, one correct only under A, four correct only under
C, and six jointly wrong. A/B's PC4 split is one jointly correct, one A-only,
six B-only and four jointly wrong. These descriptive joins do not change the
predeclared primary all-four-axis accounting.

All signed gaps and absolute gaps are retained for all 144 items. The 48
distinct axis–pair gaps are shared across arms after the common input swaps.
Observed absolute-gap ranges in the native score units are:

| Axis | Minimum absolute gap | Maximum absolute gap |
|---|---:|---:|
| PC1 | 0.06848130915460159 | 1.024879242427889 |
| PC2 | 0.001870290516011952 | 0.8944650169170152 |
| PC3 | 0.0024028641231854664 | 0.7951139427080703 |
| PC4 | 0.012427535429066383 | 0.261749456596974 |

PC4's ten direct-direction errors span its entire displayed gap range, so the
adverse result is not an exact-tie grading artifact. No margin-based exclusion,
standardization, sign reversal or selected rereading was applied. These are
not cross-axis standardized effect sizes.

## What was checked

- Verified the pinned packet manifest, response lock, grades, current judging
  source, source-reference export, scalar-score export and frozen
  protocol/dataset/pair receipts. Reference and score hashes agree with the
  acquired-export identities in [acquisition-review.md](acquisition-review.md).
  Source code was hashed, not imported or executed by this audit.
- Reconciled all 24 unique source contents, six per topic, with all 12 disjoint
  pairs. Every content occurs exactly once; each of the six cross-topic
  contrast types occurs twice. Every axis–arm–pair cell occurs once: 144 rows,
  48 distinct axis–pair targets, not 144 independent examples.
- Checked all public block/item keys against their allowed public fields.
  No explicit axis/arm/source-ID/score/truth metadata appears in those fields.
  Each of three public packets has four blocks and 48 distinct item IDs;
  all 12 block IDs and all 144 item IDs are unique. The three public prompts
  match the fixed protocol text.
- Joined every public comparison to its private-map row and fixed source pair.
  Both actual prefix strings match the source dataset exactly. Each axis–pair
  has identical first/second ordering across all three arms. Every positive/
  negative reference matches the pinned acquisition interpretation export:
  two complete 12-token lists for A/B or two intact raw prefixes for C.
  This checks the reference-export join, not a new token decoding.
- Verified the prescribed per-axis allocation: rater1 A/B/C/A, rater2 B/C/A/B,
  rater3 C/A/B/C. Each rater has exactly one arm per axis and no counterpart
  for that axis. Public packet identifiers and returned response identifiers
  agree, and each item has exactly one permitted FIRST/SECOND response.
- Compared all three files under `returned/` byte-for-byte with the sealed
  copies. Verified each sealed response hash and the lock's binding to the
  packet-manifest hash. All four response-lock files are byte-identical to
  their blobs in commit `92179228bde2be856726169e76f1246749bb38bc`.
- In a short, bounded standard-library-only CPU command, independently read
  the 96 finite exported scalar scores and computed each recorded
  first-minus-second gap. Reconstructed truth and credit from that gap and
  the locked choice, without importing the grader. Every field in all 144
  graded item records agreed exactly, including source IDs, arm/axis/rater,
  first/second scalars, gap, absolute gap, truth, choice and credit.
- Independently summed all 48-item arm totals, all 12-item arm–axis totals,
  correct/incorrect/tie counts, both constant-position baselines and all
  declared paired differences. All match `graded/grades.json` exactly.

The acquisition's calculation of U32 scores from residuals and its neural
state invariants are outside this scalar audit. They remain the responsibility
of the separate acquisition/saved-array checks; no NPZ, tensor, checkpoint,
model or activation archive was opened here. The saved randomization mapping
was checked against the protocol's pairing/allocation constraints, not
regenerated through the packet builder.

## Response chronology and transport attribution

The lock records sealing at **21:53:33.215958 UTC**. The commit's timestamp is
**21:53:33 UTC**, and grading starts at **21:54:10.107586 UTC**. Git timestamps
have whole-second resolution; seal and commit share that same second. The
commit contains the exact completed lock and all 144 responses, and its
recorded second is earlier than grading. The grader's provenance records that
same immutable commit and lock hash.

My first timing assertion incorrectly compared Git's whole-second timestamp
as though it had the lock's subsecond precision. It failed after all scalar,
mapping and committed-blob checks had passed. Inspecting the timestamps
identified this audit-comparator mistake; the resolution-aware check passes.
No packet, response, source, grade or scientific result was altered, and no
pipeline stage was rerun to address it.

The transport record (artifact not distributed in this public snapshot) states that three fresh, no-history
forks received only their corresponding public packets plus mechanical
instructions, used no tools/files/browsing, and each returned one complete
answer without feedback, repair or rerating. I checked the retained returned
and locked bytes, **not** the platform's original spawn-message/agent-event
stream. Transport isolation and the single-judgment history are therefore
attributed to that main-agent record, not independently certified by this
file audit. Raw-prefix formats can reveal the arm; the raters share a model
configuration, and there is no per-item or human replication.

## Interpretation that the evidence supports

**Useful positive:** direct-direction readouts support correct fresh-content
ordering on 11/12, 9/12 and 9/12 comparisons for PC1–3. This is a genuine
new-content correspondence test, not a repeat of selecting the easiest new
extrema or merely judging whether word lists look meaningful. PC2's 9/12 is
only one above its better constant-position baseline of 8/12, which remains
visible. Useful absolute examples can be reported without requiring an
overall method win.

**Important adverse finding:** PC4's direct readout gives 2/12, below both
fixed-position baselines. The direct-direction arm has no aggregate advantage
over the two exemplar arms, and raw fit prefixes match all its PC1–3 choices.
The observations favor treating readable direction labels as fallible
interpretations rather than reliable names for every high-loading example.
They do not prove that the geometry itself is wrong or that reversing this
axis's interpretation would generalize.

This tests four fixed activation directions of one fixed model/layer on
24 authored contents within four familiar topics. It does not test four
distinct concepts, new-topic generalization, population interpretability,
PCA versus no PCA, or an optimizer/safety benefit. The directions, comparisons
and readers are dependent; small method differences supply no significance,
equivalence or superiority claim. No automatic new experiment, prompt tuning,
axis selection or rerating follows from this audit.

## Principal immutable identities

| Artifact | SHA256 |
|---|---|
| `packets/manifest.json` | `2be4392f32e800c2f8c4cadbf0dcc04759c01bc2caacfc6565d29e41ef445ef4` |
| `responses/lock.json` | `957f3d2846a6d92a76516ec54332b7668166c5a0fe257c5827caa2908bfb6364` |
| `graded/grades.json` | `1ace0f90f0d8f539094ece65e6acce1517ea2159ef80f21ea143cfc086a7aa29` |
| `judging.py` | `1a1db646d202f579a6668c4b2c84f0f6a336fd7edc0babfef1cac4c55fd06cf4` |
| Worktree `references/interpretations.json` | `0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c` |
| Worktree `fresh-features/scores.json` | `f1e4d2aa1ace39d9d482681a3e672f60c279348cf62884042a9ee93409a19a46` |
| `rater-transport.md` | `5b8873ac1c4396f56d8f6bba48de32af6bfb78d22e12a892e5231f4e3c25a135` |