Codex — Spectral Optimizer Investigation · 10 September 2026

**PASS: no numerical or substantive interpretation blocker.** Two short
presentation precisions are recommended below; no new evaluation, rerating or
experiment is required. This is an unblinded report corroboration, not a new
blind judgment or an independent reconstruction from scientific archives.

## Scope and direct checks

Read the complete main `report.html` and `preparation-review.md`, and the
worker's complete `comparison-decision.md`, `grades.json`, `results.md`,
`report-checks.json`, `rater1-locked.json` and `rater2-locked.json`. Used the
researcher-review skill's evidence-to-claim guidance only; no interactive
workflow gate or new research task was added.

Manually matched all 16 locked responses to the corresponding per-card grade,
including assigned rater, choice and confidence. Checked each declared key
against the recorded token-panel/prefix sign permutations, then reconciled
all per-axis outcomes and aggregate counts. The responses cover C01–C16 once
each, eight per rater. Each rater has one method per axis and two cards per
method within each family. The two raters are the same model and do not
replicate judgments on the same card.

Here `12` means `A1_B2`, `21` means `A2_B1`; the key is the recorded geometric
polarity key, not a new semantic assessment.

| Card | Axis / decoder | Rater | Locked choice | Key | Result | Confidence |
|---|---|---:|---|---|---|---|
| C01 | PC3 / J-Lens | 1 | 12 | 12 | Match | High |
| C02 | random4 / J-Lens | 1 | 21 | 21 | Match | Low |
| C03 | random1 / plain | 1 | 21 | 12 | Miss | Low |
| C04 | random3 / plain | 1 | 21 | 21 | Match | Low |
| C05 | random3 / J-Lens | 2 | 21 | 21 | Match | Low |
| C06 | random4 / plain | 2 | 21 | 21 | Match | Low |
| C07 | PC2 / plain | 1 | 21 | 21 | Match | Medium |
| C08 | PC3 / plain | 2 | 21 | 12 | Miss | Low |
| C09 | PC4 / J-Lens | 2 | 21 | 12 | Miss | Medium |
| C10 | random1 / J-Lens | 2 | 21 | 21 | Match | Low |
| C11 | PC2 / J-Lens | 2 | 12 | 12 | Match | High |
| C12 | random2 / J-Lens | 1 | 12 | 21 | Miss | Low |
| C13 | PC1 / plain | 2 | 21 | 21 | Match | Low |
| C14 | PC4 / plain | 1 | 12 | 21 | Miss | Low |
| C15 | PC1 / J-Lens | 1 | 21 | 21 | Match | High |
| C16 | random2 / plain | 2 | 12 | 21 | Miss | Low |

Thus PC J-Lens is 3/4 and plain is 2/4, with PC1/2 correct under both,
PC3 correct only under J-Lens, and PC4 missed under both. Random J-Lens is
also 3/4 and plain 2/4: random1 only J-Lens, random2 neither, random3/4 both.
Each paired difference is exactly one correct axis. Confidence labels in the
worker results and inspected summary plot match the locked answers. No
confidence weighting, p-value, equivalence or population claim is warranted
or supplied.

The selected endpoint IDs and signed scores match between the two methods
for every axis after accounting for shuffled order. No axis has the same
content under opposite framings as its extrema. Random4 does select two
different cooking contents: same topic is not the same-content exclusion.
The PC extrema use six distinct contents drawn from the eight-content
candidate panel; repeated extrema are not independent test examples.

The current locked-response hashes match those recorded in `grades.json`.
Its own hash matches `report-checks.json`. Lock-before-grading commit
`fc185c7210b5cd4c611a524c4490a4c204056aaa` and one grading invocation are
reported consistently by the main preparation receipt and worker artifacts.
I did not audit the platform's rater-message transport or rerun the grader.

## Positive case and limits

The useful positive is concrete: for these fixed direction/example pairs,
J-Lens makes some contrasts interpretable and supplies one additional correct
PC match. All four PC displays remain visible, including PC4's adverse result.
This is a meaningful small correspondence check, not merely a token-familiarity
score. The report does not need to discard it because it is not yet a broader
advantage.

The same random-control tally does not demonstrate a spectral-specific benefit
or prove spectral/random equivalence. Those random controls are secondary:
their seed/QR recipe was reconstructed without verified original numerical
environment or saved original basis identity. The primary PC score records do
not depend on that reconstruction.

Geometric extrema make this a limited, favorable two-endpoint matching task.
Four dependent PCs from an already inspected single-model authored panel,
one same-model AI judgment per card, and no human/per-item replication do not
establish stable concepts, general readout reliability or an advantage over
individual examples. The report correctly separates activation covariance
from parameter-gradient clustering and does not turn prior classification
compression into a J-Lens result. The stated individual-example comparison is
a future question, not a result of this pole-matching task.

## Two concise presentation precisions

1. Replace “direction 4 produces plausible words but points the wrong way in
   this test” with “direction 4 produces plausible words, but its endpoint
   match was reversed with both decoders.” The measured error is the raters'
   forced matching, not independently incorrect geometry or proof that the
   direction itself has a single wrong meaning.
2. Prefer the confidence opening: “Exploratory reuse of previously inspected,
   single-model readouts: four related directions, eight candidate contents,
   and one AI judgment per item.” This keeps the reuse limitation adjacent to
   the claim and avoids reading “eight content examples” as eight unique
   selected PC extrema. The existing sentences about uncalibrated confidence
   and the original random environment should remain.

## Visual scope and inspected identities

Worker base:
`/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-10-j-lens-simple-comparison/`.

| Inspected artifact | SHA256 |
|---|---|
| Main `report.html`, initial reviewed version | `1ed3112d83b76ad8abb302b4c414d4e07b268b4b956b08bcb74144fb1869ff78` |
| Main `preparation-review.md` | `115c98fd0903202da24a882bf4b61bdad43928b6d739386f66e7acf713e50a98` |
| Worker `comparison-decision.md` | `913c04d5e322232ac93cfa3f1bd179466ecd576d8d6a3ed5842d19055f5ba6be` |
| Worker `grades.json` | `67732cb032856ac9a027aff1e3d66caef6de43e507e9585ae365c287bdc5cf38` |
| Worker `results.md` | `5e2b853a163878dfd1681e8f7ada596e93a27ecb3f44496ec51d753389b9fa99` |
| Worker `report-checks.json` | `fd7263f92a4b43679a1f9b5759c482452412d959aedd1059afb211535480e58e` |
| Worker `rater1-locked.json` | `a9a2b6f5b953be78d694b944f4ec4643e60ced215a41602d55deb529c40f43cf` |
| Worker `rater2-locked.json` | `600176e12c99559a3e353f0bf341a82fcd2f67ebd07a8885792fa86946138b8b` |