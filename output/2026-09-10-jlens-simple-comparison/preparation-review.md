# Main review: the smallest retained J-Lens comparison

Codex — Spectral Optimizer Investigation · 10 September2026

## Scope

The user's20:12UTC redirection is authoritative. Use the existing worktree and
saved layer11 activation outputs; no model inference, training, feature
acquisition, PCA refit or paid judge. Main read the original acquisition and
analysis source, the latest clarification, the proposed comparison and the
complete first packet-builder implementation. Final revised source/fixture
and one packaging execution remain to be receipted below.

The selected comparison holds the individual direction fixed and changes only
its decoder: J-Lens transport versus plain unembedding. All four PCs and both
signs remain visible. Mechanically selected held-out extremes provide a small
correspondence task without a post-hoc topic dictionary. Four random axes are
secondary controls, not guaranteed semantic nulls. The raw individual/full
vector and fit-mean readouts exist but do not define the same pole-matching
task; no claim that PCs outperform individual vectors is justified here.

## Accepted judging design

Sixteen public cards: four PCs and four random axes, each through two decoders.
All16 held-out rows (eight contents, two framings) are eligible for deterministic
max/min-score selection. Keep same-content/opposite-frame extrema and report
them rather than substituting more interpretable examples. Each card contains
both complete12-token lists and two prefixes, independently order-shuffled.
Decoder, axis identity, signs, scores, source IDs and geometric key remain
private. Forced-choice polarity matching is graded, confidence only described.

Two fresh same-model Astra raters receive eight cards each, one method per
axis. Within each PC/random family each rater sees two J-Lens and two plain
cards; the other rater receives the complementary methods. They see public
cards in ID order only, receive no prior interpretation or source/context,
and must not use tools or files. Lock all responses before releasing the key.
This avoids giving a rater both methods' answer for the same axis; it is still
one judgment per card, not rater replication or human validation.

Primary report: correct out of four PCs by decoder, all paired axis outcomes,
and the actual token/example display. Random results separately. No p-value,
population interval, equivalence, validated distinct-concept or new-data claim.
Extrema selection favors correspondence; results need not describe all
examples or establish semantic clustering. This is outcome-informed reuse of
an already inspected single-model panel.

## Source and implementation checks

The original acquisition's normalized PC±/random± ordering and per-method
loop match the new lookup. Existing PC scores are read, not refitted. Random
scores reconstruct the original seed20260910 Gaussian1024×4/QR recipe and
project saved centered activations. Required refinements from main review:
verify fixed input hashes before and after reading; reject an existing packet
target before reads; validate finite shapes, exact candidate IDs and unique
direction names; fix balanced rater allocation before judging.

The best-practices check used current official documentation:

- [`numpy.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
  supports the selected `allow_pickle=False` and context-manager handling of
  the numeric NPZ inputs; no object/pickle input is required.
- [NumPy random compatibility](https://numpy.org/doc/stable/reference/random/compatibility.html)
  makes reproduction conditional on build, environment, machine and exact
  generator calls—not the seed alone.
- [`numpy.linalg.qr`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html)
  uses LAPACK. Since the original random vectors were not saved separately,
  record the current numerical environment and any available original receipt;
  do not claim independently verified vector identity from a matching seed.
  This qualification concerns the secondary reconstructed random controls,
  not the primary PCs with saved signed scores.

Only source and small receipt metadata were read for this main review. No new
scientific score or packet has been generated at this review checkpoint.

## Execution and locked judgment checkpoint

Main accepted the revised source freeze
`75c59f86e2e7da3e3f7b450be9be67bf77f4f446`, builder SHA256
`e06a5f471e440268840e49dcfdca9daeb7051b0bedb35bb26988ea2c78afa3c3`,
protocol SHA256
`913c04d5e322232ac93cfa3f1bd179466ecd576d8d6a3ed5842d19055f5ba6be`.
The committed synthetic fixture passes. Both archived source functions and
the entire new builder were inspected, including the revised allocation,
pre/post hashes and numerical-environment qualification.

One packaging invocation completed successfully, exit0 in0.093s, under a
60-second timeout with CUDA hidden and OMP/OpenBLAS/MKL threads each1.
All16 public cards and separate private key were created. The original NPZ
analysis and model acquisition were not rerun. This small new extraction
uses saved PC scores and reconstructed secondary random scores only.

Two fresh gpt-6-astra/high raters received the public panels and prefixes as
inline JSON, with tools/files/browsing forbidden and no prior conversation
fork. Each returned exactly eight choices/confidences without using a tool.
Responses were saved verbatim to `rater1-locked.json` / `rater2-locked.json`
in the worktree and committed with the packet at**fc185c7**, before grading.
No evaluator received a counterpart card for an axis or a geometric key.
The public JSON is preserved; no automated byte-exact audit of the platform's
recorded spawn-message text is claimed. No judge was retried or corrected.

Main independently joined the16 frozen answers to the16 private-key rows:
PC J-Lens3/4, plain2/4; random J-Lens3/4, plain2/4. PC1/2 are correct with
both decoders, PC3 only with J-Lens, PC4 with neither. No selected extrema
share the same content ID. These are preliminary corroborated grade counts,
not a reviewed final report or evidence of a PCA-specific advantage.
The worktree Astra leaf is now checking the complete grading and preparing
all-four-PC paired displays. No new model work or paid spend is selected.
