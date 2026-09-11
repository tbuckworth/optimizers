# Frozen initial feasibility pilot

Written before observing synthetic outcomes. The adjacent recipe is the complete
roster: 27 cells, three seeds shared across three sizes and three patterns. No
search, outcome-dependent tuning, or repeated acquisitions. Pure correctness tests
may be repaired before acquisition. The acquisition records the recipe and source
hashes and refuses to overwrite its start marker or results.

Question: can a different, anchor-based graph support spectral clustering and
cluster averaging without a P by P allocation? This is an implementation and
conditional synthetic recoverability test, not a new learning or safety result.

Synthetic profiles have 16 coordinates. Balanced: four equal groups around the
first four positive coordinate axes. Rare: eight approximately equal common
groups on the first eight axes and a 3% group on the ninth axis. Signed: eight
equal groups around positive/negative versions of the first four axes. Add
independent Gaussian profile noise with SD .15 to all entries. Shuffle groups.
Generate 32 independent standard Gaussian latent observations and gradient vectors
h = profiles @ observation. All three methods see the same observations:
(1) exact uncentered EMA represented via a 16 by 16 latent second moment;
(2) its one-shot best rank-8 factor; (3) recursively truncated rank-8 sketch.
Start the moment at zero and use beta .95, including the first observation.

For each factor independently: normalize nonzero rows, choose 16 current row
indices by farthest-point sampling (seeded first point), construct Gaussian anchor
weights with sigma .35 and row normalization; remove zero-mass anchors. Keep
self-loops. Compute up to the known group count leading positive eigenvectors
from the small Gram matrix, retaining only eigenvalues > 1e-10. Include the
constant eigenvector; normalize embedding rows and run deterministic farthest
initialization / Lloyd k-means, at most 30 iterations, no restarts. The true number
of groups is supplied; this does not test selecting cluster count. Labels are
used only to score and determine that declared count, not select anchors or tune.

Record adjusted Rand index, each true group's best predicted-cluster F1, rare-group
F1, retained moment trace, streaming relative Frobenius error via small cross-Gram
matrices, runtime, array-state bytes and tracemalloc peak. Trace retention does
not imply useful-direction retention. Compare methods within each cell; retain
all adverse outcomes. There is no minimum clustering score needed to declare the
storage construction feasible. Production mathematical invariants must pass pure
tests against dense P<=128 references first; stop for failed invariants or guards.

One CPU math thread; systemd user unit with CPUQuota=100%, MemoryMax=2G,
MemorySwapMax=0, TasksMax=16, RuntimeMaxSec=900; shell timeout=890s; no GPU/cloud,
credentials, real-model work or existing experiments. Total output below 500MiB.
All large-P code uses factors; dense reference tests are excluded from memory
claims. Tracemalloc is an observed allocation measure, not a proof of every BLAS
workspace allocation; capture cgroup peak if supported. No independent audit is
performed by this leaf; root review is required before durable KB integration.

Steelman: anchors preserve nonlinear distinctions of row directions, so graph
rank can exceed the sketch rank; low-rank factors still retain rich profiles.
Failure cases: truncation discards a rare coherent direction; anchors miss a group;
bandwidth merges or disconnects groups; weak eigenmodes are unstable; finite-sample
second moments distort population patterns. Coherence and size say nothing about
semantic usefulness. Training integration and choosing refresh cadence are deferred.
