# Anchor graph pilot: feasible storage, conditional group retention

Status: initial three-seed synthetic feasibility evidence and independently
derived mathematics; no real-model training, optimizer integration, or independent
results audit. The original optimizer and knowledge base are unchanged.

The construction works without a stored P by P matrix. All 11 pure correctness
tests pass. Across the frozen 27 synthetic cells, retained rank-8 state scales
linearly with P, and the full acquisition took 2.53 seconds. Rare-group recovery
can nevertheless deteriorate substantially. This supports implementing the
anchor variant, not exact maintenance of the previous clipped-cosine graph or
claims about useful rather than memorized behavior.

## Mathematics and implementation

The standalone [prototype](../../experiments/anchor_graph_pilot.py) maintains
M approximately equal to F F^T. For each h it computes a thin QR and a small SVD
of the augmented factor [sqrt(beta) F, sqrt(1-beta) h], retaining r singular
directions. This is the best rank-r truncation of the *currently represented*
rank-one update. Recursive truncation is generally different from the best
rank-r approximation of the full historical EMA. The first observation starts
from zero and receives its correct (1-beta) weight. This is uncentered second
moment estimation, not the repository's centered-gradient implementation.

Normalize nonzero rows of F into profiles e_i. Farthest-point sampling selects
m current profile row IDs. Anchor IDs can be retained between refreshes, but
their coordinates are recomputed from the current factor. Thus an orthogonal
change of factor basis does not stale the anchors. This addresses coordinate
gauge changes, not arbitrary model drift. Zero profiles are excluded; their
graph adjacency is zero and their cluster-mean update is passed through.

For active rows, form nonnegative Gaussian weights normalized into Z 1 = 1.
Remove zero-mass columns, set Lambda = diag(Z^T 1), R = Z Lambda^(-1/2),
and define A = R R^T implicitly. Directly:

- A is symmetric, entrywise nonnegative and PSD.
- A 1 = Z Lambda^(-1) Lambda 1 = 1, so degree is identity on active rows.
- Its normalized Laplacian is I - A on those rows; self-loops are retained.
- If (lambda, w) is an eigenpair of R^T R with lambda > 1e-10,
  u = R w / sqrt(lambda) is a normalized eigenvector of A.

This is established anchor/landmark graph machinery. Liu, He and Chang introduced
[anchor graph construction (ICML 2010), equation 9](https://icml.cc/Conferences/2010/papers/16.pdf),
including the same A = Z Lambda^(-1) Z^T factorization and unit degree.
Chen and Cai's [LSC paper (AAAI 2011), equations 4–7](https://ojs.aaai.org/index.php/AAAI/article/download/7900/7759)
gives normalized Gaussian anchor representation and spectral extraction from the
small Gram matrix (their matrices use the transposed orientation). This pilot
uses all anchors, farthest-point row selection, and explicit embedding row
normalization. It is an adaptation, not a reproduction of either paper's
benchmarks. Both papers' equations were inspected directly; the ICML PDF required
a direct download after browser retrieval timed out.

Clipping a low-rank cosine matrix entrywise at zero can increase rank; the test
suite includes a counterexample. The proposed A is therefore a different graph,
not an exact factorization of that clipped-cosine graph. Graph rank is at most m,
and can exceed r because Gaussian anchor weights are nonlinear in profiles.
A graph refresh is not an exact rank-one graph update: normalization, anchor
selection, weights and degrees can all change after one new h.

For disjoint clusters, the action replaces each vector entry by its group's mean.
Assignments plus group sums apply the corresponding orthogonal projector in O(P)
time/storage without storing the P by P projector. Isolates are singleton identity
actions. No cluster weighting by usefulness is included.

Storage is O(P(r+m)+m^2), including transient factors and selected embeddings
when their dimension is at most m. Thus it is O(Pk) for r,m=O(k), k<=P.
The prototype retains both Z and R, so its constants are not memory-minimal.
One sketch observation costs O(Pr^2+r^3). Anchor selection/weights and spectral
refresh cost O(Pmr+Pm^2+m^3), plus O(I P c d) for c clusters, embedding width
d<=m, and I bounded Lloyd iterations. Graph application is O(Pm), and fixed
cluster action is O(P). Periodic refresh can amortize graph work; no refresh
cadence is tested. None of this is a strict O(Pk) bound on each whole update.

## Frozen synthetic results

[Plan](plan.md) and [recipe](recipe.json) were committed before acquisition in
`1345097`. The [raw results](results.json) and streaming cell journal (artifact not distributed in this public snapshot)
agree exactly. All code and recipe SHA-256 hashes match the start record.

Each of three patterns uses seeds 17, 29 and 43 at P=1024,4096,16384. These are
three shared seeds, not nine independent replications per pattern. Profiles have
16 latent coordinates with Gaussian noise .15; 32 latent Gaussian observations
drive an EMA with beta .95. Methods share observations. Exact means the exact
finite-sample EMA represented through its 16-dimensional latent moment; it is
not the population moment. Batch rank 8 truncates that final exact moment once;
stream rank 8 truncates every observation. All use m=16, sigma=.35 and the known
group count. Numbers below are descriptive means across the nine cells per
pattern, not confidence intervals or inferential tests.

| Pattern | Exact EMA ARI | Batch rank 8 ARI | Stream rank 8 ARI |
|---|---:|---:|---:|
| Four balanced groups | .9930 | .9946 | .9958 |
| Eight common + 3% rare group | .9687 | .9054 | .9357 |
| Four positive/negative group pairs | .9685 | .9779 | .9763 |

The rare-group best whole-cluster F1 averages .9550 exact, .7943 batch,
and .7426 streaming. At P=16384, seed 29, it is .9928, .2877 and .2855.
This loss is not universal: at P=1024, seed 29, streaming is .9836 versus
.9062 exact. The all-group ARI can obscure the rare group's much larger loss.

In rare cells, batch retains 90.33% of moment trace, versus streaming's 88.74%,
with mean relative Frobenius errors .1101 and .1236. Yet streaming's mean ARI
is higher. Better moment approximation does not force better clustering under
this fixed nonlinear pipeline. Changing rank also changes anchor selection and
embedding geometry; these comparisons do not isolate a single lost direction
as the causal mediator. No claim that bigger/coherent groups are more useful
follows. Synthetic signed patterns are group labels, not useful/harmful labels.

## Memory and runtime audit

No large-P path forms a P by P tensor. Dense references are confined to
[pure tests](../../tests/test_anchor_graph_pilot.py), with largest P=61 here,
and were run separately from the acquisition. Measured state below comprises
rank-8 F, Z, R, active row IDs, anchor IDs, masses and cluster assignments.

| P | Retained state bytes | Cell tracemalloc peak bytes (range) | Mean 32-observation sketch seconds | Mean stream graph+cluster seconds |
|---|---:|---:|---:|---:|
| 1024 | 344,320 | 1,217,268–1,230,794 | .0092 | .0042 |
| 4096 | 1,376,512 | 4,633,332–4,634,012 | .0204 | .0094 |
| 16384 | 5,505,280 | 18,297,537–18,298,217 | .0747 | .0343 |

State is exactly 336 P + 256 bytes for this implementation/configuration.
Observed allocation scales roughly by four for fourfold P. The traced peak is
for the comparison harness, including multiple method factors, not a clean
single-method peak. Profiling begins after synthetic input creation, so those
inputs and imported libraries are excluded; native library workspaces may also
be missed. Process high-water RSS reached 62,736 KiB (about 61.3 MiB), and is
cumulative across cells. Systemd did not expose MemoryPeak on this host, so no
cgroup peak is claimed. These measurements support the source-derived linear
storage bound at the tested sizes; they are not large-model capacity guarantees.
Largest active-graph row-sum error across all methods/cells is 1.03e-14 or less.

The once-only CPU acquisition finished normally (exit 0), with 2.529 seconds
elapsed in its result timestamps and 2.644 seconds cgroup CPU accounting. See
execution provenance (artifact not distributed in this public snapshot) for exact command, guard settings and unit.
Short timings on a shared host are descriptive, not optimizer speed benchmarks.

## Limits and next decision

Implementation feasibility passes; useful-learning efficacy remains untested.
The research guidance influenced the frozen roster, inclusion of signed/rare
controls, retention of reversals and explicit limits on interpretation. This is
a bounded leaf pilot, not completion of the full research workflow or an
independent audit. Root should review raw claims before updating durable KB pages.

The cheapest next step is independent code/math review. Only after that, a
separately authorized fixed-model probe stream could test whether actual
per-example gradient profiles contain stable groups and how much sketch rank
discards. Ordinary training-step gradients instead test temporal co-movement.
Neither synthetic recovery nor moment trace alone justifies optimizer integration,
a generalization/safety claim, or a broad bandwidth/rank search.
