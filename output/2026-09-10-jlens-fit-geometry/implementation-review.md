# Saved-fit implementation acceptance

Main acceptance before the one primary calculation, 10 September 2026.

Source in the J-Lens worktree is frozen at
`7b2905481ff835658c3a380a0eea10594860fbeb`, committed through normal hooks.

- `output/2026-09-10-j-lens-fit-geometry/analyze.py` (147 lines):
  `05f3805a510038abed62aeef2df98c3c78cdf637b80960a14121dfecb2b38d73`.
- `output/2026-09-10-j-lens-fit-geometry/test_analyze.py` (121 lines):
  `fe12e1d186e624e58ef356ea8c8541ac965d467e02a886b54fddc02365d2ac93`.

Main read both files completely and independently ran all eight fabricated
tests: PASS, exit0. Tests cover pure/mixed/zero decompositions, sign/scale/
translation, content-dependent framing, malformed input/metadata/order,
held-out text exclusion, existing/dangling output guards and failed-attempt
consumption. No test reads scientific arrays. The separate protocol reviewer
accepts the algebra and exact-score provenance, noting that the old ddof1
standard deviation must not be substituted for the population variance.

The source accesses only `fit_scores64` with `allow_pickle=False` and a
closed NPZ context, after verifying three exact pins. All metadata and the
literal paired-framing relationship are checked. It uses float64 row-weighted
moments, retains all rows/axes, validates both variance identities, and writes
exclusive attempt/results/receipt paths. Source and input hashes are checked
again before publishing results. No original producer import, model/PCA,
eigendecomposition, held-out projection or grading entrypoint exists.

**Admit one primary CPU invocation** from that worktree:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 output/2026-09-10-j-lens-fit-geometry/analyze.py analyze`.
Use a 60-second external timeout. Its inputs are a 95KB archive plus metadata,
not a model; no paid resource or external service. The exclusive stage paths
and resulting process/receipt are authoritative if observation is interrupted.
Do not restart a consumed invocation. An independent scalar-output audit may
subsequently verify the saved calculation without rerunning the producer.

The continuation and implementation-check skills supplied existing-state,
safe archive and normalization discipline. The user's autonomous authorization
governs; no approval gate, new workflow issue or scheduler was introduced.
