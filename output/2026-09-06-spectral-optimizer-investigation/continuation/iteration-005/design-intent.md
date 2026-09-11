# Common-gradient covariance replay: prospective design intent

Prepared after iteration004's audited result and commit c52d2ce, before any
iteration005 run. This is a design request, not pilot/full execution approval.

Question: when the gradient stream is identical, does widening the stable
estimator from 32 to 128 better approximate an uncapped finite-history
covariance and its leading rank-32 space? Does that change independently
measured clean/fixed-corruption gradient retention? This separates estimator
approximation from feedback through changed training trajectories. It does
not establish which policy would learn better or identify semantic denoising.

Proposed economical scope: replay the same three AdamW baseline trajectories
(seeds 3,4,5, nominal replacement .9, 2,000 steps) used in iteration004, with
the unchanged model/data/optimizer/RNG recipe. Two canonical observers ingest
each identical raw gradient but never alter model gradients or optimizer state.
Require bitwise matching retained baseline checkpoint parameters and warmup
hashes against iteration004. No new test evaluation or checkpoint selection.
Record raw gradients/centered innovations for independent replay; estimator
states and new clean/noisy/residual/disjoint-clean probe gradients at a small
fixed schedule such as steps 200,500,1000,2000. Freeze the final schedule,
probe RNGs, state timing and all metrics before a runtime-only pilot.

Reference: build the untruncated covariance from the same rounded innovations
and canonical initialization (first nonzero innovation is overweighted, not
silently regularized). Compute its leading space using the weighted-innovation
Gram matrix, not a 50,890-square matrix. Distinguish this reference from a
population covariance, an exact infinite-history object or the production
estimator's residual/eigenvalue pruning rules. Check initialization, orientation,
weights, reconstruction, finite values, eigen-residuals and spectral ties on
synthetic small dense cases before relying on a large dual calculation.

Candidate metrics: relative covariance reconstruction error, captured reference
covariance energy versus optimal rank32, projector/subspace discrepancy with
explicit eigengap/positive-rank gates, and common-state probe retention plus
clean-minus-fixed-corruption retention. Better covariance fidelity need not
improve useful-gradient selectivity. Current-gradient self-inclusion must be
kept separate from independent-probe retention. Report all seeds and scheduled
states; no outcome-selected window, independent-step inference or equivalence.

Resource constraint: the workspace disk has only about 2.2 GB free. Bulky full
gradient/state artifacts must go in a uniquely scoped directory beneath
`/tmp/spectral-experiment-artifacts/`, with tracked hashes/sizes and explicit paths. Do not
delete prior evidence. Available RAM is about 49 GiB; the existing RTX3090 has
24 GiB. Require a bounded runtime/memory pilot and source/data/artifact bindings
before confirmation. Local compute only, no paid APIs/cloud, no production or
previous frozen source edits. Parent owns the final GO and commits.
