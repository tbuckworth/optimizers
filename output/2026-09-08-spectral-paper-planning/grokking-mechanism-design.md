# Minimal grokking mechanism measurement and intervention

## Construct-validity gate

The local model is not Nanda et al.'s analyzed model. It takes two tokens rather
than `[a,b,=]`, and uses LayerNorm, GELU, biased MLP layers, a final LayerNorm and
a biased unembedding. In particular, `W_U W_out` is not a linear neuron--logit
map through GELU and final LayerNorm. Nanda et al.'s final-model key frequencies
and phase boundaries therefore cannot simply be imported. Their restricted and
excluded losses are useful precedents, not already validated measures here.

Do not define “rule evidence” by choosing frequencies at the final checkpoint
and projecting logits from the training grid onto
`cos(k(a+b-c))`. Such averaging can turn sparse memorized training outputs into
an apparently rule-shaped component. A Fourier restricted loss becomes an
optional secondary analysis only if the current architecture first passes an
independent final-model necessity/sufficiency ablation on never-trained pairs.

Do not run the causal branch until the readouts distinguish initial from grokked
checkpoints beyond fixed nulls. If they fail, the architecture has not supplied
a valid circuit ruler: revise the measurement on calibration material and test
the revision on fresh states rather than declaring the research question false.
Accuracy curves alone cannot answer it.

## Observational readouts on the fixed confirmation checkpoints

Use only the already fixed steps `0, 100, 500, 1000, 1500, 2000, 2500, 3000,
4000, 6000`. Preserve raw train/held-out CE, accuracy and correct-class margin.
The following measures add resolution without modifying model outputs.

Let `l(a,b)` be the `p` logits, `C=I-11^T/p`, and `z=C l`. Let the circular class
shift be `(S_delta z)_c=z_(c-delta mod p)`. On deterministic pairs for which both
inputs are in the never-trained set, measure the normalized defects

```
E_a = sum ||z(a+delta,b) - S_delta z(a,b)||^2
      / (sum (||z(a+delta,b)||^2 + ||z(a,b)||^2) + eps)
```

and analogously `E_b`, for fixed deltas `{1,2,4,8,16,32}`, plus the exchange
defect for `z(a,b)-z(b,a)`. These do not average training outputs by their known
answer, but they measure a stronger property than correct classification: an
ideal equivariant score function obeys them, whereas a correct classifier with
input-dependent margins or temperature need not. Treat them as supporting
diagnostics, never a necessary-rule veto. Always report held-out logit RMS,
because uniform logits satisfy every symmetry vacuously. Fixed random
token/output permutations and mismatched output shifts supply same-size nulls.

For a representation readout, save the final-token hidden vector immediately
after `ln_final`, just before unembedding. Split the never-trained pairs once,
deterministically and stratified by sum, into probe-fit and probe-evaluation
sets. For every `k=1,...,56`, fit a fixed-ridge linear map on the fit set from the
hidden vector to

```
[cos(2*pi*k*(a+b)/p), sin(2*pi*k*(a+b)/p)].
```

Rank frequencies by fit-set `R^2`, freeze the top five, and report their mean
evaluation-set `R^2`; do not reselect on evaluation. Repeat the identical nested
procedure for (i) the pre-attention residual, which controls for trivial input
decodability, and (ii) at least 20 preregistered permutations of the sums. This
cross-fitted, wholly never-trained probe can reveal a sum representation, but it
is still decodability evidence rather than proof of the model's algorithm.

For a cleanup diagnostic, precompute fixed train-to-held-out symmetry pairs and
matched held-out-to-held-out pairs. Define the **training-specific excess
equivariance defect** as their difference in normalized defect. A decline after
the sum probe has risen is evidence that exceptional train-point behavior is
being removed. It is not called a memorization-circuit ablation: representation
formation also changes both terms, and only the matched difference partly
controls that confound.

Report all trajectories and paired seed values. The expected descriptive
patterns are:

- earlier probe improvement and lower held-out symmetry defects: candidate
  circuit formation;
- similar probe trajectories but an earlier decline in excess defect and
  raw held-out loss after formation: candidate cleanup;
- both changes: mixed mechanism;
- neither, despite an accuracy-timing change: reject this ruler or seek an
  optimizer/weight-decay explanation.

These labels describe temporal ordering across already-diverged optimizer arms;
they are not causal until the branch below.

## One prospective common-state branch

Use seed 100 only to check the readout and choose one **early-plateau fixed
checkpoint** from `{500,1000,1500}`: training accuracy must already exceed 99%,
held-out accuracy must remain below 20%, and the cross-fitted sum probe must not
yet exceed its permutation-null envelope. Freeze the earliest qualifying step
and every numerical rule before inspecting branch outcomes for seeds 101--104.
If no step qualifies, do not substitute an outcome-favorable checkpoint.
Do not exclude seeds 101--104 when the common step lands at a different phase;
show their branch-point readouts. A formation interpretation is unavailable for
any seed whose probe is already above the frozen formation criterion, and is
not supported in aggregate unless at least three branch points are pre-formation.

For each of seeds 101--104, clone the AdamW checkpoint at that same step into
three continuations:

1. **AdamW reference:** reuse the already saved unfiltered continuation and its
   checkpoints; do not restart the completed AdamW run.
2. **Stable-on:** introduce the registered stable hard filter, with its estimator
   newly and explicitly initialized and its registered rank-200, decay-0.99 and
   100-step-warmup settings retained.
3. **Scalar-dose control:** run an identical shadow estimator, compute the hard
   projection `P_t g_t`, but deliver
   `q_t g_t`, where `q_t=||P_t g_t||/(||g_t||+eps)` after warmup.

The scalar arm matches the filter's incoming-gradient norm schedule on its own
trajectory while retaining the raw direction. It does not exactly match the
post-AdamW parameter displacement, decay contribution, or covariance history
after branches diverge. Measure those rather than claiming exact dose matching:
pre-Adam gradient norm/cosine, total delivered parameter-update norm/cosine,
separate AdamW decay norm, basis rank, retained-energy ratio and direction
turnover.

This scalar control is not a substitute for Grokfast, the closest temporal
filtering prior. Before making any grokking-specific novelty claim, add a
same-model Grokfast comparison with its implementation and hyperparameter
selection frozen independently and the same selection budget, steps, timing and
readouts. It need not be folded into this minimal common-state branch unless its
state initialization and switch semantics are first made commensurate.

At the branch point, model parameters, Adam first/second moments and step count,
data split/order, scheduler state, RNG states and all counters must be identical.
There is no Adam reset. The new empty filter state is part of the intervention,
not hidden inherited history; stable-on and scalar-dose receive identical copies.
Before execution, freeze a common absolute checkpoint grid and final step
available in the archived AdamW continuation (for example the registered
checkpoints after the selected branch point, through step 6,000). Run the two
new intervention branches to that fixed endpoint, without threshold stopping
or an outcome-driven extension. Do not rerun AdamW merely to manufacture
missing offset checkpoints. Its already observed outcomes limit the
prospectivity of the comparison, which must be stated explicitly.

The causal estimands are paired fixed-offset differences and fixed-grid AUCs in
the held-out sum probe, symmetry defects, excess defect and raw held-out CE. Show
all four seeds, mean paired difference, SE and signs; no checkpoint is an
independent replicate. Report optimizer steps and synchronized training wall
time separately.

Evidence for **faster formation after intervention** requires stable-on to move
the validated never-trained sum representation readout before the raw-held-out
transition, beyond both AdamW and the scalar-dose arm. Non-degenerate
equivariance measurements can corroborate this, but are not a necessary-rule
test. Evidence for
**later cleanup** instead requires comparable early formation readouts but an
earlier subsequent reduction in training-specific excess defect and held-out
loss. This branch identifies effects only from the chosen checkpoint onward. It
cannot establish why the from-initialization stable trajectory reached that
checkpoint state, nor does temporal ordering constitute a formal mediation
analysis.

## Steelman and scope

The strongest plausible result is that a centered temporal gradient subspace
preferentially transports updates that build an equivariant sum representation,
or preferentially removes train-specific departures once that representation
exists, beyond a matched scalar response. The strongest limitation is equally
important: this filter observes batch-mean variation over time, not agreement
across examples, and AdamW plus strong weight decay can change the competition
without semantic selection. Sparse-parity reversal remains an adverse boundary.
Even a clean positive would be a small-model, within-benchmark mechanism result
with safety motivation about rule versus exception learning—not evidence of
safer language models or general capability improvement.

Primary precedent: [Nanda et al., *Progress measures for grokking via mechanistic
interpretability*](https://arxiv.org/abs/2301.05217). Local implementation and
scope: [`experiments/grokking_model.py`](../../experiments/grokking_model.py),
[`spectral_filter.py`](../../spectral_filter.py), and the
paper-viability assessment (artifact not distributed in this public snapshot).
