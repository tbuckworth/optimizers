# Next mechanism design: preserve mean, project innovations

7 September 2026. This is a prospective concept note, not a frozen protocol,
implementation, launch decision, or optimizer recommendation. It uses the
completed I12 evidence without treating its adverse reset result as proof that
all constructive variants must fail.

## Why this is now the sharpest small intervention

[I12](results.md) completed all96 registered branches and its independent audit
passed. Deleting only Adam's second moment caused extreme but finite first-step
shocks: cohort-mean decay-adjusted step norms were about66--624, versus
0.036--0.044 with inherited moments. The four positive registered contrasts
mostly say that current32 collapsed less than raw, not that resetting `v`
restored useful filtered learning. Full moment resets were milder: soft-target
current32 accuracy sometimes improved, but raw improved much more; redraw
current32 worsened. Thus inherited Adam state matters but does not explain away
the continuing adaptation restriction.

The canonical estimator learns a subspace of **centered** raw-gradient
variation and then projects the uncentered raw gradient. A persistent useful
mean can therefore be absent from the learned covariance directions and be
discarded by delivery. The next constructive question is not another optimizer
reset. It is whether retaining the observer's own mean outside its covariance
subspace restores useful adaptation without surrendering its protection
against persistent-label fitting.

A bounded repository search found this failure mode in the mathematical audit
and project synthesis, but no completed neural-training arm with the rule below.
The prior soft-residual, scalar-norm, lagged, frozen-basis, direct-displacement,
and Adam-state interventions are not the same rule.

## Exact candidate and self-inclusion control

Let `g_t` be the raw minibatch gradient. Preserve the canonical update order:
observe `g_t` once, update the EMA mean and covariance basis, then deliver a
gradient to the unchanged inherited AdamW. Let

`mu_t = beta mu_(t-1) + (1-beta) g_t`, with `beta=.99`,

and let `P_t=V_t V_t^T` be the post-ingest rank-32 projector on that arm's own
state. The candidate is

`h_mean,t = mu_t + P_t(g_t-mu_t) = P_t g_t + (I-P_t)mu_t`.

It preserves the current EMA mean in the complement while projecting the
current deviation from that mean. It does not enlarge the learned rank or call
the mean clean, useful, or unbiased.

Because the canonical mean is post-ingest, this expands exactly to

`h_mean,t = [P_t + (1-beta)(I-P_t)]g_t`
`           + beta(I-P_t)mu_(t-1)`.

Thus the current sample receives outside-subspace coefficient `.01`, not zero.
Moreover `P_t` itself includes the current centered observation: the code uses
`c_t=g_t-mu_t=beta(g_t-mu_(t-1))` in its covariance update before projecting.
The nonlinear eigentruncation means there is no single scalar describing all
of the sample's influence through `P_t`.

Use a second new arm that deletes only the historical-mean term:

`h_leak,t = [P_t + (1-beta)(I-P_t)]g_t`
`         = P_t g_t + .01(I-P_t)g_t`.

Both arms use the same post-ingest `P_t` and the same `.01` direct complement
leak. At a common state their exact difference is

`h_mean,t - h_leak,t = beta(I-P_t)mu_(t-1)`.

This is a more interpretable control than an arbitrary random complement or a
retuned leak coefficient. Across a 500-step branch the contrast is still a
whole-policy trajectory effect: parameters, Adam moments, means, and bases
diverge after the first update.

## Minimal empirical design

Reuse the six late current-trained I9 parents used by I12: seeds100--102 at
steps1500 and2000. Reuse the exact I10 frozen plans and the three target regimes:

- fixed realized corrupted labels;
- deterministic soft expected targets;
- fresh redraw targets from the saved redraw plans.

Run only `mean32` and `leak01_32` for500 updates. That is
`6 parents x 3 targets x 2 new policies = 36 new branches`, or18,000 new
updates. Compare against the36 exact inherited-moment I10 raw/current32 cells;
do not rerun those baselines. Keep Adam moments, counter, learning rate, weight
decay, rank, observer update, target plans, model state, and all RNG streams
unchanged. There is no moment reset, new source training, new seed, or selected
horizon.

Evaluate at the existing horizons0,1,10,50,100,250,500. Keep clean CE and
accuracy on auxiliary and validation splits, fixed and soft training losses,
the fixed-minus-soft realization residual, confidence summaries, every finite
step diagnostic, and complete terminal states. Missing or nonfinite cells stay
missing; no survivor substitution, clipping, retry, or learning-rate repair.

Four separate primary estimates at horizon500 are

`U(mean32)-U(current32)` for soft and redraw, separately for auxiliary clean CE
utility and clean accuracy. Average the two parent steps within seed before the
three-seed mean, retaining every seed and parent. A favorable relative contrast
must be accompanied by positive absolute progress from the common h0; worsening
current32 less is not a learning rescue.

The corresponding `U(mean32)-U(leak01_32)` contrasts test whether the historical
outside-mean term adds value beyond the shared one-percent current-sample leak.
They are specificity contrasts, not proof that any gain is caused by a clean
mean. Fixed-target outcomes are a mandatory safety boundary rather than pooled
with the soft/redraw primaries: report clean utility and additional realization
fitting separately.

## Diagnostics needed for interpretation

At every update retain norms and squared energies for `g_t`, `P_t g_t`,
`(I-P_t)g_t`, `(I-P_t)mu_t`, `(I-P_t)mu_(t-1)`, `h_mean`, and `h_leak`, plus
their exact algebra residuals. Record the actual AdamW total and nominal
decay-adjusted data-step norms and leakage against `P_t`. The observer must
advance exactly once from the raw `g_t`; constructing either delivery must not
observe again or mutate the other arm.

Pre-Adam gradient matching does not match Adam displacement. At each common h0,
add a diagnostic-only reciprocal direct-displacement comparison: from identical
model and inherited Adam state, compute the actual candidate and current32 data
displacements, rescale each direction to the other's actual data-step norm where
both norms are nonzero, re-add the identical nominal decay, and evaluate the
same independent losses. Undefined zero-norm cases remain explicit. This adapts
I9's audited immediate control; it is an artificial one-step directional probe,
not a training policy or evidence that long-run step magnitude is controlled.

At evaluation horizons, measure alignment of the outside-mean term with fixed,
soft, clean, and fixed-minus-soft gradient probes on frozen indices. These are
local signed diagnostics, not semantic labels for individual directions.

## Steelman and falsifiers

**Steelman.** Centering can place a slowly varying useful gradient largely
outside the variance-dominant rank-32 basis. Current32 deletes it every step.
`mean32` restores precisely the observer's persistent complement while still
restricting new deviations; it could recover soft/redraw adaptation without
fully reopening high-variance nuisance directions.

**Persistent-label risk.** Under fixed labels, the realized target force is
persistent rather than mean-zero conditional on the saved dataset. It can enter
`mu_t`, so mean preservation may directly restore memorization. Immediately
after a switch to soft/redraw, the inherited mean also contains old fixed-label
history. Its direct coefficient is `.99^H` (about`.366` at100 and`.00657` at500),
but those numbers are not causal effect fractions because weights, means, bases,
and Adam state co-evolve.

The most persuasive result would be seed-consistent absolute soft improvement,
some redraw improvement, superiority to `leak01_32`, and substantially less
fixed-realization fitting than raw. Improvement shared equally by `mean32` and
`leak01_32` instead favors generic complement leakage. A soft gain coupled to
raw-like fixed memorization reveals a tradeoff rather than a selective rescue.
No gain weighs against this exact mean-preserving rule, not against every
helpful low-rank variant. Soft/redraw disagreement remains informative because
equal conditional mean gradients do not imply equal Adam or self-inclusive
projector updates.

## Status

This design has not been implemented, resource-bounded, source-frozen, or
audited. It is therefore **not ready for GO**. A prospective protocol would
still need exact artifact limits, branch-failure semantics, implementation
tests for update ordering and the two algebra identities, and independent
review before any execution.
