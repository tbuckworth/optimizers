# Construct review: modular-sum representation measurements

9 September 2026. Independent design review only; no checkpoint features or
outcomes were read. This specifies a numerical ruler for the saved-state
analysis. It does not preregister a full-model mechanism or authorize training.

## 1. Exact cross-fitted Fourier probe

### Rows and features

For each training seed, use only its never-trained pairs. Sort rows
lexicographically by `(a,b)`. Within each true-sum stratum `s=(a+b) mod 113`,
sort by the bytewise SHA-256 digest of
`probe-split-v1|seed|a|b`; assign the first `floor(n_s/2)` rows to fit and the
rest to evaluation. This gives disjoint full coverage and a per-sum count
difference of at most one without a library-dependent RNG. Freeze the same row
sets across every arm and checkpoint of that seed.

Extract two feature matrices without mutation:

- `X_final` (128 columns): `ln_final(h[:, -1])`, exactly the vector passed to
  `unembed`;
- `X_pre` (256 columns): concatenate the two token-position residuals
  `tok_embed(x)+pos_embed(position)` before attention.

The second is an additive-input control: a linear probe sees both token
identities but no learned cross-token interaction at that point. Its larger
dimension makes it a strong control, not an exactly capacity-matched one.

### Ridge convention

For each representation and frequency `k=1,...,56`, construct

`Y_k=[cos(2*pi*k*s/113), sin(2*pi*k*s/113)]`.

All arithmetic after feature extraction is float64. From fit rows only, compute
`mu_X`, `mu_Y`, and the single scalar
`r_X=sqrt(mean((X_fit-mu_X)^2))`. If `r_X <= 1e-12`, use the intercept-only
predictor and mark the features degenerate. Otherwise set
`Z=(X-mu_X)/r_X` and solve the mean-loss ridge problem

```
min_W  mean_i ||Z_i W-(Y_i-mu_Y)||_2^2 + 0.001 ||W||_F^2,

(Z_fit^T Z_fit/n + 0.001 I) W
    = Z_fit^T (Y_fit-mu_Y)/n.
```

Use a symmetric linear solve, never an explicit inverse. Predict with
`Yhat=mu_Y+Z W`. This is invariant, up to numerical tolerance, to a common
feature offset, nonzero global feature rescaling and orthogonal rotation; it is
not invariant to arbitrary coordinatewise rescaling or added features.

Define joint paired-frequency R-squared on a set `Q` by

```
R2_Q(k) = 1 - sum_Q ||Y_k-Yhat_k||^2
                  / sum_Q ||Y_k-mu_Y_fit||^2.
```

The fit-target mean is the reference even on evaluation. Retain negative values;
do not clip or replace them. Reject only a nonpositive denominator, which should
be impossible under the stratified production split. Rank frequencies by
`(-R2_fit(k), k)`, so smaller `k` breaks an exact tie. The state-level score is
the arithmetic mean of `R2_eval(k)` for the five fit-selected frequencies.
Save all 56 fit/evaluation values and the selected indices.

### Twenty row-permutation nulls

For null `r=0,...,19` and split `q in {fit,eval}`, hash-sort that split's rows by
`probe-null-v1|seed|q|r|a|b` to obtain a permutation `pi_qr`. Replace each row's
sum by the sum from row `pi_qr(i)`. Permute globally within each half, **not
within sum strata**, and use the same permuted-sum vector for all 56 frequencies.
Fit, select five on null-fit R-squared, and evaluate against the independently
permuted null-evaluation sums. This preserves target marginals and the relations
among Fourier frequencies while destroying the rowwise input--sum relation.
Save permutation hashes, all scores and selected indices. The maximum of 20
null state scores is a finite descriptive envelope; its one-sided rank
resolution is only `1/21`, and it is not a multiple-comparison-corrected test
over 150 states.

### Prospective six-state calibration criterion

Let `S_final`, `S_initial`, `S_pre_final`, and `Nmax_final` be the top-five
evaluation scores for a seed-100 arm at step 6000, its step-0 final-hidden
state, its step-6000 pre-attention control, and the largest step-6000 null.
Before reading those six states, fix the following construct-suitability rule:

```
S_final >= 0.50
S_final - Nmax_final >= 0.20
S_final - S_initial >= 0.20
S_final - S_pre_final >= 0.20
```

Require all four inequalities separately in AdamW, legacy and stable if this is
to be a common cross-arm ruler. `0.50` means at least half of held-out target
variance is explained; `0.20` is a deliberately large descriptive separation,
not a calibrated significance cutoff. Failure asks for a revised measurement
validated on separate states, not rejection of the optimizer hypothesis.

For the remaining seeds, report both (i) the prescribed per-state fit-selected
top-five score and (ii) a fixed-frequency sensitivity score. Freeze the latter
five frequencies by ranking the mean fit R-squared across the three seed-100
step-6000 final-hidden states, then never reselect them for seeds 101--104 or
earlier checkpoints. The first score asks whether *some* five sum modes are
readable; the second tracks the same coordinates over time. Always show the
selected identities and per-frequency curves so frequency switching cannot be
mistaken for growth of one circuit.

## 2. Symmetry and cleanup diagnostics

Let logits be `l(a,b)` and remove their softmax-irrelevant row mean:
`z=l-mean_c(l)`. Define `(S_delta z)_c=z_(c-delta mod 113)`. For each
`delta` in `{1,2,4,8,16,32}`, use all ordered never-trained edges for which both
`(a,b)` and `(a+delta,b)` are never trained, and compute

```
E_a(delta) = sum_edges ||z(a+delta,b)-S_delta z(a,b)||^2
             / sum_edges (||z(a+delta,b)||^2+||z(a,b)||^2).
```

Define `E_b` analogously. For exchange, use each unordered never-trained pair
once (`a<b`) and compare `z(a,b)` with `z(b,a)`. Accumulate numerator and
denominator in float64. Also save centered-logit RMS. Mark a ratio undefined if
its centered RMS is at most `1e-7 * max(1, raw-logit RMS)`; uniform scores are
not perfect equivariance evidence. Verify the shift sign with a synthetic peak.
Mismatched output shifts (`delta+37 mod 113`) and fixed hashed token/output
permutations are supporting nulls.

For the training-specific excess diagnostic, form train-to-held-out (TH) edges
under each `(axis,delta,source-sum)` and held-out-to-held-out (HH) edges in the
same group. Hash-sort each group independently with a frozen namespace and take
`min(n_TH,n_HH)` from each, yielding equal counts and matched source-sum,
transformation and axis distributions. Freeze these edges from split membership
alone. Compute the same aggregate energy-weighted defect in each pool and save

`X_excess = E_TH - E_HH`.

Report `E_TH`, `E_HH`, their denominators/RMS and `X_excess`, not only the
difference. A decline in excess is compatible with cleanup only when the sum
probe was already elevated, train fit remains saturated, held-out behavior
improves, and the decline is not merely train-logit collapse. It remains an
observational train-membership asymmetry, not a memorization-circuit ablation.

## 3. Required deterministic fixtures

1. **Extraction:** manually recompute both-token pre-residuals and the final
   `ln_final` hidden state in a tiny deterministic model; verify hook/manual
   parity and `unembed(X_final)==logits`. Hash parameters/buffers before and
   after to prove no mutation. Assert shapes `(n,256)` and `(n,128)`.
2. **Split and null integrity:** every never-trained row occurs exactly once in
   fit/evaluation, no training row occurs, per-sum counts differ by at most one,
   and repeated construction is byte-identical. Each null is a bijection,
   fit/evaluation permutations differ, and global permutation changes sums; pin
   representative SHA-256 digests.
3. **Ridge reference:** compare the implementation with the displayed normal
   equations on a small rational matrix. Test intercept handling, fit-only
   preprocessing, negative R-squared retention, deterministic tie-breaking and
   equality after common offset, global rescaling and orthogonal rotation.
4. **Known modes:** on a full `p=113` synthetic corpus, put exact sine/cosine
   features for five known frequencies plus deterministic nuisance columns in
   both splits. The five must be fit-selected and have near-one evaluation
   scores under the fixed nonzero ridge; all reported values must match an
   independent direct solve.
5. **No re-selection:** construct fit features favoring frequencies
   `{1,...,5}` and evaluation features favoring `{6,...,10}`. Assert the reported
   score uses `{1,...,5}` and differs from the illicit evaluation-selected score.
6. **Leakage/null:** a fit-row identity feature can obtain high fit scores but
   must fail on unseen evaluation rows. Constant/zero features use the
   intercept-only path. Row-permuted targets must run through the same top-five
   selection pipeline, preserve marginal sums, and not acquire the true
   synthetic-mode score.
7. **Symmetry orientation:** logits
   `z_c(a,b)=cos(2*pi*(c-a-b)/p)` give zero `E_a`, `E_b` and exchange defect,
   even after arbitrary rowwise scalar logit offsets. The wrong shift gives a
   nonzero defect.
8. **Necessary caveats:** a positive input-dependent temperature times the same
   template keeps 100% classification while producing nonzero equivariance
   defect; a constant wrong class offset can be equivariant while inaccurate;
   uniform logits trigger the undefined-energy path.
9. **Excess matching:** adding deterministic non-equivariant perturbations only
   to training-pair logits makes matched `X_excess>0`; removing them returns it
   to zero. Check equal group counts, no outcome-dependent edge choice, and show
   separately that train-only temperature changes can also raise excess without
   bespoke memorization.

## 4. Construct limits and defensible interpretation

- The 56 real sine/cosine pairs span every zero-mean function of modular sum.
  A fit-selected top five therefore measures cross-fitted **linear decodability
  of some modular-sum modes**, not discovery of Nanda et al.'s five-frequency
  circuit. Selection on fit rows is legitimate but searches 56 targets; mirrored
  null selection and the fixed-frequency sensitivity are essential.
- Since the final hidden state feeds a linear unembedding, a large late score
  may partly restate successful held-out classification. The scientifically
  useful observation would be a reproducible rise before raw held-out loss or
  accuracy changes. Even then it is a progress marker, not causal use.
- The top-five set may change across checkpoints and optimizers. Without the
  saved identities/per-frequency trajectories, comparing scalar summaries can
  conflate representation growth with frequency switching.
- Cross-fitting prevents training-row averaging from manufacturing evaluation
  R-squared, but the target is still chosen from the known task rule. It tests a
  hypothesis; it does not discover an algorithm. Row permutations test arbitrary
  association, not every structured alternative such as difference, one-token
  shortcuts or another group representation.
- Low symmetry defect is supporting evidence for a canonical equivariant score
  function, not necessary for correct modular addition; input-dependent margins
  can violate it. Conversely an equivariant but output-shifted classifier can be
  wrong. Raw held-out CE/accuracy/margin must remain alongside it.
- Final-hidden improvement over the additive-input control supports learned
  cross-token sum information, but neither probe localizes it to attention/MLP
  components nor proves the model uses it. A future activation intervention is
  needed for causal circuit attribution.

The strongest defensible endpoint is therefore: **the optimizer arms differ in
when cross-fitted modular-sum information becomes linearly readable, and this
timing is or is not accompanied by loss of a matched training-specific
equivariance excess.** Across already-diverged trajectories that is descriptive
evidence consistent with formation or cleanup. It is not mediation, a Fourier
circuit proof, or a safety/general-optimizer claim.
