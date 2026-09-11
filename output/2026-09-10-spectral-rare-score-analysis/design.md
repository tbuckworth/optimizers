# Fixed design: post-hoc rare-class score analysis

Written before reading numerical saved logits, 10 September 2026. This is a
**POST-HOC saved-logit analysis** of completed trajectories, not a new trained
policy, fresh replication, or a prospectively registered training endpoint.
The earlier reports and audit remain unchanged.

## Inputs and scope

Analyze every one of the 36 completed trajectories: seeds 202609111,
202609112, 202609113; cells clean, diffuse, shared, sham; policies raw,
native32, norm_raw. Use unpatched held-out logits at updates 100 (common
warmup) and 2000 (fixed endpoint) only. There are 500 held-out examples per
true class, hence 500 rare digit-8 positives and 4500 other-digit negatives
per seed. Training true-label counts are 50 digit-8 and 550 per other digit;
warmup excludes digit 8. Corrupted assigned-target priors need not equal these
true-label priors.

Accepted acquisition:
`/tmp/spectral-experiment-artifacts/spectral-selectivity-boundary-20260910.2IruKP/acquisition-001`.
Before opening any arrays, verify completion SHA256
`de66b0331bec90c9c24585ab9874af447de17cc653661984c0237fb65da66ceb`,
sibling `audit-001/result.json` SHA256
`d1b3a47371d8eed4d8fe46ec4b0c1cd3586f788a69cd003b264a7c0b39dd3907`,
and the byte lengths and SHA256 of all 36 logits archives and three plan
archives against completion receipts. Check the frozen acquisition source
against its completion pin to establish layout. Open only the steps member,
unpatched held-out slices at the two selected steps, and held-out labels/IDs.
Check all 12 warmup slices per seed are identical; repeated logical rows are
not independent replicates. Do not rerun the original audit.

## Fixed metrics

Compute in float64, with stable log-sum-exp and exact equality for score ties.
For each example with ten logits z, define the rare score
`s = z[8] - logsumexp(z[j] for j != 8)`. This is the softmax log-odds of digit
8 against the other nine classes. One-vs-rest AUROC is
`P(s_positive > s_negative) + 0.5 P(s_positive == s_negative)`.
Use a tie-aware average-rank implementation, validated before reading real
arrays against independently coded pairwise comparisons on fabricated known
perfect, reversed, all-tie, mixed-tie and permuted fixtures. No threshold is
fitted or selected; no best held-out threshold is computed.

For original logits and a single fixed analytical transform
`z_adjusted[8] = z[8] + log(11)` (other logits unchanged), report:

- rare digit-8 accuracy and mean multiclass cross-entropy;
- common accuracy and CE, macro-averaged over the nine other true classes;
- balanced accuracy and CE, macro-averaged over all ten true classes;
- the fraction of common examples predicted as 8 (a decision tradeoff);
- rare AUROC, including a check that the fixed shift preserves its ranking.

The multiplier 11 is fixed from 550/50 training true-label counts and equal
held-out counts; it is not chosen from held-out scores or performance.
Ordinary argmax uses the lowest class index if logits tie, matching NumPy and
the acquisition's original argmax convention. CE uses natural logarithms.

## Reporting and interpretation

Retain all seeds, cells, policies, both steps and both analytical views in
numeric JSON. Summaries give mean, sample SD, and the three seed values;
contrasts include endpoint minus warmup and endpoint native32 minus each
control under each view. There are only three paired seeds, not 36 independent
replicates. Describe rare/common tradeoffs and adverse results together.

Higher-than-warmup rare AUROC establishes improved continuous rare-versus-rest
ranking on these saved examples even if argmax accuracy is zero. A fixed
logit shift can expose decision-rule sensitivity; it does not identify the
training cause or establish a calibrated or Bayes-optimal prediction rule.
Especially under diffuse corruption, label noise and model misspecification
invalidate a simple automatic prior-shift interpretation. A remaining AUROC
gap cannot be removed by any additive constant on the specified rare log-odds
score. Neither result proves a pure class-prior explanation or its absence.

## Execution bounds