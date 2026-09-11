# Prospective aggregation specification

Prepared before the development pilot or confirmatory outcomes. This completes
the estimands in [protocol.md](protocol.md), without changing the training recipe.

Co-primary metrics are `test.max_val_accuracy.accuracy` for hard32-minus-AdamW
and hard32-minus-scalar32_norm. Report both, every seed, and the descriptive
mean, median, minimum, maximum and sample SD. Accuracy is a fraction in JSON
and percentage points when reporting accuracy differences in prose. Three seed
pairs are three observations; there are no step-level or example-level p-values,
confidence intervals, significance or equivalence claims.

Mandatory secondary learning metrics include test accuracy and CE at all four
checkpoints, checkpoint steps, selected validation accuracy and CE, final clean
and noisy training metrics, and each final/selected checkpoint's test difference
from its own common step-100 warmup checkpoint. The latter is a within-run
comparison, not another independently trained arm. Report all six ordered
contrasts: hard32-AdamW, hard32-scalar32, wide-hard32, wide-AdamW, wide-scalar32,
scalar32-AdamW. Both selectors and endpoints remain visible regardless of rank.

Verify the complete 0,100,...,2000 validation grid and rederive each earliest
strict selector from its raw values. Verify all 2,000 step records, fixed arm
and seed labels, all 12 training runs completed before official test loading,
48 complete test evaluations, and common warmup test metrics within seed.
The summarizer refuses incomplete runs or output overwrite and does not load
development data or checkpoints. Independent checkpoint replay is a separate
post-run audit, not a substitute for the launch gate.

Secondary geometry uses the same predeclared windows as iteration003: all
postwarmup steps 101-2000, early 101-500, late 1501-2000. Aggregate each seed
first. Report mean raw/candidate/applied norms and squared norms, per-step
candidate/raw and applied/raw norm ratios, raw/applied cosine and scalar alpha
where defined. These are own-trajectory quantities. Do not interpret ratios
of separately averaged quantities as paired counterfactual norm matching.

For total and nominal decay-subtracted displacement, report mean norm and
squared norm, raw/applied gradient dots and cosines, and tolerance-qualified
positive-dot frequencies. All scheduled steps belong in frequency denominators,
including near-zero classifications. Decay-subtracted displacement is the
primary update diagnostic; total displacement is mandatory, not a favorable
alternative chosen later. Neither removes historical effects of decay.

Use arithmetic means of finite per-step values, retaining null counts and exact
null-step masks. A paired metric average requires all three seed pairs and
identical null-step masks within each pair; otherwise its group statistics are
null with the unavailable seed list. Undefined hard-arm alpha is not zero.
Scalar norm/direction gate maxima are descriptive diagnostics. No scalar
orthoprojector leakage, pure mediator identification, learning-rate matching or
semantic-denoising inference is authorized by this analysis.

[summarize_results.py](summarize_results.py) is independent of training imports;
[test_summary.py](test_summary.py) uses synthetic records only. Both files and
this document are included in the pilot/source/commit binding before launch.
