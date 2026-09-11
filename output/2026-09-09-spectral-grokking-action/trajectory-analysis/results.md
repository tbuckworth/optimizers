# What the saved continuation histories show

Codex · Spectral Optimizer Investigation · 9 September 2026

**Exploratory, five-seed saved-JSON evidence.** All ten completed policy histories
and every step1502–2500 are included. This is new descriptive analysis of existing
logs, not new training or a new endpoint evaluation. The
[fixed plan](plan.md) was written after the accepted endpoint findings. Main had
also already performed a read-only preliminary rank/minimum-k scan before this
summarization. The three intervals below were fixed from the original endpoints,
not selected for favorable trajectory behavior.

## Main observations

All9,990 recorded thin bases have full **column** numerical rank under the
frozen tolerance. The rank equals the stored column count k throughout:117–200
in orthogonal histories and153–200 in norm-matched histories. This is at most200
directions among227,313 parameters, not a full-parameter-space rank claim.
Orthogonal histories' lower k is therefore not an additional numerical-rank
loss introduced by the QR/SVD action substitution. It is a difference in the
stored legacy estimator along the already diverged policy trajectories.

The norm-matched branch's multiplier c is not a nearly fixed replacement for
a learning-rate setting. Whole-window within-seed medians are1.023–1.172, but
within-seed means are1.401–1.997; all-step c ranges0.963–216.714. Thus the means
and maxima reveal substantial upward excursions that a single initial or median
scale would omit. These observations do not show whether those excursions help,
harm or merely accompany learning, and c is not an actual Adam movement ratio.

Equal-seed means of the per-step **delivered/raw incoming gradient norm ratio**
are0.8240 for orthogonal and1.3583 for norm-matched over the whole interval.
These are descriptive summaries of different branch-local states, not a
common-gradient dose comparison. Both reduce from the early to the late window
in the five-seed mean, but this is not universal per seed: orthogonal seed102
and norm-matched seed103 increase. Every per-seed/window result is retained.

All4,995 norm matches are nondegenerate, unclamped and within the original
float32 tolerance. Maximum relative post-cast mismatch is2.0403×10⁻⁸ versus
the frozen1.1921×10⁻⁶ tolerance. The recorded “exact” flag describes the
denominator domain (unclamped, or both norms zero), not exact floating-point
equality. Post-cast matching is verified by the separate tolerance check.

Within the norm-matched histories, hypothetical native-versus-orthogonal action
cosines have whole-window seed means0.8210–0.8776; individual values range
0.01133–0.99999999. Unequal gains can therefore change incoming direction
substantially at some branch-local states. There is no corresponding cosine
recorded along the orthogonal branch: its action computation intentionally
does not form native. These are not cross-branch or across-time span angles.

The projected/raw norm fraction rho along norm-matched histories has a
whole-window equal-seed mean of0.84942956 and minimum0.343163. This is a norm
fraction, not squared energy. In exact orthogonal geometry it determines the
local equal-length raw-versus-projected action cosine; it does not establish
equivalence of a future raw-direction continuation.

## Every seed and fixed window

W=1502–2500 (999 updates), E=1502–2000 (499), L=2001–2500 (500).
Each row's numerical rank equals k at **every** update. Ratio means/medians are
across that row's updates, not independent-seed estimates. All exact values,
raw/delivered norms, singular-value summaries and undefined counts are in
[summary.json](summary.json). There are no undefined reported ratios/cosines.

| Seed | Policy | Window | k range | k mean; median | Delivered/raw range | Ratio mean; median |
|---|---|---|---|---|---|---|
|100|orthogonal|W|159–200|195.612;199|0.3474–1.0000|0.7688;0.8004|
|100|orthogonal|E|159–200|191.862;196|0.6998–1.0000|0.8797;0.8724|
|100|orthogonal|L|192–200|199.354;200|0.3474–1.0000|0.6581;0.6070|
|100|norm-matched|W|196–200|199.829;200|0.3535–52.7113|1.1573;0.9750|
|100|norm-matched|E|196–200|199.657;200|0.5418–52.7113|1.5177;1.0463|
|100|norm-matched|L|200–200|200.000;200|0.3535–1.0000|0.7976;0.8491|
|101|orthogonal|W|135–200|175.187;177|0.6767–1.0000|0.9201;0.9325|
|101|orthogonal|E|136–200|176.317;175|0.8068–1.0000|0.9455;0.9474|
|101|orthogonal|L|135–200|174.060;179|0.6767–1.0000|0.8948;0.8912|
|101|norm-matched|W|153–200|195.939;200|0.4938–82.3147|1.6505;1.0367|
|101|norm-matched|E|153–200|192.162;199|1.0000–82.3147|2.1533;1.3971|
|101|norm-matched|L|194–200|199.708;200|0.4938–41.1831|1.1487;0.9284|
|102|orthogonal|W|131–200|177.678;181|0.6686–1.0000|0.9008;0.8949|
|102|orthogonal|E|138–200|179.337;177|0.6901–1.0000|0.8975;0.8912|
|102|orthogonal|L|131–200|176.022;182|0.6686–1.0000|0.9041;0.8981|
|102|norm-matched|W|188–200|199.207;200|0.4922–36.6027|1.2101;1.0096|
|102|norm-matched|E|188–200|198.427;200|0.8603–36.6027|1.4569;1.1617|
|102|norm-matched|L|199–200|199.986;200|0.4922–10.5478|0.9638;0.9865|
|103|orthogonal|W|117–200|175.815;187|0.3730–1.0000|0.7319;0.7043|
|103|orthogonal|E|153–200|193.172;199|0.4743–1.0000|0.7683;0.7492|
|103|orthogonal|L|117–198|158.492;152|0.3730–1.0000|0.6956;0.6627|
|103|norm-matched|W|195–200|199.822;200|0.4658–20.1687|1.1780;0.9999|
|103|norm-matched|E|200–200|200.000;200|0.7007–1.7591|1.0297;1.0002|
|103|norm-matched|L|195–200|199.644;200|0.4658–20.1687|1.3261;0.9994|
|104|orthogonal|W|122–200|185.583;192|0.4219–1.0000|0.7984;0.8169|
|104|orthogonal|E|166–200|192.096;198|0.5379–1.0000|0.8606;0.8963|
|104|orthogonal|L|122–200|179.082;188|0.4219–1.0000|0.7364;0.7257|
|104|norm-matched|W|174–200|198.900;200|0.4419–215.2948|1.5958;1.0022|
|104|norm-matched|E|174–200|198.160;200|0.4679–60.4890|1.6283;1.0609|
|104|norm-matched|L|192–200|199.638;200|0.4419–215.2948|1.5633;0.9864|

The c statistic below divides hypothetical native norm by orthogonal norm
**within each norm-matched branch state**; the previous table instead divides
the actual delivered norm by that branch's raw gradient norm. Neither divides
by the archived native run or by the orthogonal branch's gradient.

| Seed | Window | c minimum–maximum | c mean | c median |
|---|---|---|---|---|
|100|W|0.9632–58.0830|1.4005|1.0228|
|100|E|1.0000–58.0830|1.7876|1.3497|
|100|L|0.9632–1.2833|1.0142|1.0010|
|101|W|0.9633–82.4915|1.7810|1.0834|
|101|E|1.0000–82.4915|2.2457|1.4198|
|101|L|0.9633–41.1895|1.3173|1.0170|
|102|W|0.9818–38.6952|1.4221|1.1724|
|102|E|1.0000–38.6952|1.6510|1.3737|
|102|L|0.9818–13.2563|1.1936|1.0186|
|103|W|0.9809–20.8904|1.4526|1.0819|
|103|E|0.9984–3.0612|1.3703|1.0407|
|103|L|0.9809–20.8904|1.5348|1.1907|
|104|W|0.9779–216.7137|1.9965|1.1088|
|104|E|1.0000–103.3330|2.1260|1.1609|
|104|L|0.9779–216.7137|1.8673|1.0430|

Whole-window native-versus-orthogonal cosines, norm-matched branch only:

| Seed | Minimum–maximum | Mean | Median |
|---|---|---|---|
|100|0.05837–1.00000|0.87763|0.98127|
|101|0.01370–1.00000|0.82224|0.93218|
|102|0.05460–1.00000|0.83938|0.87315|
|103|0.06103–1.00000|0.85318|0.94303|
|104|0.01133–1.00000|0.82102|0.90449|

## What this changes, and what it cannot answer

The logged post-fork differences are not explained by numerical column-rank
truncation in the alternative action: no such truncation occurs. The trajectories
do differ in the estimator's retained column count, incoming gradient scale and
counterfactual within-span weighting. Those are descriptions of simultaneously
evolving systems, not separable causal fractions.

The scalar control remains dependent on learned V and on the evolving branch
gradient. The sizable mean/median gap and large extrema sharpen a requirement
for a future discriminator: a single initial norm match is not an adequate
description of the intervention. These histories alone do not identify the
right replacement control law, nor establish that spikes are necessary or
beneficial. Any additional control must be specified prospectively, preserve
inherited state and measure useful learning, not just match a scalar summary.

There are no raw gradient vectors, oriented bases or full Adam updates recorded
for these999 later steps. Hence these JSONs cannot recover cross-branch span
angles, subspace turnover, later actual parameter displacement, carried-moment
mediation, curvature alignment or feature semantics. The already audited common
step1501 is copied into a separate JSON section; its actual Adam displacement
measurements must not be presented as if observed throughout the continuation.

This adds no new generalization, memorization, wall-clock or safety endpoint.
The [accepted action report](../../../research/grokking_action_mechanism_2026-09-09.md)
still owns those findings and their limitations.

## Provenance and verification scope

The [summary](summary.json) binds all10 history JSON hashes before/after
processing, the accepted batch and five completion hashes, the original ten
scientific source hashes, and the preexisting accepted first-step tensor audit.
It verifies all seed/policy/step/checkpoint metadata and999 contiguous rows per
history, plus rank and norm-matching arithmetic. It does not reopen checkpoints.

**The original completion receipts did not individually hash the history JSON
files.** Current hashes and accepted-checkpoint metadata provide a strong
consistency check, not retrospective cryptographic proof of unchanged historical
scalar contents. Later histories have not received an independent tensor audit.
This report must not inherit the stronger audit label of the prior endpoint and
first-step tensor reports merely because it uses the same completed batch.