# Independent report review

9 September 2026

**Status: PASS, with interpretive qualifications.** I found no blocking
arithmetic error or unsupported central causal claim in
`research/grokking_representation_mechanism_2026-09-09.md`. This review used
only the committed scalar `summary.json` and `paired-descriptive.json`; it did
not read checkpoint arrays or rerun inference.

## Numerical cross-check

The report's principal values reproduce the accepted summaries:

- At steps 2,000 / 2,500 / 3,000 / 6,000, the cross-seed mean selected-probe
  R² and held-out accuracies in the table agree with `summary.json` after the
  displayed rounding.
- Legacy minus AdamW selected R² is +0.0897727 ± 0.0381261 SE at step 2,000,
  +0.4559583 ± 0.1062833 at 2,500, and +0.6109934 ± 0.0469347 at 3,000.
  Each contrast has five positive paired seed differences.
- The largest legacy held-out accuracy at step 2,000 is 0.0121938, supporting
  the stated 1.22% maximum. The seed-101 and seed-100 examples at step 2,500
  also agree: R² 0.207835 / accuracy 0.9509%, and R² 0.863223 / accuracy
  95.380%, respectively.
- Stable minus AdamW is +0.0891892 ± 0.0537202 at step 2,500 (3/5 positive)
  and +0.2628488 ± 0.1183677 at step 3,000 (4/5 positive).
- The complete-state pre-attention selected-R² range is
  [−0.0805369, −0.0580854], and the final-hidden permutation-null maximum
  range is [−0.0327609, −0.0202805].
- The fixed panel is `{9, 33, 32, 49, 11}`. Its final cross-seed means are
  0.361287 / 0.331883 / 0.372086 for AdamW / legacy / stable. Legacy seed 103
  selects `{15, 34, 40, 1, 30}`, with selected R² 0.979748 and fixed-panel R²
  −0.0336234, exactly supporting the no-overlap example.
- The reported correct- and wrong-shift means at steps 3,000 and 6,000 agree.
  The step-2,500 TH−HH means are −0.0079580 / −0.0363064 / −0.0231682;
  all three approach zero from below after their transition minima.
- Stable's final mean selected R² (0.985977) and correct-shift defect
  (0.0386384) are the most favorable of the three, while its mean margin is
  6.81086 versus AdamW's 8.45254. Its final cross-entropy exceeds AdamW's in
  four of five paired seeds, as stated.

The linked prior behavioral thresholds and the independent raw-array
calibration residual are provenance claims outside this two-JSON arithmetic
check. The three displayed seed-100 calibration scores themselves are present
in and agree with `summary.json`.

## Scientific interpretation

The strongest defensible finding is earlier **cross-fitted linear readability
of some modular-sum Fourier modes** in the legacy trajectory. At step 2,000,
legacy has mean selected R² 0.089625 versus AdamW −0.000147, all five paired
differences are positive, and every legacy seed remains at or below 1.22%
held-out accuracy. This is genuine readability-before-useful-classification
evidence. It is not evidence that nothing behavioral has changed: legacy's
held-out loss and margin have already improved, which the report explicitly
acknowledges.

The headline score selects five frequencies from fit rows separately for each
state and evaluates them on disjoint held-out rows. That avoids evaluation-row
selection leakage, but it means arm and seed comparisons need not concern the
same Fourier modes. The report should therefore continue to describe the
finding as an internal progress marker, not a common five-mode circuit. The
weak fixed-panel transfer is consistent with different dominant modes across
seeds, but does not by itself identify distinct circuits or make the seed-100
panel a universal ruler. The report handles this limitation well. If edited,
the opening phrase “more linearly readable earlier” would be maximally precise
as “more linearly readable earlier under the per-state fit-selected readout.”

The two negative controls have appropriate scope. The shuffled-target result
shows the late score is not reproduced by these fixed permutations. The
pre-attention result rules out a linear additive readout from the two separate
token residuals; it does not rule out nonlinear information already implicit
in the inputs. Neither is a multiplicity-corrected significance test, as the
report states.

The symmetry result is supporting functional evidence, not an independent
mechanism measurement: an equivariant logit template algebraically makes sum
modes readable, while input-dependent margins or temperatures can separate
classification from exact equivariance. The negative TH−HH excess during the
transition fails to show the proposed positive “extra training-origin defect”
signature. Because the normalized ratio can also change with
input-dependent logit energies and temperatures, its negative value does not
establish reverse cleanup. The report retains the sign and makes neither
claim.

Stable's mixed middle-checkpoint contrasts and its favorable final readout /
symmetry but worse final CE provide an important internal contradiction to a
simple “higher probe score means better optimizer” story. Preserving that
result materially strengthens the report's scope discipline.

## Step-1,500 intervention decision

Keeping the already fixed step 1,500 is justified by the completed curves and
should not be changed to chase a larger effect. At that checkpoint all five
legacy seeds have 100% training accuracy, held-out accuracy ranges only from
0.0671% to 0.2349%, and selected R² ranges from −0.02272 to 0.00226. It is thus
before the marked saved-state legacy readability separation, while lying after
memorization. A branch there can distinguish effects on subsequent emergence
of readable information from effects appearing only in later task use.

One non-scientific edit remains: “recorded90%” in the report should be rendered
as “recorded 90%.”
