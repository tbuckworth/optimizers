# Independent audit: modular-addition confirmation

8 September 2026. Audit of the complete prospective five-seed comparison. I
did not run another scientific training job or select/reject any completed arm.

## Disposition

**SUPPORTED, with material qualification.** All 15 registered seed-arm results
are present, source-consistent, complete to 6,000 updates and arithmetically
sound. The fresh legacy arm reproduces the historical modular-addition step
advantage. The current stable update has a smaller and substantially less
consistent first-crossing advantage over AdamW, and it is later than legacy in
every seed. It therefore preserves some descriptive evidence of the phenomenon
but does not improve the historical result or establish a robust stable-over-
AdamW effect with n=5.

No acquisition defect warrants discarding, replacing or rerunning an arm.

## Independently reconstructed results

The endpoint is the first scheduled post-update evaluation with held-out
accuracy at least 90%; observations are spaced by 50 updates.

| Arm | First crossing, mean step | Sustained crossing, mean step | Final accuracy | Mean final loss |
|---|---:|---:|---:|---:|
| AdamW | 3,930 | 4,040 | 1.000 | 0.004382 |
| Legacy filter | 2,810 | 2,810 | 1.000 | 0.002355 |
| Stable filter | 3,460 | 3,980 | 1.000 | 0.009321 |

The planned paired first-crossing contrasts are:

| Contrast | Seed-wise differences in steps | Mean +/- SE | Favorable to stable |
|---|---|---:|---:|
| Stable minus AdamW | -500, +250, -1200, -650, -250 | -470 +/- 238 | 4/5 |
| Stable minus legacy | +500, +1500, +50, +350, +850 | +650 +/- 248 | 0/5 |

Negative means an earlier stable crossing. For context, legacy minus AdamW is
`-1000, -1250, -1250, -1000, -1100`, mean `-1120 +/- 56` steps, favorable in
5/5 seeds. That closely reproduces the old mean difference of about 1,170
logged epochs on fresh seeds.

First crossing can be transient. Stable seed 100 first crosses at step 3,000
but does not remain above threshold until step 5,600. Consequently the stable-
minus-AdamW sustained-crossing contrast is only `-60 +/- 468` steps (3/5
favorable), while stable minus legacy is `+1170 +/- 541` (0/5 favorable). This
secondary endpoint is a necessary counterweight to describing the mean first-
crossing result as a general 470-step acceleration.

All arms reach 100% recorded final held-out accuracy. The comparison is thus
about when generalization appears and persists on the measurement grid, not
final capability at 6,000 updates.

## Integrity and arithmetic checks

I independently read the two completion records and all 15 hash-linked metrics
files, without importing the collector. Checks passed for:

- the exact Cartesian roster, seeds 100--104 by AdamW/legacy/stable, with no
  duplicates or missing arms;
- identical frozen acquisition-source hashes across the seed-100 and seed-
  101--104 batches;
- exact configuration, software/hardware record and parameter layout across
  arms except the registered seed, arm and stable/legacy flag;
- identical within-seed split hashes, 3,830 training and 8,939 held-out pairs;
- 121 ordered evaluations per arm, step 0 then steps 50--6000, with the stated
  initial/post-update phases;
- all embedded first and sustained thresholds against the raw accuracy rows;
- all 15 per-seed final metrics, arm means, paired differences and sample-SD /
  square-root-5 standard errors against the summary.

This reconstructs 1,815 evaluation rows. The accepted compact
`summary.json` has SHA-256
`d90c264a75416dc4ede5fce272af406bf1c6224892b5469b5bf7e57d9014eebf`.
The collector source reviewed had SHA-256
`9e4407ca158df4318ec41bd08d523162203342b5451a239048c80c21dc54f9a6`
and correctly refuses a selected subset, cross-batch source drift, recipe or
environment drift, split mismatch, receipt mismatch, or threshold mismatch.
It also preserves hash-checked compact raw copies and produces both accuracy
and loss curves.

I did not redundantly rehash the full checkpoint corpus after the
launcher had streamed and verified every receipt. I did independently rehash
and safely load, with `weights_only=True`, all 15 step-0 checkpoints. For every
seed, model state, full AdamW state, split identity, and CPU/all-CUDA RNG states
are bitwise identical across the three arms. This directly establishes the
registered paired initialization.

## Warmup numerical check

For seed 100 I additionally rehashed and safely loaded all three step-100
checkpoints. RNG states remain bitwise identical, filter counters equal 100,
and the source applies no filtered gradient until step 101. Nevertheless,
model and AdamW states are not bitwise identical at step 100:

- AdamW versus legacy: maximum parameter difference `1.19e-6`, parameter L2
  difference `5.89e-5`; maximum first-moment difference `1.11e-6` and maximum
  second-moment difference `8.80e-8`.
- AdamW versus stable: maximum parameter difference `7.15e-7`, parameter L2
  difference `2.70e-5`.

The step-100 accuracies are identical; the largest corresponding reported loss
difference is about `5.6e-6`. Static inspection finds no pre-warmup gradient
mutation: the filter observes the gradient and updates its private state, while
gradient replacement is guarded by `step_count > warmup`.

The exact source of this microscopic drift is unidentified. The run did not
enable PyTorch's deterministic-algorithm mode, and PyTorch explicitly limits
same-seed reproducibility in the presence of nondeterministic CUDA operations
([reproducibility notes](https://docs.pytorch.org/docs/2.11/notes/randomness.html)).
It should not be attributed specifically to embedding backward: the v2.11 CUDA
embedding source uses ordered accumulation to ensure determinism for repeated
indices ([PyTorch v2.11 source](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Embedding.cu)). No further GPU diagnostic or
post-outcome rerun is justified here. The correct limitation is that paired
initial conditions are exact, whereas the independently executed CUDA
trajectories are not bitwise counterfactuals even before filtering activates.

## Saved-basis geometry check

I also reviewed the bounded read-only seed-100 basis analysis. It safely loads
five fixed checkpoints for each filtered arm by their acquisition receipts and
computes the eigenvalues of the exact saved fp32 basis Gram matrix in fp64 on
CPU. This is the right calculation: the nonzero gains of the delivered
`V V^T` operator are the eigenvalues of `V^T V`. For these full-column-rank
saved bases, all gains equal one exactly when the columns of `V` are
orthonormal. All ten reported extreme-gain /
spectral-error identities check arithmetically.

The distinction is large. Across steps 100, 1000, 2500, 4000 and 6000, the
legacy operator's minimum gains are `0.000214, 0.00892, 0.0853, 0.0312,
0.0000643`, while its maximum gains are `11.41, 54.84, 7.28, 172.53, 14.45`.
The stable operator stays close to a projector: its gains range from
`0.9999975` to `1.0000112` over these checkpoints, with spectral
orthogonality error between `3.53e-7` and `1.11e-5`.

This supports a numerical interpretation, not a causal learning claim. The
historical legacy operation called a projection is often a strongly
anisotropic gain map on its retained span; the stable path largely removes
that scale distortion. It is therefore plausible that some of legacy's larger
step advantage arose from unintended amplification, but these saved operators
alone do not establish that mechanism. The check covers one seed, samples only
five times, does not measure the realized gradient coefficients in each basis,
and does not intervene on a common training state. Stable and legacy also
retain different basis ranks at several checkpoints, so the behavioral gap
cannot be reduced to orthogonality alone.

The reviewed geometry JSON has SHA-256
`4460ab072636e4d90ed177f0e9c2d9abfd1580800fe1809fba6ed1af02c8b71a`;
its analysis source has SHA-256
`1f8b18fb443260b224df3f924ee380ad8c923bb331b2124858f77c46f0c151c2`.

## Timing and inferential limits

At the fixed 6,000-update horizon, mean synchronized training time is 34.95 s
for AdamW, 57.73 s for legacy and 65.06 s for stable: 1.65x and 1.86x the
AdamW training time. Mean end-to-end time, which includes the much larger
filtered checkpoints, is 37.46 s, 79.01 s and 83.54 s respectively. These
shared-desktop measurements rule out presenting this run as a demonstrated
wall-clock speedup; they are not a dedicated timing benchmark.

The SEs above describe variation across only five paired seeds. They do not
include the 50-step observation interval, benchmark-selection uncertainty,
software nondeterminism, or task/architecture variation. In particular:

- the benchmark was chosen after a historical positive result, so fresh seeds
  address seed reuse but not task-level selection;
- stable versus AdamW has one adverse seed and a mean only about two SEs below
  zero; no headline significance claim is supported;
- stable is later than legacy in all five seeds, but n=5 still does not support
  a universal ordering;
- the seed-100 transient crossing makes the stable sustained result much weaker
  than its first-crossing result;
- modular addition with one small transformer cannot establish general
  optimizer, neural-network, or safety benefits.

The defensible update is therefore: **the legacy modular-addition advantage
replicates strongly on fresh seeds; the stable numerical update retains a
smaller, uncertain first-crossing advantage over AdamW, loses the legacy
advantage in every tested seed, and shows essentially no mean advantage on the
sustained threshold.** Mechanistic claims still require the separately planned
checkpoint analyses and must not be inferred from these learning curves alone.

## Final report and endpoint-supplement cross-check

I independently cross-checked
`research/grokking_stable_confirmation_2026-09-08.md` and
`grokking-confirmation-results/endpoint-supplement.json` against all 15 compact
raw metric files. The endpoint supplement does use the actual
`cumulative_training_seconds` value from each first/sustained evaluation row;
it does not scale the final runtime by the crossing step. Its implementation of
sustained attainment as the row after the final observed sub-90% row is correct
for these data, where every arm finishes above threshold.

The following all reproduce exactly from the raw rows:

- all 15 first and sustained steps, including stable seed 100's last sub-90%
  observation at step 5,550 and sustained attainment at 5,600;
- every per-seed time to first and sustained crossing;
- all arm means and sample-SD / square-root-5 SEs;
- all paired differences and favorable-sign counts;
- every final held-out loss and its paired sign.

In particular, measured training time to both endpoints is greater for each
filter than AdamW in every seed. Stable final loss is lower than AdamW only for
seed 100 and higher for seeds 101--104, exactly matching the report's “worse in
four of five” wording. Legacy final loss is lower than AdamW in three of five;
stable final loss is lower than legacy only for seed 102. The report correctly
keeps these loss comparisons separate from the identical 100% final accuracy.

The report's step percentages, timing table, sustained contrasts, geometry
qualification and inferential cautions are numerically and semantically
supported. I recommend no material wording correction. One operational phrase
was premature at audit time: the report called the compact raw copies
“committed” while Git still showed them as untracked. That becomes true once
the evidence/report commit is made; before then, “collector-created raw metric
copies” is the exact wording. This does not affect the scientific result.

The reviewed endpoint supplement has SHA-256
`02b4d829a63ab4c22476a8cbab785d9160aa74a4ecb7134ab69967da04521354`;
its source has SHA-256
`a279a5bfdd57b37764ac516ac4566ec04519501340c10fa13e2080fe4d594280`.
