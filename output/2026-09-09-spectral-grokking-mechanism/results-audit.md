# Independent audit of the complete saved-state results

9 September 2026

## Verdict

**Pass.** The acquisition contains the exact 150 registered states, and the
saved-array analysis preserves their scalar evidence and applies the frozen
fixed-frequency panel and five-seed aggregates correctly. I found no receipt,
roster, split, leakage, row-null, probe, behavior, margin, held-out shift, panel,
or aggregation discrepancy.

This audit used only preserved JSON/NPZ files. It did not load a model
checkpoint, run inference, retrain, resume, or replace any completed output.

## Complete acquisition integrity

I independently hashed every acquisition result JSON and NPZ in the calibration
and remaining directories. The two manifests and completion records bind an
exact disjoint `6 + 144` roster covering:

- seeds 100–104;
- AdamW, legacy, and stable arms;
- steps 0, 100, 500, 1,000, 1,500, 2,000, 2,500, 3,000, 4,000, and 6,000.

All 150 result and activation receipts match their files, identities, sizes,
stage directories, recipe, environment, and source hashes. The remaining-stage
manifest contains the exact calibration completion hash. Stage byte accounting
also reconstructs exactly: 112,791,618 bytes for calibration and 2,708,992,019
bytes for the remaining states, or 2.628 GiB combined. Both completion records
report success; the remaining 144 states took 576.016 seconds.

For all 300 stored probes (150 states by two representations), I checked the
complete 1–56 frequency sequence, recomputed the selected five solely from fit
R-squared with the registered tie break, checked the corresponding evaluation
values and mean, and repeated the selection check for every one of the 6,000
null fits. Each stored null mean, population standard deviation, and maximum is
internally exact. This all-state check uses the preserved per-frequency scalars;
the independent raw-array equation check below covers the fixed 30-state subset.

## Prespecified independent raw-array subset

Before inspecting intermediate outcomes, the arithmetic subset was fixed as all
five seeds, all three arms, and steps 1,500 and 2,500: 30 states. For each state I
loaded one NPZ at a time and independently verified:

- the full modular grid and finite activation shapes;
- 3,830 training and 8,939 never-trained rows as a disjoint full partition;
- disjoint/full-coverage probe halves of 4,443 fit and 4,496 evaluation rows;
- exact within-sum stratification and exclusion of every training row;
- 20 complete row permutations, their per-null and aggregate hashes;
- identical split/null inputs across arms and checkpoints of the same seed.

Using independent NumPy fp64 normal equations, I recomputed all 56 observed
frequency fits for both final-hidden and pre-attention features in all 30 states.
I also recomputed fixed null indices 0 and 19 for every state/feature, including
fit-only selection. Maximum absolute differences from acquisition scalars were:

| Quantity | Maximum absolute difference |
|---|---:|
| Fit/evaluation R-squared | 4.22e-15 |
| Fit-target means | 3.65e-16 |
| Fit feature transform | 2.22e-16 |
| Train/test CE or accuracy | 2.66e-15 |

All independently selected-frequency lists matched exactly. Behavioral CE and
accuracy were recomputed from logits with a stable NumPy log-sum-exp expression,
not copied from the scalar files.

This fixed subset spans materially different intermediate behaviors rather than
only final successes: held-out accuracy ranges from below 0.1% to 95.4% and the
selected final-hidden probe ranges from negative values to 0.863. This is a
coverage description, not an outcome-selected comparison or a claim about which
arm is preferable.

## Saved-array analysis verification

The completed `analysis-001` bundle contains 150 state receipts, 150 byte-identical
copies of the acquisition scalar JSONs, a 150-row summary, and 30 arm-by-step
aggregate rows. I verified every analysis receipt/hash and its 191,373,962-byte
pre-completion accounting. All seven pinned analysis-source hashes still match.

For every summary row, the following fields equal their source acquisition JSON
exactly:

- train/test loss, accuracy, and count;
- selected frequencies and selected-five evaluation mean for both features;
- null maximum for both features;
- all 56 fit and evaluation R-squared values for both features;
- source scalar and NPZ hashes.

Every analysis scalar copy is byte-identical to its source JSON. Each compact
summary row also matches the corresponding full state JSON after excluding the
deliberately omitted full block-level symmetry tree and the summary-only state
receipt.

I independently reconstructed the fixed panel from mean **fit** R-squared across
the three seed-100, step-6,000 final-hidden calibration records. The result is
`{9, 33, 32, 49, 11}` in ranked order, exactly as stored. For every state and
both feature types, each fixed-panel evaluation value and its five-frequency mean
matches direct lookup from the source per-frequency curve. The panel therefore
does not use seeds 101–104 or evaluation outcomes for selection.

All 30 aggregate rows were independently regenerated from their five constituent
summary rows for every registered metric. Means, sample-standard-error values,
defined counts, and total counts match exactly. This confirms aggregation treats
the five seeds as replicates; it does not treat checkpoints, frequencies, or
symmetry edges as independent observations.

## Independent margin and held-out-shift check

For the same fixed 30-state subset, I independently recomputed correct-class
margins from the archived logits for all training and held-out pairs. Every mean
matches the summary exactly.

I also independently constructed all held-out-to-held-out edges for both input
axes and shifts `{1,2,4,8,16,32}`, row-centered logits, applied the registered
positive output roll, and pooled raw numerators and denominators before division.
Counts match exactly. The maximum difference in a normalized defect or RMS was
`7.99e-15`. The largest raw-sum difference was `4.47e-8` on accumulators of order
`1e8`, consistent with floating-point summation order rather than a scientific
discrepancy.

This independent symmetry check is intentionally partial. It covers the simple
pooled correct-shift held-out diagnostic for 30 prespecified states. I did not
independently reconstruct the wrong-shift, exchange, or hash-matched
training-membership trees from raw arrays; their synthetic fixtures, full stored
trees, receipts, and all-state summary aggregation remain intact, but those
diagnostics should not be described as independently numerically duplicated by
this audit.

## Reproducibility and resource use

The audit is reproducible with
[`audit_results.py`](audit_results.py), SHA-256
`32e6ec1acf69422b269875c3f423edc48d7f0e12fe9a4263c395f8b65d6be36d`
at review time. The final run restricted OMP, MKL, OpenBLAS, and NumExpr to one
thread, completed in 37.64 seconds, used 324,760 KiB maximum resident memory,
and recorded zero swaps. It stayed well below the requested 2-GiB working-memory
bound.

## Interpretation boundary

The audit establishes integrity and arithmetic agreement, not the optimizer
mechanism. The per-state-selected and fixed-panel curves answer different
descriptive questions. High never-trained decodability is not proof that the
unembedding uses the decoded coordinate, that a five-frequency circuit has been
identified, or that a filter caused formation. Symmetry is neither necessary nor
sufficient for correctness, and the training-membership excess is not a direct
memorization measure. Temporal differences across already-diverged arms remain
observational; a causal claim requires the separately specified common-state
intervention.
