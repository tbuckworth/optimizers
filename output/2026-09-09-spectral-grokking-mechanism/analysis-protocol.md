# Saved-array analysis specification

9 September 2026. Written after the six-state calibration and before examining
intermediate-state probe outcomes or computing any scientific symmetry score.
This extends the frozen [acquisition protocol](protocol.md) without altering it.

## Readouts and fixed presentation

Analyze all150 saved states, preserving every arm, seed and recorded step.
Verify complete manifests, result/array hashes, source pins and disjoint stage
rosters. Keep byte-identical scalar result copies for committed evidence; large
arrays remain at their hash-bound large-volume paths. No model inference or
training is required for this analysis.

Report final-hidden selected-five evaluation R², pre-attention control,
twenty-shuffle maximum, and all56 frequency scores. The secondary fixed panel
uses the five frequencies with greatest mean **fit** R² across the three
seed100step6000 final-hidden calibration states. Selection and tie breaking
are unchanged from acquisition; apply the same panel to all150 states.
Both panels are readouts of sum information, not counts of model-used circuits.

Show every checkpoint and seed, with equal-weight mean and sample standard
error over the five seeds where aggregation helps. The seed, not the edge,
frequency or checkpoint, is the replication unit. Align on training step,
not an outcome-selected event time. No hidden smoothing, best-seed selection,
interpolated grokking times or newly claimed confirmatory threshold test.

Keep original train/test CE and accuracy beside the readouts. Additionally,
compute mean correct-class logit margin (true-class logit minus largest other
logit) separately for train and never-trained pairs. This is descriptive and
does not turn a correlation into causal mediation.

## Output symmetry and training-membership excess

Use the independently reviewed `experiments/grokking_symmetry.py`, five passing
synthetic fixtures and [review](symmetry-review.md). For each axis and input
shift δ∈{1,2,4,8,16,32}, row-center logits z and compute

`E = Σ_edges ||z(destination) − roll(z(source), +δ)||² /
     Σ_edges (||z(destination)||² + ||z(source)||²)`.

The correct output transformation is `roll(+δ)`, since its class-c coordinate
is the original class-(c−δ). Keep an equal-edge wrong-shift comparison δ+37,
and unordered exchange pairs `(a,b)↔(b,a)` with output shift0.

For a compact held-out shift curve pool the twelve axis/shift numerators and
denominators separately before dividing. This energy-weights the edge blocks;
it is not their equally weighted mean. Keep each block's raw statistics too.

Compare train→held-out and held-out→held-out edges, hash-selected to equal
counts within each axis/shift/source-sum group. Pool each role's raw sums and
take `E_TH − E_HH`. This balances sum transitions and counts, **not** logit
energy, difficulty or physical input identities. Keep matching hashes/counts,
energies and RMS. The reviewed code requires centered RMS>1e−12 for a defined
ratio. This absolute numerical floor preserves row-offset invariance; it
supersedes the construct review's suggested relative-to-raw-RMS floor, not an
outcome-based choice. Uniform/sub-floor outputs have an undefined score.

Symmetry is neither necessary nor sufficient for accuracy: a consistently
wrong output shift can be equivariant, while input-dependent temperature can
break exact equivariance without changing a prediction. A train-specific
temperature change alone can create positive TH−HH excess. Therefore neither
score is a direct memorization measure or a circuit ablation.

## Interpretation and subsequent decision

Earlier held-out sum decodability **before** high held-out accuracy would
support earlier readable information formation. Similar readout emergence
with a later behavioral transition is compatible with downstream cleanup,
readout alignment or margins. At final accuracy100%, high decodability is
partly expected from the model's existing linear unembedding and is weak
stand-alone mechanism evidence. Differences between already-diverged runs
remain observational; the next causal test must start from a common saved
state and perturb a precisely defined component, not merely compare curves.

Do not interpret a frequency-panel mismatch as absence of a rule. The panel
is selected from seed100 and may miss other seeds' learned frequencies.
Preserve both successful and contradictory diagnostics in the report.

Implementation/fixtures and this specification must be source-pinned before
the first full saved-array analysis. Use one CPU math thread, verified16GiB/
zeroSwap/30min cgroup bounds, exclusive fresh large-volume output, and no
retry/overwrite of completed work. Paid spend remains zero.
