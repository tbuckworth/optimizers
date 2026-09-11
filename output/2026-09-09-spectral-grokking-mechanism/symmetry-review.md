# Independent review of the symmetry diagnostics

9 September 2026. This was a source review plus small synthetic fixtures only.
No scientific checkpoint, saved activation array, or training process was read
or run.

## Verdict

No blocking correctness defect was found in
`experiments/grokking_symmetry.py`. The output-shift orientation, edge-pool
normalization, outcome-independent matching, zero-energy handling, and stated
accuracy/temperature limitations agree with the registered diagnostic. The five
dedicated unit tests pass.

Source reviewed:

- `experiments/grokking_symmetry.py`, SHA-256
  `e7d2921499cdc7047337f309096e40e0f01ebcdb78d14e8fc7debbda87c93407`.
- `tests/test_grokking_symmetry.py`, SHA-256
  `599cd9401140b7acd935b350b652cd45b82772f469f12078dde8345afbf9e2e4`.
- `output/2026-09-09-spectral-grokking-mechanism/protocol.md` and the earlier
  `output/2026-09-08-spectral-paper-planning/grokking-mechanism-design.md`.

## Checks

### Output-shift sign

The registered operator is `(S_delta z)_c = z_(c-delta mod p)`. NumPy
`roll(left, +delta, axis=1)` implements exactly this convention. For a synthetic
ideal modular score at `p=11`, direct comparison gave maximum error
`3.89e-16` for the positive roll and `1.80` for the opposite roll. The existing
tests also give essentially zero correct-shift defect and a positive wrong-shift
defect. The sign is correct.

### Per-block and pooled normalization

Each defect is a ratio of the total squared residual to the total source-plus-
destination centered-logit energy. `_pooled_metric` sums numerators and
denominators before division; it does not incorrectly average already-normalized
block values. An independent reconstruction over both axes and deltas `(1,2,4)`
matched each pooled value exactly. The pooled train→held-out and
held-out→held-out counts were both 152 in the fixture, and each equalled the sum
of its component counts.

This ratio-of-sums intentionally energy-weights mixed blocks. It is not an
equal-weight mean over axes, shifts, source sums, or edges' individual ratios.
That interpretation should remain explicit when plotting a pooled value.

### Matching and leakage

Candidate construction and selection depend only on the fixed train/test
partition, axis, delta, source sum, edge IDs, role, and a fixed hash namespace.
They never inspect logits, labels predicted by the model, defects, or other
outcomes. Within every `(axis, delta, source_sum)` group, both roles retain the
same `min(TH candidates, HH candidates)` count. Since the destination sum is
determined by source sum and delta, this balances the sum transition as well as
the explicit grouping variables. Changing every logit leaves matching metadata
and selected-edge hashes unchanged in the unit fixture. I found no selection
leakage.

The two pools contain different physical edges and are hash-selected
independently using role-specific keys. “Matched” therefore means exact balance
on the registered grouping/count variables, not paired equality of input IDs,
logit energy, margin, or difficulty.

### Zero energy and scale

Constant logits become exactly zero after row centering. The implementation
returns `value=None` and `energy_defined=False`, rather than calling this perfect
symmetry. In an additional scaled ideal fixture with `energy_floor=1e-12`, scale
zero and `1e-13` were undefined, while scale `1e-10` was defined. Thus the gate
is explicitly an RMS threshold (`centered_rms > energy_floor`), not merely an
exact-zero test. Both centered and raw RMS are retained, so a large common row
offset cannot hide the centered degeneracy.

### Classification and temperature caveats

The tests correctly establish both directions of the construct warning:

- a perfectly accurate classifier with input-dependent temperature can have a
  nonzero symmetry defect;
- a consistently wrong circularly shifted classifier can have essentially zero
  correct-shift defect.

A separate fixture shows train-only temperature scaling can create positive
training-specific excess without changing predicted classes. The diagnostic is
therefore supporting evidence, not a necessary/sufficient rule-circuit or
memorization measure. This module does not itself report accuracy or
correct-class margin; acquisition/reporting must keep those registered readouts
beside the defect and logit-energy values.

## Nonblocking hardening notes

1. `deltas` are converted with `int(delta)` before validation, so a caller could
   accidentally pass a non-integral float such as `1.5`, which would silently
   become `1`. The frozen scientific caller uses exact integer constants, so
   this does not affect the registered acquisition, but strict public input
   validation would reject non-integral values before conversion.
2. Finite individual logits are checked, but adversarially enormous finite
   float64 logits could overflow the subsequent squared-energy reductions.
   Registered neural logits are nowhere near that regime. If this becomes a
   general-purpose API, post-reduction finiteness checks would make the JSON
   contract fully explicit.

## Fixture execution

`PYTHONPATH=. python3 -m unittest discover -s tests -p
'test_grokking_symmetry.py' -v` completed 5/5 tests successfully in 0.066 seconds.
The additional read-only script independently checked the roll sign, pooled
ratio reconstruction, equal pooled counts, and the zero/sub-floor energy gate.
