# I14 independent scalar corroboration and report review

7 September 2026. This is a CPU-only, standard-library re-derivation from the
36 confirmation `curve-*.json` files in the retained external artifact root. It
did not import or call the runner or analyzer, load tensor checkpoints, replay a
model/optimizer step, or rerun any experiment or full audit.

## Provenance qualification

The original [audit](analysis-001/audit.json) remains `fail` because the
artifact root contained one unlisted entry. The separately retained
supplement (artifact not distributed in this public snapshot) accepts only that
exact empty, non-symlink `torchinductor_titus` directory after rehashing all 502
declared artifacts and binding the original audit, summary, completion, attempt,
and source records. It does not rewrite the audit or replay its semantic checks.
The calculations below therefore corroborate selected scalars; they are not a
replacement full audit.

## Independent calculation

For each seed/base/target pair, endpoint CE benefit is
`aux_clean_CE(raw)-aux_clean_CE(current32)` and accuracy benefit is
`aux_clean_accuracy(current32)-aux_clean_accuracy(raw)`. Positive values favor
filtering. All 12 endpoint effects match the retained
[summary](analysis-001/summary.json).

| Base | Target | CE s200 | CE s201 | CE s202 | CE mean | Acc s200 | Acc s201 | Acc s202 | Acc mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD | clean | -1.083056 | -1.057449 | -0.971889 | -1.037465 | -0.3476 | -0.2682 | -0.2726 | -0.2961 |
| SGD | fixed | -0.198147 | -0.206285 | -0.249395 | -0.217942 | -0.0174 | -0.1394 | -0.1956 | -0.1175 |
| SGDm | clean | -0.306139 | -0.334261 | -0.274506 | -0.304969 | -0.0736 | -0.0732 | -0.0720 | -0.0729 |
| SGDm | fixed | +0.234272 | +0.092254 | +0.137729 | +0.154752 | +0.2920 | +0.1658 | +0.1038 | +0.1872 |
| AdamW | clean | -0.156225 | -0.197906 | -0.182775 | -0.178969 | -0.0488 | -0.0538 | -0.0518 | -0.0515 |
| AdamW | fixed | +0.102282 | +0.095503 | +0.075904 | +0.091230 | +0.2952 | +0.2474 | +0.1642 | +0.2356 |

I independently selected each arm over the six nonzero checkpoints, using
minimum validation clean CE or maximum validation clean accuracy with the
earliest horizon on an exact tie. All 72 arm/selector horizon choices match the
summary. The resulting 24 paired auxiliary effects also match:

| Base | Target | min-val-CE: aux CE | min-val-CE: aux acc | max-val-acc: aux CE | max-val-acc: aux acc |
|---|---|---:|---:|---:|---:|
| SGD | clean | -0.311862 | -0.0936 | -0.326584 | -0.0715 |
| SGD | fixed | -0.216273 | -0.1523 | -0.148797 | -0.1110 |
| SGDm | clean | -0.189650 | -0.0567 | -0.189780 | -0.0570 |
| SGDm | fixed | -0.162773 | -0.0353 | -0.205765 | -0.0082 |
| AdamW | clean | -0.178789 | -0.0503 | -0.179052 | -0.0506 |
| AdamW | fixed | -0.076892 | +0.0393 | -0.057494 | +0.0130 |

For geometry, I omitted only steps whose saved leakage reason was
`basis_unavailable`, summed saved outside squared energy and squared energy
within each trajectory, divided those sums, then averaged the three seed-level
fractions equally. The table gives percentages. Its 12 rows x four columns are
all 48 summary fractions.

| Base | Target | Policy | Full data | Full total | Steps101+ data | Steps101+ total |
|---|---|---|---:|---:|---:|---:|
| SGD | clean | raw | 13.7987% | 13.1048% | 13.6387% | 12.9403% |
| SGD | clean | current32 | 0.8318% | 0.9511% | 0.0000000004% | 0.1749% |
| SGD | fixed | raw | 16.6722% | 16.5589% | 16.8141% | 16.6805% |
| SGD | fixed | current32 | 0.8109% | 1.2584% | 0.0000000006% | 0.4134% |
| SGDm | clean | raw | 30.5196% | 29.3067% | 23.8115% | 22.2660% |
| SGDm | clean | current32 | 9.3223% | 9.2714% | 0.7884% | 0.8969% |
| SGDm | fixed | raw | 30.9306% | 30.3112% | 30.8265% | 30.2042% |
| SGDm | fixed | current32 | 4.4636% | 4.5702% | 1.9258% | 2.0700% |
| AdamW | clean | raw | 80.4784% | 80.4516% | 78.9181% | 78.8816% |
| AdamW | clean | current32 | 79.0835% | 79.0848% | 77.5849% | 77.5862% |
| AdamW | fixed | raw | 68.9116% | 68.8793% | 68.4215% | 68.3877% |
| AdamW | fixed | current32 | 60.6094% | 60.6079% | 59.1656% | 59.1646% |

Across all 84 comparisons, the maximum absolute difference from the summary's
stored values was `4.22e-15`, attributable to floating-point summation order;
there was no material numerical or membership discrepancy.

## Interpretation checks

- The fixed-label endpoint benefit under SGDm is positive for every seed and
  both metrics. In this recipe, that falsifies a necessity claim for AdamW's
  adaptive second moment. It does **not** identify momentum as the cause: SGD
  and SGDm used separately calibrated rates (`.1` versus `.03`), different
  per-update decay factors, and SGDm selected the upper edge of its grid.
- All clean-target endpoint and selected effects are adverse. Fixed-target SGDm
  is favorable only at the fixed endpoint, not in the three-seed mean under
  either validation selector. AdamW's fixed selected CE remains adverse while
  mean selected accuracy is weakly positive. The latter is not uniformly
  replicated under the maximum-validation-accuracy selector: its per-seed
  effects are +0.24, +4.36, and -0.70 percentage points. Endpoint preservation
  against late raw deterioration is thus a better description than uniformly
  faster or better learning.
- Near-zero steps101+ SGD data leakage is the expected direct consequence of an
  SGD step using the projected delivered gradient. SGDm's buffer and AdamW's
  diagonal transform reintroduce outside-current-span motion. Leakage magnitude
  alone has no semantic sign and does not mediate the outcome causally.

## Report verdict

PASS. The final [results report](results.md) reproduces the registered endpoint
and selected effects, absolute-progress signs, horizon crossovers, realization
diagnostics, and current-versus-raw geometry without a material numerical or
sign error. Its headline claim is appropriately conditional: SGDm supplies a
counterexample to necessity of AdamW's adaptive second moment for the measured
fixed-label endpoint benefit, while the report explicitly refuses momentum
causality, a universal memory requirement, optimizer ranking, generic denoising,
or a production recommendation. It also preserves the adverse plain-SGD and
all-clean results rather than treating them as secondary exceptions.