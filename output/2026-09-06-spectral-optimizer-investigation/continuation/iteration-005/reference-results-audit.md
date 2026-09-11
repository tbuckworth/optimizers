# Independent saved-reference and operator audit

6 September 2026. **PASS, first attempt.** The independent checker
[audit_reference_results.py](audit_reference_results.py) reconstructed the
scheduled calculations directly from the saved tensors. It imports only the
standard library, NumPy and PyTorch—not the replay harness, reference module,
summarizer, canonical optimizer or other production helpers.

The source hash and tolerances were declared to the parent before execution.
Execution required that exact source hash and exclusively created its output;
there was no retry, tolerance change or source change after execution began.
The complete machine-readable record is
[reference-results-audit.json](reference-results-audit.json).

## Checks completed

- Verified all **12,000 raw/innovation row hashes** across the three 2,000-step
  streams. Independently reconstructed all **6,000 sequential GPU float32 mean
  updates and innovations** using the prescribed `mul_(.99).add_(raw, alpha=1-.99)`
  sequence. Innovation equality used integer views of the float32 bits, so
  signed zeros were not silently identified. First acceptance and preceding
  zero innovations were checked rather than assuming startup at step 2.
- Checked all **12 snapshot** raw/innovation hashes, independently reconstructed
  means, both saved observer means, first-acceptance indices, pre-Adam parameter
  hash links, observation phase, ranks and repair counters. The snapshot
  parameter hash was linked to the preceding post-step hash, not the current
  post-step hash. Previous-basis column counts were linked to the preceding step.
- Independently formed all **12 float64 weighted history prefixes**, including
  the overweighted first accepted innovation, and reconstructed their Grams.
  Compared saved weights, Gram entries and full descending spectra; verified
  leading saved dual-vector residuals and orthogonality, dual-to-covariance
  mapping, mapped-vector residuals and orthogonality, and orientation-invariant
  leading rank-32 subspace overlaps. No parameter-by-parameter covariance matrix
  was allocated.
- Recomputed **384 observer metric values**: both storage widths at all twelve
  states, using full saved V/S for covariance reconstruction and only the leading
  32 columns for native and matched-rank diagnostics. Covariance norms, inner
  products and traces used factor-product identities that retain actual basis
  nonorthogonality. QR was confined to the separate span diagnostic.
- Recomputed **96 reference metric values**, all probe input/output energies,
  represented-operator energy, span capture/distance, native current/previous
  gradient retention and their difference, clean–corruption cross terms/cosines,
  closure residuals and native-versus-float64 action checks. The saved rounded
  corruption residual was checked against noisy-minus-clean bitwise.

All twelve references passed their positive-rank and boundary-gap conditions;
there were no scientific nulls in these completed rows. This run does not
replace the earlier synthetic tests of null handling.

## Numerical agreement and resource limits

Float64 comparisons used absolute tolerance `1e-11` plus relative tolerance
`1e-9`; native float32-derived comparisons used absolute tolerance `1e-6`.
All original protocol residual gates were retained.

| Comparison family | Comparisons | Largest absolute discrepancy |
|---|---:|---:|
| Float64 scalars/arrays | 972, covering 36,023,352 elements | `1.412203687323199e-13` |
| Native float32-derived scalars | 456 | `3.3306690738754696e-16` |

The largest tolerance-normalized discrepancy was `0.004888213808727184` of its
allowed limit, for the width-128 relative covariance error at seed 3, step 200.
The maximum saved dual eigen-residual was `1.5325898186050109e-15`; maximum saved
mapped covariance residual was `1.148646133029119e-15`. Native action roundoff
peaked at `1.7440157312521782e-7`, and relative probe closure error at
`2.615899599902236e-8`; both passed their original `1e-5` gates.

Execution ran **13:30:51.867337–13:31:13.203406 UTC**, 21.336 seconds, on the local
RTX 3090 with one CPU thread, deterministic algorithms and TF32 disabled. The
preflight found only allowed desktop processes, 23,467 MiB free GPU memory and
52,697,092,096 bytes available host RAM. Peak allocated GPU memory was
**2,373,529,088 bytes**, reserved memory **2,455,764,992 bytes**, and RSS
**2,312,978,432 bytes**, below the frozen 8 GiB allocated / 12 GiB RSS / 300-second
limits. The process exited successfully and released its GPU allocation.

## Independence and limits

The separate [raw-summary-audit.json](raw-summary-audit.json) establishes full
file byte binding and provenance: 91 unique artifact hashes/sizes and 19 source
bindings. This checker records that audit's hash and independently rechecks row
bytes and mathematical contents; it does not repeat every whole-file hash.

No MNIST IDX file, test set or model was opened or executed. Saved probes were
reprojected, not regenerated by autograd. Neither observer was replayed, so the
ancestry of a saved previous basis is not independently reconstructed beyond its
phase/rank links. Snapshot parameter hashes are checked against saved trajectory
hashes; historical unsaved full parameter/gradient vectors are not reconstructed
or claimed to match. There was no new scientific snapshot, policy, learning
replication, bulk output, paid API call or optimizer-source change.

This audit supports the reported **saved-stream numerical diagnostics**. It
does not establish that production/reference discrepancies are truncation alone,
that wider storage always improves fidelity, or that better covariance fidelity
causes better learning or semantic gradient selection.

## SHA256 bindings

| Artifact | SHA256 |
|---|---|
| Independent audit source | `85fa7e1a952b6d2c5b7f9cc248c34ed8e6ac94e1d48aa5ff298439e56758eb0d` |
| Independent audit JSON | `785802a37b0edee79f62947fda2ba6f64f0da84bebf25efbe25e99be8e4b4db5` |
| Separate raw/provenance/summary audit JSON | `e1ba8337806a17620a1308662db0fe2c4919b3ba263d7a3f98b75f031dcc9cf0` |
| Original completed execution JSON | `b8368fe5ee628dee648a2167e909a0380c6abc2b3fcc7d11976a707dd86e3478` |
| Frozen iteration005 protocol | `eb8f49f4a5247e9c33a23a0d15a572f2c5cec08355f2b4d05ada50f435ce0035` |
