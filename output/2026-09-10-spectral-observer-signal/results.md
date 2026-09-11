# Saved observer signal: rare input is present; the relative boundary starts at delivery

Codex — Spectral Optimizer Investigation, 10 September 2026.

**Completed once; post-hoc three-parent evidence.** The incoming gradient
has positive alignment with the true rare probe in every Clean and Diffuse
case. The adverse rare Grouped-minus-Interleaved contrast in two Diffuse
parents is already visible in the delivered gradient, before Adam. It is
not a reversal first introduced by Adam for this particular contrast.
All native absolute rare delivered dots nevertheless remain positive.

The fixed [protocol](protocol.md) used only saved vectors from the
[matched-parent observer study](../2026-09-10-spectral-observer-pathway/results.md):
three reused seeds, Clean/Diffuse labels, rare/common true-label training
probes, and Interleaved/Grouped/raw/zero actions. There was no new model,
gradient or prediction computation. New dot products join previously
audited Adam and heldout-loss scalars. This analysis has source review and
internal compensated numerical checks, **not a new independent audit**.

## Main evidence

For probe `q`, common block input `g` and delivered action `h`, define
`B=qᵀg`, `F=qᵀh`, `J=−qᵀ(actual Adam displacement)` and
`U=heldout CE_before−CE_after`. Positive F describes a hypothetical small
gradient-descent step; positive J describes the actual step's first-order
probe utility; positive U is finite heldout improvement. Their magnitudes
have different units.

Seed order is **202609121 / 202609122 / 202609123**. Every number below
comes from the [archived scalar JSON](results/result.json).

| Rare grouping contrast | Clean, all seeds | Diffuse, all seeds |
| --- | --- | --- |
| D_filter = F_grouped−F_interleaved | +0.128563 / +0.250851 / +0.252053 | +0.044021 / −0.061640 / −0.136978 |
| D_Adam = J_grouped−J_interleaved | +0.001370 / +0.004277 / +0.003641 | +0.000653 / −0.000982 / −0.001782 |
| D_U = U_grouped−U_interleaved | +0.001414 / +0.004464 / +0.003399 | +0.000752 / −0.000962 / −0.001638 |

The six sign agreements constrain this local grouping comparison; they are
not a mediation estimate. Rare input B is positive in every parent, with
means **2.298926 Clean / 4.886571 Diffuse**. Meanwhile mean rare-input
cosine is **0.420966 / 0.169797**, and mean input norm is
**0.365267 / 1.916107**. Larger Diffuse dot therefore does not mean better
orientation. The matched label change increases the total rare dot in all
three parents and reverses the common dot from positive to negative. It
does not isolate a rare-example contribution or establish a benefit of
label corruption.

![All-seed rare grouped-minus-interleaved contrasts in filtered delivery, actual Adam utility and held-out loss. Separate units.](results/rare-signal-chain.png)

## Controls prevent a stronger claim

- Absolute rare F is positive for all nonzero actions, but Clean parent 121
  has negative actual J and U: grouping reduces damage there. Clean parents
  122/123 obtain genuine positive rare U. Diffuse Grouped has positive rare U
  in all three despite its adverse relative contrast in two.
- There is no general order-preservation claim. For rare Grouped-minus-raw,
  F signs `−−−` become J/U signs `+−+` in Clean; F signs `−−+` become
  `+−−` in Diffuse. Common grouping also has stage-specific sign differences.
- All Diffuse nonzero common F values are negative, yet actual common J/U
  are positive. Zero-gradient Adam itself moves and improves common CE;
  every nonzero Diffuse action improves it **less than zero**. Raw beats
  both native schedules on common CE in every case. Native rare utility
  does not consistently beat raw.
- Rare training and heldout accuracy remains **0% before and after every
  action**. Wrong-target training CE improves under every nonzero Diffuse
  action; this is not evidence of selective suppression of label noise.

All-seed absolute outcomes, both probes, filter changes and raw/zero
comparisons are retained in the [interpretation](interpretation.md),
[48-row table](results/all-actions.csv) and JSON.

![Input alignment separates magnitude and orientation.](results/input-signal-v2.png)

## Scope and record

These are three reused, outcome-informed early parents, not fresh endpoint
replications. The input is a smoothed block mean, not a new minibatch; the
probes are finite training subsets, not heldout gradients. The observer
intervention includes EMA mean/order, finite-rank history and native
direction/gain. No isolated covariance/Adam mechanism, endpoint mediation,
calibration, safety or production conclusion follows.

The completion receipt (artifact not distributed in this public snapshot) records one execution in
**8.389 seconds**, high-water RSS **536,060 KiB**, under one CPU and a
4 GiB/no-swap cap. Source freeze is
`ec3285c4fc9f0991ac0307e70cde726a1861b2de`.

Result SHA256:
`20d0514e3e6d5b0453d781957e38ac549c4c8269892657c5b76ee594e3d8cb2c`.
Joined original independent-audit SHA256:
`d9b05160c9a58952d4f0899699f79de2ee2d919c5319f2bb96d687ea448d0d35`.
