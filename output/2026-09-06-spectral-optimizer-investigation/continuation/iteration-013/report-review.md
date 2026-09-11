# I13 report review

Codex — Spectral Optimizer Investigation, 7 September 2026.

## Verdict

The [I13 results report](results.md) is numerically consistent with the frozen
[audited summary](analysis-001/summary.json) on the registered effects and the
substantive supporting comparisons checked below. Its main conclusion is
appropriately bounded: preserving the outside-space EMA mean restores
substantial soft/redraw adaptation, but also restores nearly raw-like fitting
of the fixed corruption realization. This supports a mixed restoration
tradeoff, not selective denoising or a new optimizer default.

This was a read-only scientific and narrative review. I did not rerun training,
the CPU audit, model/data forwards, or optimizer steps. The existing
[independent audit](analysis-001/audit.json) passed with all 36 new branches,
36 inherited references, 18 paired probes, 258 alignment records, and no
numerical failures. Its stated limitations remain operative.

## Substantive checks

- All four registered mean32-minus-current32 effects at horizon 500 are
  transcribed correctly and are positive in every seed average: soft CE
  `+0.186380`, soft accuracy `+27.28` points, redraw CE `+0.127622`, and redraw
  accuracy `+23.56` points. The report correctly treats the two parent anchors
  within a seed as correlated states rather than six independent seeds.
- The gains are genuine absolute learning, not merely relative protection.
  Mean32 improves from the common parent by `0.129853` CE utility and `38.08`
  accuracy points under soft targets, and by `0.029880` and `25.023` points
  under redraw. The report preserves the adverse seed-102 redraw CE change and
  does not imply that mean32 generally beats raw: it remains slightly behind
  raw at the endpoint on both soft metrics and on redraw accuracy, with redraw
  CE nearly tied.
- Historical-mean specificity is supported only against the declared one-percent
  leak. Mean32-minus-leak is large and favorable for all four soft/redraw
  endpoint comparisons, whereas leak remains close to native current32. The
  text correctly leaves larger leakage, alternative momentum, decay, and
  smoothing rules untested.
- Fixed-label protection is substantially lost. The residual
  `R_zeta=L_fixed-L_soft` moves from `-0.052355` at the parent to `-0.261183`
  under mean32, versus `-0.270663` raw, `-0.048870` native, and `-0.050811`
  leak. Mean32 therefore closes about 96% of the raw-versus-native endpoint
  realization-fitting gap. The report correctly describes this as relative
  fitting at a model, not a causal force or a clean-generalization metric, and
  retains the disagreement between favorable fixed-target CE and mixed
  accuracy.
- The common-state probes do not overstate immediate utility. Mean32 produces
  a less adverse auxiliary-clean loss change than native in all six parents
  for each target, and the advantage remains after matching the native
  data-step norm. Both actual proposals still worsen auxiliary-clean loss.
  Mean32 also improves fixed, soft, and clean probe losses, so the local
  direction is broad rather than clean-specific.
- Step magnitude is handled as a mediator rather than ignored. Mean32/leak
  average data-step norm ratios over the trajectories are approximately
  `1.394` fixed, `2.436` soft, and `0.966` redraw. This rules out one universal
  “larger step” account because redraw improves with a slightly smaller mean
  step, but it does not identify a direction-only causal effect over 500
  updates.
- Signed alignments support mixed content without proving Adam mediation. At
  the common parents the outside mean aligns positively with both clean and
  fixed-realization gradients. At fixed-target mean32 endpoints it becomes
  strongly realization-aligned, while soft/redraw endpoints align with clean
  gradients and oppose the fixed-minus-soft residual. The report correctly
  labels these as local gradient geometries, not parameter-step or long-run
  causal decompositions.

## Remaining limits

The evidence is conditional on three reused seed bundles, six late
current-filter-trained MNIST states, inherited AdamW history, rank 32, and one
fixed EMA decay. There is no fresh dataset, official-test confirmation,
independent mean-decay intervention, moving-basis decomposition, long-horizon
protection guarantee, or current-stable non-Adam comparison. The fixed-basis
low-pass/heavy-ball mathematics supplies a coherent prospective explanation,
but I13 does not identify temporal smoothing as the unique mechanism. The
report preserves these boundaries and the negative evidence, so I found no
remaining material numerical or scientific overclaim.
