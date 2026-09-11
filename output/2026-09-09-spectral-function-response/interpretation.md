# Preliminary function-response interpretation

## Evidence and conventions

This is a readout of the completed, independently audited one-step diagnostic at the five fixed step-1500 states (seeds 100--104). The saved summary (artifact not distributed in this public snapshot) has SHA-256 `8fda0239230ab0c21a85722236e33aec97f5084809ecd8cc7cf924c729b4b472`. The audit receipt (artifact not distributed in this public snapshot) has SHA-256 `161cc406834ba7a499e032210296d88254db11145e672978681aa73e182c20cf` and reports `PASS`, 7,675 checks, and no errors. The linked summary contains every seed-level value; all numbers below are stored means and sample SEs, without numerical recomputation.

For named action contrasts, finite scalars are right minus left: lower CE and higher margin are favorable. Positive linear utility predicts a CE reduction at the shared before logits; it is the CE derivative along the finite logit difference, not a parameter JVP. In the separate `responses_from_before` object, positive CE `finite_improvement` means a reduction from the common before state. `R` denotes the sum-averaged component and `I-R` the within-sum component; the latter is not identified with memorization.

## Primary finite effects

| Action contrast | Split | CE, mean ± SE | CE signs | Margin, mean ± SE | Margin signs |
|---|---|---:|---|---:|---|
| raw → truncated | test | -0.001178 ± 0.000730 | favorable 5/5 | +0.001465 ± 0.000908 | favorable 5/5 |
| raw → truncated | train | +0.00004121 ± 0.00002081 | adverse 5/5 | -0.001266 ± 0.000871 | adverse 5/5 |
| truncated → projected | test | +0.0002009 ± 0.0001737 | adverse 5/5 | -0.0002644 ± 0.0002327 | adverse 5/5 |
| truncated → projected | train | -0.000008409 ± 0.000003691 | favorable 5/5 | +0.0001926 ± 0.00008247 | favorable 5/5 |
| raw → projected | test | -0.0009771 ± 0.0005569 | favorable 5/5 | +0.001200 ± 0.0006763 | favorable 5/5 |
| raw → projected | train | +0.00003280 ± 0.00001719 | adverse 5/5 | -0.001073 ± 0.0007939 | adverse 5/5 |

Thus, at this inherited state, supplying the truncated incoming action, which removes the raw action's off-span component before AdamW, improves heldout CE and margin while slightly worsening the training metrics. Amplifying the retained incoming component from truncated to norm-restored projected has the opposite local effect: a small heldout cost and a small training benefit. The total raw-to-projected contrast remains heldout-favorable because the suppression benefit is larger than the amplification cost. Accuracy is essentially insensitive at this one-step resolution: the two first contrasts change test accuracy in zero and one seeds respectively, and train accuracy in none.

The five seed-level test CE values are:

- raw → truncated: `[-0.0004067824, -0.0004782739, -0.0006886339, -0.0040839428, -0.0002324227]`
- truncated → projected: `[0.0000002964, 0.0000517863, 0.0000535202, 0.0008941998, 0.0000047553]`
- raw → projected: `[-0.0004064861, -0.0004264876, -0.0006351137, -0.0031897431, -0.0002276674]`

Seed 103 is much larger in magnitude than the other seeds and therefore inflates several SEs, but it does not create the primary CE or margin sign agreements.

## First-order utility decomposition

| Action contrast | Split | Total utility | `R` utility | `(I-R)` utility |
|---|---|---:|---:|---:|
| raw → truncated | test | +0.001171 ± 0.000719 (positive 5/5) | -0.0003173 ± 0.0002083 (negative 5/5) | +0.001489 ± 0.000926 (positive 5/5) |
| raw → truncated | train | -0.00004237 ± 0.00002143 (negative 5/5) | -0.000007622 ± 0.000003823 (negative 5/5) | -0.00003475 ± 0.00001767 (negative 5/5) |
| truncated → projected | test | -0.0001533 ± 0.0001313 (negative 4/5) | -0.000002800 ± 0.00003099 (positive 4/5; seed 103 negative) | -0.0001505 ± 0.0001022 (negative 5/5) |
| truncated → projected | train | +0.00001107 ± 0.000005742 (positive 5/5) | +0.000002886 ± 0.000001488 (positive 5/5) | +0.000008185 ± 0.000004258 (positive 5/5) |

The raw-to-truncated heldout benefit is therefore not an increase in favorable sum-consistent utility. Its `R` component is adverse in every seed; a larger favorable within-sum component dominates it. Likewise, the small heldout cost of retained-direction amplification is concentrated in the within-sum utility, while its mean `R` utility is near zero and mixed. These statements concern directional utility of finite function responses, not whether either component carries reusable rules or example-specific information. Response energy alone has no helpfulness sign.

## Zero-gradient Adam reference

The zero-gradient Adam step is not a no-op: inherited moments and weight decay still move the parameters. From the common before state, it improves heldout CE by `0.005607 ± 0.001238` in 5/5 seeds and heldout margin by `0.004604 ± 0.002492` in 4/5. It worsens train CE by `0.002862 ± 0.003224` in 4/5 and train margin by `0.01395 ± 0.01955` in 4/5. Its heldout total linear utility is positive in 5/5 (`0.01149 ± 0.00512`), with mixed `R` utility (`0.000118 ± 0.002994`) and mostly positive within-sum utility (`0.01137 ± 0.00776`, 4/5).

Relative to that drift reference, each active gradient action has the same qualitative tradeoff:

| Contrast | Test CE | Test margin | Train CE | Train margin |
|---|---:|---:|---:|---:|
| zero → raw | +0.002532 ± 0.001271 (adverse 4/5) | -0.002374 ± 0.002165 (adverse 4/5) | -0.007621 ± 0.007217 (favorable 5/5) | +0.04960 ± 0.03958 (favorable 5/5) |
| zero → truncated | +0.001354 ± 0.000729 (adverse 4/5) | -0.0009096 ± 0.001589 (adverse 4/5) | -0.007579 ± 0.007220 (favorable 5/5) | +0.04833 ± 0.03987 (favorable 5/5) |
| zero → projected | +0.001554 ± 0.000833 (adverse 4/5) | -0.001174 ± 0.001711 (adverse 4/5) | -0.007588 ± 0.007219 (favorable 5/5) | +0.04852 ± 0.03982 (favorable 5/5) |

For all three zero-to-action heldout contrasts, total utility is negative in 4/5 and within-sum utility is negative in 5/5, despite favorable `R` utility in all five seeds for raw and four of five for truncated/projected. Train total, `R`, and within-sum utilities are positive in 5/5 for every active action relative to zero. This supports a local train-versus-heldout tradeoff at step 1500 and shows that carried Adam history is functionally consequential. It does not make the zero action a generally preferable optimizer and does not isolate momentum, decay, or their nonlinear interaction with the supplied action.

## Bounded conclusion

The strongest constructive conclusion is local: at the same five inherited step-1500 states, suppressing the incoming raw action's off-span component consistently improves its immediate heldout function response, and that benefit is carried by the within-sum utility contrast rather than by the sum-consistent component. Norm-restoring amplification of the incoming retained direction then slightly reverses that heldout benefit while improving training response. Consequently, the previously observed long-horizon projected-policy advantage cannot be explained simply as an immediate heldout benefit from amplifying the retained direction at this state. Cumulative trajectory feedback, optimizer-state interaction, and effects at later states remain viable explanations.

This one-step diagnostic does not establish 1,000-step mediation, identify a pure memorization component, prove that learned geometry is generically necessary, or transfer beyond these five states and this benchmark.
