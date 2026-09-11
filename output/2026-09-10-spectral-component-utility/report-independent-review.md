# Independent interpretation and presentation review

Status: **PASS**, after the two wording corrections below. This is a bounded review of completed evidence and its presentation, not another scientific acquisition or data audit.

## Evidence and scope

Read [protocol.md](protocol.md), [audit.json](audit.json), [results.md](results.md), report.html (artifact not distributed in this public snapshot), [interpretation-analysis.md](interpretation-analysis.md), [make_plots.py](make_plots.py), [test_make_plots.py](test_make_plots.py), and plots-manifest.json (artifact not distributed in this public snapshot). Visually inspected both existing PNGs. Used only audited scalar JSON and file receipts; no model, checkpoint, numeric archive, raw logits, optimizer replay, experiment, or audit execution. The research skill contributed evidence tracing, not new approval gates or replication requirements.

The audit receipt is `e16a7774b853060d80a3393d1bc0626b2b8965a13b451552989356a18cbb3971`: PASS, 50,388 checks, 12 saved parents, 24 paired draws, 48 raw/native actions, and 144 endpoint readouts. This review does not independently certify model inference.

## Quantitative findings checked

- Primary translated-final, full-step native-minus-raw original clean CE improvements are `[+6.595461e-6, -1.189737e-5, -8.814843e-6]` nats in seed order 171–173. The positive first contrast is less damage, not absolute clean progress. Both choices make positive clean progress in the other two seed averages.
- Primary consistency contrasts are `[-4.295152e-8, +1.507674e-7, +3.963079e-7]`. Both choices nevertheless **increase** consistency loss in every primary seed average. Relative attenuation of deterioration must not be called absolute consistency improvement.
- Verified all 12 full-step parent comparisons for clean CE, consistency, and the fixed-assignment component, together with primary linear/tenth values, wrong-target CE and accuracy, and displacement norms. Native movement is not uniformly smaller.
- Across the eight cells and six objectives, all 144 seed-level finite/linear contrasts have the same arithmetic sign. Six finite contrasts meet the registered `1e-8` tiny threshold; no linear contrast meets its `1e-10` threshold. These are numerical/descriptive guards, not statistical significance or equivalence tests.
- The two action draws are averaged within a parent, not counted as independent seeds. Seed 172's primary consistency contrast changes sign between draws; seed 171 has one draw with absolute raw consistency improvement despite its negative seed average. The detailed reports preserve these qualifications.

## Wording corrections resolved

1. The HTML primary heading now specifies **final states trained with augmentation**, and its absolute consistency statement explicitly says **averaged over the two batches**. It no longer accidentally generalizes to all saved final states or every draw.
2. The independent interpretation now explicitly makes the adverse warmup wrong-label ordering **relative to raw**. At none/warmup/seed 173, both choices unfit the wrong targets; native merely unfits them less. The corrected opening and conclusion distinguish this from absolute wrong-label fitting. The detailed results likewise identify a relative ordering.

No remaining material numerical or interpretation error was found. The reports correctly distinguish S from clean CE, F from isolated wrong-label memorization, and lower C from competence or safety. Their useful-learning claim remains grounded in the earlier trajectory, not these tiny local contrasts.

## Plot presentation

The primary figure visibly separates absolute objective decreases from paired differences and labels their different nats scales. Every primary seed is shown with zero references. The all-parent figure names its signed-log scale, marks the central linear/tiny region and tiny observations, and distinguishes full steps from independently evaluated materialized tenth paths. No point is omitted or clipped. Neither figure treats the tenth path as simply one tenth of the full effect.

The plotting source and both PNG hashes match the manifest:

- Plot source: `df2d1ff3a69d3dcf032c42cd9e4f91d76a047fb1b0f2305eef929e47f22ad48b`.
- Primary PNG: `d8d338f9d1c3d9fc56ee2dce7148917d5e8aa2ef7f3da7fbe5da2afa79144ddb`.
- All-parent PNG: `6bb5b588254078b1751ca19a4fc53d60bbc9a1c0fea801a6a9a9b430dcb72e7e`.

## Limits preserved

Reviewed corrected prose receipts: results `3b28d77808d98a1c1f81c7c6690e099b1ef296cce0f1b8f3fcf784e78aef7e00`; HTML `c3660c381836d182dbcaa7bf6dc1f6974907a66116bc9799ab83fbc2230119da`; interpretation `d08330c796146b5bb656d4d38e0c19242e050266de04ade2e33761b02d347d09`.
