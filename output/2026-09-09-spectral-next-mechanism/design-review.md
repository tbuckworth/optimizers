# Selectivity design review

Verdict: **proceed after the small specification fixes below**. The construct is
worth testing: a competent common state must acquire genuinely useful rare-class
information while exposed to either diffuse error or a reusable wrong cue. The
native filter's answer is not fixed by the generator. Keep the favorable Diffuse
cell, clean acquisition check and full fixed roster; no broad sweep is needed.
This is a source/design review only, not experiment execution.

## Fix before implementation

1. **Distinguish a balanced sham from independent random assignment.** Keeping
   poison IDs and modified labels identical between shared and sham is correct.
   However, target-stratified largest-remainder patch allocation is conditional
   randomization, not an unconditional independent draw. Nor does it guarantee
   exact empirical independence from poison membership within target 0. Integer
   counts generally prevent exact independence anyway. Choose and name the
   estimand: either sample exactly 500 sham patch IDs uniformly with a separate
   RNG, accepting finite-sample correlations without redraws, or retain a
   prospectively balanced sham and specify quotas over modified-target × poison
   status, rounding/tie rules and sampling without replacement. Record patch ×
   poison × true-label × modified-label counts. Do not call approximate balance
   exact independence.

   Also fix the eligible patch universe. Shared patches exclude true 0 and rare
   8, whereas the stated sham can patch true 0. Either use the same eligible
   universe for both, or explicitly acknowledge this difference. A simple
   strong matching option is sham sampling independently within each nonzero
   majority true class, with patch counts equal to shared counts for that class.
   That preserves patch prevalence and true-class composition while breaking
   patch–poison association probabilistically. Do not add all variants as arms.

2. **Use patch-excess susceptibility for the primary coherence interaction.**
   Plain patched ASR can change because the models' unpatched target-0 biases
   differ, even with identical training target counts. Define, on the same fixed
   held-out nonzero images, `E = mean[1(pred_patched=0) − 1(pred_clean=0)]`, then
   `D_E = (E_native,shared − E_native,sham) −
          (E_raw,shared − E_raw,sham)`.
   Keep plain ASR and the originally written D as secondary descriptions.
   Freeze whether the primary population excludes rare 8 (recommended majority
   nonzero digits, with 8 separately), and its class weighting. Do not condition
   evaluation on being correctly classified by each policy: that changes the
   compared population.

3. **Complete the rare-label exclusion contract.** The proposed counts are
   consistent: 9×550 + 50 = 5,000. Assert no true-8 training image, replacement
   target 8 or patched image enters warmup; no rare image is corrupted/patched
   after it. For Diffuse, replacements draw uniformly from the nine digits
   excluding 8, including the example's true label. Report selected replacements
   separately from actually changed labels; the latter define wrong-label fit.
   Preserve disjoint held-out class populations and clean rare metrics.

4. **Specify every probe's target and undefined cases.** Common digit-3 and
   rare digit-8 useful gradients must use their **true** labels, even if a common
   example's training label was corrupted. State whether cell-modified images
   or their unpatched counterpart is the probe input. In shared/sham, use the
   same frozen poison IDs and each cell's corresponding input for both wrong
   and corrected gradients; that makes the same-input residual well defined.
   Diffuse probes use actually wrong labels, not merely selected replacements.
   Clean wrong-gradient/residual probes are absent: null plus an explicit reason,
   not zero. Define handling if fewer than 32 eligible rows exist.

   Diffuse wrong probes contain mixed targets; poison wrong probes all target
   0. Their coherence values therefore are not a clean cross-cell comparison.
   Keep them descriptive and emphasize within-cell policy effects and paired
   shared–sham IDs. Specify whether retention means squared norm of projected
   mean divided by mean-gradient energy, or summed projected individual energy
   divided by individual energy; these answer different questions. Record both
   if desired, with separate names and zero-denominator rules.

5. **Fix probe/Adam temporal alignment.** A post-step-500 group gradient dotted
   with the displacement from step 499→500 is not the before-state first-order
   utility of that step. Name the gradient's parameter state, observer state and
   update index. Either capture group gradients before the corresponding actual
   update, or use an explicitly separate cloned diagnostic step from a saved
   post-update state. Restore all state/RNG and do not advance the real observer
   through probes. The step-100 action is still common warmup, not active native
   filtering. This distinction must survive the result schema.

6. **Define the norm-law action without treating the native operator as an
   exact projector.** Write `n_t = native_filter_action(g_t)` and
   `h_t = ||n_t|| g_t / ||g_t||`. Preserve the stable implementation's centering,
   current-gradient inclusion, numerical drift and identity fallback. Use
   explicit zero tensors when the action is zero, not `None`; freeze cast-aware
   norm tolerance and degenerate handling. A branch-own shadow observer matches
   a function, not other branches' actual norm histories or Adam steps.

## Resource and interpretation checks

The stated 68,400 branch updates plus 300 warmup updates is correct. The previous
24-trajectory pilot took 368 seconds despite expensive clustering; removing its
graph refreshes makes 30 minutes plausible, not guaranteed. This design also
adds about 4,032 individual diagnostic gradients under the proposed probe counts;
include that work in the enforced deadline, not an unbounded postprocessing job.

The 2 GiB output proposal needs an explicit inventory. Saving every complete
observer state at all 21 evaluations would exceed it. A workable scheme is three
shared warmup states, native states at 500/2,000, other-arm final states, saved
float32 logits for train/clean-heldout/patched-heldout, and bounded probe arrays
or clearly specified sufficient statistics. Logits for all 36×21 evaluations
on 15,000 images with 10 outputs alone are about 454 MB. All 4,032 full float32
gradient vectors add about 821 MB. Account for checkpoints, bases, metadata and
reserve before deciding what the independent arithmetic check will read;
summaries alone cannot corroborate their own underlying gradient calculations.

Finally, freeze any binary criterion behind “raw cannot learn the rare class”
or “no cue effect,” or simply report the continuous acquisition contrasts.
Do not choose assay-validity thresholds from outcomes. Rare versus common is
also digit identity, and patch-triggered rare images are a new input condition;
the proposal already correctly avoids calling either distinction a frequency-
only or human-values safety result. These limitations do not weaken the value
of observing favorable Diffuse learning alongside a coherent-harm boundary.

Reviewed source: selectivity proposal (artifact not distributed in this public snapshot). Resource
comparison: [completed neural pilot](../2026-09-09-spectral-clustering-mnist/results.md).

## Resolution: executable protocol reviewed, 9 September 2026

The [executable protocol](../2026-09-09-spectral-selectivity-boundary/protocol.md)
resolves the design-level issues above. **No remaining construct-validity
blocker was identified.** This is prospective acceptance of the specification,
not acceptance of implementation or experimental results.

- Sham and Shared now have identical eligible true digits and per-digit patch
  counts, with explicitly rounded within-digit poison-status balance. The
  protocol correctly avoids claiming exact or unconditional independence.
- The primary interaction uses paired patch excess on nonzero majority digits;
  rare/nonzero populations and ordinary ASR remain separately reported.
- Rare exclusion, actually wrong labels, true-label unpatched common/rare
  probes, same-input residuals and absent/undersized probe handling are explicit.
- Native diagnostics now use pre-update parameters at updates 101/500/2,000,
  after the actual training-gradient observation, then the one real Adam step.
  The native-action norm rule, fallback and cast tolerance are specified.
- There is no binary outcome-dependent admission gate. The output cap increases
  prospectively from 2 to 3 GiB to retain auditable raw evidence; complete
  observers are not saved at every evaluation.

Implementation still must freeze an exact artifact schema, byte inventory,
retention denominators/numerical-span convention and numerical tolerances before
launch. These are implementation details to review against the protocol, not
new experiments or reasons to add arms. In particular, shared held-out logits
can be reused at initialization/warmup, whereas training logits for different
cell-modified images cannot be treated as identical. The
[independent saved-data audit plan](../2026-09-09-spectral-selectivity-boundary/audit-plan.md)
states the required raw inputs and the limits of arithmetic corroboration.
No protocol or scientific source was edited and no experiment was run here.
