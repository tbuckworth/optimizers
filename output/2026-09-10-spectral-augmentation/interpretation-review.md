# Augmentation: outcome interpretation review

Codex — Spectral Optimizer Investigation · 10 September 2026

**Bottom line:** ordinary masking improves useful learning with raw AdamW, but does not rescue native spectral rare learning. Targeted masking produces a genuine, modest reduction in the registered spectral cue contrasts, accompanied by worse unpatched competence. The strongest positive is conditional protection/regularization, not successful selective defense.

This is interpretation of the completed [scalar summary](audit/summary.json), not another scientific audit. All numbers below are stored endpoint values or prespecified paired contrasts. Three fresh paired seeds, not 72 independent replicates, support this one fixed small-MNIST recipe. Both scientific jobs are finished; none was rerun.

## 1. Useful effects first

The primary Random−None comparison establishes a real augmentation benefit for raw AdamW: rare accuracy and rare CE improve in **every seed in each of Clean, Shared and Sham**. Common CE also improves in every seed in each cell. Common accuracy improves in two seeds under Clean/Shared and all three under Sham.

| Cell | Raw rare accuracy, None → Random | Native rare accuracy, None → Random | Native rare CE, None → Random |
| --- | ---: | ---: | ---: |
| Clean | 56.07% → 61.40% | 11.60% → 10.40% | 2.3333 → 2.4127 |
| Shared | 57.27% → 63.80% | 9.20% → 3.40% | 2.5550 → 2.7285 |
| Sham | 58.07% → 61.67% | 0.73% → 0.20% | 3.1205 → 3.2335 |

Native rare CE worsens in all three seeds in every cell; its rare accuracy falls in all Clean/Shared seeds and the sole nonzero Sham seed. The relative rare-accuracy and rare-CE augmentation interactions favor raw in all three seeds within every cell. This is neither a benefit hidden by a difficult raw baseline nor merely a failure of all augmentation to work.

Sources: summary keys `groups`, `augmentation_contrasts`, `interactions`, and `policy_contrasts`, with metric paths `heldout_unpatched/...` and `heldout_patched/...`.

## 2. Targeted masking: a real cue change with a competence cost

For native Shared, Targeted−None reduces patch excess E by 2.250, 2.300 and 6.425 percentage points: mean **−3.658 points**, favorable in all seeds. Its Shared−Sham association excess Q falls by 3.925, 0.625 and 5.950 points: mean **−3.500 points**, also favorable in all seeds. These registered reductions are real; they should not be discarded because the larger learning objective is unmet.

But native Shared common accuracy falls 87.96% → 87.03%, common CE rises 0.4676 → 0.4987, rare accuracy falls 9.20% → 0.47%, and rare CE rises 2.5550 → 2.9487. Every paired seed is adverse on all four competence measures. Rare accuracies change from **0.8%, 5.0%, 21.8%** to **0%, 0%, 1.4%**: the mean loss is dominated by the third seed, but the direction is not.

Targeted also worsens all four native Shared competence measures relative to Opposite in every seed. Native E improves relative to Opposite in all seeds, but Q improves in only two: its Targeted−Opposite Q changes are **−4.500, +0.975, −5.725 points**. Thus cue coverage matters to the observed contrast, but the location control does not establish uniform association-specific protection or pure mediation. The Targeted optimizer interaction in Q is favorable in only two seeds, not three.

## 3. Keep E, Q, and their components distinct

E is patched minus unpatched target-0 prediction rate on the same nonzero common images. Q is E in Shared minus E in Sham. Neither is raw patched target-0 frequency.

The primary Random result is an important trap: native Q falls **96.275 → 95.433 points**, and its optimizer interaction is favorable in all seeds. Yet native **Shared E rises 95.717 → 95.917 points**. Q falls because Sham E increases more (−0.558 → +0.483 points), not because Shared susceptibility falls. The native absolute Q change itself is favorable in only two seeds.

Even Targeted's all-seed E improvement needs its components:

| Shared target-0 rate | Native None → Targeted | Raw None → Targeted |
| --- | ---: | ---: |
| Patched images | 97.475% → 96.083% | 99.392% → 99.800% |
| Unpatched images | 1.758% → 4.025% | 0.500% → 2.725% |

Native patched target-0 rate improves in two seeds, while unpatched bias worsens in all three. Raw's E decreases even though its patched target-0 rate worsens in all three. Therefore neither E nor Q alone supports “the attack stopped working.” Native Targeted patched common accuracy/CE improvements versus None are also only two-seed effects. Report absolute competence and component rates beside every favorable interaction.

Sources: `cue_association/{Q,delta_Q,I_Q,targeted_minus_opposite_Q}`, `augmentation_contrasts/shared/.../cue/majority_nonzero/...`, and `location_contrasts`.

## 4. Learning and mechanism limits

Native still improves rare CE from the inherited common-only warmup in every cell/mode/seed. Poor recognition is not no learning or a frozen model. Conversely, native Shared/Sham common CE remains worse than warmup in every mode/seed: relative preservation must not be called unqualified acquisition. Tiny strict-sign artifacts such as the approximately −1.1e−16 Clean Targeted common-accuracy interaction are effectively ties, not substantive seed effects; preserve the stored values rather than promoting their machine sign.

The prospective conceptual note (artifact not distributed in this public snapshot) supplies the crucial boundary: surviving Shared cues remain perfectly associated with their assigned wrong target. Erasure reduces frequency, not that reliability, and retains wrong labels. Ordinary masks also rarely fully cover this corner. These results constrain this erasure recipe; they do not prove that augmentation increased nuisance covariance or that reliability-changing/invariant-view interventions cannot help. No gradients or observer products were measured in this study to establish such mediation.

## 5. Single next research decision

**Do not tune the erasure sweep further; prioritize designing one reliability-changing augmentation contrast.** A concrete candidate is occurrence-wise cue reassignment independent of assigned labels, at a prospectively matched cue prevalence, compared against the existing static association-breaking Sham construction with the same wrong labels and useful-learning readouts. The new question would be whether removing cue reliability, rather than just hiding it intermittently, changes the learning tradeoff. The existing Sham already shows that low cue association alone does not rescue native rare recognition, so this is not a promised rescue or a reason to suppress the present negative interaction. Define the single contrast and its scope before any acquisition; this review launches or authorizes no new run and requires no compulsory local diagnostic first.

## Evidence scope

The [audit record](audit/audit.json) is PASS: 315 artifacts, 1,512 evaluation records, 72 trajectories and 668,052 checks. It independently rebuilds prediction metrics from saved logits and checks receipts; it **does not** independently recompute internal checkpoint-tree hashes, execute state restores, or reread training IDX labels. Fixed contrasts use the shared pure analysis function. Preserve these explicit limitations rather than describing this as a second full-state experiment or independent contrast implementation.