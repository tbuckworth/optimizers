# Augmentation study: prospective design review

Codex — Spectral Optimizer Investigation · 10 September 2026

**Disposition: proceed with implementation; no material construct-validity blocker.** The exact protocol addresses the initial design concerns. One minor cue-interaction wording correction is recorded below. This note reviews the full protocol, delivered rationale, accepted selectivity design/results, and inert mask helper. No scientific data, model, or saved tensors were loaded; no experiment, old audit, or mask fixture was executed. Only this review file was written.

## Strongest constructive case

Augmentation might weaken an easy misleading association enough to improve useful common or rare learning, including under the restricted optimizer. Whether it does so, and whether the filter benefits more than raw AdamW, are separate questions. Ordinary random masking versus none is an appropriate first learning-level comparison. Shared/Sham retain a valuable association-breaking control, and Clean makes a competence cost visible. The targeted/opposite comparison gives the idea a fair cue-coverage test when ordinary masks seldom reach a corner.

The outcome is not fixed by construction: training-only erasure on half the occurrences leaves many visible cues, never repairs their wrong labels, may remove useful digit information, and may add centered gradient variation. The simple visibility identity `Cov(bc) = q(1−q)ccᵀ` cautions against assuming that randomized visibility removes the cue from a centered-covariance filter. No local mechanism assay is required before measuring these learning outcomes.

## Initial checks — resolved by the exact protocol

1. **Evaluate without augmentation.** Apply no random, targeted, or opposite erasure to evaluation inputs. Evaluate the original unpatched and white-patched versions of the same held-out images, with true labels unchanged. Otherwise targeted erasure during evaluation would trivially remove the tested cue and answer a different question. Training-fit readouts should also specify whether their images are the canonical cell inputs without erasure.
2. **No label-conditional augmentation.** Gate and center arrays are indexed by update and batch slot, not example ID, poison status, true/assigned label, policy, or cell. Repeated presentations of one image may receive different masks. Use the same occurrence plans across every cell/policy/mode, with the none arm ignoring its gate; apply masks after cue insertion, including to rare and uncued images. Do not patch again after erasure or mutate a shared source image array.
3. **Preserve targets and allocation.** Shared/Sham must reuse identical poison IDs and targets and the accepted within-true-class sham allocation. Clean remains correct-label data before augmentation. Digit 8 is correctly labeled and uncued in every cell, but is still eligible for erasure. The intervention does not repair poisoned labels or make the sham unconditionally independent.
4. **Fix the evidence units and chronology.** Three fresh seed bundles, not 72 independent replicates; 72 paired continuations have 136,800 continuation updates plus 300 shared warmup updates. Warmup is the same unmasked, common-only 100 steps; no arm-specific retraining. Restore complete model/Adam/observer state before each fork. There are 1,512 logical evaluation rows for 21 states per trajectory; initial/warmup predictions may be reused only when their exact inputs and states match.
5. **Show absolute usefulness before interactions.** Give all-seed endpoints and changes from warmup for common/rare accuracy and CE, alongside patched results and target-0 excess. A positive interaction caused by raw deterioration is not a native improvement. Conversely, a useful gain in both optimizers remains useful even with no differential benefit. Low patch excess caused by failing to classify is not selective protection. Do not select endpoints or masks from these results.

## Corner geometry is a substantive interpretation limit

The prospective helper fixes the random rectangle to `[max(c−4,0), min(c+4,28))` on each axis, with integer centers 0…27 and gate probability 0.5. Its source/fixture arithmetic agrees with this paper calculation:

| Geometric event | Conditional on an active random mask | Including the 0.5 gate |
| --- | ---: | ---: |
| Any overlap with the 3×3 corner cue | 49/784 = 6.25% | 3.125% |
| Full coverage of the cue | 25/784 ≈ 3.189% | ≈ 1.594% |
| Mean erased area | 43,264/784 ≈ 55.18 pixels | ≈ 27.59 pixels |

Targeted and opposite masks each erase 64 pixels when active. Targeted fully covers the cue whenever active; opposite never covers it. Thus targeted/opposite are area-matched to each other, **not** to random masking. Half-open even-width rectangles also have the stated edge asymmetry; this is a fixed implementation convention, not a reason to change it after outcomes.

Report realized geometric overlap/full coverage and erased-area distributions, preferably also over the actual cued occurrences in Shared/Sham. Geometric erasure is not the number of previously nonzero pixels changed. A random-mask null would constrain this weak-coverage augmentation, not augmentation generally or a more frequently hidden cue. Do not increase the rate or move the mask after observing a null.

## Interpretation boundaries to retain

- Targeted masking knowingly uses the synthetic cue location. Applied to all examples, this is not covert label leakage, but it is a privileged coverage diagnostic rather than discovery of an unknown shortcut.
- Opposite-corner erasure matches area and gates, not erased digit content. Shared/Sham subtracts an association-related difference but does not make targeted-versus-opposite a pure causal mediation estimate. Different training pixels and nonlinear trajectories remain.
- Clean/common/rare competence must stay visible because masking may destroy label-relevant strokes. Label preservation is a recipe choice, not a theorem that every masked image remains unambiguous.
- New seeds make this a fresh paired test of the augmentation recipe, but prior task-informed design, the same dataset family, three seeds, one fixed rank, one learning rate, and inherited common-only warmup limit generalization. No emergent-misalignment or generic safety claim follows.
- Runtime includes augmentation, evaluation, logging, and any observer work. It is descriptive resource accounting, not a matched optimizer speed comparison. Freeze host/GPU/time/output limits, exact source/input pins, complete roster, and once-only handles before launch; independently verify saved evidence afterward without replaying training.
- There is no need to add Diffuse, a norm arm, a mask sweep, or another local diagnostic gate to answer this selected question. Those would answer additional questions rather than repair the proposed comparison.

## Sources inspected

## Exact protocol review

Read `output/2026-09-10-spectral-augmentation/protocol.md` completely, SHA256 `afcb7caa954b6182723f6b4dca65edd136911e59c04e3766c873c800d19dcf97`. Sections 3–5 resolve every initial execution-definition concern above: exact shared starting state and schedules; independent occurrence-level mask RNG; cue insertion before masking; unchanged targets; inclusion of rare/uncued images; unmasked evaluation; explicit geometric coverage among actual cue occurrences; absolute and warmup-relative competence; and secondary optimizer interactions. The 72-trajectory/three-seed roster and 1,512 logical row count are correct.

**Minor wording correction requested:** section 5 follows `I_Q = ΔQ(Native) − ΔQ(Raw)` with “negative means reduced excess association.” A negative interaction means a *relatively more favorable change for Native than Raw*. It does not by itself mean Native's association excess decreased: for example, Native +1 point and Raw +2 points give a −1-point interaction although both worsened. Absolute Native reduction requires `ΔQ(Native) < 0`. The specified absolute `Q`, component rates and `ΔQ` already supply the necessary evidence; no new metric or arm is needed.

The theoretical law of total covariance is correct with the stated fixed-parameter and independent-occurrence conditions. It is not an estimator identity for the moving finite-memory observer. The distinction between constructive augmentation benefit and differential optimizer benefit is appropriately explicit. Targeted coverage is not a known favorable neural outcome because wrong labels and half the visible-cue exposures remain.

The proposed 1-CPU, 16-GiB-host/no-swap, 8-GiB-allocated-GPU, 25-minute cooperative/30-minute hard, 2-GiB-output envelope is coherent for a bounded local test. It is **not yet a verified storage or runtime certificate**. The source must complete the protocol's promised exact byte estimate before launch. For scale, all 1,512 logical rows with three 5,000×10 FP32 logit arrays would occupy 907,200,000 bytes; deduplicating initial/warmup arrays reduces this. Full endpoint observers/Adam/model state and provenance also count toward the cap. Keep required evidence and the frozen limit rather than discovering an overrun and dropping artifacts. Reserved once-only acquisition/audit handles are appropriate; later source review and admission must certify executable resource enforcement, not treat this design review as a launch certificate.

The inert mask helper reviewed here has SHA256 `3a17d98db89754e118b52722392fc3108d4de330211a0fe25e5b1f4471166f0f`; its source-inspected fabricated fixture file has SHA256 `0ed9202dc225cf60e33cd11a14f027187402c2a434396068d1bd319defceb83b`. No test outcome is claimed from this review. External reference verification is owned by the main agent; this note checks the explicitly specified geometry and mathematics rather than adding another literature search.

The researcher guidance informs construct-validity and evidence discipline; user-authorized autonomy and this bounded task override workflow approval, fail-fast, delegation, or rerun defaults. No broader experiment, mechanism gate, or new user-approval gate is recommended.

### Resolution read-back

The main agent corrected the cue-interaction wording and explicit logical-logit byte accounting before implementation freeze. Read-back of protocol SHA256 `4eaeb66a4d87ad0f59f5164449deff3d3dee6e81df659f6b757d8a60b5c091be` confirms both. No design correction remains outstanding; exact artifact budgeting and executable resource checks remain implementation/admission work, not an additional research gate.
