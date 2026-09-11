# Action/history synthesis: bounded mathematical review

**Final verdict: PASS for this bounded mathematical/evidence-scope review.** No sign, telescoping, or summation-index error was found in the identity. Two documentation corrections were identified, implemented by the source owner, and verified below. No unresolved issue remains within this review's scope; this is not scientific acquisition or audit acceptance.

Reviewed source: [spectral_action_history_synthesis_2026-09-10.md](../../research/spectral_action_history_synthesis_2026-09-10.md), final SHA-256 `a77559826570fba4948403211e0e8026ecb216f5ccc3cba1925ee702019cd26a` (125-line version, read fully after correction). Initial reviewed SHA-256: `1bf82331af3a8973dbf0b004acb1461f87abc30557b35e781c7473f7351a8351` (119 lines). The research-workflow skill was used for evidence discipline and equal treatment of constructive/adverse evidence, not to restart its workflow. Review was confined to documentary sources and elementary derivation; no literature search, datasets, arrays, checkpoints, models, new numerical scientific calculation, experiment, or audit replay.

## Resolved corrections

1. **Evidence scope, lines 18–19:** “The studies use different seeds” incorrectly suggests that all three studies have distinct seed bundles. The component diagnostic deliberately reuses the strong bridge's seeds 202609171–173 and native warmup/final states. The observer-history study uses seeds 202609121–123. Suggested replacement: “The observer-history study uses different seed bundles and operating points; the component diagnostic deliberately reuses strong-bridge native states. The interventions and estimands differ, so their headline numbers cannot be joined into one measured causal chain.” This preserves the valid non-mediation conclusion without implying fresh independence.

   **Verified resolved:** final lines 18–23 explicitly distinguish the observer study's seed bundles from reused strong-bridge native states and retain the different-intervention/estimand qualification.

2. **Diagnostic/horizon indexing, near lines 73–86:** the algebra's “final transition” is t = T−1. For the reported training endpoint T = 56,304, the diagnostic's *final parent* is x_T, and its hypothetical probe update is 56,304→56,305, corresponding to F_T—not a summand in the displayed t = 0,…,T−1 identity. Likewise h100 probes 100→101. Suggested sentence: “For the reported T = 56,304 training horizon, the final-parent probe is beyond that endpoint, not the last transition in this identity.” The existing identity is correct; this is an important precision about what the measurements do not estimate. Calling that probe a final-transition c term would be incorrect unless a different horizon and corresponding terminal comparison were defined.

   **Verified resolved:** final lines 86–90 give both 56,304→56,305 and the actual last transition 56,303→56,304, identify the 100→101 warmup probe, and explicitly deny that the final-parent probe measures the reported gap's last summand. Finding line references above refer to the initial version.

## Independent derivation and signs

Fix the exogenous draws, a deterministic full-state transition for each policy, and a common initial state x₀. Define

```text
G_t = V_t^R(x_t^N),                         t = 0,...,T.
```

G_t is the terminal loss of the hybrid policy using native for its first t transitions and raw thereafter. By the recursion for V and the native trajectory,

```text
c_t = V_(t+1)^R(x_(t+1)^N) − V_t^R(x_t^N)
    = G_(t+1) − G_t.
```

The endpoints are G₀ = V₀^R(x₀) = H(x_T^R) and G_T = V_T^R(x_T^N) = H(x_T^N). Therefore the sum is exactly H(x_T^N) − H(x_T^R). There are T terms, no omitted initial term under the stated common-start assumption, and no terminal continuation after t = T−1. Without a common start, an additional initial-value difference would be needed; the source already states the necessary condition.

The immediate quantity d_t = H(F_t^R(x)) − H(F_t^N(x)) is a **benefit**: positive means native gives lower immediate loss. The c_t ordering is a **cost**: positive means that one native choice, followed by the common raw future policy, gives higher terminal loss. Consequently c_(T−1) = −d_(T−1)(x_(T−1)^N), because V_T^R = H. For earlier steps, replacing V_(t+1)^R by H is not generally valid. No differentiability, linearity, convexity, stationarity, or expectation over seeds is required for the telescoping statement.

## Limits and evidence checks

The full state must retain everything affecting later transitions, including inherited moments/counters and observer memory/counters, with any remaining randomness fixed. Two branches then share the **future raw policy and draws**, not necessarily future gradients or parameter displacements. The source makes that distinction correctly. These terms are reference-policy-dependent hybrid contrasts, not unique additive allocations to representation, Adam, or observer memory. Small immediate H differences at a few states do not bound changes in the future-evaluated V terms. The claim concerns a fixed horizon and terminal objective; it should not silently be applied to independently validation-selected checkpoints.

The source also correctly distinguishes the exact fixed-panel identity D_L = D_S + D_F + D_C from clean held-out D_H. Different labels, panels, and objectives prevent substituting the former decomposition for the latter. Neither identity identifies a mediation percentage or supplies a truth label for a retained direction.

Documentary scope checks:

- [Strong bridge](../2026-09-10-spectral-strong-augmentation/results.md): supports actual multi-seed post-warmup native progress, an adverse native-plus-translation comparison, and the separate validation-selected accuracy/CE boundary. It does not identify the cause of the trajectory difference.
- [Observer-history intervention](../2026-09-10-spectral-observer-pathway/results.md) and [saved-vector accounting](../2026-09-10-spectral-observer-signal/interpretation.md): support a fixed-model/Adam/input local pathway, favorable relative Clean rare effects in all three reused parents and mixed Diffuse effects. One Clean parent is damage reduction rather than positive absolute rare progress. Saved-vector accounting is not a separate fresh scientific replication. The synthesis does not overstate these points.
- [Component utility](results.md): supports the primary two-draw seed averages, unsupported local lost-useful-consistency account, opposing warmup relative C/clean ordering, and the mismatch between F and actual wrong-subset fitting. The small reused-state action contrasts are not measurements of V_t or accumulated policy effects.

No new experiment, mediation claim, or novelty claim is needed. With both wording clarifications verified, the note is suitable as an elementary explanation of why the three kinds of evidence must remain distinct.
