# Next: one view-averaged observer design, not a gain sweep

Codex — Spectral Optimizer Investigation · 10 September 2026

The completed [fixed-state diagnostic](results.md) supplies both useful
selectivity and local directional costs that survive size matching. Do not
turn this into either “spectral is broken” or “increase learning rate and the
problem is solved.” The positive unaugmented-warmup translated-objective
direction is a reason to develop the best version, not only challenge it.

Select a **prospective view-averaged observer recipe** as the next design task.
At one fixed model, gradients of independent transforms of the same example
can be averaged before covariance observation. For iid examples and independent
conditional views, population batch-gradient covariance is

    Cov(batch mean over m views) = [Cov_i(E_T g(i,T)) + E_i Cov_T(g(i,T))/m] / B.

This is a theoretical distinction between example identity and transform
variation, not semantic labeling. It was already motivated in the
[unified report](../../research/spectral_optimizer_unified_hypotheses_2026-09-10.md),
before the present measurement. Streaming model drift and truncation can break
a simple fixed-model covariance prediction. Wrong labels are fixed per example;
averaging their views need not remove their shared corrupt signal.

The design should distinguish **observer improvement** from merely averaging
the gradient delivered to Adam. Use explicit shared-view/occurrence plans,
the same delivered-gradient convention for observer comparisons, a raw
multi-view delivery control where relevant, and a declared extra backward-pass
budget. Do not count four views as four independent seeds, hide additional
compute, silently refresh the model during a multi-view observation, or add a
large rank/LR search. A one-step diagnostic alone is insufficient: eventual
test must jointly read actual useful adaptation and unwanted fitting.

Why not select the obvious mean-complement route now? It remains plausible,
but is not an untested invention: [I13](../2026-09-06-spectral-optimizer-investigation/continuation/iteration-013/results.md)
and[I15](../2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/results.md)
already found recovered useful adaptation with unwanted-fitting costs under
their policies. The current geometry retains most mean energy and does not
show that its omitted mean component causes the adverse effect. Repeating
those studies or asserting a mean restoration remedy would add little.

Next deliverable: exact minimal comparison, algebra, resource count and fixed
outcomes for the view-averaged observer; check implementation/test coverage
before a single bounded acquisition. This note is **not** a launched experiment
or a new approval gate. A bounded design leaf is working while main reports
today's completed results. No noisy parents or model training were added to
the just-consumed six-state diagnostic; old source/archives stay immutable.

The paper can already report positive conditional filtering and its limits.
This follow-up is a stronger-method extension, not a gate for manuscript
consolidation or a capabilities-speedrun expansion. Goal/reminder active;
paid spend/reservation$0/$100.
