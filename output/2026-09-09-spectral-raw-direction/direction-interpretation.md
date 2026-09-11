# What the raw-direction comparison can identify

Codex · Spectral Optimizer Investigation · 9 September 2026.
Prospective interpretation written while the final acquisition seed runs;
no new endpoint results used. Exact-arithmetic theory unless explicitly noted.

## Three distinct geometric questions

At one common state, let g be the raw gradient, A=VVᵀ the updated legacy
operator, and P=QQᵀ the orthogonal projector onto its retained numerical span.
Write a=‖Ag‖, q=Pg and ρ=‖q‖/‖g‖. For nonzero g,q,a, the two norm-matched
incoming actions are

<pre>
d_raw  = a g / ‖g‖
d_proj = a q / ‖q‖
‖d_raw‖ = ‖d_proj‖ = a
cos(d_raw,d_proj) = ρ
‖d_raw − d_proj‖² = 2a²(1−ρ)
</pre>

This distinguishes directional restriction from the action norm. Here a uses
the full stored operator VVᵀ, while P uses the threshold-retained numerical
span. If thresholding discards a nonzero singular direction, a can include its
contribution; exact separation into gains on the same span requires retaining
all nonzero directions. The new raw policy does not discard the learned
operator: it still sets the scale from that full operator, at its own evolving state.
If α=a/‖g‖, then

<pre>
α² = (gᵀA²g)/(gᵀg)
</pre>

Thus raw delivery retains a direction-dependent scalar observation of learned
geometry. It does not implement a constant or precomputed learning-rate
schedule, and it is not ordinary unfiltered AdamW. After branches diverge,
neither g, V, a, nor Adam history is held numerically common. The intervention
compares policies sharing a functional scale law, not matched scalar sequences.

## Raw is steepest locally for SGD; that does not predict the experiment

Cauchy–Schwarz gives the exact common-state first-order identity

<pre>
gᵀd_raw = a‖g‖ ≥ a‖Pg‖ = gᵀd_proj
projected / raw first-order SGD decrease = ρ
</pre>

Among all vectors of length a, raw gives maximal immediate descent of the
training objective. Among vectors restricted to the span, projected gives the
maximum. This says nothing by itself about held-out loss or rule information.
It also does not guarantee the better finite training step.

For a concrete quadratic, choose a=1, g=(1,1), P=diag(1,0), and local Hessian
H=diag(1,100). This can be realized by L(x)=gᵀx+½xᵀHx at x=0. Then

<pre>
L(−ηd_raw) − L(0)  = −√2 η + 25.25 η²
L(−ηd_proj) − L(0) = −η + 0.5 η²
</pre>

At η=0.05, raw changes loss by approximately−0.007586 and projected by−0.048750.
Both descend, but removing the high-curvature direction helps the finite step.
This is a mathematical possibility, not evidence that this learned subspace
estimates curvature or that curvature explains any result.

Even a frozen positive diagonal preconditioner changes the first-order ruler.
For g=(1,1), the same P and a=1, take D=diag(100,1). Updates−ηDd give
gᵀDd_proj=100 versus gᵀDd_raw=101/√2≈71.42. Here projected wins the local
preconditioned comparison despite raw winning the Euclidean one. This is a
counterexample to transferring the SGD ordering, not an approximation asserted
to describe the actual run: Adam additionally changes its moments as a function
of the delivered action and inherits nonzero history.

## Reading the fixed endpoint comparison

If projected has better endpoints, that supports a useful post-fork effect of
the directional-restriction policy relative to this specified alternative.
It would not identify semantic feature selection: curvature, temporal effects,
different scalar feedback and Adam history remain possible causal pathways.

If raw has better endpoints than projected, learned span restriction is not
required to achieve those raw-policy endpoints. Improvement relative to native
or a baseline requires that separate contrast. This result would not show that
learned geometry or pre-fork filtering is useless: both the scalar feedback
and inherited model/Adam/filter state retain that history.

Mixed seed/metric/endpoint outcomes must remain mixed. Small differences are
not equivalence. Large effects at2500 are not time-to-grokking measurements;
only the predeclared2000/2500 endpoints are compared. The archived reference
adds CUDA trajectory sensitivity beyond the ideal common-state equations.

The strongest safety-relevant contribution remains insight into when a
history-dependent restriction helps useful learning and when it hurts. None
of these geometric statements labels a direction as benign, harmful, general
or memorized, and none establishes language-model or alignment transfer.
