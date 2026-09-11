# Targeted coherence-prior check

Codex — Spectral Optimizer Investigation · 10 September 2026

**Scope:** a method-level check of two close precedents, not a comprehensive
novelty certificate. Read Sections 2.4 and 3 of Coherent Gradients, and the
metric, minibatch and empirical-evolution sections of its alignment follow-up.
These sharpen attribution for the current [paper core](spectral_paper_core_2026-09-10.md).

## What the inspected prior experiments already establish

[Coherent Gradients](https://arxiv.org/html/2002.10657) analyzes pristine/corrupt
group contributions using their inner products with the total gradient.
Its intervention clips each coordinate's per-example minibatch gradients at
lower/upper percentile thresholds before aggregation. The MNIST experiment
therefore already studies weak-component suppression and the resulting
fitting/generalization tradeoff. It is coordinatewise robust aggregation,
not temporal covariance eigenspace selection. Its group inner-product
accounting also predates our local signed-gradient diagnostics.

[Making Coherence Out of Nothing At All](https://arxiv.org/html/2008.01217)
defines α = ‖E[g]‖²/E[‖g‖²] and m-coherence = mα. It explicitly distinguishes
example gradients from minibatch gradients, whose averaging changes coherence.
Its ImageNet experiments find rising coherence even under randomized labels.
Thus neither batch-dependent agreement nor the observation that coherent
gradients can support memorization is new here. Their coherence statistic is
not the eigenstructure of a temporally centered observer.

## What our controlled comparison adds as an empirical object

Our [methods supplement](spectral_paper_methods_2026-09-10.md) specifies a
different intervention: conserve complete indexed training exposure while
changing its batch grouping; then replay those histories through observer
copies at fixed model weights and deliver one common gradient from identical
Adam states. Separating incoming signal, native delivery, actual displacement
and held-out loss measures a particular observer-history pathway, with both
positive and adverse cases. The contribution must be this linked empirical
characterization and its boundaries—not a new gradient inner product,
coherence metric, suppression principle or first demonstration of noise
protection.

The distinction is substantive but not a proof of priority over every subspace
or robust-learning method. It also does not establish that temporal filtering
is practically preferable to coordinatewise aggregation. That would require
an explicitly matched method comparison if practical robustness became the
headline. The current conditional-mechanism paper can acknowledge the useful
observed regime without making that stronger claim.

## Claim edits and remaining work

The main draft now explicitly attributes both robust aggregation and the
earlier group-inner-product diagnostic. Keep the positive selective-learning
construction, rare/cue boundary, schedule intervention and fixed-state result.
Do not reframe the entire paper as merely finding that coherence is imperfect:
the follow-up already makes that point, whereas our question is what this
specific deployable observer/action policy does with the available signal.

Nearest temporal-subspace methods and the exact comparator scope remain to
be checked before a novelty judgment. Existing Grokfast, GaLore, GradPCA and
NeuralGrok/EGD comparisons are separate source notes; this small check does
not silently upgrade their coverage. No experiment, analysis rerun, optimizer
change or new scientific audit was performed here.
