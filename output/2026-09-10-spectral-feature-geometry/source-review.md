# Feature-geometry connection: bounded primary-source check

Codex — Spectral Optimizer Investigation · 10 September 2026

This is a conceptual supplement prepared while the fixed strong-augmentation
acquisition runs. No scientific data, model, checkpoint or partial outcome was
used. It does not select an experiment, change the protocol or establish novelty.

## What was checked

- **Martens and Grosse (2015), K-FAC**, [conference paper](https://proceedings.mlr.press/v37/martens15.pdf),
  §1.2 and §2. The paper supplies the layer-gradient outer-product/vectorization
  identity and approximates a joint Kronecker expectation by the product of
  activation and derivative second moments. Its Fisher samples labels from the
  model. Our fixed-last-layer, uniform-prediction/uniform-label calculation is
  an exact special case of this familiar algebra, not a new factorization.
  Hard top-subspace projection is not inverse-Fisher preconditioning.
- **Kunstner, Balles and Hennig (2019)**, [paper](https://arxiv.org/html/1905.12558v2),
  §2, §4 and §5, especially Eq.18. Observed-label gradient second moments,
  centered stochastic-gradient covariance, Fisher and curvature are distinct
  objects in general. Their variance-adaptation discussion is relevant but
  does not justify retaining the largest-variance directions. The exact
  uniform-label/uniform-prediction coincidence in our construction must not
  be extended to trained networks or the native temporal estimate.
- **Li, Soltanolkotabi and Oymak (2020)**, [conference paper](https://proceedings.mlr.press/v108/li20j/li20j.pdf),
  definitions1.1–1.2, Theorem2.2 and §4/Theorem4.1. This is a direct conceptual
  precedent for useful learning preceding corruption fitting through a
  low-rank, sufficiently diffuse Jacobian. The stated results use their
  clustered-data or smooth nonlinear least-squares assumptions and bounded
  sparse corruption. They do not prove our multiclass cross-entropy algorithm,
  a90%-replacement result, or identification of semantic directions by a
  learned temporal covariance. We did not audit all supplementary proofs.

No quotations are needed. The bibliography check is targeted, not exhaustive.
Earlier project synthesis already cited empirical-Fisher limitations; the new
connection is the explicit feature-second-moment specialization and its limits.
The separate derivation is
[spectral_feature_geometry_2026-09-10.md](../../research/spectral_feature_geometry_2026-09-10.md).

## Interpretation boundary

The promising hypothesis is conditional: noisy-label gradients may reveal
input/representation geometry even when their mean class signal is weak.
Useful learning then additionally requires that the retained geometry overlap
the desired task. High input-feature energy is neither truth nor importance.
The zero-signal fully random-label case cannot recover class labels merely
because its covariance has structure. The live experiment measures practical
complementarity with augmentation, not this mechanism's mediation.
