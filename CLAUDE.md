# Optimizers & Generalization

## Maintained knowledge base

For the current synthesis of this project, begin with
[`knowledge/index.md`](knowledge/index.md). Follow the shared maintenance rules
in [`knowledge/schema.md`](knowledge/schema.md) when ingesting evidence,
answering research questions, or updating conclusions. New durable findings
must update the relevant knowledge pages and append an entry to
`knowledge/log.md`. Validate changes with `python3 scripts/lint_knowledge.py`.

## Project Overview

Research project investigating how different optimizers affect generalization in deep learning, with a particular focus on:

- **Lion vs Muon vs Adam/SGD** — how optimizer mechanics lead to different generalization behavior
- **Emergent misalignment connection** — Jason Brown's finding that Lion increases emergent misalignment (stronger generalization) while Muon decreases it
- **Random labels experiments** — can optimizer choice affect the ability to memorize random labels (Zhang et al. 2016)?
- **Deep learning theory** — why overparameterized networks generalize, implicit regularization, flat/sharp minima

## Key Research Questions

1. How do different optimizers (Lion, Muon, Adam, SGD) differ in their generalization properties?
2. Is there an optimizer that struggles to fit random labels? What would that tell us?
3. Can we reproduce the optimizer-dependent emergent misalignment findings?
4. What does the loss landscape look like under different optimizers?
5. Is emergent misalignment a unique form of generalization, or do the same optimizers that generalize better there also generalize better in standard benchmarks?

## Key People & References

- **Jason Brown** — found Lion increases emergent misalignment, Muon decreases it
- **Zhang et al. (2016)** — "Understanding deep learning requires rethinking generalization" (random labels paper)
- **Chen et al. (2023)** — Lion optimizer (Google Brain, symbolic discovery)
- **Keller Jordan** — Muon optimizer (spectral/SVD-based)
- **LessWrong posts** — series on deep learning theory and "papers that killed DL theory"

## Experiment Plans

- Small-scale random label memorization experiments across optimizers
- Generalization gap measurements (train accuracy vs test accuracy curves)
- Loss landscape visualization under different optimizers
- Emergent misalignment reproduction with different optimizers

## Technical Notes

### Lion Optimizer
- Sign-based updates — uses sign(momentum) for weight updates
- Effectively constrains update magnitude (implicit regularization?)
- Better generalization in emergent misalignment context

### Muon Optimizer
- Takes gradient momentum matrix, conceptually does SVD: M = U Sigma V^T
- Replaces Sigma with identity: update = U V^T
- This fixes the condition number problem (Sigma can have very high condition number)
- Geometrically: projects onto the closest orthogonal matrix
- Worse at emergent misalignment generalization (more conservative?)

## Repo Structure

```
research/          -- notes, literature review, findings
experiments/       -- experiment code
results/           -- experiment outputs, plots
```
