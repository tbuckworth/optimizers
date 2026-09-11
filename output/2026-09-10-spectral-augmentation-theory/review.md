# Theory supplement: review and scope

Codex — Spectral Optimizer Investigation · 10 September 2026

**Disposition: theoretical derivations checked; no new scientific acquisition.**

Main wrote the [source-linked note](../../research/spectral_augmentation_loss_geometry_2026-09-10.md)
while the existing rank200 experiment continued. A separate leaf derived the
identities and limits in [its independent review](independent-math-review.md)
before reading the integrated note. Main read that review in full and checked
the binary-CE realization by hand. This is mathematical review, not a fresh
independent replication or an audit of the running experiment.

## Checks

- Exact finite-view cross-entropy decomposition and KL orientation agree.
  Softmax of mean logits is the normalized geometric probability center.
- Fixed label versus corruption-law expectation gives the stated linear
  logit force; adaptive fitted parameters invalidate simply averaging that
  force away. Derivatives include the moving logit center.
- The projector counterexample is realizable by two affine binary-logit
  views. Both total-objective steps descend locally, while consistency changes
  have opposite signs despite 100/101 gradient-energy retention. Full retention
  is correctly treated as a special exception for an orthoprojector.
- Main additionally used a one-off, standard-library-only arithmetic check:
  three explicitly fabricated three-class logit views with weights .2/.3/.5;
  CE decomposition for all three hard targets; KL identity within 1e−12.
  The independent review's explicit binary construction at θ=0 and step
  size 1e−7 gave ΔR=−1.0019998475385705e−7 for the raw step and
  +1.0000000993937164e−7 for the projected step. Both total losses decreased.
  These are checks of stated algebra, not neural experiment evidence.
  No files, scientific arrays, model checkpoints, optimizer or GPU were used.
- Main inspected primary PDF sections: Wager2013 §2/equations4–6;
  Dao2019 §§4.1–4.2; Wood2023 AppendixB.3/equations32–33. Exact URLs and
  contribution boundaries are in the note. Existing theory is credited;
  no theorem novelty or imported external empirical result is claimed.

## Interpretation retained

Augmentation and filtering need not be redundant: consistency can constrain
view-specific fitting but leave transformation-consistent memorization intact.
Further useful spectral restriction is possible. The small recipe's practical
negative and the stronger historical protection evidence both remain unchanged.
Neither this decomposition nor a favorable consistency score proves truth
recovery, useful Adam delivery, safety improvement or trajectory mediation.

## Readable artifact