# Cross-study compatibility check before another experiment

Codex parent, Spectral Optimizer Investigation, 6 September 2026.
Status: completed **static source/provenance comparison**, not a training replay.
No experiment was restarted and no model, dataset or checkpoint was loaded.

## Finding

Iteration 004 hard32 and iteration 006 current32 implement the same nominal
delivery rule and small-MNIST recipe; their AdamW baselines likewise share the
same implementation. No changed nominal learning configuration or checkpoint
selector was found to explain the selected-accuracy sign reversal. This supports
investigating bundle variability, but does not prove that seeds are the only
cause: a matched-input cross-harness trajectory replay has not been performed.

## Direct sources

- Iteration 004 execution (artifact not distributed in this public snapshot), revision
  `e9126179a010323111ded3dad37ba1c0ba0af244`, and
  [harness](../iteration-004/norm_control_harness.py).
- Iteration 006 execution (artifact not distributed in this public snapshot), revision
  `bdeb4078edfd7e2a7df7576f7bbee96c06221dfa`,
  harness (artifact not distributed in this public snapshot) and
  [policy](../iteration-006/policy_math.py).
- Shared [helper](../iteration-003/neural_harness.py), SHA-256
  `5c0bb4bdbf0c561eef43686aaf86f9608d564f4df46cda8c7f4ebf8d7f7bb1b7`.
- Shared [canonical optimizer](../../../../spectral_filter.py), SHA-256
  `9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`.

All nine iteration-004 and fifteen iteration-006 bound scientific-source hashes
match the current files. The two common source entries have identical hashes
in both execution manifests. This is a check against recorded execution bindings,
not an independent reconstruction of every instruction executed historically.

## Same nominal components

| Component | Source-level comparison |
|---|---|
| Split, corruption and batch plans | Both call unchanged `make_plan`, namespace `[20260906,3,seed,stream]`, and `dataset_for_plan`; bundles differ (3–5 versus 6–8). |
| Data content | All four training/test IDX size/hash pairs match. Path strings differ because one manifest resolves a symlink; both directories resolve to the same location. |
| Network and initialization | Both call unchanged `make_model`: 784–64–ReLU–10, 50,890 parameters, fixed per-bundle initialization. |
| AdamW | Both call unchanged `make_optimizer`: LR .001, decay .01, betas .9/.999, epsilon 1e-8, foreach/fused false. |
| Current hard filter | Width 32, decay .99, stable recurrence, repair schedule 100, eigenvalue floors 1e-8/0; both call the same `observe_and_project` exactly once per raw gradient. |
| Activation | Raw delivery through step 100; current native float32 basis projection from step 101. Missing basis has identity action. |
| Training | Same raw cross-entropy loss, batch 64, 2,000 steps, gradient assignment and AdamW call. No dropout, batch norm, schedule or augmentation. |
| Validation | Same helper evaluates all 5,000 validation examples on grid 0,100,…,2000. Both preserve the earliest strict accuracy maximum and CE minimum independently. |
| Test | Same helper evaluates four retained states only after all training/selection in that study. Same 10,000 official examples; no new independent dataset. |
| Environment | Both record Python 3.12.3, Torch 2.11.0+cu128, NumPy 1.26.4, CUDA 12.8, RTX 3090, one CPU thread, deterministic algorithms and TF32 off. |

For current32, iteration 006 obtains the same current candidate as iteration
004 hard32, computes an additional previous-basis candidate, then delivers a
clone of the current candidate without rescaling. The additional previous-basis
calculation does not alter the declared current-delivery rule. Within-study
instrumentation-on/off pilots passed previously; that is not a cross-study
same-seed equivalence test.

## Differences retained rather than hidden

- Primary bundles change all sampled factors together: initialization, split,
  replacement labels and minibatch order. “Seed variability” does not isolate
  which factor causes a contrast, and several factors may interact.
- Iteration 006 adds clean conditions and lagged arms, stronger gates, per-step
  hashes, different optional measurement schedules and exclusive pretest records.
  These add verification, not proof of cross-harness bitwise equivalence.
- The **scalar controls are not byte-identical policies**: iteration 004
  multiplies native float32 by its scalar; iteration 006 explicitly multiplies
  in float64 and casts once. Their norm gates also differ. Do not extend the
  current32/AdamW compatibility finding to scalar trajectory equivalence.
- Final training scoring uses two helper passes in iteration 004 and shared
  logits in iteration 006. It occurs after training and is not a selector.
- Test reuse and adaptive research history remain. A future fixed bundle study
  would be a conditional stability check on this recipe and benchmark, not an
  untouched test of generalization or a retroactive pooled confirmatory result.

## Consequence for planning

There is no identified defect warranting a restart of either completed study.
Retain the prior negative and later positive selected-baseline comparisons.
Include AdamW in a future ordering stability study if that study is also meant
to address the baseline reversal; three filter-only arms cannot answer it.
An additional matched-input check, if required by design review, must be a
separate bounded compatibility diagnostic, not a relaunch of either experiment.
