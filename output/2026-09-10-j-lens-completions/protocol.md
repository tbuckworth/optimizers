# Meaningful completion follow-up — prospective

Status: preparation only; parent reviews source and releases GPU before acquisition.
Exploratory single frozen model, fixed authored panel; no training or paid compute.

## Question and design

Do meaningful completion-loss gradients give topic-related directions under
the Euclidean residual-space convention, and does PCA add value beyond individual
vectors, the mean, full vectors and random projections? This addresses the previous
common ` The` target and topic/style confounds together; it cannot isolate their
separate causal contributions and is not an optimizer experiment.

48 rows: four topics × six distinct prefix/completion pairs × two identical
cross-topic framing styles. First four pairs/topic fit; final two held out, in
both styles (32 fit, 16 held out). Content pairs never cross split. Every
completion is a meaningful authored clause, at least two model tokens. These
are teacher-forced reference completions, not generated answers or sampled
natural-corpus observations. Their topic vocabulary deliberately enters loss.
No chat wrapper. Raw text tokenized jointly; boundary offsets exclude any token
straddling the prefix/completion boundary by rejecting such inputs. No truncation;
reject total length >96. A newline separates framing from content and the
completion starts with a space.

Retain original pinned model/lens and layers 11/17, rank four. Extract activation
at the final prefix token and g = derivative of mean completion-token NLL at
that same post-block residual. All completion positions contribute by causal
backpropagation; vectors remain width1024 regardless of completion length. Also
save first-completion-token gradient as a paired diagnostic, without another
forward. Save per-token NLLs and exact IDs/boundaries. Frozen parameter checks.

## Correct mathematical objects

Paper: https://transformer-circuits.pub/2026/workspace/index.html (Methods).
Source: https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e
`jlens/lens.py:135` transports row vectors as h @ J.T;
`jlens/hf.py:166` applies final normalization then unembedding.
`jlens/fitting.py:166` seeds output residual coordinates and backpropagates to
assemble Jacobian rows. It sums valid future output contributions and averages
valid source positions, then prompts: implementation weighting is more specific
than a uniform average over all source/target pairs.

J is a forward map from intermediate activation perturbations to final
**pre-normalization** residual perturbations. Official activation readout is
W_U norm(J h). J-Lens does not PCA training gradients. Completion gradients obey
g_l = mean_t J_context,l,t^T Dnorm(h_final,t)^T W_U^T(p_t-e_target).
J^T pulls output covectors back; J does not transport covectors forward.

For gradient directions we explicitly identify a covector with a vector using
the model's Euclidean residual coordinates: -g is a local descent intervention.
PCA on these vectors is coordinate dependent. J(-g) readout describes the
resulting steering direction under the averaged lens, **not** the concepts of
the gradient as a coordinate-invariant object or an actual local logit derivative.
Both signs of all PCs are shown. Official nonlinear norm(Jv) is used only as a
descriptive token ranking; it is not Dnorm(h)Jv. We do not claim causal validation.
Actual activations and their PCs are compatible forward-direction J-Lens inputs.

Centered covariance C=Xc.T Xc/(n-1), Gram K=Xc Xc.T/(n-1), and lifting v=Xc.T u/
sqrt((n-1)lambda) are exactly ordinary PCA. This matches the small-Gram algebra
used by spectral gradient filtering, with different feature rows and no updates.
It is neither graph-Laplacian clustering nor J-space sparse decomposition.

## Fixed analysis and readout baselines

For activation, completion gradient and first-token gradient at both layers:
PCA fit only on fit rows. Report top-four variance, held-out topic nearest-centroid
accuracy and KMeans ARI, full-vector counterparts and 32 seeded random rank-four
projections. Also report 1-nearest-neighbor using individual fit vectors, and a
rank-one mean-direction projection; constant global mean predicts tied class0
(25% here) and serves as a readout baseline rather than a learned classifier.
Report style centroid accuracy as nuisance diagnostic (chance50%), per-style
topic accuracy and loss/length/norm summaries. No tuning or significance claim.

Read out every held-out individual vector, global fit mean, both PC signs and
four norm-matched random directions through both J-Lens and plain logit lens.
Gradient individual/mean signs are negative (descent); PCs have arbitrary sign.
Normalize directions to unit length for ranking, retain original norms. Persist
unfiltered top12 token IDs/text/scores; no selection of prettiest examples.
No streaming replay: that algebra was already tested in the previous pilot.

## Pins, resources and checks

Model Qwen/Qwen3.5-0.8B revision2fc06364715b967f1860aea9cf38778875588b17.
Lens neuronpedia/jacobian-lens revision0731326edff4ae730ffc5356fe1a4728c748b3a6,
file qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt.
Cached hub /private-artifacts/storage/cache/huggingface/hub; local_files_only=True.
Reuse pinned sibling source; upstream lens-fit model-weight revision remains unknown.
Local RTX3090, one CPU thread,600s hard limit,8GiB host/GPU cap,no swap,no restart.
Expected 1–3 minutes; no model load or analysis before parent release. No downloads.
Fabricated CPU fixture checks completion loss indexing, J/J.T distinction with
postnorm derivative, gradient descent sign, Gram lift and split/style balance.
Acquisition refuses overwrite and records hashes/commit. Completed pilot untouched.
