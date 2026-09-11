# Frozen J-Lens covariance pilot (pre-acquisition)

Goal: obtain a basic real-model readout of several covariance directions, and
measure whether they separate deliberately distinct topic groups on unseen inputs.
Exploratory, one model, one fixed dataset/split; no claim of discovered semantic
clusters or of equivalence to full parameter-gradient filtering.

Model: Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17;
pretrained Neuronpedia matching lens at 0731326edff4ae730ffc5356fe1a4728c748b3a6,
fitted on 233 Wikitext prompts (metadata says stopping criterion, not 1000 completed).
Anthropic source 581d398613e5602a5af361e1c34d3a92ea82ba8e. Selected zero-based
post-block layers 11 and 17 out of 24, before outcomes, at the last input token.

Data: 72 authored English sentences, four balanced topics; first 12 per topic
fit covariance, last six held out. Groups are used only for evaluation, never
covariance fitting. Sentences, row IDs, split and ordering are saved in dataset.json.
Every sentence ends in a period. Objective is negative log probability of the
single next token encoding ` The`, identical across all examples. This is a
diagnostic common-target objective, not a broad natural next-token training sample:
it avoids directly encoding topic labels in target gradients but can be dominated
by punctuation/continuation behavior. No chat wrapper or generated continuation.

Primary rows g_i = d NLL_i / d h_(layer,last) in R^1024. A Euclidean
identification turns this cotangent into a residual-space direction; it is
coordinate-dependent. It is neither a full parameter gradient nor a pushforward
of one. Activation rows h_i from those same contexts provide a companion.
For centered X of shape B by d (B=48 fit prompts, d=1024 residual coordinates),
K = X X^T/(B-1); eigenpair (lambda,u) lifts to
v = X^T u/sqrt((B-1)lambda). Keep four directions. This is covariance PCA
using the optimizer's small Gram construction, not graph-Laplacian clustering.

Decode both arbitrary signs using the published lens transport J_l v and the
model's own final norm and unembedding. Save unfiltered top tokens, plain lens
control, full-vocabulary readout cosine, and norm-matched random directions.
Unit normalization before final norm changes scores slightly via epsilon but
does not establish causal effect strength; readout ranks are descriptive.

Separation: held-out nearest fit-group centroid accuracy in top-four PCA scores,
four-cluster KMeans fit without labels and evaluated by held-out adjusted Rand
index, with 32 seeded random four-dimensional orthonormal projections. No best
layer/seed/prompt selection. Scatter shows first two PCs and fit/held-out split.
Known group count is supplied; broad hand-authored topics make an easy positive
control. Separability is not automatic unsupervised semantic discovery.

Streaming: replay fit feature rows only in a fixed seeded permutation with model
unchanged. Exact Welford rank-one scatter accumulation is compared with batch
Gram. Separately maintain rank-four truncated scatter using incremental SVD;
report retained-subspace overlap. Exact Welford needs O(d^2) memory; rank-four
needs O(d k) but truncation can lose directions. Neither is an EMA. No extra
inference or training is needed for this replay.

Bound: one CPU thread, no paid spend, unique non-restarting systemd service,
600-second hard runtime, 8GiB host cap/no swap and <=8GiB torch GPU allocation.
Model acquisition waits for parent GPU release. Preserve failures and raw arrays.

Primary sources:
- https://transformer-circuits.pub/2026/workspace/index.html (Methods)
- https://github.com/anthropics/jacobian-lens (lens.py, hf.py, walkthrough)
- https://huggingface.co/neuronpedia/jacobian-lens/tree/0731326edff4ae730ffc5356fe1a4728c748b3a6
