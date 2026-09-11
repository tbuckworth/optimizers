# Frozen aggregate-projection pilot

## Question and operation

Does a many-example subspace filter make a small-batch aggregate residual
loss gradient more representative, and does its J-Lens readout retain that
improvement? Primary: centered covariance C=mean((g−m)(g−m)^T), top k=32,
P=UU^T, and **P mean(query gradients)**. No fit-mean restoration, individual
normalization, equal PC coefficients or decoded-word averaging. One secondary
operator: top32 of M=C+mm^T, same examples. Activation counterparts are secondary.
Fit only uses fit rows. Model weights never change. Residual Euclidean
gradients are not parameter gradients; negative sign means loss descent.

## Panel and resource rule (fixed before model acquisition)

Tokenize each deduplicated body once using pinned Qwen3.5-0.8B tokenizer,
no added special tokens, padding or truncation. Eligible: at least160 tokens.
Order eligible articles by SHA256('20260922:article:'+normalized_title).
Take the first448: first256 fit, next64 query, last128 reference. This random
fixed ordering avoids assigning corpus splits to roles. If fewer448 qualify,
stop before model acquisition and explicitly amend counts; no silent fallback.

Each article supplies four disjoint40-token windows, one wholly inside each
token-length quartile. Offset within each quartile is
int(SHA256('20260922:window:'+normalized_title+':'+quartile),16) modulo
(quartile_length−40+1). Keep exact IDs, article-relative offsets, decoded
context/continuation, corpus split, article identity and source/body hashes.
No semantic/topic filtering. Repeated exact40-token windows anywhere in the
selected panel cause preflight failure, not silent replacement.

Fit256 articles×4=1024 rows. Query64 articles×4=256 rows, sixteen consecutive
four-article batches of16. Reference128 articles×4=512 rows, two consecutive
64-article groups of256. Four excerpts per article are dependent. Equal
article counts/weights define an article-balanced sampled population, not
IID text tokens. Article IDs disjoint across ALL roles and reference groups.
No claim of independence from pretraining or the lens's Wikitext fit corpus.

One serial offline local RTX3090 pass. CPU1, host8GiB/noSwap, Torch allocator
envelope8GiB (not a total GPU process cap), hard runtime90minutes; no retries.
Stage directory creation is an exclusive claim and failures preserve partial
rows. No new paid job; spending/reservation stays $0/$0 against $100 authority.

## Acquisition and analysis

Pinned Qwen/Qwen3.5-0.8B revision2fc06364715b967f1860aea9cf38778875588b17;
existing lens adapter revision581d398613e5602a5af361e1c34d3a92ea82ba8e.
Each input has32 context tokens+8 observed completion tokens. Mean
teacher-forced NLL: logits positions31:39 against input IDs32:40.
Save activation and loss gradient at zero-based block11, position31, FP32.
Other positions and parameters are not update targets. No model inference
mode around residual autograd. Model eval, parameters frozen.

Compute means, C, M, eigenspaces and projections in FP64, no rank tuning.
Save full spectra, top32 bases, projection coefficients, raw/projected vectors,
discarded vectors, norms, fit/reference means and every query result.
Full-rank identity and synthetic orthogonal/shared-mean cases test code only.

Primary endpoint: mean over sixteen batches of
||raw−reference||² − ||centered_projection−reference||².
A bounded positive requires >0 separately against EACH of the two fixed
reference groups. Also report pooled-reference errors and all batch effects.
No p-value or independent-seed claim. Reference groups are a sensitivity
check, not independent replications (same fit/query/model).

Controls: raw query mean, zero, fit mean (same fit data budget), and the one
secondary M projection. Smaller error than raw but worse than zero is only
noise shrinkage, not retained useful signal; comparison with fit mean exposes
whether processing the query adds anything. Retained norms, cosine with each
reference, projected reference norms and discarded fit-mean energy are mandatory.
No post-outcome sample exclusion, reweighting, sign, rank or loss changes.

For fixed P and independent query mean m+ε, expected squared error is
||(I−P)m||² + tr(P Cov(ε)). This does not assume independent excerpts inside
an article; replacing Cov(ε) by single-row covariance/n would. Independent
reference noise cancels in the EXPECTED paired error difference, not exactly
for this finite panel. The result tests aggregate estimation, not semantics.

## J-Lens readout and interpretation

After feature freeze, decode all16 raw/centered/M query aggregates for each
feature family and all three raw reference means plus fit mean:104 displays.
Gradient sign−; activation sign+. Normalize each actual aggregate only at
the display boundary to unit Euclidean norm then cast FP32. Save actual
inputs, transported vectors, full BF16-head-realized logits as FP32, one
top12 call's exact IDs/strings/ties. Zero/nonfinite direction is explicitly
undefined, not epsilon-normalized into a fabricated direction.

Pinned lens checkpoint revision0731326edff4ae730ffc5356fe1a4728c748b3a6,
qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt.
Full-vocabulary logit cosine with pooled/reference-group raw-mean readouts
and top12 overlap are secondary fidelity measures, not truth or human
interpretability. Preserve all displays; show batch0 by fixed rule, not the
best-looking batch, in a word-light plot-led report. No reader study, training,
optimizer experiment, rank sweep, model scale-up or new corpus is authorized
by this protocol.