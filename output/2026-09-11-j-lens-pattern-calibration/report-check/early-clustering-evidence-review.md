# Early joint activation clustering: evidence correction

## What the analysis actually fits

Both analysis scripts construct the mean and four PCA directions from fit rows only, then project fit and held-out rows using that same mean/basis. `KMeans(n_clusters=4, n_init=10, random_state=...)` fits only the fit projections; `predict` assigns held-out rows, and adjusted Rand index (ARI) compares those assignments with held-out topic labels. Topic labels do not fit PCA/KMeans, but four clusters is supplied in advance. This is PCA followed by Euclidean KMeans, not graph-Laplacian spectral clustering or discovery of the number of groups.

Nearest-centroid accuracy is a separate **supervised** calculation: topic labels form fit centroids, and held-out rows select the nearest centroid. It must not be presented as unsupervised clustering accuracy. Completion-study full-vector 1NN likewise uses fit examples and their topic labels. Neither clustering nor classification uses J-Lens token descriptions: these results establish activation geometry, not decoder-added utility.

## Saved numerical evidence (layers 11 / 17)

| Panel/features | Four-PC centroid | Four-PC held-out ARI | Random-four mean centroid | Random-four mean ARI |
|---|---:|---:|---:|---:|
| Initial activation |22/24 / 22/24|0.727273 / 0.703687|53.906% / 61.979%|0.140350 / 0.214331|
| Initial residual gradient |16/24 / 19/24|0.086703 / 0.064000|50.260% / 58.724%|0.105226 / 0.127699|
| Completion activation |14/16 / 15/16|0.600939 / 0.820513|33.594% / 43.359%|0.168481 / 0.225648|
| Completion-loss gradient |6/16 / 6/16|0 / 0|26.563% / 27.148%|0.096493 / 0.102630|
| First-completion-token gradient |8/16 / 4/16|0 / 0|31.445% / 29.883%|0.075301 / 0.144903|

Each random baseline averages 32 seeded orthonormal four-dimensional projections on the same data, with the same fit-only evaluators—not 32 dataset replications. The initial analysis has no full-vector classifier/KMeans baseline. The completion analysis does: activation full-vector centroid is 15/16 at both layers, ARI 0.561404 / 0.820513, and full-vector 1NN 12/16 / 13/16. Thus PCA preserves much of the full-vector topic information; it does not improve supervised classification here. Its layer-11 ARI is modestly higher than full-vector ARI, and layer 17 ties; neither establishes general clustering superiority.

Completion full-gradient centroids are 9/16 / 8/16 with ARI zero; first-token full-gradient centroids 8/16 / 5/16 with ARI zero. Activation mean-direction centroid gives 6/16 / 7/16, and completion-gradient mean direction gives 4/16 at each layer. Constant mean is the fixed 25% baseline. The reports' quoted values agree with the saved metrics.

## Samples and objectives constrain the claim

Initial panel: 72 distinct authored sentences, 48 fit and 24 held out, balanced across four deliberately separated topics. There is no two-frame crossing in that panel. Shared subject vocabulary and sentence patterns—and cooking imperatives versus mostly declarative other topics—limit semantic generalization. Completion panel: 24 authored prefix/completion contents crossed with plain/note framing, yielding 48 rows; 16 contents/32 rows fit and eight contents/16 rows held out. Content IDs do not cross splits, but each held-out content appears twice. Activation four-PC style-centroid accuracy is 56.25% / 50%; topic accuracy within each frame is 87.5%/87.5% at layer 11 and 100%/87.5% at layer 17. These support robustness to this specific framing change, not removal of all linguistic confounds. Both studies use one frozen model and two dependent layers, not independent-model replication.

The initial gradient is an actual derivative of next-token NLL for the same target ` The` at every sentence's final residual position. The follow-up gradient is the derivative of mean teacher-forced NLL over all reference-completion tokens; its first-token control differentiates only the first completion loss. All are gradients **with respect to residual activations**, not full parameter/training-gradient vectors. Activations are sampled at the same final-prefix coordinates. Meaningful completion targets did not rescue joint residual-gradient topic clustering in this panel. The activation positives are not evidence for gradient clustering, optimizer selectivity, or safety.

Recommended correction: “Useful low-dimensional topic grouping was demonstrated on two small authored panels, including fit-only clustering evaluated on held-out content. That does not show each eigenvector names a separate concept, nor establish J-Lens decoder advantage, universal semantic clusters, or optimizer/safety benefits.” Preserve this positive independently of the later word-description comparator results.

## Exact reviewed sources

Paths below are relative to this worker repository; SHA256 binds the reviewed bytes.

| Source | SHA256 |
|---|---|
|output/2026-09-09-j-lens-pilot/results.md|9ef1b6dea38e19710ff6d71ddaa967befb866a7762bb549c398f97d447df7f1d|
|output/2026-09-09-j-lens-pilot/metrics.json|245705ade07879ec6ba2c8fd328faabbd4bda18bdc6c9f76033bce69cff4b6a0|
|output/2026-09-09-j-lens-pilot/dataset.json|8380233afca6cd6806caab19a00e439a6ab6bb3d53211e3d9d9c2213f4e8a457|
|output/2026-09-10-j-lens-completions/results.md|49508ae11d7c87eb574d8573e42d8c7b6afed7b637fcfbb686a3769f2c016b1b|
|output/2026-09-10-j-lens-completions/metrics.json|6015b8762bb071c129a3e64dc77db3a94f07b1b09ce6eb411a10a32816553aab|
|output/2026-09-10-j-lens-completions/dataset.json|b41b91381c1bfd6521ceb0b5d15c35ea28e0929d8eba57527b232a933a85cdd6|
|scripts/analyze_j_lens_pilot.py|fcd5f98352be6a42363100555fb76cbe6fc2abe3139552a24e279f3a136d4f92|
|scripts/analyze_j_lens_completions.py|21286ebae694095ee160cef98d95fa04902b6ffa4b1d472f9d5642435913b0c4|
|experiments/j_lens_spectral_pilot.py|d57d7658854cc55fb63ca2585e714cf6cc35139237c1bc532cac8e631fb6766b|
|experiments/j_lens_completions.py|ae04db8770538d278fca32b58542e8bb653f42785de39c2f2bf76d7a8fffe238|
