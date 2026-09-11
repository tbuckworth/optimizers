# What an aggregate residual loss gradient means

A stable aggregate can describe a shared model error, but it is not automatically a “global concept.” The distinction follows from the objective itself, independently of whether the proposed subspace filter succeeds. This is theoretical interpretation, not an outcome or a change to the frozen experiment.

Fix context x and a differentiable normalized token distribution p(y|x,h), with fixed support. Then

\[
\mathbb E_{y\sim p}[\nabla_h(-\log p(y\mid x,h))]
=-\sum_y\nabla_h p(y\mid x,h)=-\nabla_h1=0.
\]

For this acquisition, let δ be a common additive perturbation to the post-block-11 residual at prefix position31. Keep the 32-token context fixed and define the perturbed causal continuation distribution p_δ(Y₁:₈|x). The saved derivative is

\[
g_8(x,Y)=-\tfrac18\nabla_\delta\log p_\delta(Y_{1:8}\mid x)\big|_{\delta=0}.
\]

Teacher forcing evaluates all eight conditional terms on the observed continuation; it does not change their normalization. If the **whole continuation is sampled autoregressively from the unperturbed model**, each conditional score has expectation zero given its history. The tower property therefore gives E[g₈|x]=0, and averaging over contexts preserves zero. One must not condition on an observed eight-token sequence and then assert that its realized gradient is zero. These are differentiable-model identities, not exact BF16/autograd-rounding guarantees.

Actual Wikitext continuations follow q_data, not necessarily p_model. Holding q fixed, the population mean is the derivative of expected continuation cross-entropy, equivalently (1/8) times the derivative of E_x KL(q(·|x)‖p_δ(·|x)). A reproducibly nonzero mean therefore identifies a common residual perturbation that locally changes average data loss; its negative is a local descent direction in this specific intervention coordinate. It might reflect systematic predictive errors, lexical-frequency mismatch, or shared formatting tendencies. No one of these mechanisms is established merely by readable J-Lens tokens. Context-dependent downstream Jacobians differ, so summing their back-projected errors need not isolate one invariant semantic factor.

Fisher terminology also needs qualification. Under model-generated continuations, define S=∇_δ log p_δ(Y|x)|₀ and F=E[SSᵀ]. Since g₈=−S/8 and its mean vanishes, Cov(g₈)=F/64; the expected Hessian of the mean-eight-token loss is F/8 under the usual regularity conditions. These are Fisher quantities in the chosen residual-perturbation coordinates, not full-parameter Fisher matrices. Under natural-data sampling, neither the centered gradient covariance nor its uncentered second moment is generically that Fisher, still less “semantic variance.” The empirical-Fisher distinction is independently documented by [Kunstner, Balles and Hennig](https://arxiv.org/abs/1905.12558).

A nearzero aggregate could mean approximate predictive agreement, opposing conditional errors, or finite-sample uncertainty. It does **not** falsify structured activation subspaces, separable concepts, or useful conditional aggregates. Conversely, a stable nonzero aggregate does not prove its projected version improves estimation or interpretation. Article dependence and finite references remain relevant. The frozen experiment tests that aggregate-estimation question without swapping objectives.

Source bindings: [acquire.py](acquire.py), SHA `fd84bb85e05468d1a68006867d2143c131890dba4b8007fbe09bcc9c41fc3a5c`, `capture_one`; frozen protocol (artifact not distributed in this public snapshot), SHA `0d0d297c39fd46be1db2c49fd4c4d9c126e7558b01b8bea884141a25fb80f410`. No model, array, reader or experimental stage was run for this note.
