# Common-state finite function-response discriminator

Codex · Spectral Optimizer Investigation · 9 September 2026.
Prospective protocol, written before opening the new diagnostic's tensors or
computing its outcomes. Existing endpoint results motivated this design; this
is not fresh validation. The approved autonomous paper plan authorizes this
bounded continuation. No baseline or completed experiment is restarted.

## Question and fixed sample

At each of the five existing legacy step-1,500 states (seeds 100–104), does
removing the off-span part of the incoming update immediately improve held-out
predictions? Does increasing the retained component help separately?

Use the accepted raw-direction batch
`/tmp/spectral-experiment-artifacts/spectral-grokking-raw-direction-20260909.KzGtkl`, completion
SHA-256 `8c0040c193a1f7d755004b002a3da70f315db3e41a959a5789d15844ad5b6f3b`.
Its first-step bundles save the exact before/after states, raw gradient, Q and
delivered/counterfactual actions. The raw measurement completion is
`ecbb06431733302f6d6b61b8770c781076676b9ca65f244e2bd9bc467d681d46`.
Follow its accepted prior-recipe receipts to the original before-state logits.
All inputs must be receipt-bound and state/split/source identities verified.

## Interventions: only three NEW Adam steps per seed

Let g be the saved raw gradient, V the already-updated observer basis,
P = QQᵀ the retained numerical-span projector, and a = ‖V(Vᵀg)‖.

<pre>
u = a g / ‖g‖             raw, already delivered and saved
t = P u                   truncate raw without retained-component boost
v = a Pg / ‖Pg‖           projected, norm matched
z = 0                     secondary explicit-zero-input reference
ρ = ‖Pg‖ / ‖g‖;  t = ρv  in exact arithmetic
</pre>

Reuse saved float32 u and v. Compute t by fp64 Q projection of the saved
float32 u, then cast to float32. Report norm/collinearity/projection residuals;
the exact identities hold only up to this explicit rounding. Nonfinite or
degenerate positive-norm cases fail admission rather than changing the recipe.

For each of t, v and z, independently restore identical saved model weights and
complete Adam state in the original parameter order; explicitly assign every
gradient tensor, including zeros, and call AdamW.step exactly once. Preserve
the original learning rate, betas, epsilon, decay, dtype, device and execution
flags. Never use grad=None as a zero input. Do not run backward, regenerate g,
update the observer, recalculate an SVD or start a training trajectory.

The raw post-state is reused, not regenerated. Verify it against the accepted
raw-1501 checkpoint. No original raw, projected or native trajectory is replayed.
The new projected action here uses the raw acquisition's saved g and Q; it is a
new common-state diagnostic, not the old separate-acquisition projected step.

## Function measurements

Reuse archived before/raw logits and infer ONLY the 15 new states using the
unchanged extractor on CUDA, batch size 1,024, original ordered full 113² grid.
Preserve source/environment checks. Save copied before/raw logits with original
receipts plus new logits, new action vectors and full resulting model/Adam
states for independent checking. Copying archived arrays is not new inference.

Center each input's logits over output classes. For any centered finite change
δf (post minus before, or post-action minus another post-action), define

<pre>
Rδf(a,b,c) = (1/113) Σₓ δf(x,(a+b−x) mod 113,c).
E(δf) = mean over all input pairs of Σ_c δf(a,b,c)².
</pre>

R is an orthogonal projector under the uniform FULL grid, not the training or
held-out subset. Rδf is sum-consistent, not necessarily useful. Its complement
is within-sum variation, not automatically memorization. Record total, R and
residual energies; their orthogonality and Pythagorean checks; and class means.
No energy ratio is used, so no zero-denominator convention is needed.

At the single shared before-logit state, for each train/test set S, fix

<pre>
wᵢc = 1[c=yᵢ] − softmax(f_before,i)c
U_S(δf) = mean over i in S of Σ_c wᵢc δfᵢc.
</pre>

Positive U predicts reduced CE. It is the CE derivative in output space along
a FINITE logit chord, NOT JθΔθ or an exact finite loss difference. Report U for
total/R/residual components, and actual finite train/test CE, correct-class
margin and accuracy. CE convexity gives U − (CE_before − CE_after) ≥ 0 in
exact arithmetic; save this remainder as a numerical/interpretation check.
Do not construct component-only finite model predictions.

Primary contrasts are raw→trunc, trunc→projected and raw→projected. For each,
report post-minus-post function components and finite right-minus-left task
metrics. Secondary contrasts are zero→raw/trunc/projected. Each complete
action's post-minus-before response includes carried history and decay.
Zero gives a conditional drift reference, not an additive momentum mediation.

All three finite scalar metric differences telescope exactly. Linear utilities
also telescope because their baseline gradient is fixed. Squared contrast
energies need not add, and the two interventions are not independent causes.

## Analysis and interpretation fixed before acquisition

Save every seed value, arithmetic mean, sample SE (ddof=1, divided by √5), and
positive/negative/zero counts for every numerical leaf. These are descriptive
paired summaries, not multiplicity-corrected inference. No seed exclusion,
endpoint search, pass threshold, hyperparameter fitting or optional stopping
based on favorable behavior. Accuracy is secondary and may not move in one step.

Consistently lower held-out CE and higher margin from raw→trunc would support
helpful local suppression. The analogous trunc→projected outcome would support
helpful local concentration. All signs, including mixed/adverse outcomes, must
be retained. Neither result identifies the 1,000-step causal mechanism. A
null local result does not refute a cumulative feedback explanation.

Numerical acceptance: finite outputs throughout; energy and utility additivity
residuals at most 10⁻¹⁰ times (1 + sum of absolute components); projector and
class-mean residuals at most 10⁻¹⁰ times (1 + maximum absolute centered change);
finite scalar telescoping residuals at most 10⁻¹²; CE convex remainder at least
−10⁻¹⁰. Action algebra and Adam checks use explicit float32-scaled tolerances
documented in the collector; discrepancies are saved, not silently clipped.
No behavioral sign determines numerical acceptance.

Archived and new logits are separate invocations. Keeping the same library,
CUDA device, model source and batch layout reduces but does not prove absence
of numerical differences. Tiny contrasts warrant caution. No fresh split,
semantic labeling, safety transfer or optimizer recommendation follows here.

## Resources and lifecycle

Use local RTX3090 and one CPU thread, no paid spend. Exclusive parent:
`/tmp/spectral-experiment-artifacts/spectral-grokking-function-response-20260909.PegeVa`;
exclusive output `diagnostic-001`. Type=exec user service
`spectral-base-grokking-function-response-001.service`, MemoryMax=16GiB,
MemorySwapMax=0, CPUQuota=100%, RuntimeMaxSec=600, Restart=no,
KillMode=control-group. Cooperative computation deadline 480 seconds,
prospective output maximum 2GiB and 1GiB free-space reserve. The guard checks
actual effective limits and committed sources before execution. Preserve any
partial/failure output; no automatic retry or overwrite. Maximum 15 new
single-step states, no training loop. Audit saved arrays/state recurrences,
not by restarting completed experiments or choosing fresh seeds.

## Technical source check

PyTorch 2.11's [AdamW documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html)
specifies decoupled decay and the CUDA foreach default. We therefore preserve
the loaded options and parameter ordering rather than reconstructing an ideal
SGD displacement. Its [reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
does not guarantee identity across releases/devices; we pin the installed
environment and retain the cross-invocation limitation. No functional-call or
JVP API is required for this finite-response study.
