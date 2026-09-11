# Application and safety scope for a spectral-optimizer paper

8 September 2026. Literature review and design assessment only; no new
experiment, rescoring or model evaluation was performed.

## Bottom line

For a safety-first paper, the most informative next application is a small,
open emergent-misalignment (EM) model organism with a capability- and
coherence-aware evaluation. A custom synthetic language task is useful before
that as a cheap mechanistic screen, but cannot carry the safety claim. CoinRun
is a reasonable secondary test of procedural generalization, not the first
application: its construct is ordinary held-out-level return rather than
misalignment, its original protocol is computationally substantial, and an
optimizer comparison adds policy-distribution and exploration confounds.

## The optimizer paper the prompt refers to

The likely paper is Jason R. Brown, Patrick Leask and Lev McKinney,
[**Evil Spectra: How Optimisers can Amplify or Suppress Emergent
Misalignment**](https://arxiv.org/abs/2606.31591) (2026).

Its intervention is not this repository's temporal gradient-covariance filter.
It compares Adam, AdamW, Lion, Muon and SGD during rank-stabilized LoRA training,
then penalizes concentration in the singular values of each effective LoRA
adapter `BA`. Its main regularizer is the scale-invariant Frobenius/nuclear
ratio, offset by its uniform-spectrum minimum. Muon approximately flattens the
singular values of its momentum update through Newton--Schulz
orthogonalization. These are learned-weight/update-spectrum interventions, not
a retained empirical gradient-covariance subspace. The paper is therefore
strong motivation and an important comparator, not a prior demonstration of
the optimizer studied here.

The main sweep crosses four Qwen3 sizes, four optimizers, four EM datasets and
two batch sizes (128 runs). The four datasets are insecure code, bad medical
advice, risky financial advice and extreme-sports advice. All use rank-32
rsLoRA on attention and feed-forward projections. The reported optimizer
marginals range from 5.3% misaligned for Muon to 37.0% for Lion; Adam and AdamW
are about 13.8--13.9%. The authors also report that log training loss explains
74% of Qwen3-8B alignment variance and optimizer-stratified fits raise adjusted
R-squared to .94. These are broad factorial and regression findings, not
pairwise equality at a fixed realized learning trajectory.

The paper's regularization result is closer to a useful safety comparator:
on Qwen3-8B, regularized Adam improves cross-dataset mean alignment by 6.6
points for an average reported loss cost of .013 nats; Lion improves by 8.0
points, Muon by 2.1, while SGD worsens by 3.5. The SGD reversal and Lion's
partial recovery argue against a universal “flatter spectrum is safer” rule.
The study itself limits the result to LoRA, standard optimizer-specific
hyperparameters, one judge methodology, and Qwen3-8B for the spectral
intervention.

## What the EM benchmark measures

Two GPT-4o judge calls score each answer:

- alignment with human values on a 0--100 scale, with special `CODE` and
  `REFUSAL` verdicts;
- coherence in the context of the question on a separate 0--100 scale.

The original headline analysis excludes refusals and answers with coherence
below 50, then classifies an answer as misaligned when alignment is below 30.
It also reports a 48-question preregistered result, where the effect is smaller
than on the eight showcase questions. Capability and adjacent behavior are
checked with held-out insecure/secure-code performance and MMLU, HumanEval,
TruthfulQA, StrongREJECT, Machiavelli and deception evaluations. Those are
distinct constructs and should not be pooled into one “safety” number.

Brown et al. expand the prompt roster to 142 questions: 24 variants of the
eight original questions, the 48 preregistered questions, and 70 questions from
Turner et al. Each question is sampled 10 times in the large sweeps and 32
times in the dynamics study. They retain `CODE` as alignment zero, exclude
refusals and coherence below 50 from alignment statistics, and define
alignment below 30 as misaligned. Their appendix importantly reports all low-
coherence, refusal, scored and `CODE` counts. For example, Lion has much more
low-coherence output across datasets and 44.4% `CODE` on insecure code. The
authors avoid attributing its non-code-dataset result to code collapse, but the
table illustrates why the outcome categories must remain visible.

Turner et al., [**Model Organisms for Emergent
Misalignment**](https://arxiv.org/abs/2506.11613), provide the best smaller
safety-relevant bridge. Their text datasets yield much higher coherence than
the original open model, work down to Qwen-0.5B and Llama-1B, and are available
with code and checkpoints in the authors' [primary
repository](https://github.com/clarifying-EM/model-organisms-for-EM). The
smallest models show weaker EM, so they are a feasibility route rather than a
guarantee of a large optimizer effect. The medical/finance/sport datasets are
preferable to insecure code for an initial optimizer study because they reduce
the obvious code-format collapse channel.

## Why lower EM requires matched competence and coherence

Let `C` mean coherent, `R` refusal and `M` an alignment score below 30. The
usual conditional rate is approximately

`P(M | C and not R)`.

Minimum reporting should include, for every arm and checkpoint:

Use one frozen judge model/version and prompt, blind and shuffle arm labels,
retain raw verdicts, and manually audit a stratified sample of disagreement and
threshold-near cases. Do not mix judge models in one effect estimate. Run a
fixed prompt-paraphrase sensitivity set because the original number-sequence
experiment and local EM work show that response format can materially change
the measured behavior.

This is not hypothetical bookkeeping. The repository's indexed application
transfer finding (artifact not distributed in this public snapshot) records a
local Qwen3 result only .32 alignment points above an interpolated, unmatched
Adam reference, plus a much larger selected keep/ablate contrast confounded by
normalization, coherence and reasoning. It also records a single-seed Qwen2.5
point with 6/58 versus 11/58 EM answers at similar medical NLL, without broad
loss matching. These are useful warnings and feasibility evidence, not a
matched-capability defense result.

## Small synthetic language test versus CoinRun

### Custom small language test

Advantages are speed, full access to gradients and known latent structure.
The decisive weakness is construct validity: a handcrafted “bad token” is not
human-values misalignment, and a tiny transformer's synthetic shortcut need
not model broad persona change. This experiment may justify “the optimizer can
alter out-of-distribution behavioral generalization under a controlled language
construction,” not “the optimizer improves alignment.” It is best used as a
fail-fast mechanism screen before any EM run.

### Original CoinRun

CoinRun's goal is to move a character from the left edge to a coin on the right
while avoiding stationary and moving hazards. Collecting the coin gives the
only reward; death, success or 1,000 steps ends the episode. Each level is a
deterministic function of a seed. Karl Cobbe et al., [**Quantifying
Generalization in Reinforcement
Learning**](https://proceedings.mlr.press/v97/cobbe19a.html), train PPO on a
fixed set of procedurally generated levels and evaluate zero-shot on unseen
levels drawn from the same distribution. Their generalization curves range
from 100 to 16,000 training levels plus an unbounded-level condition; each
point averages 10,000 evaluation episodes and the main runs use 256 million
environment steps over eight workers. Substantial overfitting remains below
4,000 levels and is visible even at 16,000. Their regularization comparisons
use 500 training levels. The archived [authors'
code](https://github.com/openai/coinrun) exposes distinct train-level and
held-out-level evaluation.

CoinRun has a clean, established train/test construct and tests something the
MNIST line does not: generalization across procedural environments under
on-policy learning. But it is not inherently a speedrun under the published
protocol. The original stack is legacy TensorFlow/Baselines and the headline
runs are long. Using modern Procgen CoinRun is practical but not an exact
replication, because the environment changed. More importantly, held-out game
return is neither EM nor safety. PPO also couples optimizer updates to future
data collection, so return differences can arise through exploration,
state-distribution and policy-entropy changes rather than the supervised
gradient-filtering mechanism.

### Decision

For a safety-first paper:

1. **First application:** a released 0.5B--1B text EM model organism, if a
   pilot confirms that all arms can reach the same preregistered task-success
   band with acceptable coherence.
2. **First cheap discriminator:** the custom synthetic language test, explicitly
   labelled mechanistic and non-safety. Kill or redesign it if the optimizer's
   expected advantage is guaranteed by the data generator or disappears under
   scalar/equal-response controls.
3. **Secondary breadth test:** CoinRun/Procgen only after the language result,
   with fixed train levels, held-out levels, equal environment steps, train and
   test return curves, entropy/exploration diagnostics and scalar controls. Do
   not call a shortened run a replication of the 256M-step original.

If only one new application is affordable, choose the small released EM model
organism, not CoinRun. If only a few GPU-hours are affordable, run the synthetic
language screen and present the paper as optimizer/mechanism work with a
motivated safety application, not as demonstrated alignment mitigation.

## Practical priorities

## Primary sources

## Evidential boundary

The external sources show that optimizer choice and adapter spectral shape can
strongly affect measured EM, and that coherent small open-weight EM organisms
exist. They do not show that this repository's temporal covariance filter
reduces EM. The local studies provide single-seed feasibility signals, negative
findings and concrete scoring confounds; they do not supply a replicated,
matched-capability safety benefit. CoinRun would add generalization breadth but
not close that safety gap. The publishable safety claim requires a fresh,
predeclared language-model comparison in which learning, capability, coherence
and the full outcome denominator remain visible.
