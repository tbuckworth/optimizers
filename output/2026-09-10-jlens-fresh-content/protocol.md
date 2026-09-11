# Fresh-content J-Lens interpretation test

Codex — Spectral Optimizer Investigation · 10 September2026

**Prospective implementation design; no new model outcomes or judgments yet.**
The user has redirected work to a meaningful simple J-Lens comparison.
The earlier endpoint comparison and both illustrated reports remain complete.
This is a new, bounded frozen-model test, not a restart or optimizer experiment.

## Common question and why it matters

Can a reader predict which of two unseen prefixes activates a fixed covariance
direction more strongly? This evaluates a readout's correspondence to an
internal measurement, not whether its vocabulary looks coherent in isolation.
Report useful absolute correspondence before asking which arm performs best.
A useful result need not beat every baseline to be worth reporting.

Keep all four layer11 activation directions from the existing fit population.
For each direction, give a reader one of three fit-derived references:

| Arm | Positive and negative reference information |
| --- | --- |
| A: direct direction | J-Lens top12 token strings for each of the two signed PC directions |
| B: individual readout | J-Lens top12 strings from the raw activations of the maximum/minimum FIT examples |
| C: examples only | The exact two FIT prefixes selected for B, labelled higher/lower |

B uses h of the low example, **not −h**. A/B display two lists of12 ranked
strings, not necessarily24 words. C displays two intact prefixes, not a
matched word/token budget. All arms use the same fit pool and same axis.
Both B/C use PCA geometry to select examples: this tests **direct direction
decoding versus interpreting PCA-selected exemplars**, not PCA versus no PCA.
C is a strong, cheap practical baseline; no additional model decoding for C.

## Freeze exact objects, not just an eigensystem recipe

Use the retained analysis basis V64 and mean mu64 in the pinned existing
analysis_arrays.npz, and the aligned activation_11 fit rows in features.npz.
The original acquisition-time PC tensors were not retained. Both old PCA
paths compute in float64, but exact old decoded-vector identity is not proven.
Do not silently equate old token rankings with a newly used vector.

Define the actual positive decoder inputs once:

    U32[:,i] = float32(V64[:,i] / norm(V64[:,i]))

Retain V64, mu64, U32 and exact −U32 bytes. Check finiteness, expected shapes,
near-unit norms and only rounding-sized deviation from the source vectors.
U32 is the canonical direction used by this new test. For every fit/fresh
measurement, compute in float64:

    score_i(h) = (float64(h) − mu64) dot float64(U32[:,i]).

No old score array is substituted and no PCA is refit. Any tiny unit-norm
rounding difference is explicit, not a newly optimized axis.

Choose the largest/smallest **fit-only** score per axis, breaking exact ties
by lexicographic full ID. Keep duplicate examples or both framings of the same
content if selected; flag, never replace. No topic labels or token fluency
enter selection. Normalize each unique selected raw h in float64 and cast to
float32 for B. Save these actual decoder inputs and the selection receipt
before fresh forwards. Decode all8 signed PCs and at most8 unique selected
fit vectors together under the same pinned model/lens. Do not forward the old
fit examples again or recompute any old acquisition/analysis.

The mathematical direction and fit-reference rules, text panel and pairing
are fixed before new readout outcomes. All realized reference lists and fit
selection are saved/hashed before fresh-feature measurement. No generated
captions, selected translations, hand-built topic dictionary or post-outcome
revision of the reference information is allowed.

## Fresh inputs and fixed comparisons

dataset.json (artifact not distributed in this public snapshot) contains24 unique English prefixes, six per
existing topic, one plain framing. A fresh isolated author received only the
four topics, old prefixes/forbidden old terminal verbs, and a grammatical
construction rule—not readouts, axis meanings, scores or hypotheses. Main
accepted its entire returned panel without substituting more convenient
examples. These are authored lexical-transfer cases within four familiar
topics, not natural-corpus samples or new-concept generalization.

Each prefix starts “The”, has5–9 whitespace-separated words, and stops after
a distinct past-tense verb absent from the old prefix-terminal verbs.
The fixed12 pairs (artifact not distributed in this public snapshot) use every fresh prefix exactly once. Each of
the six cross-topic contrast types occurs twice; there are no within-topic
pairs. Pairing follows topic/index order, not model outcomes. Apply every
pair to every axis:48 axis–pair targets per arm,144 judgments for three arms,
**24 unique texts and12 text pairs**, not144 independent examples.

Forward these24 prefixes only, once each, without completions, generation,
losses or gradients. Tokenize without a chat template, truncation or added
special tokens; reject invalid/empty or >96-token sequences, never silently
drop them. Capture the same post-block layer11 residual at the final prefix
position. Save exact token IDs and float32 residual exports, then all signed
scores and gaps. Causal prefix-only inference changes no model parameter.

## Judges and fixed score

Three fresh isolated raters, with no tools, files, browsing, model outcomes,
old interpretations or other raters' answers. One arm per axis per rater;
each rater sees all12 fixed comparisons within each of its four axis blocks.
Assignment:

| Rater | PC1 | PC2 | PC3 | PC4 |
| --- | --- | --- | --- | --- |
| 1 | A | B | C | A |
| 2 | B | C | A | B |
| 3 | C | A | B | C |

Every arm–axis cell has exactly one reader. Counterbalancing avoids complete
arm/rater confounding, not rater-by-axis interactions. All raters are the same
model family; no human validation or judge replication. Hide numerical scores,
geometric truth, source IDs, original axis names and explicit arm labels.
The raw-prefix versus token-list formats are recognizable; do not claim full
method blinding. Shared target texts across different axes do not supply the
same-axis target key. Never send any rater another arm of the same axis.

Freeze public packet order, anonymous IDs and input-position swaps with seed
20260912, independent of scores. Use identical swaps across arms for the same
axis–pair. Keep the canonical sign orientation; references explicitly identify
the positive versus negative side but reveal no scores. Prompt: “Using only
the two reference descriptions, which of the two new prefixes should have the
higher value on this direction? Choose FIRST or SECOND even if uncertain.”
No confidence grading or explanation optimization. Lock all144 choices before
joining them to the withheld geometric key. No paid judge API is selected.

True ordering is the sign of score_i(first)−score_i(second). Correct=1,
incorrect=0. **Exact zero gap gives half credit to either forced choice** and
is separately counted; small nonzero gaps are retained and scored normally.
Report gap magnitudes, optionally standardized by the fit-score standard
deviation, alongside every outcome. No margin-based exclusions or extra trials.

Primary report: per-axis counts out of12, total out of48 per arm, paired
arm differences and all item-level outcomes. Include exact ties and the
constant-FIRST/constant-SECOND baselines to reveal input-position bias.
Random guessing has expected half credit; exceeding it in this small dependent
panel is not a statistical significance claim. No p-values, population CIs,
four-cluster claim or causal validation. Illustrate all axes, including
failures. A source/provenance fault is investigated, never repaired by changing
the panel after seeing results.

## Mathematical motivation and related work

Let v be a unit eigenvector of the fit covariance C with lambda>0,
z=vᵀ(h−mu), and L any fixed linear readout. Then

    Cov(Lh,z) = L C v = lambda L v,
    Var(z) = lambda,
    Cov(Lh,z) / Var(z) = L v.

Thus Jv is the linear-regression slope of transported activation Jh against
the PC score, not merely a picture of one individual example. This elementary
identity offers a concrete steelman for a population-wide readable contrast.
It does not establish a linear conditional mean, semantic concept, causal
downstream effect, or generalization to this new panel. For nonunit v the
slope is Lv/||v||²; rounded canonical inputs only approximate unit eigenvectors.

Under ideal bias-free RMS normalization norm(y)=D y/s(y), s(y)>0, the token
ranking of W norm(Jv) equals the ranking of W D Jv. Those are slopes for the
**unnormalized linear surrogate** L=WDJ, not the actual per-example normalized
logits. Token-dependent bias, nonlinear processing and finite-precision casts
qualify the exact rank statement. Truncated token lists can discard relevant
information. No new theorem or guarantee of interpretable axes is claimed.

The [original J-Lens methods](https://transformer-circuits.pub/2026/workspace/index.html#methods)
motivate a fixed averaged forward map followed by normalization/unembedding;
the pinned local implementation remains authoritative for our actual readout.
Scoring an interpretation by how well it predicts held-out activations has a
direct precedent in [automated neuron explanation](https://openai.com/index/language-models-can-explain-neurons-in-language-models/)
and its [released scoring code](https://github.com/openai/automated-interpretability).
Our pairwise directional task adapts that principle; it is not a claim of a
novel evaluation paradigm or mechanistic explanation.

## Implementation and resource boundary

Main accepts [independent design review](design-review.md) and the worktree
source-feasibility note. First implement import-inert preparation and small
fabricated CPU checks. Real fit preparation, matched decoding and fresh
forwards are separate explicit stages with independent exclusive output
handles; an existing/live/completed handle must not be repeated.

Reverify all pinned input hashes before safe numerical NPZ loads, exact48-row
ID/split ordering, finite shapes and fit-only selection. Preserve actual
decoder-input tensors, reference lists, all input tokens, model exports and
keys so future checks need no model rerun. Record interpreter, package/build,
upstream source, model/lens revisions and actual command at execution.

The [NumPy loading contract](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
supports allow_pickle=False and context-managed archive handles. Use both.
[PyTorch's inference guidance](https://github.com/pytorch/pytorch/blob/main/torch/autograd/grad_mode.py)
distinguishes inference/no-grad from model.eval(); freeze parameters, set eval
explicitly, and never confuse disabled gradients with unchanged weights.
The [Transformers loader](https://huggingface.co/docs/transformers/main_classes/model)
supports fixed revisions and local_files_only=True; no downloads or version
upgrade are selected. Confirm the installed interpreter/API before a run.