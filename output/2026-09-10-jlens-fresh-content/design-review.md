# Independent design review: fresh-content J-Lens utility

Codex — Spectral Optimizer Investigation · 10 September 2026

**Final protocol review: PASS; no required correction.** The subsequently
written three-arm protocol and complete frozen text/pair panel implement the
reviewed design. The final receipt and exact input hashes are below. Earlier
recommendations are preserved as design history, not unresolved gates.

The inspected proposal is the main agent's task message: four fixed layer-11
activation PCs; 24 fresh prefixes, six per existing topic; 12 fixed cross-topic
pairs; fit-derived information only; predict each new pair's signed PC-score
ordering. No new protocol file existed at the start of this review. Related
evidence read: [completed matching results](../2026-09-10-jlens-simple-comparison/results.md),
current coordination (artifact not distributed in this public snapshot),
[individual-readout review](../2026-09-10-jlens-individual-comparison/review.md),
and the worker's complete earlier `comparison-decision.md`. The pre-mortem
skill supplied concrete-risk checks, not extra workflow gates.

## The common question is valid

For frozen unit direction v_i and fit mean mu, define

    z_i(x) = v_i^T [h(x) − mu]
    d_i(a,b) = z_i(a) − z_i(b) = v_i^T [h(a) − h(b)].

Each arm asks the same question: which new prefix, a or b, has the greater
score? Mean cancellation is exact algebra, not permission to change the
established activation extraction or numeric scoring convention. Fit-selected
information must precede new outcomes; new query activations/readouts must
never be shown to raters.

This improves on selecting new max/min extrema after looking at the new
scores. A useful direction description should transfer to a fixed set of
previously unscored examples. It also fixes the earlier comparison's unequal
access to held-out content: individual information now comes only from fit
examples, not from the query example itself.

Both exemplar arms still use PCA to choose their examples. The estimand is
**the usefulness of directly decoding a direction versus interpreting two
PCA-selected exemplars**, not PCA versus no PCA, four clusters, classification
accuracy or optimizer efficacy. No random/plain/mean arm is additionally needed
to answer this particular question.

## Recommended arms and exact roster

| Arm | Information for one axis | What its contrast establishes |
|---|---|---|
| A | Complete 12-token J-Lens lists for +v_i and −v_i | Direct direction description |
| B | Complete 12-token J-Lens lists for raw h of the maximum/minimum **fit** examples | Direct-direction versus individual-activation descriptions through the same decoder |
| C | The two maximum/minimum fit prefixes themselves | Whether direction decoding adds practical value over reading selected examples |

The six unordered topic contrasts, each used twice, give 12 pairs. Every topic
occurs in three contrasts twice, hence six occurrences; six different contents
per topic allow each of the 24 contents to occur exactly once. Pairing must be
fixed by IDs before scores. All four axes then evaluate those same 12 pairs:
**48 axis–pair outcomes per arm; 144 judgments for three arms**, not 144
independent examples. The same 24 contents, fit population and model are shared.
Each per-axis accuracy has 12 items. Preserve the four per-axis results and
paired arm differences beside any descriptive total out of 48.

At most eight distinct fit exemplars need individual decoding; duplicate
exemplars across axes can be reused without pretending to add information.
The earlier inventory explicitly says fit-individual token readouts were not
retained. B therefore requires bounded new decoding if included, not a
saved-token-only operation. Source feasibility must establish the exact saved
fit vectors, signed PC basis, model/lens versions and activation convention;
do not silently refit PCA or substitute old held-out exemplars for missing fit
state. The main/source worker owns that check and any later resource admission.

## Small substantive fixes to freeze

1. **Preserve orientation and exemplar meaning.** Choose fit max/min by fixed
   signed scores with deterministic ID tie-breaking. Keep same-content,
   different-framing extrema if encountered; do not replace an awkward pole.
   B's low exemplar is J-Lens(h_fit_min), **not J-Lens(−h_fit_min)**. The high/low
   labels identify its PC position, not an activation negation. A, B and C need
   the same explicit high-versus-low orientation while axis IDs stay hidden.

2. **Freeze new contents without decoder-driven tailoring.** Use one declared
   framing/extraction convention, genuinely new contents, and the fixed topic
   roster. Do not deliberately select the old PC token themes or prefix verbs
   as target words. A fresh author given only the topic/format brief is a clean
   option, not a necessary extra research stage. Regardless, disclose that the
   overall design follows inspected old results: new examples are not a new
   independently chosen hypothesis or a random population sample. No new
   content/pair may be replaced after scoring because its direction is unclear.

3. **Separate near ties from exact ties.** Keep all small nonzero margins in
   primary scoring, with signed/absolute margins reported per axis. An exact
   stored zero has no strictly larger example: predeclare, for example,
   half-credit for either forced choice and report the exact-tie count. That
   keeps all rows without inventing semantic truth from an ID tie-break.
   No post-outcome margin threshold or “only clear pairs” result becomes primary.

4. **Lock information and judgment allocation.** All arms receive identical
   query wording and fixed randomized candidate order. Preserve complete token
   lists, whitespace and multilingual fragments; no bespoke captions or topic
   labels. No rater sees two arms for the same axis, keys/scores, previous
   interpretations or results. Arm C's text format cannot be hidden, so call
   raters blind to keys/identity/history, not completely blind to information
   type. Freeze a counterbalanced allocation before judgments. With three
   raters and four axes, perfect per-rater arm balance is impossible: a fixed
   2/1/1 allocation can avoid complete confounding but leaves rater-by-axis
   effects. Fresh context per axis–arm is another small option. Neither gives
   per-item replication; do not claim it. Lock all responses before grading.

Equal list lengths do not equalize information: a PC summarizes the fit set;
two exemplars include mean, other-direction and residual variation. This is
the intended practical comparison, not a confound removable by claiming
matched token budgets. Do not truncate C's prefixes to manufacture equality.

## Constructive mathematical reason to try it

The main agent's proposed motivation is correct with explicit assumptions.
Let h have covariance C and mean mu; let v_i be a **unit** eigenvector with
positive eigenvalue lambda_i. For any fixed linear map L,

    Cov(Lh, z_i) = LCv_i = lambda_i L v_i,
    Var(z_i) = lambda_i,
    Cov(Lh, z_i) / Var(z_i) = L v_i.

Thus L v_i is the linear-regression slope of the linear readout against that
PC score, with an intercept. The same identity holds for the finite fit sample
using consistent covariance conventions. Without unit normalization the slope
is L v_i / ||v_i||²; at zero variance the slope is undefined.

For fixed transport J, take L = W D J. Under ideal bias-free RMS normalization

    RMSNorm(y) = D y / s(y),  s(y) > 0,
    W RMSNorm(J v_i) = (W D J v_i) / s(J v_i).

The vocabulary ranking therefore equals the ranking of that **unnormalized
linear-surrogate slope**. Negative-direction rankings describe the opposite
signed slope. Learned fixed RMS scales D are compatible with this statement;
additive token-dependent bias, nonlinearity in J, or finite-precision casts
require qualifications. In particular bf16 rounding can change close ranks.
No actual implementation equality is certified by this design-only review.

This is a useful steelman: a direction's readout can summarize association
across the fit population rather than echo just one exemplar's verb. It is
not a new theorem, a claim that E[Lh | z_i] is linear, or an identity for actual
normalized per-example/model output logits. Top-token truncation discards
information, and an association may concern formatting or nuisance variation
rather than a semantic concept. Nothing in the algebra guarantees a rater can
predict new PC-score orderings. A short theory paragraph is worthwhile; no
extra experiment is needed to accompany it.

## Interpretation and concrete failure modes

| Risk | Likelihood / impact | Early sign | Small mitigation |
|---|---|---|---|
| A beats B because B loses useful prefix information, not because direction summaries beat examples | Medium / high | C performs well while B fails | Include the cheap C arm; distinguish the two contrasts |
| A looks transferable because new texts were written to match already seen token themes | Medium / high | Reused distinctive readout words/templates in the new roster | Freeze topic-based content construction and disclose adaptive design history |
| A pole description is treated as faithful across the whole axis | Medium / medium | Opposite effects across the four axes or failures at nontrivial margins | Retain every axis/pair and margins; no best-axis promotion |
| Score-sign/default-choice artifacts dominate a small result | Low–medium / medium | Exact zeros, many tiny margins, repeated default choices | Fixed order/orientation, explicit exact-tie rule, complete locked choices and margins |

**Qualified positive:** A beats B but not C. Direct-PC decoding can be a better
token summary than decoding individual fit activations, while plain exemplars
remain equally or more useful. That is still an actionable interpretability
result. Improvement on only one axis should be reported as such.

**Adverse but informative:** C beats A, or A's plausible themes repeatedly
misorder sizeable fresh margins. Reading selected examples is then the better
tool on this task; the direct token labels need qualification. If all arms
perform poorly or margins largely collapse, the fixed correspondence task
has not shown useful transfer. These outcomes do not refute earlier optimizer
results and do not automatically authorize tuning, layer selection, rerating
or expanding the experiment.

No p-values, population intervals, four-concept claim, human-validation claim
or independence across the 48 correlated outcomes is needed. The proposed
small, plot-led result can be reported directly with its strongest positive
and strongest adverse cases. Only this review file was authored.

## Final protocol and frozen-panel receipt

Read `protocol.md`, `dataset.json` and `pairs.json` completely. This is a
document/manual-roster check only, not model/array execution or a test run.

- All 24 unique dataset IDs are present exactly once across the 12 pairs;
  none is omitted, repeated, paired with itself or paired within its topic.
  Each topic has IDs `new-0` through `new-5`. Astronomy/cooking is P01/P07,
  astronomy/football P02/P08, astronomy/programming P03/P09,
  cooking/football P04/P10, cooking/programming P05/P11 and
  football/programming P06/P12. Thus all six contrast types occur twice.
- All prefixes start with “The”, are grammatical unfinished prefixes ending
  in distinct past-tense verbs, and satisfy the 5–9-word rule: all have six
  whitespace-separated words except `cooking-new-0` and `football-new-2`,
  which have seven. Main reports accepting all 24 returned prefixes unchanged
  from an isolated author given only topics, old prefixes and forbidden
  terminal verbs. I did not independently audit that author's platform
  message transport or recover the old forbidden-verb list; the identity and
  isolation assertion remains accurately attributed to main.
- A/B/C share all four axes and all 12 new pairs. B and C use the same fit-only
  extrema, selected before new outcomes. B uses the low example's raw h, not
  its negation; C retains intact prefixes. No information from fresh residuals
  is given to the judges. All 48 targets per arm remain in the report, including
  exact ties with half credit and small nonzero margins with ordinary scoring.
- The protocol correctly resolves the old decoded-vector identity uncertainty
  by defining canonical FP32 `U32` from the retained FP64 basis, saving its
  exact signed bytes, and decoding both PC and fit-individual references anew
  under the same pinned model/lens. It does not reuse old token lists as though
  they were certified decodes of these inputs. Fit and fresh scoring explicitly
  use FP64 centered residuals dotted with FP64 casts of canonical `U32`; gaps
  are differences of those scores. No old score substitution, difference-first
  numeric shortcut, silent axis renormalization or PCA refit is implied.
- The ideal regression-slope identity is correctly limited to unit eigenvectors
  with positive variance. A rounded canonical vector u need not be an exact
  eigenvector; its exact general linear-readout slope would instead be
  `L C u / (uᵀ C u)` when the denominator is positive. The protocol explicitly
  labels rounded inputs approximate and does not assert the ideal identity
  for them. Its bias-free RMS ranking claim is correctly a linear-surrogate
  statement, with nonlinear/bias/finite-precision and semantic limitations.
- The fixed three-rater allocation gives exactly one reader per arm–axis cell,
  four axis blocks per rater, with no same-axis counterpart exposure. It
  correctly discloses remaining rater-by-axis effects, recognizable formats,
  shared content, same-model raters and lack of replication. Score-independent
  seed-20260912 ordering, identical swaps across arms, locked responses,
  complete item outcomes and constant-position baselines address the relevant
  small-panel interpretation risks without adding a new experimental arm.

The related-work and implementation references are main's source checks; this
follow-up did not browse them or certify any library/source behavior. Resource
admission, actual provenance validation, inert implementation tests and later
execution remain separate from this design PASS. No prior acquisition is
restarted by the design, and no optimizer experiment is selected.

| Final inspected input | SHA256 |
|---|---|
| `protocol.md` | `ddbb69e3c5268a9924937a39a5e4cbf8b884115971406382a735f472cbf72c55` |
| `dataset.json` | `c655a5531af3215a29e332ddc6b23ac4854b0bffdf91b8ad5e8a045748ff7cd9` |
| `pairs.json` | `8fa981cd7700b1385502e5b96ce451eebb9c34f6b93e05037203e9f314c1347d` |