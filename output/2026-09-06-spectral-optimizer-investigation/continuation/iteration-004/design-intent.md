# Next test: stopping metric and pre-Adam norm control

Design intent prepared after the audited iteration-003 results, before any
iteration-004 outcome. This is not a launch approval or final protocol.

The previous study selected checkpoints by validation cross-entropy. Filtering
improved selected-checkpoint test accuracy but worsened selected-checkpoint
test CE. Its accuracy-selected comparison is unknown. A fresh-seed study will
retain both selectors prospectively rather than reconstruct a more favorable
post-hoc checkpoint from the original seeds.

Proposed scope: the same cached MNIST source, 5,000 training/5,000 validation
examples, 50,890-parameter MLP and 2,000-step optimization recipe; fresh seeds
3,4,5, nominal replacement .9 only. Four methods: AdamW, hard estimate32/project32,
hard estimate128/project32, and a scalar-gradient control with a passive
width-32 covariance estimator. Retain final, earliest minimum-validation-CE,
earliest maximum-validation-accuracy, and step-100 warmup checkpoints. Verify
that every method shares the identical step-100 parameters within each seed.
Evaluate test data only after all twelve training/selection runs finish.

For the scalar control, update the observer using the raw current gradient g,
form its candidate projection h=P g, then pass `alpha*g` to AdamW where
`alpha=||h||/||g||` (identity during warmup/without a basis). Its gradient norm
matches the candidate projection **on its own trajectory** while direction
remains collinear with g. Freeze zero-gradient and numerical-tolerance handling.
Do not describe this as matching another arm's realized gradient norms or
actual AdamW step norms: trajectories and moment histories differ, and Adam's
scaling can cancel a constant gradient rescaling. Log those distinctions.

Primary questions: does hard rank32 beat accuracy-selected AdamW on fresh-seed
test accuracy, and does hard filtering differ from the scalar-gradient control?
Report both selected-checkpoint test accuracy and CE, all seeds, endpoint and
warmup-stop comparisons, and the width contrast. No tuning or checkpoint-rule
choice based on pilot accuracy; no step-level significance inference.

Reuse the validated low-level implementation helpers without editing any
frozen iteration-003 or production source. Before full execution: finalized
protocol/analysis definitions, unit tests for the new scalar/checkpoint rules,
instrumentation-preservation development pilot, independent review, committed
source hashes and explicit parent GO. Local RTX3090 only; no paid APIs/cloud.
