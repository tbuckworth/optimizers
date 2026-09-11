# Same examples, different coordination: a useful but incomplete effect

Codex — Spectral Optimizer Investigation ·10 September2026.
**Three fresh paired seeds;36trajectories;72events; independent audit PASS.**

## Finding and implication

Grouping rare correct examples at exactly unchanged exposure improves the
spectral policy's preservation of held-out common-digit recognition under
diffuse wrong labels.
It does **not** restore rare correct recognition in that noisy condition.
Clean rare recognition improves, but the mean is dominated by one seed.
This is a useful sampler-dependent learning effect, not a successful general
rare-learning rescue or semantic-selection result.

The[protocol](protocol.md) fixed seeds202609121/122/123, Clean/Diffuse,
interleaved/grouped and raw/native32/own-norm raw,100warmup/2000total updates.
Every50update block uses the exact same3,200 indexed examples, including
repeats, and identical labels. Their batch composition and presentation order
change. Each fresh
seed's12branches share full warmup state. No oversampling, extra loss weight,
selected checkpoint or production filter modification. Neither schedule is
the older iid baseline, and the fresh seeds differ from that study.

## Complete endpoint means

Accuracy in percent; CE lower is better. “Own norm” applies the native norm
function to its own raw gradient, not the spectral trajectory's numeric norm.
All seeds, sampleSD/SE, signs, warmup changes and paired contrasts are in
[all-seed tables](all-seed-results.md); all56 metrics in
[independently checked JSON](results/checked-summary.json).

| Condition | Schedule | Policy | Rare acc | Rare CE | Common acc | Common CE | Wrong-label fit |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean | Interleaved | AdamW |51.67|1.7296|93.60|.2340|—|
| Clean | Grouped | AdamW |52.07|1.4875|93.74|.2273|—|
| Clean | Interleaved | Spectral |2.27|3.1825|88.87|.3894|—|
| Clean | Grouped | Spectral |10.27|2.6963|88.73|.3963|—|
| Clean | Interleaved | Own norm |51.93|1.7032|93.50|.2343|—|
| Clean | Grouped | Own norm |53.47|1.4394|93.72|.2279|—|
| Diffuse | Interleaved | AdamW |51.80|2.2428|35.19|1.8586|31.59|
| Diffuse | Grouped | AdamW |17.40|3.0067|35.57|1.8531|30.60|
| Diffuse | Interleaved | Spectral |0.00|4.8185|47.92|1.9095|7.93|
| Diffuse | Grouped | Spectral |0.00|4.6245|56.64|1.8941|6.40|
| Diffuse | Interleaved | Own norm |52.67|2.2028|35.45|1.8613|30.99|
| Diffuse | Grouped | Own norm |21.40|2.8280|37.03|1.8364|30.05|

![All three seeds and their means for each policy and schedule, at every fixed evaluation.](results/learning.png)

### Useful protection strengthens; noisy rare recognition does not

Spectral Diffuse common accuracy gains **+9.20,+10.07,+6.91percentage points**
under grouping (mean+8.73±.94SE). Its schedule-minus-raw interaction is
**+8.34±1.08points**, favorable in all three seeds. Wrong-target fit falls
in all three, mean7.93→6.40%. Balanced accuracy43.13→50.98% and
balanced CE2.2004→2.1672 both improve in all seeds. Common CE improves in
two seeds but worsens slightly in the third. This preserves a real benefit
without declaring every metric uniformly improved.

Both schedules nevertheless leave spectral rare accuracy at **0% in every
Diffuse seed**. Rare CE improves from warmup everywhere, and grouping improves
it in two seeds; zero recognition is not a frozen model. Meanwhile grouping
hurts raw rare accuracy by29.6,58.4,15.2points (mean51.80→17.40%) and its CE
in every seed. Therefore the favorable **+34.4point rare-accuracy interaction
is entirely a control deterioration**, not positive spectral rare learning.
The own-norm control also loses noisy rare recognition in all seeds.

Even grouped spectral remains worse than raw on mean common CE1.8941versus
1.8531 and balanced CE2.1672versus1.9684, despite its large accuracy benefit.
All noisy policies degrade from the common warmup; the result concerns
relative preservation, not improvement beyond initial common competence.

### Clean rare learning improves unevenly and has costs

Grouped spectral rare accuracy is .8%,.6%,29.4%, compared with0%,0%,6.8%:
changes+.8,+.6,+22.6points, mean+8.0±7.3SE. Every sign is favorable but most
of the effect comes from seed123. Rare CE changes by−.9274,+.1607,−.6918;
the second seed gets slightly better recognition but worse probability loss.
Common accuracy falls in every seed (mean−.133points), and common CE rises
everywhere. The rare native-minus-raw interaction is mixed across seeds.
Grouped native still has worse rare/common accuracy and CE than both raw
policies in every Clean seed. This is partial useful learning, not a cost-free
rescue or a competitive clean-data optimizer.

![Every paired schedule change, with accuracy and CE signs clearly separated.](results/schedule-effects.png)

## Mathematics and limits

All18registered Grouped high-count events (9Clean,9Diffuse) improve the fixed
rare training-probe CE after the actual Adam update. The useful local signal
is real, even where final rare recognition remains zero. Yet all9Diffuse
Grouped high events worsen common-probe CE while the long-run common endpoint
improves. Late Diffuse Grouped rare mean-action retention is96.29–99.18%,
so permanent exclusion of that incoming mean cannot explain the endpoint.
These observations motivate competition/persistence and history hypotheses;
the sparse anchors do not measure forgetting between bursts or identify
which intervening updates matter. Main independently cross-checked all72JSON
receipts, finite-loss sign counts and late retention ranges for this report.

At fixed parameters, exact occurrence conservation preserves a block's mean
gradient. In the deterministic two-group approximation,

    g_t = mu_common + q_t (mu_rare − mu_common)
    Cov_block(g_t) = Var_block(q_t) (mu_rare − mu_common)(mu_rare − mu_common)^T.

## Evidence and execution

One acquisition completed246.464333s, all36trajectories/72diagnostics,
1,774,641,956artifactbytes,128,130,048GPUpeakbytes and1,439,700KiBRSS.
The frozen independent CPU-only saved-data audit passed**514,404checks**,
all756evaluations and72events in34.387918s,718,408KiBRSS. It checks saved
arithmetic and source/provenance, not an independently repeated forward/backward
pass. No outcomes, limits, tolerances or roster changed after launch.

Parent:`/tmp/spectral-experiment-artifacts/spectral-batch-composition-20260910.H2Ow40`.
Acquisition completionSHA:
`39b4ef091e691780b46c1d7f5dcb12e108dbfd5b3b85c6abba612a93a393b60a`.
Audit resultSHA:
`3640269898d6c4115adbac925586fb54a6f1d47b0000b7e511a2fa1f1fcdeb6b`.
Checked summarySHA:
`4f437004dda5244e2b0ff18e4c072905417354cc551abeff39b5c662c0c8691b`.
Both jobs are terminal; seelaunch receipts (artifact not distributed in this public snapshot). No paid spend.

Main visually inspected both150dpiPNG outputs; all labels/legends and seed
coverage are visible, no clipping. Their exact hashes are in
render-receipt.json (artifact not distributed in this public snapshot). Separate
[old-score findings](../2026-09-10-spectral-rare-score-analysis/interpretation.md)
remain post-hoc PARTIAL/RESOURCE_FOOTER_FAILED; this successful new audit
does not certify that earlier analysis's missing resource footer.
