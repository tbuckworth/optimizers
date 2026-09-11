# Rare-class ranking is present in three cells, absent in Diffuse

**POST-HOC saved-logit analysis; three paired seeds, one task, provisional
interpretation.** The single execution computed all planned numerical rows,
but its terminal resource check failed; this is not an unqualified completed
PASS. Input hashes and numerical checks completed before that failure. See
provenance and exact failure (artifact not distributed in this public snapshot).

Low rare argmax accuracy hides considerable positive rare-versus-rest ranking
in Clean, Shared and Sham. It does not hide useful positive ranking in the
pre-specified score under Diffuse. A constant rare-logit bias is consequently
an incomplete description of the policy difference.

The score is digit-8 logit minus log-sum-exp of the other nine logits. Warmup
AUROC is 0.551828, 0.541868, 0.570006 across seeds 202609111/112/113
(mean 0.554567); rare accuracy is zero in all seeds. Endpoint AUROC is:

| Cell | Raw AdamW mean | Native spectral, all three seeds | Native mean | Own-norm raw mean |
| --- | ---: | --- | ---: | ---: |
| Clean | 0.979520 | 0.947463, 0.933524, 0.907575 | 0.929521 | 0.979275 |
| Diffuse | 0.952554 | 0.420306, 0.396155, 0.407517 | 0.407992 | 0.948870 |
| Shared | 0.977775 | 0.942987, 0.908849, 0.883835 | 0.911890 | 0.977696 |
| Sham | 0.979297 | 0.902814, 0.899937, 0.844209 | 0.882320 | 0.978973 |

Every native Clean/Shared/Sham endpoint improves rare ranking over warmup,
including all three Sham runs with zero rare argmax accuracy. Native Diffuse
instead falls below warmup and below 0.5 in every seed, despite improved rare
CE. Rare CE improvement therefore need not mean better rare-versus-common
ranking. Native AUROC remains below both controls in every seed and cell.
[All numerical rows and contrasts](metrics.json).

## The fixed log(11) adjustment exposes decision sensitivity and its cost

The transform adds log(550/50) to logit 8, based only on the fixed training
true-class counts and balanced heldout distribution. It was specified in
[design.md](design.md) before numerical-logit access. It was not fitted or
selected using held-out performance. It is a purely analytical readout
transform, not a newly trained policy or a calibration procedure.

Native spectral means before → after; accuracies are percentages:

| Cell | Rare accuracy | Common accuracy | Rare CE | Common CE | Common predicted as 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 4.93 → 62.87 | 88.69 → 85.65 | 2.8118 → 0.9997 | 0.3932 → 0.4892 | 0.007 → 5.659 |
| Diffuse | 0.00 → 35.87 | 62.01 → 36.57 | 4.5669 → 2.2811 | 1.8750 → 2.0100 | 0.000 → 42.489 |
| Shared | 3.53 → 63.60 | 87.34 → 83.69 | 2.9298 → 1.0594 | 0.4718 → 0.5847 | 0.007 → 6.622 |
| Sham | 0.00 → 56.27 | 85.07 → 81.42 | 3.2715 → 1.2586 | 0.5634 → 0.6778 | 0.000 → 6.941 |

Adjusted native rare accuracies, all seeds, are Clean 79.8/55.4/53.4%,
Diffuse 37.4/43.0/27.2%, Shared 81.4/57.0/52.4%, and Sham 71.4/60.6/36.8%.
The transform increases rare recognition and reduces common recognition in
every endpoint, with a particularly severe Diffuse tradeoff. Mean native
balanced accuracy changes from 55.81% to 36.50% in Diffuse, despite balanced
CE improving from 2.1442 to 2.0372. Accuracy and CE remain distinct outcomes.

Apply the same transform to every control when comparing policies: adjusted
raw rare accuracy means are 75.87/81.27/75.80/80.67% for
Clean/Diffuse/Shared/Sham; own-norm raw means are 76.00/79.93/75.53/80.87%.
Native's adjusted rare accuracy has mixed seed signs against the controls in
Clean and Shared, and loses in every Diffuse and Sham seed. These details
prevent treating the large native shift response as a uniquely successful
restoration. All before/after accuracy and CE values for all 36 trajectories,
plus the three distinct warmups, appear in [all-seed-results.md](all-seed-results.md).

AUROC is exactly unchanged by this transform in every row: adding a constant
to this log-odds score cannot change its ordering. Consequently, a pure
constant rare-logit offset cannot remove the observed native-control AUROC
gap. Clean/Shared/Sham support both partial useful discrimination and strong
decision-rule sensitivity; Diffuse supplies a stronger ranking failure that
a constant shift does not repair.

## What remains unresolved

These are descriptions of saved outputs, not causal proof that class-prior
mismatch caused the original effect. They do not prove or rule out a broader
role for imbalanced training. Neither the fixed transform nor any model here
is asserted to be calibrated or Bayes-optimal. In particular, corrupted
assigned-target distributions, class-conditional noise, finite optimization
and model misspecification prevent automatically applying an ideal prior-shift
account. Rarity, digit-8 identity, missing warmup exposure and novelty remain
entangled. No threshold search, sign reversal, model computation or new
training was performed. Existing favorable common-class preservation and
adverse rare recognition results remain valid for the original decision rule.

## Provenance and bounded-execution failure

Before any numerical array was opened, all 36 logits archives and three plan
archives matched the accepted completion receipts. The completion, accepted
audit and acquisition-source pins also matched. Five fabricated tie-aware
rank/pairwise AUROC fixtures and independent analytical CE/argmax examples
passed; no original audit was rerun. All 72 selected unpatched slices were
read once, and the 12 warmup copies per seed were verified identical.

The script then wrote 144 metric rows and their summaries before raising
`RuntimeError: runtime/memory envelope` at its final check. It used
`ru_maxrss` as a present-program RSS measure, which was a mistake in this
launcher. A separate no-data Python process reported `ru_maxrss` 1,659,276 KiB
while its own `/proc/self/status` reported peak address space 22,452 KiB and
peak resident memory 11,456 KiB. This exposes historical process accounting
that the footer failed to distinguish from the current executable's memory.

The scientific script installed a 1 GiB address-space cap, one-CPU affinity,
single-thread environment and 120-second CPU/wall limits before NumPy import.
However, its own elapsed/RSS values and `/proc` memory peak were not saved,
so its terminal resource certification is missing. The observed start-file
to metrics-file write interval is 1.209 seconds, not a full runtime measure.
The original source and failure are preserved, with no scientific reread,
retry, relaxed limit or claim that the footer passed. The saved JSON is
interpreted provisionally with that limitation attached.

[Frozen analysis source](analyze.py), fixtures (artifact not distributed in this public snapshot),
verified input receipts (artifact not distributed in this public snapshot), [numeric JSON](metrics.json),
post-failure provenance supplement (artifact not distributed in this public snapshot).
Metrics SHA256: `f849489f3d96752033500b66e1606a89caa91552a5a08ebe34d872ed183ee7ec`.
Preparation commit: `41647af888ac8ccfc0eb799a6686ed62162b09ac`; no commit was created.
