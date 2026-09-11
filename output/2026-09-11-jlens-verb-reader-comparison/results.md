# J-Lens descriptions communicate a useful local distinction

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

Fresh readers using the original direction-token lists got **116/128** saved
orderings correct; readers using the original fit-example prefixes got
**98/128**. Both token-list readers scored **16/16 on the prespecified PC4**.
The PC4 example readers scored10/16 and16/16: useful prediction is consistent,
but added benefit on that primary axis is one win and one tie, not two wins.

This is a meaningful simple comparison on a selected local panel, not a
population benchmark. Researchers chose the panel after its earlier positive
result; the four fresh readers did not see scores, the later gloss or prior
results. All256 choices were locked before the new grade. No neural forward,
training, refit, decoder, tokenizer, archive reload or paid compute was used.

## All four directions, unchanged references

| Fixed direction | Direction tokens A | Original fit examples C | A−C |
|---|---:|---:|---:|
| PC1 | 28/32 | 28/32 | 0 |
| PC2 | 26/32 | 14/32 | +12 |
| PC3 | 30/32 | 30/32 | 0 |
| PC4 — primary | 32/32 | 26/32 | +6 |
| All directions | 116/128 | 98/128 | +18 |

Always choosing FIRST or always choosing SECOND gives16/32 per axis/arm,
64/128 overall/arm, by complementary orientation. This is a fixed-position
control, not a full null distribution. There are no exact scalar ties.

| Reader | References | PC1 | PC2 | PC3 | PC4 | Total |
|---|---|---:|---:|---:|---:|---:|
| 1 | Tokens | 14/16 | 13/16 | 15/16 | 16/16 | 58/64 |
| 2 | Examples | 14/16 | 7/16 | 15/16 | 10/16 | 46/64 |
| 3 | Tokens | 14/16 | 13/16 | 15/16 | 16/16 | 58/64 |
| 4 | Examples | 14/16 | 7/16 | 15/16 | 16/16 | 52/64 |

The prespecified primary descriptive checks have different answers:

- Each A reader exceeds8/16 on PC4: **yes**, both16/16.
- A beats C on PC4 in each cohort: **no**; cohort1 is+6/16, cohort2 is0/16.

Overall A−C is+12/64 in cohort1 and+6/64 in cohort2. The consistent additional
gap is PC2, +6/16 in each cohort, but PC2 is a secondary result, not a newly
substituted primary winner. PC1/PC3 examples are equally predictive here.

Agreement after mapping choices to underlying text is16/16 for A on every
axis and for C on PC1/PC2/PC3. C agrees10/16 on PC4. Agreement includes shared
errors, so it is not accuracy or independent model-family validation. The
different readers inherit the same parent configuration; the resolved backend
build and sampling parameters are not exposed by the collaboration interface.

## Every form and adverse case retained

| Axis | Active A / C, each out16 | Passive A / C, each out16 |
|---|---:|---:|
| PC1 | 14 / 14 | 14 / 14 |
| PC2 | 12 / 6 | 14 / 8 |
| PC3 | 16 / 16 | 14 / 14 |
| PC4 | 16 / 13 | 16 / 13 |
| All, each out64 | 58 / 49 | 58 / 49 |

The six PC4 example-reader errors are reader2 on copied/supplied,
transcribed/mailed and scanned/sterilized, each in both forms. Reader4 succeeds
on those same cases. Neither examples nor targets were changed to fix this.
The original PC4 references remain goalkeeper/blocked versus recipe/combined.
Their mismatch with a later human gloss is not permission to tailor them
after the result. All item-level scores, choices and gaps remain in the JSON.

## Interpretation and limits

The strongest supported positive is **communicable local correspondence**:
fresh readers recover a useful signed ordering from the original token lists,
without being given the researcher-authored observation/provision gloss.
This goes beyond the earlier16/16 fixed-gloss forecast. It does not prove the
readers reasoned via one particular concept; no explanations were solicited.

The pooled comparison favors token lists, while the prespecified PC4 advantage
does not repeat strictly across both cohorts. Reference formats carry different
information: both are fit-only but are not information-equivalent. Separate
homogeneous packets prevent cross-arm clues, but mix arm with reader variation
and allow borrowing across axes within the same representation. This is not
a within-reader causal experiment, PCA-versus-no-PCA test, or comparison with
plain unembedding. No inference of four distinct semantic clusters follows.

There are **eight chosen contents, expressed in two forms, on one frozen
model/layer**, not256 independent semantic observations. The panel was reused
after its positive result, so this is exploratory comparison, not fresh-data
confirmation. Active/passive prefixes encode different partial propositions;
only text through the saved verb capture was shown. The source model was not
rerun on shortened inputs. No p-values or population confidence intervals.

The old later-ending1/8 failure and external-text19/24 result remain separate
and unchanged. Earlier broader comparisons did not establish general token-
list superiority; this targeted positive does not erase them. No optimizer,
alignment, safety or capability-performance result is claimed. Next test the
same reference formats on genuinely new, pre-frozen content at a matched
target role; do not replay these readers or claim their replications are data
replications.

## Direct evidence

Worker root: `/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree`.

- All256 grades and summaries (artifact not distributed in this public snapshot),
  SHA `fe245ec2786414d83d9f109fe26e52c57668684a03fe49f52d66bfecd28a1f35`,
  frozen worker commit `cb68aa6`.
- Committed response lock (artifact not distributed in this public snapshot),
  commit `f9a46889a24f289939ff3a2dedba6f077b1b3653`, SHA
  `23a4132da102f6a091f5aae694becc4e073d0af65cf6315b5ecc0ecdb5ea0421`.
- [Independent arithmetic audit](audit.json), [auditor](audit_saved.py),
  [protocol](protocol.md), execution (artifact not distributed in this public snapshot), [source review](source-review.md),
  actual reader delivery (artifact not distributed in this public snapshot).

The audit reconstructs every exact grade row and summary from committed
responses and the unchanged scalar key without importing the grader. Main
and the worker are different authors. Physical reader isolation is not
independently certified by this numerical audit.
