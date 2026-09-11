# Independent I20 reporting review

8 September 2026. **Accepted-summary review only.** I verified the supplied
SHA-256 values for `analysis-001/summary.json` and `analysis-001/audit.json`,
then read only the accepted summary. I did not open an I19/I20 stream array,
rerun reconstruction or audit, import the observer, draw randomness, edit an
acquisition/analysis source, or inspect unregistered outcomes.

Accepted hashes:

- summary: `d8aee4475c0b18e85f09e2daa63aad84cfd0e484eda0f37fd0c90fb6c3daed69`
- audit: `01ab83a51ab4610f374bad2a1fb086513650ca5ea5ec04890254e5bc1e4a982d`

The summary reports PASS on all192 streams and contains the registered27,648
per-seed metrics,864 means,2,160 contrasts,270 identity/whole primaries,4,608
per-seed direction diagnostics,144 direction summaries and472 reused parent
means.

## Strong-cell primary and late candidate

The candidate is `ew999/rstar/rec9`. Effects below are paired
`comparator MSE - candidate MSE`, so positive favors the candidate. Counts are
positive/negative/zero over all32 reused seeds.

| Window | Candidate MSE | Comparator | Effect ± seed SE | Signs |
|---|---:|---|---:|---:|
| Whole, primary | .620736634 | parent EMA .9 | +.073496881 ± .023536928 | 27/5/0 |
| Whole, primary | .620736634 | parent common Kalman | +.042337170 ± .021945857 | 23/9/0 |
| Whole, primary | .620736634 | matched EW .99 | +.128841665 ± .014036218 | 30/2/0 |
| Late, secondary | .567252982 | parent EMA .9 | +.117901129 ± .024973765 | 29/3/0 |
| Late, secondary | .567252982 | parent common Kalman | +.090945775 ± .023611167 | 29/3/0 |
| Late, secondary | .567252982 | matched EW .99 | +.164273809 ± .016782624 | 30/2/0 |

The whole common-Kalman margin is modest, only23/32 favorable, and comes from
an outcome-informed reused-seed grid. It supports a constructive candidate,
not confirmed superiority or a multiplicity-adjusted result. The accepted
`results.md` states that boundary correctly.

## Startup and transition boundary

The favorable result is not uniform over time. Against EMA/common Kalman/
matched EW .99 respectively, the candidate has:

| Window | Effects ± seed SE | Signs in the same order |
|---|---|---|
| Startup | −.063270206 ± .051695129; −.234469772 ± .031317992; −.004704716 ± .003115118 | 15/17/0; 3/29/0; 11/21/0 |
| Transition | −.059320936 ± .030402336; −.088935187 ± .028147199; +.025573003 ± .016555391 | 13/19/0; 12/20/0; 18/14/0 |
| Late | +.117901129 ± .024973765; +.090945775 ± .023611167; +.164273809 ± .016782624 | 29/3/0; 29/3/0; 30/2/0 |

Candidate MSE is.964500 at startup and.760820 in transition, versus
.730030/.671884 for common Kalman. The positive whole-horizon primary therefore
survives adverse early windows because the late window is longer and favorable;
it is not evidence of uniformly better tracking.

## Direction-memory contrast

The strong identity-cell paired EW `.999 - .99` useful-alignment differences,
computed from the accepted per-seed direction summaries, are:

| Window | Paired difference ± seed SE | Signs |
|---|---:|---:|
| Whole | +.191342422 ± .024003637 | 30/2/0 |
| Startup | −.005961843 ± .003544493 | 10/22/0 |
| Transition | +.031919266 ± .028095713 | 19/13/0 |
| Late | +.245680410 ± .028948981 | 30/2/0 |

Late mean alignment is.861536 ± .038076 for EW .999 and
.615855 ± .019982 for matched EW .99. Both are present at all96,000 late
seed-steps; each is absent only at the first step per seed over the whole
horizon. The late improvement is therefore neither survivor averaging nor an
initialization mismatch. Startup is slightly adverse and transition mixed,
consistent with a longer-memory acquisition delay.

## Native response contrasts

For the same saved native direction, `rec99` is worse than CP throughout the
strong cell. Shortening the complement to `.9` is much better than `rec99`:

| rho | Window | rec99 versus CP | Signs | rec9 versus rec99 | Signs |
|---|---|---:|---:|---:|---:|
| .9 | Whole | −.242248619 ± .019052396 | 0/32/0 | +.999358688 ± .109036371 | 31/1/0 |
| .9 | Late | −.216029143 ± .022244416 | 0/32/0 | +.770763170 ± .110846504 | 29/3/0 |
| rho* | Whole | −.329990217 ± .025735682 | 0/32/0 | +1.078981328 ± .111077338 | 31/1/0 |
| rho* | Late | −.294884839 ± .030031177 | 0/32/0 | +.843450831 ± .115058048 | 29/3/0 |

The strong rho-star native means are1.427746 CP,1.757737 rec99 and.678755
rec9 over the whole horizon; late they are1.181915,1.476799 and.633349.
Shortening the complement is therefore an effective response intervention on
this stream. It does not by itself establish a whole-horizon win over common
Kalman: the native rho-star rec9 whole effect is −.015681546 ± .024213115,
18/14/0. Its late effect is +.024850097 ± .024159647,23/9/0 and remains
secondary. At rho=.9, rec9 is the direction-blind EMA `.9` identity and cannot
support a necessity claim for spatial selection.

## Weak/no-signal adverse cases

The same candidate is decisively non-robust across process cells:

| Process | Window | versus EMA .9 | versus common Kalman | versus matched EW .99 |
|---:|---|---:|---:|---:|
| 0 | Whole | −.414819586 ± .002548360 (0/32/0) | −.676179466 ± .004154003 (0/32/0) | +.002177950 ± .000107803 (32/0/0) |
| 0 | Late | −.419255364 ± .002639607 (0/32/0) | −.681899184 ± .004442608 (0/32/0) | +.002024287 ± .000098947 (32/0/0) |
| .01 | Whole | −.414083229 ± .002495996 (0/32/0) | −.507493618 ± .004166273 (0/32/0) | +.001945966 ± .000182265 (32/0/0) |
| .01 | Late | −.419142695 ± .002647061 (0/32/0) | −.510498968 ± .004199509 (0/32/0) | +.001556805 ± .000147398 (32/0/0) |

Thus the longer-memory direction gives a tiny matched-estimator improvement in
those cells while the resulting policy remains far worse than both common
controls in every seed. At weak change, late EW .999 useful-axis alignment is
.000560 versus.005082 for EW .99: longer memory more stably selects the
high-variance nuisance direction. With no changing signal, e1 alignment is a
reference-axis statistic, not accuracy at finding a real useful direction.

These adverse cases are essential. The result is conditional constructive
tracking evidence, not a universal optimizer improvement, a semantic-usefulness
detector, fresh replication or neural efficacy.

## Paired rotation diagnostic

Using the accepted per-seed rows, the maximum absolute **mean** paired
rotated-minus-identity MSE over all three process cells, four windows and36
policies is `2.47024623e-14`; the largest absolute per-seed difference in that
scan is `6.39488462e-14`. For the strong candidate, rotated-minus-identity mean
differences are `-1.42247325e-16` whole and `-2.04697370e-16` late, with maximum
per-seed magnitudes `2.44249065e-15` and `2.77555756e-15`.

The maximum absolute mean rotation difference in alignment is
`2.99673480e-16`. The rotated-minus-identity change in the paired late EW
alignment effect is `3.21791205e-16`. These are numerical rotation-consistency
checks on paired transforms, not independent replication or a general
invariance theorem.

## Report cross-check

Every MSE, effect, SE and sign count inspected in the current `results.md`
matches the accepted summary, including native rec99 strong-late MSE1.476799.
The report correctly presents whole identity risk first, retains startup,
transition, late and weak/no-signal boundaries, and explicitly forbids plugging
moving native/EW alignment into the fixed-angle risk formula. It also avoids a
causal-mediation, fresh-confirmation, optimizer-general or neural-efficacy
claim.

During review, two reporting gaps were corrected: the late paired alignment
contrast now includes its seed SE and30/32 signs, and the rho=.9 identity is
described as zero measured numerical discrepancy rather than an unsupported
byte-equality claim.

## Presentation and delivery review

The final `plots-002` bundle closes exactly against the accepted summary. All
1,336 CSV data rows match the complete ordered roster of864 I20 and472 reused
parent equal-seed mean rows, including origin, policy, process cell, rotation,
window,32-seed count, mean and seed SE. The manifest's48 MSE records and8
direction-alignment records match their source summary rows exactly. Its three
generated-file hashes and sizes close, and its renderer hash matches the
reviewed renderer.

The MSE figure uses the eight policies fixed in `presentation-roster.md`, shows
whole and late identity-rotation mean +/- SE in every no/weak/strong process
cell, labels the useful-direction oracle as privileged, and visibly retains the
adverse weak/no-signal cases. The alignment figure uses the four fixed
estimators in the strong identity cell and labels alignment as a diagnostic,
not a native-risk plug-in. Both figures are readable without clipped captions
or label overlap and prominently disclose reused seeds, exploratory scope and
the lack of fresh confirmation.

The HTML faithfully compresses the accepted result: the10.6% EMA reduction and
6.4% common-Kalman reduction are mean-MSE percentages in the strong cell; the
23/32 sign count and modest-uncertainty warning are retained; weak/no-signal
failure, toy scope, reused seeds and the fresh-seed/direction-switch next test
are explicit. Its two CID references exactly name the two reviewed PNGs.

The reviewed one-shot sender requires every communication/evidence file to be
regular, committed at one evidence commit and hash-bound; pins the accepted
audit, summary and plot manifest; checks complete registered rosters and all
manifest input/output/source bindings; requires this PASS review and all
reviewed hashes; validates the exact recipient, From display, subject,
Message-ID, size, HTML and two inline PNG MIME parts; and writes an exclusive
attempt record before external delivery so an ambiguous attempt cannot be
automatically retried. Its original MIME checks used optimization-removable
assertions; these were replaced before delivery by explicit fail-closed checks.
I did not execute the sender or send a message.
