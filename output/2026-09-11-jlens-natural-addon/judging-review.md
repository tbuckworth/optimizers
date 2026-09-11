# Prospective judging and checker acceptance

Codex — Spectral Optimizer Investigation · 11 September 2026

Main fully read all 756 lines of the new judging source and all 515 lines of
its fabricated tests, including the final metadata-contract change and its
explicit test. Source freeze: main `dd4d20074358a9c0cefe3d2a133441cff61d9a64`.
No new measured scores or activations were inspected during this review.

| Artifact | SHA256 |
|---|---|
| judging.py | `b11302567336bfed8b7ace1b21734f2144e8c393fab47abb2a2edabe8e6636b8` |
| test_judging.py | `8b6df4de01e7ef54f90bb732caeedc406c64de8cb76faca6e7429c3e81f3e679` |
| prepare_delivery.py | `63214e34dccfdb381c6f7524259a5b09e1d3c62736b652a454d9a45499a71480` |
| worker forward.py (unchanged) | `23c9c7cf77f2239bd16b219c440fbf056628730cd2bdf51f66b812ab0e8f580c` |
| worker check.py (now bound) | `879431802cae6c883140cbf9fbf368976b62e2e667fc9c94b662edcc34eef108` |

The code uses the frozen original C example strings in both E and EA, and
adds only unchanged A token lists in EA. Public pole meanings never reverse.
All four readers receive the same axis/pair display order and neutral field
definitions; cohort 2 reverses only target FIRST/SECOND. Anonymous IDs differ.

Exact random algorithm: one Python `Random(20260920)` instance; draw 64 Boolean
target swaps in PC1–PC4/original-pair order, shuffle block IDs 1–16 and item
IDs 1–256, shuffle the four-axis display order once, then shuffle each axis's
pair list once in PC1–PC4 order. For raters1–4 in order, pop anonymous IDs while
walking that shared display order, reverse targets according to the common
swap XOR cohort2 rule, and draw one 64-bit packet ID after each reader. No
other random stream or reroll. The exact public JSON and actual wrapper will
also be frozen before dispatch; a seed alone is not the transport specification.

Package creation opens the committed release and six public/reference/token
inputs only. Producer, forward receipt and scores are **metadata-only** at
that stage. The public-only lock verifier validates all 256 choices and all
five exact committed response/lock blobs before any private-map, measurement
or release-input access. The grader then reconstructs all public/private joins
from immutable inputs before the sole score read. The separate saved-array
checker uses the same public lock gate before loading producer or arrays.

Main caught a contract mismatch during review: producer was in FROZEN but
initially absent from manifest.inputs. The agent added its metadata there,
while explicitly skipping producer contents in the public input loader, and
added a fixture covering every FROZEN name after deleting all private/scientific
fixtures. This was fixed before any actual packet, neural pass or reader.

Primary scoring is all-four-axis EA−E out of 128 per arm, with two cohort
differences out of 64. Both must be positive and each EA reader must exceed
its own stronger realized constant FIRST/SECOND control. Exact scalar ties
receive 0.5; tiny nonzero gaps remain. PC4 is secondary. All 256 item credits,
128 matched gain/harm rows, topic counts (including empty football), controls,
reader/cohort/axis/category splits and chosen-text agreement are retained.
No population-significance or causal-interpretability claim follows from this
small reader-dependent package comparison.

Independent main 23 fabricated tests passed in 17.826 seconds on the pre-comment
source. Agent's final contract-fixture run passed 23 in 17.429 seconds. The
normal stop-hook then committed only two exact inline allowances for the
independently verified public token-artifact SHA; stripping those comments
reproduces reviewed source hashes. No rule was disabled or credential stored.
Main reran the exact final committed 23 fixtures: PASS in 17.814 seconds.
The final bound checker passed all 10 fabricated tests in 2.901 seconds;
transport builder syntax check passed. No real checker stage has been run.

The review follows the documented [Python JSON controls](https://docs.python.org/3.12/library/json.html#standard-compliance-and-interoperability)
for duplicate/nonfinite values and bounded parsing, freezes the actual ordering
because [random-module reproducibility has limits across versions](https://docs.python.org/3.12/library/random.html#notes-on-reproducibility),
and verifies exact committed file contents with [git cat-file](https://git-scm.com/docs/git-cat-file).

The bound checker is committed at worker `7b90282`, clean. Main admits **one**
new 32-prefix forward only, using the already-consumed successful preflight.
Fresh host/GPU headroom and exclusive output/service guards remain required.
Then metadata freeze, one package, four frozen public transports, four first
reader finals, committed lock, saved-array check and grade. No old-stage replay,
model updates, PCA fit, extra decoding, retries or paid compute.
