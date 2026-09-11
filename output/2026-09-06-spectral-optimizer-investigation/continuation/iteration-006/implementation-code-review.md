# Parent code review: pre-pilot implementation checkpoint

6 September 2026. **Pass for a local implementation checkpoint; no launch approval.**

The parent read the complete policy, runner, storage, independent summarizer,
tests, protocol and result schema. Separate agents authored those code modules;
the parent authored the design/analysis and this review. This is code review,
not independent empirical replication. The research workflow caused the design
challenges and separation of implementation from review; the documentation check
informed cloning, state witnesses and rescaling safeguards.

## Findings and resolutions

- A material pilot-provenance bug was found before any experiment: the runner
  reused its pilot-manifest path variable while iterating scientific source
  files, then returned the last source file's binding. Distinct `pilot_path`
  and `source_path` variables now preserve the correct manifest binding. The
  synthetic regression checks that exact returned artifact and changed-source/
  environment rejection. No historical result was produced by the faulty code.
- The pre-test gate now independently reconstructs both strict earliest
  selectors from finite validation rows and requires identical state hashes
  for coincident checkpoint steps. Exact cell membership, saved tensor hashes
  and availability are checked before the test loader can be called.
- Plan checks cover integer split/batch indices, bounds, disjoint subsets and
  replacement arrays. Realized corruption metadata retains the inherited
  float32 means, validated against exact counts within 3e-8; labels and old
  helpers are unchanged.
- Previous bases are cloned before once-only observation; step 101 uses the
  actual stored warmup basis. Native stored-basis action is not replaced by QR.
  Delivery gates read actual assigned gradients; zero delivery still calls
  AdamW. Scalar identity and rank-limited delivery are distinguished.
- Full snapshots include NumPy-global state as well as the inherited core
  states. Optional measurement witnesses cover additional update metrics;
  mandatory delivery validation runs in both on/off paths and has separate
  statelessness tests. On/off equality is not overstated as validation of
  unprotected code by itself.
- The independent stdlib-only summarizer reconstructs selectors, step/null
  masks, sign tolerances, all four primary groups and all fifteen fixed arm
  pairs per condition. It rejects incomplete evidence and checks the exact
  source map, pilot, data and 111 prospective full-run artifact bindings.
  Source inspection reconciles immutable pre-test JSON with final JSON minus
  its test field, and the binding-size sum with the store's byte accounting.
- Exclusive bounded storage, finite failure context, cooperative resource
  limits and explicit launch flags remain in place. Existing frozen sources
  and the delivered report are unchanged.

## Checks executed

The final combined CPU suite passes: 60 tests (18 policy, 18 harness, 24 summary)
in 9.250 seconds in the parent's run. The 17 existing repository tests also pass.
All neural-code checks used `CUDA_VISIBLE_DEVICES=''`.

A separate parent integration check ran the actual `train_cell` function on
the eight-example, three-feature synthetic fixture, for all six policies and
220 steps each. The independent `validate_step` accepted every one of the
1,320 produced rows with actual current/previous rank histories; CUDA remained
uninitialized. This checks the real producer/scalar-schema boundary, not a
fabricated MNIST result or complete 36-cell empirical execution.

Final byte bindings and counts are in
implementation-checks.json (artifact not distributed in this public snapshot). The complete
scientific-source map has fifteen members; this review and its evidence record
are outside that map. Changes to bound files require renewed review before use.

## Remaining authority and evidence gates

The source commit is a checkpoint, not permission to launch. A separate parent
pilot decision, committed-byte verification and fresh resource/occupancy checks
are required before any MNIST/GPU execution. A passing pilot must then receive
independent artifact/invariant review and a separate full-run decision.
No pilot or learning result exists at this review. GPU numerical behavior,
actual runtime, full artifact production and learning outcomes remain untested.
