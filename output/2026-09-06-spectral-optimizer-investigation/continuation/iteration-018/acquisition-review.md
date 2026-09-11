# I18 acquisition and theory review

8 September 2026, pre-scientific-draw. Main wrote protocol/runner and independently
read the full core and tests. Current_report wrote core/tests and independently
reviewed main's runner/tests and response-transport algebra. I17_analysis reviewed
the protocol's estimand, optimum and adverse boundaries; it is separately
implementing the saved-array auditor, without outcome reads or importing the
acquisition core. No broad researcher approval/fail-fast gate is imposed.

## Accepted source and correctness checks

- Core: 43c33683b5e40ac17d8b975fbc895e588e9a4982200c12484f0ae4a571124841.
- Core tests: 82cac986fe04474629c305566e1076040d5cc4f752ecb7aa28cec566968f3bd0.
- Main read all initial core/tests and all subsequent changes: constant optimal
  decay hoisted outside loop; shared canonical noise versus rotated epsilon;
  literal g1 initial delivery; masked absent-basis alignment zero; input/RNG
  preservation and forbidden optimizer.step fixture. No unresolved formula defect.
- Core worker final 7/7 tests PASS, compile PASS; main earlier 3b10e4 7/7 PASS.
  Main runner 896562 7/7 PASS; independent worker runner 7/7 PASS.
- Main's exact response-transport checker 240726: 48 rational identities PASS,
  zero random draws/native updates. Independent current_report algebra review
  PASS for T, delta recursion, arbitrary-action constant preservation and the
  conditional projector norm bound. No universal tracking-harm claim follows.

Final main combined synthetic suite017beb:14/14PASS in0.820s; final source hashes
59711a match the two pins above. This is the reviewed acquisition source set.

The observer remains unchanged; one native filter_grad call per observation,
post-ingest mean and actual action are shared across all response rules. No
optimizer/forward/backward call, scientific RNG fixture, seed search or replay.
The archive saves all response/buffer states and both canonical and rotated
noise, enabling direct digest checks and independent one-step algebra.

## Resource feasibility and residual risk

Worker deterministic sinusoidal/non-scientific 1000-step fixture: 1.648 seconds
core, 722000 array bytes, 999 active-basis observations. Linear projection is
about1266 seconds for768000 observations before archive I/O, within the1600s
cooperative cap but with only about21% time margin. Approximate total numeric
arrays are555MB plus small NPZ/metadata overhead, within1GiB. No favorable
scientific outcomes or timing-based seed/arm selection were used.

This is feasibility evidence, not a guaranteed finish: current load and I/O can
consume the margin. Keep the1800s unit cap,5s shutdown grace,4GiB memory,
zero swap,1core,Restart=no and exclusive attempt. If interrupted, preserve the
partial prefix and do not silently resume/restart. Source/test/protocol freeze
and fresh mount/space/service checks are still required at actual launch.

Independent saved-array audit source/tests need final main review and a separate
freeze before their first real read. They do not alter this frozen acquisition.
No I18 scientific root/attempt exists at creation of this review.
