# Parent pilot review

The single authorized development pilot completed with process exit 0. Harness
elapsed time was 32.72470740787685 seconds; the external process
clock was 34.34 seconds. All 12 traces reached 220 steps; no validation/test or
accuracy evaluation was performed. No retry was launched.

The parent directly checked all fifteen current/committed source bindings,
both training-file bindings, the saved plan and timing artifact bytes, all
1,320 on/off trajectory digest pairs and within-condition warmup core/observer
links. The successful full-state equality assertion is live-execution evidence:
the pilot does not retain every parameter tensor or full final state, so those
cannot be independently rehashed from its saved digest strings.

Peak allocated GPU memory was 158,846,976 bytes and RSS 1,605,783,552 bytes, below
the 8 GiB/12 GiB guards. The two bulk artifacts total 1,202,839 bytes on the verified
large volume. Numerical settings match the prospective single3090 environment.

## Runtime assessment

For each instrumented arm, extrapolate its measured first 100-step sum plus
1,900 times its mean step time over 101–220. Multiply the sum over six arms by
six seed/condition cells per arm. This gives
550.773728874512 seconds for 36 training loops.

This is a scheduling estimate, not a runtime guarantee. It excludes full-study
validation/final/test scoring, larger JSON/checkpoint writing, some orchestration
and hash work, and later-trajectory changes in step cost. The fixed 900-second
full guard leaves about 349 seconds above this loop estimate; do not relax the
cap or shorten the scientific protocol if the run encounters it.

## Evidence bindings

- Pilot manifest SHA256: `99699b1da8253d2b58295c88a9528870b490f5fb6cb6ce8e21ed9ed0955c77ec`.
- Full log SHA256: `0819e753419e58df2c18e614f0f4e1411d3c14d5c0550cf87af0679a86dc2d15`.
- Manifest: pilot/execution.json (artifact not distributed in this public snapshot).
- Log: pilot-run.log (artifact not distributed in this public snapshot).
- Separate auditor's exhaustive persisted-witness counts:
  [pilot-audit-checks.json](pilot-audit-checks.json).

This parent review alone is not full-run authority. Read the separate pilot
audit before issuing and committing any full GO. No primary learning outcome
exists at this review; the pilot answers runtime/isolation questions only.
