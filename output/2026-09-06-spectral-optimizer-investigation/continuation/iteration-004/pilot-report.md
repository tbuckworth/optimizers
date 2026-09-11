# Development pilot: passed

Executed 2026-09-06 11:47:00.861421–11:47:16.737821 UTC, 15.876400 elapsed
seconds on the local RTX3090. This is runtime/invariant evidence, not a learning
result. Development seed 9877, four arms, 220 steps each with instrumentation
off/on: eight traces and 1,760 optimizer updates. No validation accuracy, test
data or development performance was evaluated. No failed attempt or retry.

For every arm, all 220 parameter hashes and the complete final training state
matched exactly with instrumentation off versus on. Every warmup step's
parameters and raw/applied gradient hashes matched across arms. The complete
step-100 optimizer/core state matched, and hard32/scalar32 observer state and
warmup history also matched. Each instrumented arm passed six before/after
measurement-state checks. Width128 reached its required width by step160.
Scalar delivery and finite/orthogonality gates passed.

| Arm | Instrumented mean seconds/step, 129–220 | Peak allocated GPU bytes |
|---|---:|---:|
| AdamW | 0.005043821 | 119,144,448 |
| Hard estimate32/project32 | 0.006243721 | 151,925,248 |
| Hard estimate128/project32 | 0.007620221 | 249,694,208 |
| Scalar32 norm control | 0.007461009 | 152,129,024 |

Maximum reserved GPU memory was 603,979,776 bytes. The steady-step extrapolation
is 158.21 seconds for 3 seeds times 2,000 steps times all four arm means. This
does not model warmup hashing, validation, checkpoint/test evaluation or JSON
serialization overhead; budget several minutes, not an exact completion time.

Execution manifest (artifact not distributed in this public snapshot) binds all nine source files and
training IDX hashes. Timing and invariant records (artifact not distributed in this public snapshot)
retain all per-step hashes and timing/rank evidence. The generated development
plan is local Git-ignored, with its size and hash in the manifest. The pilot
source hash is `21bf504ef706be8f35ae1f4c5fd1f454b0207587b8225e54f43ce2942c066375`;
the protocol hash is `40fa3dc42612176d955d944d3c052af82d09f46eac58ba893c789a47554fbe53`.

The 11 harness and 7 summarizer synthetic tests passed before this pilot and
were independently rerun by the reviewer. Confirmation remains separately
gated by unchanged source/data, committed source and explicit parent approval.
