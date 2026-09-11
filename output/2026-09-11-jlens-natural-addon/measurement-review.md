# Measurement source review — natural add-on

Codex — Spectral Optimizer Investigation · 11 September 2026

Worker source freeze `5cde0fb93bc2037e61951b7fd91c0271cec90ef9`, clean.
Main fully read the new producer, checker scaffold and both complete synthetic
test files. No actual tokenizer/model/scientific arrays were used by this review.
The worker source receipt (artifact not distributed in this public snapshot)
specifies the exact schemas and pending integration.

| File | SHA256 |
|---|---|
| forward.py | `23c9c7cf77f2239bd16b219c440fbf056628730cd2bdf51f66b812ab0e8f580c` |
| test_forward.py | `c817d86f044323266530da5249b1d1ab6cc66857b0485002a03742927d480a1d` |
| check.py (disabled scaffold) | `f6043d8884c82d9dcdd9c72ffce14264928dd84f315e0c5889a1211e7c234a0f` |
| test_check.py | `403743bcb5c641066f8c34c72d2e50da386ac4c649cf0d1f9b091b684b038379` |

Main independent producer14fixtures PASS3.236s; checker10fixtures PASS3.367s.
Both commands used `/usr/bin/python3`, CUDA hidden, offline flags, numerical
threads1 and60secondtimeout. They ran concurrently; durations include that
contention. Handles85336/79306 are terminal exit0. All four hashes match after
review/tests. Agent's separate fixture runs were14PASS2.612s/10PASS2.923s.

The producer binds the exact collected32rows/16pairs, protocol and collector
receipt/source. Whole-roster tokenizer preflight preserves input IDs, all-one
masks and character offsets without BOS/padding/truncation, cap96tokens.
Unicode byte subtokens may share character spans; overlapping ordered offsets
are retained. It captures the last actual input token, not an earlier token
chosen from character spans. No unmatched non-whitespace or partial last
prefix is permitted. A late-row failure consumes the whole preflight, not a
replacement row or shorter prefix.

This use of fast-tokenizer character offsets and explicit tokenization controls
is supported by [Transformers tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer).
The source's observation hook returns no replacement and removes its handle
after each call, consistent with [PyTorch2.11 forward-hook semantics](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).
The overlap policy is our validation choice, tested on fabricated Unicode
subtokens; official offset support alone does not certify this real roster.

The later forward preserves BF16/eager/eval/no_grad, pinned model/adapter and
the original source mean/U32. It checks parameter versions before/after and
captures post-block11 at only prefix_end:32×1×1024activations,128scores,
64original-left-minus-right gaps. Same float64 centering/projection, no refit,
decoder or model update. No actual forward is admitted by this source note.

Checker arithmetic and input/scope guards pass synthetic tests. Its
FORWARD_SHA/JUDGING_SHA are deliberatelyNone: actual execution fails closed
before key access. The new public-only `verify_response_lock` implementation
and its immutable source pin remain to be implemented/reviewed, with all256
first choices and five committed response/lock blobs preceding any key reads.
The scaffold is not an accepted live scientific checker yet.

Release at this checkpoint: ONE new whole-roster tokenizer preflight only,
CPUQuota100%,4GiBhost/no swap,120seconds, offline/CUDAhidden, no restart.
Check actual headroom and exclusive stage/unit first. Neural release additionally
requires reviewed prospective judging/analysis code and a successful frozen
preflight. Attribution publication screen is still pending. No old stages
are released, and no extra approval from the user is being requested.
