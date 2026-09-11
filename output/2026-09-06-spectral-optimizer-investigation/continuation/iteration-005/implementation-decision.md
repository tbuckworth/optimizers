# Parent implementation decision

6 September 2026, 12:25 UTC. Approve implementation and synthetic CPU unit
tests only for the current protocol, SHA256
`eb8f49f4a5247e9c33a23a0d15a572f2c5cec08355f2b4d05ada50f435ce0035`.
The parent read the full draft and all primary/null/aggregation amendments;
independent design critique and 65 small algebra checks support its target.
Official PyTorch and version-matched NumPy guidance has been checked.

This approval does not cover MNIST replay, the development pilot, the
worst-size synthetic resource test or full reference calculations. Those
remain separate decisions after implementation tests and independent review.
Do not modify the protocol to accommodate observed outcomes or resources.
No production or previous frozen source edits, paid compute/API work, public
push, credential access or deletion of prior evidence is needed or approved.

Author owns replay/reference implementation and its synthetic tests; parent
owns the independent analysis specification/summarizer/tests. Share the raw
result schema before finalizing source bindings. Bind every executed helper
and supporting math/analysis file before the runtime-only pilot, then commit
the reviewed passing source set before confirmation. All bulky artifacts and
partial arrays must remain under a verified unique large-volume attempt root.
