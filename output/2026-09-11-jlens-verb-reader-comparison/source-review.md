# Saved-verb reader implementation review

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

Main has read the complete new worker `judging.py` and `test_judging.py`,
including the revised arm-homogeneous allocation and final literal-pin test.
Final inspected SHA256 values:

- `judging.py`: `319d045220b5751f7cacc324b8a09a79f766e75b4c42bb9d01f65b66534bfd9f`.
- `test_judging.py`: `d8d54c086d44dd649925c845ca1623d6762a05fddffb31434580d57b46bbe8ac`.
- Main protocol: `d4b7ac785e7ec9ea31e642c113e1ac4ec0f56ba10ee1c41e73e033fee16ca9de`.

Main's final fabricated-only suite: **16 tests PASS, 5.019 seconds**, using
`timeout 60s env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -B output/2026-09-11-j-lens-verb-reader-comparison/test_judging.py`.
No scientific package/seal/grade, scalar parse or model execution was part
of those tests. Temporary fake Git repositories exercise committed locks.

The draft had a score-file hash corrupted by a mechanical count replacement
(`288` became `256` inside the digest). Both worker and main spotted it before
freezing or any real stage. The corrected value matches `sha256sum` on the
unchanged score file. A literal-pin regression test now protects it. This was
an implementation error, not a changed scientific score or discarded run.

Main checks:

- Pure packaging accepts no key, retains original A/C reference strings, and
  creates four homogeneous packets with all64 choices each. All32 target
  strings stop at the full verb, independently bound to saved subtoken offsets.
- Two cohorts have opposite item orientations; A/C within each cohort share
  orientation. Only anonymous public IDs, references and truncated targets
  reach readers. Private row labels do not.
- Strict JSON rejects duplicate keys, nonfinite numbers, wrong/extra response
  fields, wrong IDs and incomplete64-item responses. Empty decoded reference
  fragments, whitespace and Unicode are retained rather than filtered out.
- Every actual stage exclusively creates its output directory. Failure is
  recorded and consumes that attempt. No automatic retry or old stage reuse.
- Sealing copies exact response bytes. Grading checks the lock and allfour
  response blobs against an immutable Git commit before reading the scalar
  key. It also reconstructs the public/private map from pinned source inputs.
- Both saved score roles are schema-validated; only `verb` is graded. Exact
  ties get0.5 credit and tiny nonzero gaps are not rounded away. All256 item
  records, axes/readers/cohorts/forms, paired differences and text-level
  agreement remain in the result. The constant-position pooled controls
  must each get half credit.

Independent focused review: **PASS**, complete final source/tests read,
16 fabricated tests PASS in4.698seconds with CUDA hidden and one-thread,
offline settings. No material flaw found. Worker source-only freeze:
`92cfe481f7bc544de81f74ebf87181e0c1fa11fb`; worker clean. Worker also checked
all11 frozen pins against actual file bytes, without parsing scalars.

FINAL SOURCE ACCEPTANCE: main may now authorize the single new package stage.
All public packets and private map must be committed and checked before any
reader is dispatched. Passing fabricated implementation tests is not evidence
of reader accuracy, and the real transport/lock obligations still apply.
