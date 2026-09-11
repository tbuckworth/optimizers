# Canonical fit-only preparation — complete

Codex — Spectral Optimizer Investigation · 10 September2026

Main read the complete worktree prepare.py, checked it against the frozen
protocol and reran its fabricated CPU fixture successfully. Source freeze
`e9989311dcffc2b3c49691c3b8965f875d99ccd9`; source SHA256
`5bfa5cfadd22ad54ffa6c4073e570860faf3e9590e7a474a50da0d1b657413c4`.
Explicit ValueError checks remain active independently of Python assert mode.
No PCA, old held-out projections or model import occurs in preparation.

The real preparation was then executed **once**, finishing21:27:05UTC/exit0:

    timeout 60s env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 /tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-10-j-lens-fresh-content/prepare.py prepare

This execution is consumed. Do not invoke it again. The attempt exists before
scientific reads; existing/dangling targets/attempts and fabricated loader
failure all correctly prevent automatic repetition. Input hashes are checked
before and after safe context-managed, non-pickle archive loading. Exact
48-row ID/split/style/pair structure, mean/shape/dtype/finiteness, orthonormality,
unit-input rounding and fit-only selection checks passed.

## Saved outputs and selection

Worktree new output directory:
`/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-10-j-lens-fresh-content/`.

- directions.npz (artifact not distributed in this public snapshot),
  SHA256`47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa`.
- selection.json (artifact not distributed in this public snapshot),
  SHA256`aac75625b67d9a90ba351d979f3e1a77cd75f69706a8b3113e071630de023f4d`.
- Completion/environment receipt (artifact not distributed in this public snapshot)
  and preparation-attempt.json preserve actual command, source/input/output
  pins and NumPy/build/interpreter identity. Main read both result JSONs.

| Axis | Positive FIT example | Negative FIT example |
| --- | --- | --- |
| PC1 | astronomy-1-plain | cooking-1-note |
| PC2 | football-0-note | cooking-1-plain |
| PC3 | programming-2-plain | cooking-1-note |
| PC4 | football-1-note | cooking-3-plain |

Seven unique full row IDs, including both framings of cooking-1; six unique
content identities. All four individual positive/negative pairs have different
contents; repeated cooking-1-note across axes is retained. Realized U32-to-V64
L2 differences are about2.31e-8 to2.62e-8. These are preparation diagnostics,
not acquisition-versus-analysis vector comparisons or interpretive success.
No fresh text score, decoded token list or judgment has been computed.

## Presentation and next action