# I17 report-arithmetic corroboration design

Prospective implementation note, 8 September 2026. This check was written and
tested without reading an I17 scientific summary or outcome artifact.

## Purpose and boundary

Before decoding scientific records, the checker requires supplied SHA-256 pins
for both collection manifests, the I17 summary, and the passing I17 audit. It
verifies every gzip member against both compressed and original hashes and byte
counts, rejects symlinks, special files, unsafe paths, duplicate JSON keys, and
nonfinite JSON constants, and requires the collection roster to equal the JSON
artifacts declared by both phase completions plus their completions and root
attempt records. Artifact root, audit, phase-completion, attempt, runtime
inventory, and the direct I16 collection binding must agree. Output is an
exclusive `analysis-001/report-audit.json`; a failed comparison is retained and
cannot be overwritten by an automatic retry.

## Independent reductions

The checker reconstructs exactly 30 logical trajectories: four normalized
scalar policies and one spectral policy for each of three seeds and two
training targets. The `k=0` rows come from the bound I16 archive; the other 24
come from I17. It independently derives:

- 12 validation-selected choice records;
- eight spectral-minus-joint-scalar primary contrasts;
- 32 spectral-minus-fixed-`k` selected contrasts;
- 16 horizon-2000 fixed-`k` endpoint contrasts;
- all 30 six-point curves and 960 scheduled metric aggregates;
- 20 path/displacement summaries, covering both predeclared windows.

Joint scalar selection minimizes validation clean CE or maximizes validation
clean accuracy across all four `k` values and all six horizons. Ties are broken
by earlier horizon and then lower `k`. Spectral horizons are selected separately
with the same validation criterion and earlier-horizon tie rule. Auxiliary
clean CE is converted to utility by negation; auxiliary accuracy is already a
utility. Auxiliary outcomes never enter selection.

Every primary and fixed-`k` aggregate is seed-first. If any scalar member needed
for joint selection fails, that seed's joint contrast is unavailable. If any
seed value is absent, the three-seed mean is unavailable; no survivor mean is
formed. Path summaries likewise require every recorded step in the exact
101–2000 or 1001–2000 window.

## Synthetic evidence

The synthetic suite constructs hash-bound gzip collections without experimental
data or tensors. It checks complete section reconstruction, hand-forced joint
selection ties, no-survivor behavior, mandatory audit/hash admission,
exclusive output, and retention of a failing report when a summary scalar is
changed. The fixture includes all 1,900 scheduled step rows per logical branch,
so both path windows exercise their real boundaries.

Passing this checker will mean that the displayed summary structures agree
with an independent JSON-only computation. It will not establish external
validity, causal mechanism, optimizer equivalence, semantic denoising, or the
correctness of tensor-level acquisition state.
