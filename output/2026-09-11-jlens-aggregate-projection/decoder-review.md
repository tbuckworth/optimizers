# Decoder and saved-array checks

Root fully read worker decode.py and six synthetic tests atc99aaab;
source4f74c0e6dd7d0603c3a62dbc80fb6dc03bd0dc056f5a27d234d1bc9f8b6ab6e2.
Root test rerun PASS6tests. Exact110names/order, signed unit inputs, FP32
transport, BF16 final norm/head, one top12 call with ties and raw tokenstrings,
all three independent-reference comparisons, explicit zero undefined records.
No text forwards, tokenizer encoding, fit or parameter update. Model/lens/
tokenizer/source pins and exclusive stage claim checked before actual release.
The scalar fidelity is readout agreement, not semantics.

The implementation-validation skill prompted an independent saved-array
checker. Its initial draft incorrectly accepted NaNs and broadcast-compatible
wrong shapes; those guard bugs were caught on synthetic fixtures and corrected
before any scientific audit. Exact roster/rank and original feature precision
checks were added, plus pooled-query and mandatory mean/norm diagnostics.
Five root synthetic tests pass; no experimental outcomes were inspected to
choose these fixes. [NumPy's documented eigensystem convention](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)
agrees with the producer's descending reversal and the independent residual/
spectrum checks. No new eigenspace is substituted or rank selected by auditing.