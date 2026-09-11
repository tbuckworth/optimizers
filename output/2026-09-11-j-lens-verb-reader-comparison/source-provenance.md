# Saved-verb reader implementation freeze

Source/fabricated fixtures only; actual package, readers, seal and grade are **unrun**.
No tokenizer, model, numerical archive, projection, judge, paid call or old stage
was invoked. Main owns admission and subsequent execution. Paid compute remains
$0 spent / $0 reserved / $100 ceiling; subscription reader calls are not charged cloud runs.

## Source and checks

- [judging.py](judging.py): SHA256 `319d045220b5751f7cacc324b8a09a79f766e75b4c42bb9d01f65b66534bfd9f`.
- [test_judging.py](test_judging.py): SHA256 `d8d54c086d44dd649925c845ca1623d6762a05fddffb31434580d57b46bbe8ac`.
- Snapshot adapted after complete reading of main
  `output/2026-09-11-jlens-independent-content/judging.py`
  (`f1fb07cec971d4f9533b2c8f65e613900a3172213b1cd3b2c0aea22965979ef1`)
  and its full tests (`37ed13cf21897964d0b98ab46b1393d1e449a38a126ca6af6837d8cfa4708859`).
  No old module is imported or executed. The saved token-record contract was
  inspected in the frozen new-verb producer; validation is local standard-library logic.
- Main revised protocol commit `95d4665`, SHA256
  `d4b7ac785e7ec9ea31e642c113e1ac4ec0f56ba10ee1c41e73e033fee16ca9de`,
  was read in full. Before any actual packet, its homogeneous allocation replaced
  the initial crossover: rater1/3 see only A; rater2/4 see only C.
- Fabricated-only command, from this worker root:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 timeout 60s /usr/bin/python3 -m unittest discover -s output/2026-09-11-j-lens-verb-reader-comparison -p 'test_*.py'
```

Result: **16 tests PASS, 4.863 seconds**. Temporary fixtures include four exact-byte
response blobs plus a real fabricated Git lock, with normal commit hooks.
Coverage: import isolation, local deterministic RNG, homogeneous allocation,
opposite cohort orientations, complete 256-item coverage, exact full-verb
truncation including split verbs, unchanged reference strings, malformed
wrapper/identity/offset/receipt rejection, 64-response schema, exact ties/tiny
gaps, verb-only grading despite validated sentence-role data, all summary
denominators, agreement by selected text, constant controls, exact committed
five-blob gate before map/key access, caps/symlinks/exclusive failed stages.

During draft review, a broad count replacement accidentally changed the `288`
substring inside the score digest. This was corrected before freeze or actual
use; literal-pin regression coverage and independent byte-hash checks were added.
All eleven frozen protocol/reference/dataset/pairs/token/preflight/producer/score/
forward-receipt/checked-result/checker-receipt digests in this source were then
independently checked with `sha256sum` and matched. These checks parsed no scores.

## Interfaces and limits

`package(output, references=..., references_sha256=..., dataset=...,
dataset_sha256=..., pairs=..., pairs_sha256=..., protocol=..., tokens=...,
preflight=...)` accepts no scores. It validates pinned full-row and saved-offset
bindings, then exposes only `text[:verb_span[1]]`, exact A/C references, random
anonymous labels and the common question. It does not retokenize truncated text.

`seal_responses(output, packets=..., manifest_sha256=..., responses=...)`
preserves all four response byte strings. `grade(output, packets=...,
manifest_sha256=..., responses=..., lock_sha256=..., lock_commit=..., scores=...,
scores_sha256=...)` verifies the committed lock and all four response blobs
before private-map/scalar access. The exact pinned producer JSON is the only
key: both location schemas are validated, only `verb` is selected. Other
`KEY_PINS` record protocol-declared provenance; the grader does not reopen or
recompute the previous checker results. Their bytes were checked at this freeze.

The API itself does not prove actual reader isolation, transport or the packet
commit timing; main must record those under its frozen delivery contract.
Within-arm cross-axis context, reader/order variation and outcome-selected
targets remain limitations. The comparison is not new-data confirmation,
information-equivalence or a causal within-reader representation contrast.

## Implementation documentation check

Used the best-practices-validator skill before implementation. Current official
[Python 3.12 JSON documentation](https://docs.python.org/3.12/library/json.html)
supports bounded parsing, duplicate-key hooks and explicit nonfinite rejection;
[exclusive creation](https://docs.python.org/3.12/library/functions.html#open)
supports fail-if-existing outputs. The
[random reproducibility notes](https://docs.python.org/3.12/library/random.html#notes-on-reproducibility)
motivate recording the exact interpreter alongside seed 20260916. These are
implementation safeguards, not evidence for the scientific interpretation.
