# Prospective author preparation

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

The new [protocol](protocol.md) is fixed before authoring; commit `de27fd7`,
SHA256 `e82a2c5a7ae8756c6f549a338d047c9768b20c0f1b837e57f46030b5b896cfcf`.
Independent design review passes. A wording ambiguity was corrected before
dispatch: the author generates actor/object pairs rather than receiving them.
The useful prior result and its one-win/one-tie primary comparison remain
evidence, not a result to erase or a success criterion changed after data.

## Fixed lexical scope

Worker inventory (artifact not distributed in this public snapshot),
commit `0ff616aa8d70e944c98a233723a77cbddcc5b794`, SHA256
`fc7f28c6eabbb92a6d0d82d114656e52e8ba5c0f73ab8198c40bf6ca5fded0b6`.
Seven unique retained source rosters/232 rows,218 forms,198 canonical lemmas,
203 author exclusions including five spelling aliases. Main read the complete
form-to-lemma mapping and provenance, and independently checked all seven
source hashes and338 source-row/field/span bindings. Main also checked the
sorted author list is exactly the canonical/alias union. PASS.

Linguistic annotation uses agent judgment, not automatic or human-certified
ground truth. Conservative deverbal modifiers/nouns are explicitly marked;
five noun homonyms are not verb attestations. Completion fields are only
certified as prior stimulus metadata, not necessarily historical model inputs.
Full articles, failed acquisitions and decoded token lists are not included.
The exclusion claim is relative to this declared inventory, not complete
derivational/pretraining disjointness or independent semantic content.

## Exact author transport

Main extracted only the sorted alphabetic list, not source sentences, into
excluded-lemmas.json (artifact not distributed in this public snapshot), SHA256
`28ee8ec79b7cf8d5655d4c119fed924a548429fc24e8070727a20581d072c1d0`.
The complete author prompt (artifact not distributed in this public snapshot) was read and frozen together
at main `4d6aedd`, SHA256
`81ee4bb13c06b3b70e35f5b046dce5cc4c6f1517b7ed96e7ee030922cf5b632a`.
That hash includes its terminal newline. The collector must send the exact
decoded file text, with no additional context. It contains only the abstract
contrast, schema/grammar requirements and exclusions; no model/axis names,
scores, prior examples, token descriptions or results.

Existing Astra `/root/jlens_simple_comparison` is authorized to launch exactly
one `fresh_content_author` child with `fork_turns="none"`, no overrides. It
must preserve the first final JSON unchanged (one terminal newline allowed),
freeze `raw-author.json` plus actual delivery details, and stop. No repairing,
replacing, stimulus construction, tokenizer or model execution is authorized
to that collecting task. Verify the actual handle before any continuation;
do not launch a second author because collection is not yet reported complete.

Collection is now **complete and consumed**. The actual child was
`/root/jlens_simple_comparison/fresh_content_author`, observed03:35:30–03:37:06UTC,
one history-free dispatch, no overrides or observed tool messages. The first
raw JSON and delivery record are frozen in worker `8da96a0e7bcb1ad87d7bed969340e24d73aafb88`.
Raw SHA256 `8be4c015a4fcfdbaac08335dd18228a0c1f028d9bbb77fe82041d5a4699009f2`.
Main read the complete response and accepted all eight entries unchanged,
including documented semantic ambiguities, before any neural measurement.
See [roster review](roster-review.md), committed `c7a89fa`.

Main prepared a small [formatter](prepare_roster.py) and
[fabricated tests](test_prepare_roster.py): four tests PASS in0.003seconds.
It validates schema/exclusions and exact row/pair/spans, not English grammar
or semantics. Independent full code review and four separate fabricated tests
PASS (0.003seconds). After the raw-response freeze and complete main review,
one actual formatter stage constructed all32 rows and16 pair joins. Main
independently reconstructed every string, character span and join directly
from the raw author response: PASS. No rows were excluded or modified.
The actual stage is consumed; outputs are frozen at main `1ed3a17`:

- dataset (artifact not distributed in this public snapshot): `80749619a5bdbfd9506b8d453cc7f54d8f50c6eff73729f6a148152c0c0b35bf`.
- pairs (artifact not distributed in this public snapshot): `ec361bee2a15ec796230824c2e9125f46cebe76316a6b50ee45904cb2be02912`.
- receipt (artifact not distributed in this public snapshot): `a24ecb04087a0aa5d7b2d40fc7b7a3e557083f6477794f6bc8dd991d599acd9d`.

New source implementation is now active: Astra owns the inert one-verb-role
producer/checker and synthetic tests in the worker fresh study; the existing
review agent owns the new JSON-only reader package/seal/grader and tests in
this main study. Main will independently read and test both before release;
an implementer's own review is not counted as independent. No actual
tokenizer, forward, reader or grading stage is yet authorized by this
implementation assignment. The earlier author task must not be repeated.

The implementation-check skill was used before writing the formatter. Current
official [JSON documentation](https://docs.python.org/3.12/library/json.html)
supports bounded input, duplicate-key rejection and explicit output handling;
[exclusive creation](https://docs.python.org/3.12/library/functions.html#open)
supports fail-if-existing attempts. These implementation safeguards are not
evidence of linguistic novelty or scientific usefulness. No paid compute.
