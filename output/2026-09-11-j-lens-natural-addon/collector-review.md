# Independent collector review

Initial verdict: **FAIL for excerpt acquisition pending two bounded corrections.** Selection logic has no identified blocker. This is a source review, not an acquisition release.

1. **Expected revision continuation is rejected.** New collect.py (artifact not distributed in this public snapshot) delegates to old `Stage.request`, whose line 119 (artifact not distributed in this public snapshot) rejects every `continue` field. With `rvlimit=1`, ordinary older-revision continuation recreates the previously documented first-article failure. Add a new-study request validator accepting only the exact nonempty-string `{'rvcontinue', 'continue'}` continuation shape, preserving original response bytes, and rejecting all other continuation. Do not fetch the continuation or modify the old helper. Clarify this in the new protocol and add synthetic acceptance/rejection fixtures.
2. **Wrong returned IDs can be silently skipped when marked ineligible.** Old eligible lines 189–194 (artifact not distributed in this public snapshot) return early on redirect/missing/non-main/disambiguation before comparing page IDs. The new protocol instead defines a different returned ID as an integrity failure. Check any returned positive integer ID against the request before delegating to technical eligibility; fixture-cover a wrong-ID response carrying a redirect marker. Preserve normal same-ID technical skips.

Full reads completed: new protocol, collector (247 lines), tests (267 lines), source-review note, old helper (266 lines), and the prior continuation adapter (context only; no entrypoint invoked). The review-changes skill guided this material-findings review.

Fixed first-root ownership before exclusion, complete prior-request union, candidate/pair hash construction and tie breaks, odd-tail retention, whole-pair rejection without repair, no reservation of rejected prefixes, left-then-right visits, request caps, no-redirect transport, output reserve/exclusivity and selection/source/protocol bindings otherwise match the stated design. No candidate selection or network operation was performed.

The unchanged initial test suite passed independently: **16 synthetic tests, 0.084 s**, `/usr/bin/python3`, CUDA hidden, OMP/OpenBLAS threads 1, bytecode disabled, `timeout 60s`. Those tests do not cover the two cases above. This test run used only temporary fabricated corpus metadata and the pinned inert helper source.

Reviewed SHA256 pins:

| File | SHA256 |
|---|---|
| main protocol.md | `61271da5ce4f18d9facdf902483e15340fc960074325826e377a4ee3766a4ae0` |
| main collect.py | `dff36201d41d2e7c298adf8f4844f03b6906cb7c5b177341f17b12285911cdb4` |
| main test_collect.py | `c7bf3f80705c85e8b2adc292113a7acb1f922bcf4fc00f12e0d596cf507ad539` |
| main source-review.md | `ffa9a347f244be5f7e686439b709eb6cd4c0772b7cafaece100fcd080202f03a` |
| old collect_text.py | `337e9a208f491b22ec0deb3a591b1216669eaf42db4712e43f8e1bc97e98d347` |

Collector/test hashes were identical before and after fixtures. Both findings were sent to main before any release. No model, tokenizer, arrays, scores, readers, old stages or source edits occurred in this review.

## Corrected-source re-review — PASS

Both initial findings are closed before any real selection or acquisition. Full corrected protocol, collector, tests and main source-review note were reread. The new local `request()` preserves exact response bytes and allows only the two-key nonempty-string revision-history continuation; it never follows that token or invokes old `Stage.request`. Unexpected/empty/extra-key/numeric continuation fixtures fail as required. Before calling old eligibility, a returned positive integer page ID is checked against the request, including redirect, non-main namespace and disambiguation cases. Normal same-ID technical skips still reject the whole fixed pair without repair.

Independent corrected fixture run: **17 tests PASS in 0.090 s**, same CPU-only/60-second command and synthetic scope as above. Source/test/protocol hashes were unchanged after the run. No remaining material issue identified in this bounded review; resource admission and actual stages remain main's responsibility. No actual candidate selection, HTTP request, model or reader call was performed by this reviewer.

Final reviewed SHA256 pins:

| File | SHA256 |
|---|---|
| main protocol.md | `a510feaf2dfdd5cf6541d974d2c305188e24d04a4e9e185ac73675970b8a6246` |
| main collect.py | `decdb21a2d76b0a2d75f7e4f0753ae76570b5c75e9aac12aa8e2002743001ee8` |
| main test_collect.py | `5e0e6d9be5ccc60330e7043d6b63cdef45fc77ccf36b075ddd754cf9d6850568` |
| main source-review.md | `0d14d11f063ffca670160f1cf4f71b5f5fc5f26f0d5a22bff05e4e64d1049611` |

The old helper remains pinned to the unchanged digest above. This acceptance supersedes the initial source verdict, not the preserved account of its defects.
