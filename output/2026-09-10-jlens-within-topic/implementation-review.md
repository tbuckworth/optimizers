# Main implementation acceptance

Codex — Spectral Optimizer Investigation · 10 September 2026

Main read the entire original 488-line judging module and 335-line fabricated
test file this continuation, then the complete new-source/test diff. The
adaptation changes only the frozen protocol/pair hashes, seed, rotated arm
allocation, prompt, exact adjacent-pair validator, and new-stage schema names.
Old reference/score acquisition schemas remain unchanged. Previous scientific
files are not modified. Astra's source/test commit is `6a32df5`.

- New judging source SHA256:
  `ebb55a9f4b93fa881b28d65dc99fed98f4a8ecb5ddbdd5245cb9787914bf2dd2`.
- New fabricated-test source SHA256:
  `90de779c95fb71913179cd0aef380e55670f148091241506c7a6f85c53d52511`.
- Protocol SHA256:
  `5de33f7d781aee09678efda50b7d8793d66245f0c1af8ca4cd72acef6a2259eb`.
- Pair roster SHA256:
  `552b4090c2e2382a352a14d2b6be3bd28cad98a33f8f0c36dd49dde0a89fad7a`.

Main ran `python3 -m unittest discover -s
output/2026-09-10-jlens-within-topic -p test_judging.py -v`: **16 tests PASS**,
3.411 seconds, exit 0. These use fabricated data only, including a temporary
git repository to verify exact committed response bytes before score access.
No actual package, response lock or numerical grade was created by tests.

The exact roster check is stronger than merely requiring same-topic pairs:
alternative disjoint within-topic matches, cross-topic matches, and swapped
fixed endpoints are rejected. Tests also cover all 144 unique allocations,
shared swaps, unchanged strings including whitespace/empty tokens, local RNG,
anonymous public structure, strict response/JSON validation, exact ties and
tiny nonzero gaps, score-independent packaging/sealing, missing responses,
non-finite scores, provenance changes, and exclusive attempt directories.

The implementation-check skill led main to consult current official Python
JSON, exclusive creation, and RNG reproducibility documentation (linked in
the protocol) before accepting this adaptation. Source/version pins and
strict parsing agree with those contracts. These engineering checks do not
make reused data prospective or remove reader/content confounding.

Accepted for one actual package; subsequent fresh responses must be sealed
and committed before one grade. Do not restart any completed stage or change
the source after packet construction. The separate design review addresses
the scientific interpretation, not a license to expand this task.

## Pre-packet wording amendment

Independent design review returned PASS without a blocking issue. Before any
packet construction, main corrected “12 reused pairs” to “12 new pairings of
24 reused texts” and added the global-reference and changed-reader caveats.
Pairing, allocation, prompt and all scientific settings are unchanged. Main
changed the code's protocol pin only; tests/source logic are otherwise exactly
as reviewed above. Final protocol SHA256 is
`25b632fc7afa5a291a007ba3bf12ac9036bda9a0c5c0a6e81705f62701950c3d`;
final source SHA256 is
`3b6148b835001fa9a088ad5f70c779a12d4993966bef3d98016bf1b8de0d3f2b`.
The previous hashes above record the original acceptance, not the final
execution payload. No scientific attempt or response was repeated.
