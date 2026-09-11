# Pattern-calibration collector — source-only freeze

This checkpoint performs **no article requests, acquisition attempt, selection,
tokenization, model work or reader calls**. Root owns any later admission.

## Sources and exclusion scope

- New [collect.py](collect.py), SHA256 `5f4f764df9ff1b4483cd3918b5b7565418e4cb83ea01d6a52698d48a39df0a83`.
- New [test_collect.py](test_collect.py), SHA256 `d21f4ded2972f2429090e6c99e41f40c6e1d67d1cdcc611dff14fc89ed6e31d0`.
- old-prefixes.json (artifact not distributed in this public snapshot), SHA256 `fd66b415caca0b563425b77e2ce0ee62c3c625849e6c9829c2b64153e062f6f7`: 296 exact strings/296 source-bound rows across nine saved input rosters. Each row retains source index, original row index and ID. Completion fields, decoded references and the upstream lens-fitting corpus are excluded. The older 72-row pilot is included conservatively. Reused panels are listed only once; no newly authored exclusions or linguistic filtering.
- All 32 original direction-fit prefixes are explicitly present in that union. The inventory includes the previous 24 and 32 Wikipedia accepted prefixes; previously requested but rejected article IDs remain excluded separately by the frozen 62-ID inventory and manifest, not by reserving their rejected text.
- Exact main protocol SHA256 `4d80a9eef24e32a475c1d5016f57ca05419cacb45f2b27504f9ca7f916fae762`; frozen manifest `8287106e8e3238ccf18d0c60bc0a4b59de894c7a8fd038c6a870a2cd1e94ed63`; selection receipt `cbd406b04ce5c776c95091331cc8064cf16d38f7d50c5ca137b07e667c15ea78`. These are read from MAIN `output/2026-09-11-jlens-pattern-calibration/`; no planner entrypoint is invoked.
- Existing inventory.json (artifact not distributed in this public snapshot), SHA256 `e57b194d6ed0b6b0999097d64c12dc37198f7b73d9b27bb5c2dd31e51a54f7aa`, and planner source SHA256 `8d97d6b8412a6dd2f21a1532713d966603debc87e2df5f240f8479726eb35b5b` bind the already-reviewed prospective roles. All seven executable constants ending in `_SHA` were independently checked against their actual files at this checkpoint.

The full old MAIN `output/2026-09-11-jlens-natural-addon/collect.py` and tests were read; its SHA256 is `decdb21a2d76b0a2d75f7e4f0753ae76570b5c75e9aac12aa8e2002743001ee8`. The new source copies the no-redirect transport structure, not the old pairing/stage entrypoints. At runtime it hash-checks and imports only the inert original `output/2026-09-11-jlens-independent-content/collect_text.py` (SHA256 `337e9a208f491b22ec0deb3a591b1216669eaf42db4712e43f8e1bc97e98d347`) to call strict JSON decoding and pure article eligibility. That helper was read in full. Its old `Stage`, fetch, catalogue and excerpts functions are never called.

## Contract

The new exclusive worker `excerpts/` holds separate `calibration-dataset.json`/`calibration-pairs.json` and `evaluation-dataset.json`/`evaluation-pairs.json`, IDs C01–C32 and T01–T16, plus attribution, every candidate disposition, exact requests/responses, per-side eligibility, visited-pair decisions, bindings and terminal receipt. There is no new candidate ordering or role reassignment. Both members of a technically rejected pair are requested; integrity/API/transport failures stop immediately. Only accepted whole pairs reserve strings, across both roles and the old-prefix union. Capacity skips issue no requests.

Limits are 80 requested pairs, 160 serial article GETs, 2 MiB per response, and 32 MiB across all new stage artifacts with a 256 KiB failure reserve. Oversized bodies retain the capped byte prefix with an explicit incompleteness flag; no retry. Success audits every artifact hash/byte count; failure inventories actual files, including a partial interrupted write. No software can guarantee writing a receipt after physical disk failure, but failed files/attempts are never erased or overwritten.

The best-practices validation used installed Python standard-library source (no network in this task) and root's current official documentation check: a socket timeout is not an end-to-end deadline. The transport therefore adds a Linux SIGALRM/ITIMER_REAL 20-second wall deadline around opening and reading each response, restores the old handler and inactive timer, and refuses an already-active process timer without changing it. The admitted process must be serial/main-thread Linux. Root must also enforce the protocol's external 3,600-second, 256 MiB/no-swap process limits. Revision-only continuation is retained, never followed; redirects, retries, credentials and identity rotation are absent.

## Fabricated validation and recommendation

`env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 60s /usr/bin/python3 test_collect.py`: **18 tests PASS in 0.219 seconds**. Only helper source and temporary fabricated fixtures are read; the tests never load the actual manifest/datasets or contact a transport. Coverage includes immutable roles/IDs, exact 96-row split and all 86 dispositions, whole-pair rejection, no survivor repair, cross-role/old/within-pair duplicates, unreserved rejected text, Unicode word extraction, integrity/technical failures, request/time/output caps, raw error preservation, continuation, mocked deadline and restoration, exclusive/dangling stage paths, source joins and drift, and partial-file failure accounting.

Recommendation: source is ready for root's final exact-byte review and one separately admitted collection. No actual collector call has been made; no measurement implementation is added by this checkpoint.
