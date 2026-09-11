# I17 gain-control runner review

Prospective review, 8 September 2026. I authored `gain_core.py` and `test_gain_core.py`, while independently reviewing the main-authored runner and runner tests against the protocol; main separately read the core and its tests. This is therefore not an independent audit of my own core. The review used source and synthetic CPU fixtures only. It did not call the real `bound_inputs`, load an old scientific tensor, inspect an I17 outcome, use a GPU, or execute an acquisition.

## Verdict

**PASS for the bounded static runner and synthetic smoke-admission review, with no remaining material source blocker found.** This is not blanket approval of the acquisition: independent analysis-source review and the physical resource/service checks remain pending. The one material admission gap found during review was corrected before this verdict: `verify_smoke` originally hashed the eight terminal state files but did not bind them to the branch checkpoint records. It now requires each branch to contain exactly one h110 full-state checkpoint whose manifest record is the declared `state-<id>-h110.pt`, with a 64-hex state digest. A synthetic omission regression rejects the formerly admissible case.

Reviewed source pins:

| File | SHA-256 |
|---|---|
| `gain_core.py` | `ed05340bafcf8a0d2f7932a80c5085f212209fb7d8d0593ab5e2c38c6ea64ee7` |
| `test_gain_core.py` | `5fe3fbfbb4678c4a137024d6c8a0052a35544ebcbc842512968545edc0eff6ba` |
| `run_gain_controls.py` | `2acae37b7c83f78df6a1fd96bb01ab36f12d3edc85332f5445e23ddab2985b1a` |
| `test_gain_runner.py` | `c6810afcaf0f2293b3bf78aff95cc87d4b567c2d6dbef8ebf8ae9c0dbb13be63` |
| `protocol.md` | `aa15ff9bbdbe74d38e0a4c07e4152076f85011b9b8997113520dcc2a535f79db` |
| `predictions.md` | `a1562f60d412b9e2c516dbf8668fa2f323b8ac772828e09f02f11f7a9c5ab73c` |
| `moving-gain-caveat.md` | `f606e9b9a64ca7ef68fb9f7b2d5a05453976c29ee90e5f9ff717405ed677a55a` |

## Substantive checks

- The scientific roster is exact: 24 new branches are three normalized scalar policies plus normalized spectral mean/projected history over six seed-target parents. Scalar k0 is rejected from real acquisition and reused as exactly six original I16 references.
- The I16 reference seam is direct rather than summary-only. The runner pins the accepted audit, summary, zero-difference report corroboration, lossless collection and confirmation completion; selects exactly the six k0 branch JSON records; verifies their I14 parent/evaluation digests and full six-point curves; and binds every declared checkpoint record through its artifact hash. It performs no checkpoint-forward or training replay.
- Before the first confirmation update, all six I14 h100 parent states are loaded, digest-checked, restored, neutrally evaluated against the I14 and I16 h100 references, and checked for complete-state and RNG neutrality. Every branch repeats the complete-state restore and evaluation seam, while the original parent object is checked unchanged after execution.
- The implemented step matches the registered recurrence. It ingests the raw gradient once, retains the unnormalized momentum buffer, uses the exact I15 `native + (mu - A(mu))` spectral delivery convention, applies the actual native action without basis repair, performs common manual `.9997` shrinkage, and then applies the normalized data delivery. First-step raw-gradient, post-observer and old-buffer digests are paired across the four new policies.
- Typed numerical failures seal and retain a branch with explicit missing endpoints; structural, source, seam, resource and serialization errors propagate and abort the phase. The completion representation does not silently pool survivors. Duplicate branch artifacts and consumed attempt files prevent replay; `Restart=no` remains an external service requirement.
- Smoke admission validates the attempt/source binding, exact 20-artifact roster plus terminal completion, regular nonsymlink artifact hashes, manifest and eight-entry branch-index relationships, first-step pairs, exact 200 warmup plus 80 new updates, terminal checkpoint associations, and zero numerical failures. Its forecast uses the slowest policy-specific rate, not a pooled rate, and must fit the 1,800-second confirmation cap.
- The source manifest extends the frozen I16 transitive closure with the I17 protocol, predictions, core, runner and both test files. The runner itself enforces the exclusive `spectral-i17-001.` root on `/dev/RECONFIGURE_FOR_LOCAL_STORAGE`, declared runtime directory, per-phase attempt created with exclusive mode, cooperative wall limits and inherited 3 GiB shared artifact cap. Host-memory, zero-swap, GPU-memory, CPU-quota and `Restart=no` enforcement properly remain properties to verify in the external launch unit and receipt.

## Test evidence

With CUDA hidden and OMP, MKL and OpenBLAS limited to one thread, the combined synthetic suite passed **18/18** in **1.989 s**. The core tests include three consecutive k0 steps with exact I16 full-snapshot parity, explicitly covering gradients, observer state and RNG; scalar buffer updates against native PyTorch SGD multiply-then-add ordering; optimizer-group preservation; spectral recurrence parity with I15; diagnostics; and typed versus structural failures. Runner tests cover all eight smoke branches, seam and membership rejection, failure retention, duplicate rejection, slowest-policy forecasting, complete smoke admission, artifact corruption, source/attempt mismatch, and checkpoint-association omission.

## Scientific interpretation limits

The experiment is a useful test of whether directional temporal routing adds value after a specific conditional stationary-gain normalization. It is not a fully gain-matched dynamical comparison. I independently checked the exact counterexample linked by the amended protocol: with constant `g=mu=(1,1)`, `rho=.9`, and alternating `A1=diag(1,0)`, `A2=diag(0,1)`, the two-cycle is `b1=(1.9,1)`, `b2=(1,1.9)` and `d1=(.19,1)`, `d2=(1,.19)`. Its mean is `(1-.9^2/2)g=.595g=119g/200`, despite unit DC gain for either frozen projector. Thus a result can reflect action rotation, inherited transients, finite-time response, update energy/path or nonlinear trajectory feedback as well as spatial routing. The note's constant-preserving recurrence is a future comparator, not an admitted I17 arm.

The panel is also adaptively reused, and joint scalar selection has 24 `(k,h)` choices versus six spectral horizons. Validation selection remains legitimate for the registered comparisons, but it is not an equal whole-program tuning budget or fresh-task confirmation. I8 and I15 constructive results must remain in the synthesis irrespective of I17's signs.
