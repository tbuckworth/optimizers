# Parent review of the development/resource pilot

The single approved pilot completed successfully, with no retries or source
changes: **2026-09-06T13:05:34.488496 to13:05:48.845153 UTC**, elapsed
**14.35667285695672 seconds**. All19 bound source files match prospective commit
`de1ba26d43a6df80162ee9530d8b64c9ac08acf3`.

The source freeze was created by the workspace's review-session hook while the
parent's freeze command was pending, as explicitly recorded in its commit body.
The parent verified all19 committed bytes afterward and before launch. Its
separate pilot decision supplied the execution authority; an automatic commit
does not authorize a run. An earlier freeze command stopped before staging
because the summary audit file was still being written; this was not a failed
pilot or a source revision.

## Runtime and invariants

Two220-step traces on development seed9878 completed. All220 post-step parameter
hashes agree bitwise between uninstrumented and instrumented traces. The
source-bound pilot reports exact complete core-state equality, including NumPy
state, and all prescribed observation/probe/numerical gates pass. No learning,
accuracy, retention or fidelity outcomes were retained or used for choices.

| Measurement | Seconds |
|---|---:|
| Uninstrumented steady-step mean,129..220 | .002318184645644025 |
| Instrumented steady-step mean,129..220 | .007337000990367454 |
| Instrumented repair/probe step200 | .0926438863389194 |
| Worst-size weighted Gram | .7911789617501199 |
| Worst-size CPU eigensolve | 1.0233366978354752 |
| Worst-size mapping/checks | .08342467108741403 |
| Two full synthetic streams write | 6.400986318010837 |
| Two full synthetic streams hash | 1.5819452041760087 |

The worst-size check uses the prescribed synthetic seed9879, p50,890 and2,000
input rows. It passes all reference residual/spectral/orthogonality gates. It
does not establish conditioning of the later MNIST reference or scientific
covariance performance.

Peak host RSS is3,233,677,312 bytes; peak allocated GPU memory is2,004,121,600
bytes and reserved memory is2,527,068,160 bytes. These are below the prospective
12 GiB RSS and8 GiB allocation caps. The bulk root is the exclusively created
`/tmp/spectral-experiment-artifacts/spectral-iteration005-pilot-i4wtjcof`, on the verified large
volume. Two407,120,128-byte float32 arrays have2,000 completed rows each and
matching recorded hashes, as expected because the synthetic raw/innovation
write-path fixture deliberately writes identical values. No old files were
removed and the synthetic arrays are retained.

## Extrapolation and limits

The instrumented mean implies about44 seconds for6,000 replay updates. Charging
all12 references the worst-size Gram/solve/map time adds about23 seconds;
tripling the two-stream write/hash cost adds about24 seconds. These approximately
91 seconds exclude repeated verification hashes, full snapshot serialization,
historical scalar checks and other overhead. A few-minute run appears feasible
under the fixed900-second cap, but this is an extrapolation, not a runtime
guarantee. All three streams remain resident as memory maps, so full-run RSS
can exceed the pilot's single-stream-pair peak; the guard remains active.

The independent pilot audit and fresh occupancy/headroom checks are still
required before a separately recorded confirmatory GO. No full replay is
authorized by this review alone. Source files remain unchanged after the pilot.
