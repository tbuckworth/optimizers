# Completed analysis and independent readout audit

9 September 2026. All scientific stages are now terminal; none should be
restarted. The failed admission-only attempt001 remains preserved separately.

## Successful analysis attempt002

Source repair reviewed/tested/committed as2bf90d5. Seventeen fixturesPASS and
actual archived-JSON-only admissionPASS before this separate execution.
Unit spectral-base-grokking-raw-direction-analysis-002.service, PID2967637,
invocation73952c75d7a94838b4b2f47c1258e761,14:43:23–14:43:49UTC.
Actual /usr/bin/python3.12 and intendedcgroup inspected live. Typeexec,
2GiB/no-swap/cpu.max100000/100000/5min/Restartno/KillModecontrol-group.
Command is the same fixed analyzer and --measurement-dir measurement-001,
--output-dir analysis-001 under the existing raw parent. Attempt001 never
created that output, so no partial/completed result was reused or overwritten.

Resultsuccess/MainPID0/inactive,15new rows,six fixed contrasts,25.8970sec,
20899867output bytes before completion. SHA256:
- completion da06e6dde3ec2188f6041a54b176bc8ce8e8ebb2efcbc4bf9f9d25f61a7d6f17;
- summary 23359c850a17c51d178b431594bafbdd40e57c2cc803267c69a066cd81352165;
- manifest 41462144decc6de4c6c7fd22e26419124711ea2205736aaf270a40aec4eb97f3.

## Independent new-array and paired audit

Unit spectral-base-grokking-raw-direction-readout-audit-001.service, PID2968349,
invocation21b1da4b01d54089b0a8d6e18c0a301e,14:44:32–14:45:31UTC.
Exact reviewed audit_readouts.py and all three main-supplied input completion
hashes were passed, as recorded in actualExecStart and the result. Interpreter
/usr/bin/python3.12 and intendedcgroup verifiedlive; actual16GiB/no-swap/
CPU100%/10min/Typeexec/Restartno/KillModecontrol-group. Sources remained the
reviewed1b6d832 audit snapshot, unchanged by the separate analyzer repair.

PASS, no errors,24137savedchecks (journal includes one final output check),
59.1698sec,213266432peakRSSbytes. ResultSHA
0e6b098a50b12c5112c1fa3ba6cd1a94b6c7c3a52c92211997de883517f85c04.
All15newNPZ readouts and six14-metric paired comparisons independently checked.
Maximum probe-score absolute difference9.11e−15; paired arithmetic0. Some
unnormalized symmetry sums differ1.49e−8 within their relative tolerance;
this is not the error of a normalized endpoint metric. No old inference or
archived-array re-audit, training replay, or new random seeds.

Both audit results, complete summary and manifests were copied byte-identically
into results/ by the reviewed packaging-only builder. PackagingmanifestSHA
468459ba4a915cace2cdda04e4f70430b5ca678b14ebdaf267fe109368825d1e.
Main visually inspected the primary plot: allfive points and storedmean±SE,
bothendpoints and favorable directions visible. Independent claim/algebra review
PASS in raw-direction-interpretation-review.md. Report and current working paper
have new offline readers; old readers are preserved, not overwritten.

No new experiment is launched by these results. Next is the three-action
functional-response design described in the report and independent review.
Goal/timer stayactive; paid spend/reservations0/100USD.
