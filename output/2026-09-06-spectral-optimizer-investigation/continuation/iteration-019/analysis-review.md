# I19 independent saved-array analysis — main review

8 September 2026, before first actual scientific saved-array analysis. Main
read the complete independent NumPy auditor, its final diff and all 626 test
lines. The i17_analysis leaf authored only the auditor and tests, with no
producer/native/Torch import or random draws and no scientific-root access.

Final sources:

- audit_stochastic_tracking.py: 68915d8a4dc3c7811ad070645ab16029b84e51a6fbdae9b422135db29722680c
- test_audit_stochastic_tracking.py: e09fd9f758fb6e2226a65683655bb5dae820d56bf5466bbbf43fa88f4121282c

Main 77cefe/4caa6b: **20/20 deterministic CPU fixture tests pass in 6.319s**.
The fixtures independently construct algebraic streams rather than invoking
the producer. They cover every one of 56 saved arrays and every policy output,
all six process/rotation cells, first-observation equality, current-stage DEMA,
Kalman prior weights and covariance recurrences, native response transport,
unavailable alignment, exact envelope/source/hash closure and canonical
rebinding. A two-observation all-zero fixture covers the full 192-file roster;
separate aggregate fixtures cover 112 primary contrasts, 896 total contrasts,
472 means and 15,104 per-seed window metrics. Failed prefixes retain missing
seeds without survivor means. These are software tests, not scientific seeds.

Main source review confirms full-moment and represented rank-one covariance
relations, maximum residual accumulation across streams, explicit absent-basis
masks, paired rotations rather than additional replication, and stationary
theory separated from finite empirical risks. The registered strong-cell
prediction concerns late alignment only, not a performance conjunction.

The first actual audit must wait for terminal acquisition evidence, use a
separate analysis source freeze and exclusive analysis-001 directory, and run
CPU-only under 4GiB, zero additional swap, one core and 300s plus 5s stop bounds
with Restart=no. No repeated acquisition, observer replay, RNG or retry is
authorized by this review. Seven acquisition sources remain frozen at
840383aa5ae1c145f33c70d05793950f55b53862.

Concurrent commit 152214df577097ced9b0284b9601dbb67c842d3e tracked an earlier
auditor/checker draft at 12:21UTC. Main and the assigned leaf did not issue that
git write; its origin is unverified and is preserved. It is not the final
analysis freeze. No live acquisition source was altered by that checkpoint.
