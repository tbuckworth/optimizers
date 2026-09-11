# Main review: independent I19 report arithmetic

8 September 2026, before the first actual report corroboration. Main read
all398 checker lines and412 test lines. The i17_analysis leaf authored these
independently without reading accepted outcome JSON or scientific arrays.

- check_stochastic_report.py:8e7ec1dbd6087a332375071d7b7b2c1d7190e16f5c6044f369f199e9878064ff
- test_stochastic_report.py:7ee59f1445be52ac67c07708b33bb1722d5af13bba777b421c55e91226872898

Main04b970/d16a0e:13/13 deterministic CPU fixture tests PASS6.151s. This
includes all15,104 metrics,472 means,896 contrasts and112 primaries; exact
7+2+2 acquisition/analysis/report source closures and commit checks; all192
file-roster synthetic fixture admission before array loading; corruption,
missingness, scalar/sign/label changes; exclusive output and strict JSON/NPZ.
The duplicate-ZIP-member warning is intentional in a rejection test.

Only saved g/s/output are deserialized. Error vectors/decompositions and all
four-window risks are independently calculated, then scalar fsum statistics
check means, paired differences, SEs and exact signs. Fixed tolerance is
5e-13*max(1,abs(expected)); a subtraction residual additionally scales by its
MSE components. It is not a repeat of the full observer-state audit, a random
replay or an additional scientific seed. Source/input provenance is verified
before and after the read. No tolerance was chosen from actual discrepancies.

Main approves one first report check after a separate source freeze, exact
accepted summary135502cc7fc6a69bba7c16fd0ddcf53d10255a78065c0d2198fbe156fb76ba9f
and auditc549fc5d10dec4d644c857821e6e1e67a26ad6469e821f0d7cae4588671b28f2.
Use exclusive report-check-001, CPU1,4GiB,zeroSwap,280s cooperative/300s hard,
5s stop,Restartno and hiddenCUDA/thread1. Never rerun a consumed handle.
