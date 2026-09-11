# I18 report-arithmetic source review

Status: PASS for first bounded saved-array report check, 8 September 2026.

Main read the complete final NumPy/stdlib implementation and synthetic tests.
Source was prepared by current_report without reading scientific arrays; main
has already read the independently accepted summary to write the report.
This is an independent arithmetic corroboration, not outcome-blind reporting.

- Source SHA256: `7eb69225009c3c805890e2be533ebf4eab3d6f8f61b1dbdb324453980a53eeca`.
- Tests SHA256: `62154d3b69bd480e6d01211fa0ba581e9c52b9e34cdda225655c4c6b3c700615`.
- Main test invocation233fc8: 4/4 PASS,0.005s, CUDA hidden and numerical threads1.
- Main caught and corrected a draft fixture's standard-error arithmetic before
  admission; final fixture uses sqrt(7/3). Added duplicate membership checks and
  a genuine temporary NPZ loader fixture before actual input access.

The checker binds the accepted summary, primary audit, plot manifest, terminal
records and every raw NPZ/metadata hash. It loads only saved output/s arrays,
recomputes all registered per-seed MSEs, seed means/SEs and both paired contrast
sets/signs, checks 23 plot points and 228 unique CSV rows. Alignment plot points
are checked against the accepted primary summary, not independently derived
from alignment arrays here. The full primary audit already checked those.
Whole-horizon results are derived separately from the same raw arrays and
explicitly labelled post-hoc, not a changed primary endpoint. No RNG, native
observer, torch, optimizer or experiment replay is imported or executed.

First-run admission: exclusive report-check-001 absent (f09d5d); no loaded I18
service (6e73d0). Resource checks50a0d6/18ef1c show688GiB mounted-disk space and
48,244MiB available RAM. Existing unrelated swap use is untouched. Prospective
unit: spectral-base-i18-report-001.service, CPU1,2GiB,zero swap,120s hard cap,
5s stop timeout,Restart=no,CUDA hidden. No paid spend or new scientific draw.
Freeze these sources before the first actual invocation; record its handle and
terminal output. A completed report check must not be restarted automatically.
