# I17 archival collector source review

Status: PASS for source and synthetic checks; no real collection admitted yet.

Main read all402 lines of collect_gain.py and the complete test file. The
stdlib-only collector requires a passing, hash-pinned audit binding exact root,
phase completions, attempts and runtime inventory. It checks exact48 JSON
members, streams hashes, forbids symlinks/unexpected members/duplicate JSON
keys, writes deterministic bounded gzip, byte-compares decompression, and
rechecks provenance before the exclusive receipt. It does not load tensors,
delete source data or silently reuse a partial output directory.

Main synthetic suite944517:8/8PASS in0.163s with CUDA hidden and one thread.
Agent's separate8/8PASS in0.161s is implementation validation, not independent
scientific replication. No real I17 root/audit/collection exists or was touched.

collect_gain.py SHA256:
abf3641244978293dcac4347dd66405ddc389ab86cabc0b1352b91461d488ca4

test_collect_gain.py SHA256:
878133d8757084c24c8f82b7eaf7edc3287c6c0502adc6cf55224550442e671e

Actual tensor semantics are delegated to the prerequisite independent analysis;
collector fixture tensors are deliberately uninterpreted bytes. This review
does not admit acquisition, analysis or collection before their separate gates.
