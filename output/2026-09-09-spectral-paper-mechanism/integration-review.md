# Working-paper integration and reader check

9 September 2026, 13:33 UTC. Main-agent review; not submission-readiness approval.

## Offline reader checks

The existing reader builder packaged the working draft without modifying
previously emailed or hash-bound reports. A separate structural check passed:

- Source SHA256:
  `28bcd851c7bde98eebd4997c8ad50ef7f6cba7fa886275fb8353088c91dc2f4b`.
- HTML SHA256:
  `2f8a4b491b872b71361c3c2d0a40d77632543c98f254d5f37541ce12a3d6a21b`.
- 345,476 bytes, two embedded PNGs with nonempty alternative text; decoded
  image hashes agree with the build receipt.
- Four native-Unicode mathematical blocks, including the gain-square,
  inequality and retained-norm identities; no script/remote-math dependency.
- All internal navigation and local evidence inventory links resolve. The
  inventory names local evidence; it is not a public download service.
- Every new Python source parses. Six trajectory-analysis synthetic fixtures
  pass; knowledge lint passes for14 pages/11 indexed content pages;
  `git diff --check` passes.