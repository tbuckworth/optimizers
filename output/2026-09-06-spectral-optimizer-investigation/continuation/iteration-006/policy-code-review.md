# Parent independent policy review

6 September 2026. **PASS for the policy module and its synthetic tests.** This
does not certify the unfinished producer/summarizer, authorize a pilot, or
establish any learning effect. The parent did not author these two modules.

Read both files completely and independently ran:

```bash
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-006 -p test_policy.py
```

All 18 tests passed in 1.864 seconds, without a CUDA context or dataset access.
Source inspection agrees with the amended design: the previous native basis
is cloned before one canonical raw-gradient observation, activation is strictly
after 100 observations, and the actual assigned gradient receives mandatory
norm/direction checks regardless of optional logging.

The tests compare current delivery against both the unchanged helper and
canonical `filter_grad()` through 220 synthetic observations, including first
activation and repair boundaries. Separate adversarial in-place basis updates
distinguish current and previous candidates. All observing arms retain matching
warmup optimizer and observer histories; this policy-level synthetic check does
not replace the forthcoming full runner's model/RNG isolation pilot.

The numerical contract permits a finite restoration scale exceeding float32's
range by multiplying in float64 before one cast. It preserves failures for
undefined zero directions, nonfinite inputs, complete underflow and excessive
subnormal quantization. Scalar delivery preserves raw direction and is not
misclassified as a rank-32 output. Zero gradients still reach AdamW, so existing
moments and decay can move parameters. Baseline/warmup candidate measurements
are null rather than fictional basis observations.

Remaining integration checks belong to the full producer/summary reviews:
exact 36-cell identities, full NumPy/Python/Torch/CUDA witnesses, historical
source/data binding, test-access ordering, resource and artifact preservation,
and independent aggregation of the policy metadata. No material defect was
found in the reviewed policy modules themselves.

Reviewed SHA256 values:

- `policy_math.py`: `543e458f7d1471c5463850bfcf37e987b84d934bc1ce946cda14414b64631361`.
- `test_policy.py`: `2b5cc5f416dfb91450d8ab3151cde87e6f30860ff998a937e982fe8b50fe78ab`.
