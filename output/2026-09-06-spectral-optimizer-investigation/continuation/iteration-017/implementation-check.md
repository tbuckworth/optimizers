# I17 implementation contract and primary documentation

8 September 2026. Design-time check, not acquisition approval or results audit.

The local framework is PyTorch2.11. I17 retains the initialized SGDm state
contract: momentum buffer excludes the learning rate, and the data update
uses a separately specified delivery vector. This matches the documented
distinction between PyTorch's momentum convention and rate-in-buffer variants.
The special first-buffer initialization rule does not apply to the restored
h100 states. See [PyTorch SGD](https://docs.pytorch.org/docs/2.11/generated/torch.optim.SGD.html).

Manual parameter and momentum mutations use
[no_grad](https://docs.pytorch.org/docs/2.11/generated/torch.no_grad.html)
only around those mutations; the loss and backward pass remain grad-enabled.
This avoids recording the optimizer update in the autograd graph. The same
canonical observer ingests the single raw gradient; no extra backward pass is
needed for normalization.

Required checks before source freeze: initialized-buffer parity against native
SGD using multiply-then-add ordering; exact k0 parity with I16; unchanged
optimizer-group hyperparameters; full synthetic state/RNG seams; no accidental
storage of normalized delivery as the recurrent buffer; moving/nonorthogonal
action defects logged without basis repair; all four real string policies
accepted and test-only k0 rejected by the runner. The fixed-point theory is an
own conditional derivation, not a guarantee supplied by these library docs.

The strongest design limitation remains informative rather than repaired away:
the same adaptive panel and unequal scalar/spectral selection opportunities.
All eight primary effects, both targets and every seed remain mandatory. A
future fresh-panel confirmation is a separate prospective experiment.

The implementation validator skill guided the API check; no GPU or scientific
training is admitted by this note. Main still needs to review all synthetic
tests, independent analysis and the exact frozen source closure.
