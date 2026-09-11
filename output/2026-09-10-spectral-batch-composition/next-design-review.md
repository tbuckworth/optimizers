# Main review of the next local discriminator

Main readnext-discriminator.md (artifact not distributed in this public snapshot) completely after the
batching report. DesignSHA`2fbb538f4da88680322d64ec6506e16545fbd8a4bd7e068d9cb0599af6ed21fd`.
Selected for source preparation, not launched and not an endpoint efficacy test.

The design cleanly holds model/Adam and action input common while changing
observer histories. The gradient stream itself still uses assigned labels;
there is no separate correctness indicator. It tests the full native observer
channel, including mean, EMA weighting, order/truncation and scalar gain, not
pure population covariance or a norm-matched directional intervention.

Primary canonical self-inclusion is explicit (observer150→151), and the
one copied Adam step is separately100→101. Earlier pre-inclusion geometry is
descriptive only. The symmetric fixed block-mean action input is not a usual
next minibatch, and class-informed probe gradients never enter delivery.
Raw and zero references make absolute helpfulness visible alongside paired
contrasts. A shared zero reference requires identical full parent states and
after-logit hashes across the cell labels; label-derived metrics remain distinct.

Reuse of outcome-informed parents is declared. A local positive does not
explain the long-run result; a null need not persist at later states. The
next implementation must enumerate its sub1GiB inventory and independent
saved-state arithmetic checks before any once-only acquisition. It must
not force old snapshot validators to pretend observer and Adam clocks agree.
Frozen scientific parents and all completed training remain immutable.

No new GPU/cloud job, reservation, source implementation or scientific data
read was made for this review. Next action is bounded source and fabricated
fixture preparation, within the existing autonomous user authorization.
