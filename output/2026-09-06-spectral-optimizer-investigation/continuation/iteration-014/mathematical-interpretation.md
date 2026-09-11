# I14 mathematical interpretation

This interpretation uses the three-seed [analysis summary](analysis-001/summary.json),
not recalled chat values. The original audit remains failed because an unlisted
but empty Torch runtime directory existed. The accepted
runtime supplement (artifact not distributed in this public snapshot) rehashed
all 502 declared artifacts (1.198 GB) and bound that sole empty-directory
exception; it inherited rather than reran the original scalar/state semantic
checks. This is therefore accepted multi-seed evidence with that provenance
qualification.

## Registered predictions

The prospective predictions (artifact not distributed in this public snapshot) were meaningfully
discriminated.

1. **Fixed corruption was more favorable than clean training, but not across
   every base.** All six clean-target endpoint effects were adverse. Fixed-target
   SGDm favored filtering in all seeds (mean clean-CE benefit $+0.155$, accuracy
   $+18.7$ percentage points); AdamW did too ($+0.091$, $+23.6$ points). Plain
   SGD instead favored raw in every fixed-target seed (CE $-0.218$, accuracy
   $-11.7$ points). Thus the
   relative fixed-versus-clean prediction succeeded, while the strongest
   optimizer-independent projection prediction failed.
2. **The predicted geometric ordering was sharp.** Over filtering-active steps
   101–2000, current32 data-step outside-energy fractions were approximately
   $4\times10^{-12}$–$6\times10^{-12}$ for SGD, 0.8%–1.9% for SGDm,
   and 59%–78% for AdamW (clean–fixed range). Full-trajectory SGD ratios near
   0.8% include the deliberately unfiltered warmup and must not be used to
   reject confinement. Total-step leakage is larger because decay is not a
   projected data step.
3. **Clean restriction was worst under SGD.** From h100 to h2000, filtered SGD
   lost 0.709 auxiliary CE and 22.5 percentage points of accuracy, filtered SGDm
   lost 0.104 CE and 1.2 points, whereas filtered AdamW still gained 0.094 CE
   and 1.2 points.
   All remain worse than their raw clean endpoints, so the registered “strong
   filtered clean gains” falsifier did not occur.
4. **Fixed endpoint gains mostly exceeded selected-stop gains.** SGDm's positive
   endpoint effects became adverse in mean under both validation selectors;
   this does not assert that every selected seed effect was adverse. AdamW's
   selected CE effects were adverse and its selected accuracy effects only
   $+1.3$ to $+3.9$ percentage points. This supports preservation against late raw
   deterioration rather than an across-the-board learning-rate acceleration.

The realization diagnostics agree. Under fixed labels, SGDm raw reached 42.9%
training-label accuracy, $R_\zeta=-1.166$, and confidence 0.374; current32 gave
15.1%, $-0.022$, and 0.142. AdamW changed similarly (45.7% to 17.7% fixed
accuracy; $R_\zeta=-1.039$ to $-0.052$). Plain SGD raw fitted less of the
realization already (21.7%, $-0.125$, confidence 0.173), and filtering reduced
that further without improving clean utility. SGDm's favorable contrast is
therefore partly a filter preventing a failure mode that its raw baseline
expresses strongly. Filtered SGDm itself made only a small post-h100 CE gain
and lost 0.8 percentage points of accuracy; filtered AdamW made modest gains on
both metrics.
The raw opportunity differed sharply: SGD fixed-label auxiliary accuracy rose
from 35.6% to 51.4% after h100, whereas SGDm fell from 43.5% to 24.0% and AdamW
from 48.7% to 28.6% while their $R_\zeta$ values became strongly negative.
The sign reversal is therefore not solely evidence about projected-step leakage.

## What the SGD–SGDm split means

It establishes a useful conditional non-necessity result: AdamW's coordinatewise
second moment is not necessary for the measured fixed-label endpoint benefit.
SGDm has no second-moment state, yet its fixed effects are positive in every seed
and metric.

A constructive explanation is available. Let $P_t$ be the current retained
action and $Q_t=I-P_t$. For current32 SGDm,

\[
 b_t=\rho b_{t-1}+P_tg_t,\qquad Q_tb_t=\rho Q_tb_{t-1}.
\]

Plain SGD permanently discards $Q_tg_t$ at that update. Momentum instead
carries previously admitted directions across time; because $P_t$ rotates,
a direction admitted yesterday can lie outside today's range. Its DC gain also
integrates persistent projected signal. This yields a plausible middle regime:
enough temporal freedom to retain useful low-frequency motion, but much less
freedom than AdamW's diagonal transform to fit the fixed realization. The small
1%–2% leakage energy need not be negligible if its direction has unusually high
clean utility. This is a steelman, not a mediation estimate: leakage energy has
no semantic sign.

The comparison does **not** isolate momentum causality. Calibration selected
$\eta=.1$ for SGD and $.03$ for SGDm, giving different decay factors, finite-step
scales, and raw fixed-label failure modes; SGDm was also selected at its grid
boundary. The buffer changes temporal smoothing and effective gain as well as
outside-span motion. There is also an exact constant-state distinction. With
gradient $g$ and coefficient $\lambda$, SGD stationarity requires
$g=-\lambda\theta$. For unnormalized momentum,
$b=g/(1-\rho)$ and stationarity requires $b=-\lambda\theta$, hence
$g=-(1-\rho)\lambda\theta$. At $\rho=.9$, the same coefficient gives one tenth
the gradient-level regularization at an ideal fixed point. This identity does
not describe noisy, moving nonlinear training, but it can help explain SGDm's
greater raw realization fitting. Consequently, “momentum can coexist with
success” is supported; “momentum caused the success” is not.

## Single next discriminator

Use the saved, exactly paired SGDm h100 states in one prospective 2×2 branch
test: delivery `{current32, mean32}` × history handling `{native, projected-history}`,
with the same η=.03, decay, batches, labels, observer order, seeds, and horizon.
The intervention changes only the carried term:
$b_t=\rho P_tb_{t-1}+h_t$, instead of $b_t=\rho b_{t-1}+h_t$.
It must not project the whole updated buffer: that would erase mean32's newly
delivered $Q_t\mu_t$ and make the factorial uninformative.
Reuse the completed native-current32 I14 trajectories as one factorial cell;
do not rerun them. Only `{current32/projected-history, mean32/native,
mean32/projected-history}` are new: three arms × two targets × three seeds = 18
new 1,900-update continuations, using the original steps 101–2000 batches.
Existing raw I14 trajectories remain an additional reference. This design is
proposed, not yet frozen or executed.

This separates an outside-span memory bridge from explicit mean restoration.
All four factorial cells share the same SGDm rate and decay, holding the cross-base
fixed-point regularization difference constant.
If native-current32 succeeds but projected history removes the benefit, while
mean32 rescues the history-projected arm, the strongest explanation is that SGDm was
supplying a useful historical complement that mean preservation can replace.
If confinement leaves the benefit intact, outside-span leakage is not the key;
within-span temporal averaging or the raw-baseline dynamics become stronger
accounts. Predeclare endpoint and selected CE/accuracy, realization fitting,
and post-h100 absolute progress; do not select among them afterward.
