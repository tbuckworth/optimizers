# I15 mathematical interpretation

The independent audit passed with 18/18 complete confirmation branches, no
numerical failure, 4,357,895 checks, 395 file hashes and 30 complete-state tree
digests. Results below are three-seed means with all seed values retained in
the [audited summary](analysis-001/summary.json). I15 identifies conditional
policy effects on this MNIST/SGDm continuation; it does not identify a semantic
subspace or a mediation fraction.

## Registered contrasts

The carried-history prediction was target-dependent. Under **clean** training,
$H=U(\text{current,native})-U(\text{current,projected})$ was positive in all
three seeds: mean clean-CE utility $+0.0330$ and accuracy $+0.59$ percentage
points at h2000. Yet both current cells deteriorated from h100. Native current
changed by $+0.1036$ CE and $-1.21$ points accuracy, while projected current
changed by $+0.1366$ CE and $-1.81$ points. Thus clean $H$ is a small relative
preservation effect, not continued useful learning. Under **fixed** corruption,
$H$ was $-0.00218$ for CE in all seeds and $+1.22$ points accuracy in mean with
mixed seed signs. The leading fixed-target carried-history prediction therefore
did not replicate across metrics.

The mean-restoration prediction was much stronger. $M=U(\text{mean,projected})-
U(\text{current,projected})$ was positive in every seed for both metrics and
targets. Its means were $+0.2056$ CE utility and $+4.71$ accuracy points under
clean training, and $+0.0877$ and $+12.17$ points under fixed training. This was
not merely preservation: mean/projected progressed from h100 by $-0.0691$ CE
and $+2.91$ accuracy points on clean targets, and by $-0.1054$ CE and $+10.17$
points on fixed targets. Fixed-label realization fitting also increased, but
moderately: training fixed accuracy rose $2.61$ points and
$R_\zeta=L_{\rm fixed}-L_{\rm soft}$ fell by $0.0327$.

The registered substitution interaction was not general. Clean-target $S$ was
negative in every seed ($-0.0980$ CE utility, $-2.61$ accuracy points): mean was
more beneficial with native history. Fixed-target endpoint $S$ was positive in
every seed ($+0.1943$, $+28.60$ points), but the complete strongest pattern was
absent because fixed $H$ was not consistently positive. Moreover, under
validation-selected stopping, fixed $S$ was negative for CE and positive for
accuracy. This is a large target- and metric-dependent interaction, not a
single substitution coefficient.

Endpoint levels make the constructive case concrete. Clean mean/native reached
CE $0.2218$ and accuracy $93.51\%$, essentially raw's $0.2205$ and $93.48\%$;
mean/projected reached $0.3528$ and $90.31\%$. Under fixed corruption,
mean/projected was best at the endpoint among these five policies at CE
$1.9848$ and accuracy $53.63\%$, versus current/native $2.0747/42.68\%$, raw
$2.2294/23.96\%$, and mean/native $2.1812/26.25\%$. Under hindsight validation
selection, mean/projected had the highest mean fixed-target accuracy
($59.77\%$ under the maximum-validation-accuracy selector), while raw had the
better mean CE under the distinct minimum-validation-CE selector ($1.8843$
versus $1.9603$). These are three-seed mean levels, not claims of an all-seed
win. Clean selected contrasts retained positive $M$ but only a very small $H$.

## Geometry and the strongest valid mechanism account

The intervention acted as intended. Current/projected reduced data-step
current-action complement fractions to about $2.5\times10^{-11}$ for both
targets, from native-current $0.00788$ clean and $0.01926$ fixed. Mean/native
fractions were $0.1137/0.0992$ (clean/fixed), versus only
$0.00260/0.00141$ for mean/projected. Native-mean buffers likewise retained
substantial complement energy, whereas history projection left principally the
newly delivered mean complement.

The best positive account is therefore conditional: the EMA mean carries
temporally smoothed information that current rank-32 delivery omits. With clean
labels, native momentum accumulation makes that information useful enough to
recover raw-like learning. With fixed labels, the same accumulation can amplify
realization-specific as well as softened-objective information; projecting old
history while admitting only the new mean complement yields a better
adaptation–memorization balance here. This explains the joint clean/fixed
pattern more naturally than “all outside energy is noise.” It remains a
hypothesis because the experiment changes the total history policy, not the
semantic content of the complement.

Amplitude cannot be separated from orientation. Under fixed labels, native
mean's mean buffer energy was $5.25$ versus $1.14$ for mean/projected, alongside
far greater realization fitting. But under clean labels the ordering reversed:
$1.41$ native versus $2.36$ projected, even though native mean learned better.
The verified rotating-subspace calculation (artifact not distributed in this public snapshot) explains
why projecting every old buffer can produce a larger between-policy steady
buffer by changing phase and cancellation. It does not establish that this
caused either observed ordering.

Signed dots are also non-semantic. Mean-complement/data-step and removed-history/
data-step dots were much more negative for native mean, especially under fixed
labels, but these are training-history-derived components, not gradients of
auxiliary clean population utility. The raw-gradient dot alone uses the current
training minibatch. Neither establishes clean descent; together with norms they
describe delivered-step alignment and amplitude on each endogenous path.

Finally, the local numerical records support fidelity to the intended
recurrence: maximum basis orthogonality error was $1.49\times10^{-6}$,
action-idempotence defect $1.75\times10^{-6}$, maximum separately recorded
idempotence roundoff reference $1.16\times10^{-4}$, ideal-data-step defect
$6.22\times10^{-7}$, and manual-decay defect $4.51\times10^{-7}$. Small local
algebra errors do not bound accumulated nonlinear utility sensitivity, prove
long-run numerical robustness, or establish causality. Nor do they turn the
moving native action into a fixed oracle projector. The strongest conclusion
is that explicit mean restoration supports continued learning, while momentum
history controls its target-dependent gain and realization cost. Clean
direction, universal SGDm mediation, and transfer beyond this three-seed
setting remain unestablished.
