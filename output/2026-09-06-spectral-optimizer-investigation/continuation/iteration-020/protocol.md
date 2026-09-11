# I20: saved-stream direction-estimator × response discriminator

8 September 2026. New **exploratory, outcome-informed counterfactual** on all
completed I19 inputs; not fresh replication, a rerun, or a replacement for I19.
No RNG, native observer, optimizer step, forward/backward pass or GPU is used.
The parent files and all delivered reports remain immutable.

## Scientific question and controls

I19 demonstrates directional-oracle headroom but inadequate native whole/late
risk. Compare direction estimation and response design as separate factors.
Longer covariance memory could reduce serially correlated estimation noise,
but might delay learning; changing the complementary response could reduce
the cost of imperfect directions, but could erase the benefit of selectivity.
Neither effect is a unique causal mediation fraction of neural learning.

Keep every parent seed19000–19031, process variance0/.01/.1, rotation0/pi4,
and all4000 observations:192 streams,32 independently drawn *reused* bundles.
No early stopping, selected subset or additional tuning after outcome access.
Preserve all parent policies/472 means as explicitly reused references.

## Direction factor: six fixed members

All estimators use the parent's unchanged post-ingest mean mu_t (beta=.99).
The new non-oracle computation uses only g,mu and saved actions/masks, never s,
process variance, clean labels or future observations. The generating rotation
is used only for separately labeled oracle actions and alignment diagnostics.

1. `native`: exact saved I19 A, with saved basis-present mask.
2. `full_legacy`: saved I19 full_action when full_basis_present; identity
   fallback otherwise. Original unscaled-first-nonzero initialization retained.
3. `ew99`: zero-start standard EW moment, forgetting lambda=.99.
4. `ew999`: same initialization/normalization as ew99, lambda=.999.
5. `oracle_useful`: fixed rotated e1 projector, privileged direction oracle.
6. `oracle_nuisance`: fixed orthogonal projector, privileged adverse oracle.

For the two new EW estimators, z_t=g_t-mu_t, M_0=0,w_0=0, and for every t
including the initial zero residual:

```text
M_t = lambda M_(t-1)+(1-lambda) z_t z_t^T
w_t = lambda w_(t-1)+(1-lambda)
N_t = M_t/w_t.
```

Use the leading normalized-moment eigenvector only when the eigenvalue gap
exceeds1e-10*max(1,abs(largest eigenvalue)); otherwise use identity and mark
direction unavailable. Eigenvectors are sign-ambiguous; use their outer product.
ew999-versus-ew99 isolates forgetting with matched initialization. The separate
ew99-versus-full_legacy contrast exposes initialization/normalization effects;
do not attribute their combined change only to a longer memory. This is a dense
2D diagnostic, not an implemented scalable native rank-one estimator.

## Response factor: two fast decays × three responses

Fast decay rho is .9 or rho*=(2.1-sqrt(.41))/2=.7298437881283576, the already
chosen I19 strong-cell value. No re-estimation by cell/outcomes. For each of the
six direction sequences P_t, compute all six responses. All d_1=g_1 exactly.

- `cp`: original constant-preserving delivery-state recurrence. Compute
  h_t=P_t g_t+mu_t-P_t mu_t and
  d_t=rho P_t d_(t-1)+(I-rho P_t)h_t.
- `rec99`: B_t=rho P_t+.99(I-P_t),
  d_t=B_t d_(t-1)+(I-B_t)g_t.
- `rec9`: same two-sided recurrence with complementary decay .9.

For ideal fixed projectors, cp and rec99 agree under matched initialization;
moving actions distinguish how they carry past information. rec9 with rho=.9
is analytically ordinary EMA .9, independent of the direction: retain it as
an exact implementation control, **not a novel experimental discovery**.
rec9 with rho* retains genuine directional variation with a less slow complement.
Full-moment identity fallbacks are operational policies, not learned directions.

Exact order is estimator-major (`native,full_legacy,ew99,ew999,oracle_useful,
oracle_nuisance`), then rho (`r0p9,rstar`), then response (`cp,rec99,rec9`).
Policy names join those three parts with `/`:36 new/reconstructed policies.
Native cp outputs must match the two accepted parent CP outputs within
rtol=2e-11,atol=5e-12. Fixed-oracle cp must similarly close to the available
parent useful/.9, useful/rho-star and nuisance/.9 outputs; the parent has no
nuisance/rho-star policy. All rho=.9 rec9 outputs must close to parent EMA .9.
This algebra check does not execute or reset the original native observer.

## Outcomes and prospective comparison roster

Primary: identity-rotation mean squared vector error over steps1–4000, equal
weight per reused seed. Secondary: startup1–100, transition101–1000,
late1001–4000, and paired rotated counterparts. Report SE and exact signs across
32 seeds; do not count observations, policies or rotations as replications.
No multiplicity-adjusted or fresh-confirmation claim.

Retain the full36-policy×3-process×2-rotation×4-window mean table (864 rows),
all27,648 per-seed/window MSE values and all472 reused parent mean rows.
Direction diagnostics retain masked useful squared alignment and action
change norms for all six estimators, with absence counts separate.

For each process, rotation and window, freeze these90 contrasts, with positive
effect always `comparator MSE - target MSE` (positive favors target):

- Every non-oracle estimator/rho/response (24 targets) versus parent EMA .9
  and parent common Kalman:48 comparisons.
- ew999 versus ew99; ew99 versus full_legacy; full_legacy versus native,
  at each fixed rho/response:18 comparisons.
- rec99 versus cp at each estimator/rho:12 comparisons.
- rec9 versus rec99 at each estimator/rho:12 comparisons.

This gives270 identity-whole primaries and2160 total comparisons. No pooled
winner across cells, new policy selector or dropped comparator. The strong-cell
late direction hypothesis predicts higher ew999 than ew99 mean alignment;
whole-risk improvement remains uncertain. Interpretation must include weak/no-
signal regimes and whether an apparent remedy merely approaches isotropic EMA.

## Inputs, output schema and integrity

Parent root:/tmp/spectral-experiment-artifacts/spectral-i19-001.wnMmy7.
Parent completion6a69b9f96f81dcd1066954f53f703280794c88ea29c32beed0c01fa329b4740e,
attempt3ed5bfe9d66e94a0193e7881d7aa6bd11b61e80eb0d477aeb9cf89c3cfb2a1c1,
summary135502cc7fc6a69bba7c16fd0ddcf53d10255a78065c0d2198fbe156fb76ba9f,
auditc549fc5d10dec4d644c857821e6e1e67a26ad6469e821f0d7cae4588671b28f2,
report-check32b1629079ab5db14809f4448a2d5432afedfb11f2e51f0b201476a536f3d867.
Verify all parent inventory/source hashes and the current I20 source freeze
before computing new outputs; all parent acceptance handles remain consumed.

New stream arrays, exact order and shapes (T=4000):

1. actions:float64(T,6,2,2)
2. direction_present:bool(T,6)
3. weighted_moment:float64(T,2,2,2), ew99/ew999 order
4. weight_mass:float64(T,2)
5. normalized_moment:float64(T,2,2,2)
6. weighted_eigenvalues:float64(T,2,2), ascending
7. weighted_gap:float64(T,2)
8. output:float64(T,36,2), policy order above
9. useful_squared_alignment:float64(T,6), zero-filled where unavailable
10. action_change_norm:float64(T,6), Frobenius ||P_t-P_(t-1)||, first zero.

No object arrays or duplicated parent tensors. Metadata binds parent file
hashes, exact policy/estimator order, initialization, absence conventions and
source freeze. Save all new actions and outputs for a separate independent
algebra/risk audit; no outcome-dependent repairs to frozen source.

## Resource envelope and stopping

CPU-only, CUDA hidden, one numerical thread/core,4GiB memory,zero extra swap,
600s hard/540s cooperative,5s stop,Restart=no. Prospective cap2GiB arrays and
3GiB total new root under the verified /private-artifacts/storage mount. Fixture timing
and byte counts must fit before launch; no cuts to the roster to fit later.
Use one exclusive `spectral-i20-001.*` root and one attempt; completion records
the entire completed prefix and any failure. Never automatically retry, restart
a parent run or regenerate a seed. Paid spend/reservation remains0/100USD.

## Confirmation requirement

This tests hypotheses on already inspected inputs. Any outcome-informed remedy
needs separately frozen fresh-seed confirmation and a changing-true-direction
adverse case before an efficacy claim. It adds no neural result or default.
