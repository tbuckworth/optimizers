# Familiarity × continuation frequency: source feasibility

Codex — Spectral Optimizer Investigation · 10 September 2026

**Feasible as a new small-MNIST study, not as a continuation of the existing parents.** This is a source/metadata assessment only. No roster, warmup horizon, frequency levels, acquisition, or paid reservation is selected here. Read the [current decision](../2026-09-10-spectral-component-utility/next-decision.md) and the current root-state checkpoint before treating this as an executable plan.

## What can be reused unchanged

| Component | Exact source | Reuse boundary |
| --- | --- | --- |
| Model and AdamW | I9 core, model at line 100 and optimizer at 113 (artifact not distributed in this public snapshot) | CPU-seeded 784–64–10 ReLU model, 50,890 parameters; AdamW .001 learning rate, .01 decay, (.9,.999), epsilon 1e-8, foreach/fused false. Initialization preserves the caller's Torch RNG. |
| Canonical observer | I9 tracker constructor (artifact not distributed in this public snapshot), stable update (artifact not distributed in this public snapshot), delivery (artifact not distributed in this public snapshot) | Global centered rank 32, decay .99, repair 100, hard weighting, no normalization/adaptation. V uses gradient dtype; S is CPU FP64. Native observes the current raw gradient before delivering V(Vᵀg); absent basis means identity. |
| Complete-state copying | snapshot and restore (artifact not distributed in this public snapshot) | Includes model specification, weights, .grad, module modes, Adam state/flags, observer attributes and Python/NumPy/Torch/CUDA RNG. Restore binds the original parameter order and checks topology. It is not a generic arbitrary-model loader. |
| Input and scalar primitives | CPU normalization (artifact not distributed in this public snapshot), classification sufficient statistics (artifact not distributed in this public snapshot), array digest (artifact not distributed in this public snapshot) | Pinned training IDX only; CPU NumPy FP32 division by FP32 255; true-label counts, correct counts and FP64 CE sums. Empty subsets remain undefined, not zero performance. |
| Pairing and evidence pattern | common parent and forks (artifact not distributed in this public snapshot), final-state/logit receipts (artifact not distributed in this public snapshot) | Reuse the pattern of hashing full forks, persisting occurrence plans, reusing baseline predictions, recording first raw-gradient equality, and saving every endpoint. New identity fields and roster checks are required. |

The canonical filter, I9 core, selectivity, batching and observer source files have no working-tree changes and their present SHA256 values match their accepted pins: respectively `9280c7d3…df943`, `70685182…970ec`, `c851cb71…fac0f`, `35c4b3f5…c8b1a`, and `263d7fb0…63f63e`. Full pins are in the accepted batching manifest (artifact not distributed in this public snapshot) and observer source. The batching source was frozen in commit `c5ad0f2250d0294a71556e51f21573d41b3d7954`; its acquisition manifest records launch commit `af03a3b8ba9ae3a39a82727eafee353c3cefee45`. No old acquire/main should be invoked or globals monkeypatched.

## What must be new

**Roles and plans.** Existing make_plan (artifact not distributed in this public snapshot) fixes digit 8, uses 50 training examples for it versus 550 per other digit, excludes it during warmup, and samples continuation uniformly over that unequal pool. Group metrics (artifact not distributed in this public snapshot), corruption targets and schedule construction (artifact not distributed in this public snapshot) also hard-code 8. Those complete functions are not role-general building blocks.

A new numeric plan should retain true digit labels and explicitly record the designated group, split IDs, warmup membership/replacement, continuation occurrence lists and class counts. Keep the available training support fixed across frequency conditions; vary sampling exposure, not simultaneously 50-versus-550 unique-example support. Counterbalance each selected role across the seed bundles, rather than assigning one digit to each seed and conflating role with seed. Roles sharing initialization/data are repeated conditions, not extra independent seeds. There is no need to remap true labels to 8 merely to reuse hard-coded helpers.

For a fixed seed/role/familiarity parent, raw and native must consume exactly the same continuation IDs, ordering and targets. Coupling the same frequency-specific schedules across familiarity parents is also straightforward. Between frequency levels, group exposure necessarily differs; common-class exposure changes when total batch size/updates stay fixed. Record both. The previous grouped/interleaved same-multiset claim does **not** apply across this new factor. No grouping arm is needed simply to reuse its scheduler.

**Parents.** A fresh familiar/unfamiliar pair must start from the same initialized weights and equal update budget. Familiar warmup includes correctly labelled target-group examples; absent warmup replaces those slots according to a fixed declared rule. Replacement inevitably changes other exposure and the resulting model, Adam moments and observer history. Preserve the common slots where possible and save exact replacement counts. “Absent positive examples” does not mean untouched output weights: multiclass CE still updates the omitted class as a negative alternative.

Warmup competence is an observed phase-boundary outcome, not guaranteed by exposure. Choose a fixed warmup horizon before results, save target/common CE and accuracy there, and retain every seed if target competence is weak. Do not select a favorable checkpoint, extend individual warmups until a threshold, or call absent/unlearned-target results preservation. A separate warmup-only sample pool would change example familiarity; using the same pool primarily tests class/task familiarity. This choice needs to be explicit in the eventual estimand.

## Clocks and restoration semantics

Let H be the fixed new raw-delivery warmup length and C the continuation length. During warmup, the observer records one raw gradient per Adam update while delivery stays raw. At the fork, Adam counters and observer count are H in both familiarity conditions, but their numerical states need not match. Native then observes and steps H+1 through H+C, with current-gradient self-inclusion; raw inherits the same model/Adam parent and bypasses filtering. Do not reset Adam moments, bias-correction counters, EMA mean, basis or repair phase at the boundary.

The canonical automatic delivery warmup is 100, so a deliberately longer raw-delivery preparation must be expressed by the **new runner's phase**, not by accidentally calling native delivery after step 100 or relabelling the observer count as 100. The old explicit raw policy is a useful implementation pattern, but its hard-coded evaluation/first-action identities need replacement. For raw, either discard the observer after verifying the complete fork, as before, or keep an explicitly labelled shadow; the former avoids unnecessary observation cost.

The observer-only study (artifact not distributed in this public snapshot) deliberately produced observer 150/151 with Adam 100/101. Those envelopes are **not** matched-clock warmup parents for this study. Its copy/nonalias checks (artifact not distributed in this public snapshot) provide a pattern, but contain fixed-100 assertions. Ordinary familiarity-by-frequency learning compares different full histories; it does not isolate observer memory.

## Existing snapshots: available but scientifically insufficient

Metadata-only `stat` checks confirmed all six original selectivity/batching warmup files are regular files of **7,347,297 bytes** each. No tensor was loaded or rehashed here. Batching seeds 202609121/122/123 have completion-recorded warmup SHAs `3561a54a…41634`, `bb80bf48…06db`, `7513bea1…99b8`; the accepted completion (artifact not distributed in this public snapshot) retains full hashes. Selectivity warmups are under `/tmp/spectral-experiment-artifacts/spectral-selectivity-boundary-20260910.2IruKP/acquisition-001/` with seeds 202609111–113.

The batching scalar results (artifact not distributed in this public snapshot), `.rows` at `clean/interleaved/native32`, record warmup rare held-out accuracy **0, 0, 0%**, CE **8.5673, 9.4169, 8.4546**, versus common accuracy **87.47, 88.89, 88.60%**. These are useful absent-target references, not familiar-target parents. Accepted receipts also list final raw/native states, but those are already trajectory-dependent and cannot be substituted for a controlled familiar warmup. Availability and prior successful use do not certify restorability in a future environment; no restoration was tested during this assessment.

## Bytes and prospective local budget

At P=50,890, one FP32 parameter/gradient vector is **203,560 bytes**; a full P×32 FP32 basis is **6,513,920 bytes**, its FP64 S is **256 bytes**. Weights plus Adam m/v are 610,680 bytes. The old conservative tracked-state allowance, including populated gradients, mean, counters and container/RNG reserve, is **7,597,528 bytes** (inventory source (artifact not distributed in this public snapshot)). Actual old cleared-gradient tracked states are 7,347,297 bytes; the seed-121 raw endpoint receipt is 628,394 bytes. A dense FP32 P×P matrix would exceed 10 GB and is neither needed nor part of this plan.

With s seeds and r independently crossed designated roles, two familiarity levels × two frequencies × two policies gives **8sr continuations** and at most **2sr fresh warmups**. Shared familiar parents might reduce that count if their data/roles are genuinely identical; never assume such reuse from equal shapes. Physical update count is `2sr H + 8sr C`, not the sum of duplicated per-branch warmup timings.

For sizing only, **s=3, r=3, H=500, C=2000** would mean 18 warmups, 72 continuations and 153,000 physical updates. It is **not a recommended or selected roster**, nor evidence H=500 establishes competence. Saving train+held-out FP32 logits for 5,000 examples each at 22 states costs 633.6 MB. Conservatively budgeting all 90 parent/final snapshots as tracked adds 683.8 MB; another 256 MiB for plans, common predictions, JSON and headers gives about **1.59 GB**, below a 2 GiB cap without diagnostic per-example gradients or repeated full bases. Exact inventory must be recalculated for the chosen roster/evaluation cadence. Ten roles would require materially more time/storage than this example.

Measured references, not predictions:

| Completed acquisition | Trajectories / physical updates | Wall time | Artifact bytes before completion | Peak host RSS / allocated GPU |
| --- | --- | ---: | ---: | --- |
| Selectivity | 36 / 68,700, plus diagnostics | 258.416 s | 2,096,072,077 | 1,576,144 KiB / 190,379,008 bytes |
| Batch composition | 36 / 68,700, plus 72 diagnostics | 246.464 s | 1,774,641,956 | 1,439,700 KiB / 128,130,048 bytes |
| Observer pathway | Fixed-state diagnostic, not training comparator | 65.925 s | 615,990,257 | 1,481,024 KiB / 134,743,552 bytes |

The first two measured times include their evaluation, diagnostic serialization and differing tracked-policy counts; they are not optimizer speed measurements. Simple update-count scaling places the illustrative new roster around nine minutes, but that is not a benchmark. A **10–20 minute planning allowance**, with one local RTX3090, one CPU math thread, 16 GiB host/no swap, an 8 GiB allocated-GPU ceiling, 25-minute cooperative/30-minute hard non-restarting service and a separately inventoried 2 GiB archive, is conservatively plausible for that example. It is not a completion guarantee. A once-only saved-logit/state-arithmetic audit can provisionally use one CPU, 4 GiB and five minutes; final schema/byte admission must establish its fit. No paid resource is needed or reserved.

Direct completion SHA256 receipts: selectivity `de66b0331bec90c9c24585ab9874af447de17cc653661984c0237fb65da66ceb`; batching `39b4ef091e691780b46c1d7f5dcb12e108dbfd5b3b85c6abba612a93a393b60a`; observer `b1c4837438051dc6ed51665879b6c37d7bd35673a86f8c4f363878c3f022464f`. Batching manifest SHA is `bbc45168e5f221ffa93ad884bc9f0ae5a56826d76d401cac4958e8a289885797`. Their original PASS audits remain consumed; no audit was repeated here.

**Decision support:** implementation feasibility is strong. The remaining decision is scientific: select role coverage, fixed useful-warmup opportunity, replacement semantics and frequencies that distinguish preservation from acquisition without conflating frequency with unique-example support. Existing source and saved states do not answer that question by themselves. No additional local dot-product or observer-only test is technically required before the proposed learning-level comparison.
