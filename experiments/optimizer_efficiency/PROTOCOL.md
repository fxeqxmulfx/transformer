# Frozen compact-moment AdamW comparison

Task #191, 2026-11-08. Frozen after the source cost profile and before
candidate correctness, timing or training outcomes. Use original CE and
full backpropagation throughout. No Forward–Forward/local-learning arms.

## Candidate semantics

Keep the original model, 143360 independent parameters, FP32 forward/backward,
source AdamW equations, update clock, warmup, betas, epsilon and decay.
For complex arms, store each first/second moment only once per independent
real coordinate in `model::pairs()` owner order. Parameter matrices retain
their original expanded layout. Combine signed expanded gradients exactly
as source FP32 AdamW does. Write parameter aliases after updating owners;
do not store duplicate moment aliases. Canonical snapshots reconstruct signed
first-moment aliases and unsigned second-moment aliases for exact comparison.

For all arms, compute the uniform FP32 quantities `lr / bc1` and
`1 - lr * decay` once on the host per update, in the same rounded operations
as the source kernel. Pass these scalars to the fused candidate kernel.
Preserve the order and FP32 rounding of gradient pullback, moments, square,
square root, second-moment bias division, epsilon addition and final update.
Do not substitute reciprocal approximations, fuse/reassociate arithmetic,
change epsilon placement or share the real/imaginary second moment.

**Require bitwise equality of canonical parameters and both moments.**
Any discrepancy fails the correctness gate; no numerical tolerance relaxes
this criterion. Real SwiGLU keeps its original moment layout and tests only
uniform-scalar precomputation. The two complex changes are one implementation
candidate; this experiment does not isolate their separate timing effects.

## Correctness and paired cost workloads

Pin the original trainer/PTX and preserve all original inputs. Audit all
24 saved states from the source profile. At each, compare one actual source
CE next step against the candidate, including canonical state and metrics.
At the eight original initial states, additionally compare each of the first
20 fixed-gradient optimizer updates to cover warmup clocks. Check four arms
with both zero and tiny signed gradients, nonzero tied first moments and
zero/tiny nonnegative second moments, including final owners and aliases.
Keep every failure and require these checks before training.

Repeat the source profile's 256-update fixed-gradient and eight-step complete
CE workloads for **both** implementations at all 24 states. Three warmup and
nine measured rounds retain all samples; rotate implementation order within
each workload and workload order within each round. Every workload starts
from the saved canonical state. Compare final canonical states bitwise after
every paired workload, including warmups, and validate all ties/finite values.
No round or case may be selected for favorable timing. Report per-case medians,
nearest-rank p90, paired source/candidate wall and event ratios, and empty-event
floor samples. Canonical restore/conversion and comparison are diagnostic
work outside these throughput intervals, not part of the resident update.

Use the same CUDA-event/host-wall definitions and ownership policy as the
source profile. Cap this paired workload cohort at 600 seconds excluding
compilation/preflight. Retain all slots and failures/caps. Another GPU process
invalidates an exclusive-device speed claim, but results remain recorded.

## Full quality trajectories

Use fresh seeds **29 and 31**, four original arms and source/candidate:
**16 from-scratch runs**, each **4000 updates**. Initialize with the source
algorithm and zero moments; match data, sampler and scheduled batches.
Keep paired engines resident and alternate their order every 100 updates.
Evaluate full train/validation CE and correct counts at step 0 and every
100 updates, with finite/tie checks and canonical state hashes. Require
identical paired metrics and canonical state hashes at all 41 observations.
Save endpoint canonical checkpoints. No radius, learning-rate, seed or restart
search; no early stop at a quality threshold or budget extension.

Record all histories, first observed 99%/99.5%, endpoint/best accuracy and
CE, and first attainment of the paired source endpoint accuracy **and**
no-worse CE. Compare source's first attainment of its own endpoint criterion,
not automatically its full 4000-step cost. Identical quality is a correctness
criterion, not evidence that this implementation improves convergence.
Cap each run at 120 active seconds and the training cohort at 900 wall seconds,
checked at 100-update boundaries; preserve all 16 slots and capped/errors.

## Cost, memory and interpretation

Synchronize timed chunks and retain every raw duration. Charge setup,
all forward/backward work, scalar calculation, optimizer launches and
scheduled evaluation/state checks (including canonical conversion/hashing).
Report setup, training, evaluation and active time separately. Exclude
compilation/JIT warmup, preflight, other methods' turns, ownership polling,
report serialization and checkpoint I/O. Poll before/after every paired
benchmark case and training group, and every 400 training updates.

Report actual allocated buffer payload and moment bytes per implementation;
do not claim total driver-memory savings. Candidate setup may allocate the
source moments temporarily before replacing them: disclose this transient
allocation and do not claim reduced peak setup memory. Charge replacement
work to setup. Report state export/conversion overhead in evaluation timing.
No hypothetical memory omissions or free backward passes are allowed.

Treat the fixed-gradient ratio as a throughput diagnostic, not convergence.
Report any full-step or matched-quality slowdown as well as improvements.
No claim follows merely from fewer arithmetic operations or smaller state.
Use Rust/Kernite only; preserve ignored generated evidence outside the
temporary worktree. Commit source, protocols, generators and authored findings.
