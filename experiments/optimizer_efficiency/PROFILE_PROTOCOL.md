# Frozen source AdamW cost profile

Task #191, 2026-11-08. The user has skipped further no-gradient/local-learning
work. This profile uses the unchanged source CE and full backpropagation.
Freeze it before measurement; choose and freeze any candidate separately.

Use all **24 original states**: four architectures, seeds 7/11 and steps
0/5000/10000. Verify original source/PTX, checkpoint, data and sampler pins.
Extend the deterministic sampler to 11000 updates without changing its
original prefix. Use the actual saved moments. No saved input is modified.

At each state compare two workloads, restoring the saved state before each:

* **Update-only:** compute the next scheduled CE gradient once outside timing,
  then execute 256 source AdamW updates with that gradient fixed and clocks
  `saved_step+1` through `saved_step+256`. This measures optimizer throughput,
  not a training trajectory or convergence.
* **Complete step:** execute eight consecutive actual source CE training
  updates, with their scheduled batches and advancing optimizer clock.

Use three warmup and nine measured rounds, alternating workload order each
round. Retain every sample, median and nearest-rank p90. Record synchronized
host wall time and CUDA-event span, divided by workload update count. Event
spans can include queue gaps; do not label them pure arithmetic execution time.
Measure nine empty event brackets as a floor, without subtracting them.
State restore, fixed-gradient preparation and state validation are excluded
from measured intervals and reported as diagnostic work. Validate finite
parameter/moment ties after every workload and retain final state hashes.

Report update-only/full-step ratios for wall and event times separately.
These are workload comparisons, not an additive phase decomposition or a
certified Amdahl bound. Keep all architectures/states; no favorable subset.
Report actual source GPU buffer payload and optimizer moment bytes, including
expanded complex aliases, plus the original independent-parameter count.

Poll GPU ownership before/after every case. Keep the complete process list,
including desktop graphics. Report exclusive ownership separately from
shared-device observations; no exclusive-device performance claim with another
process present. Cap the cohort at 600 seconds excluding compilation, retain
all 24 slots/errors/caps, and verify input hashes afterward. No new training
cohort or optimizer candidate is selected from validation outcomes here.

Use Rust/Kernite and native tools only. Keep generated evidence ignored and
preserve it outside the temporary worktree. Commit source and authored findings.
