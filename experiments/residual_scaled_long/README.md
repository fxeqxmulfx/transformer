# Fixed-scale continuation to 99% validation

Task154 asks whether the early advantage of post-RMS scaling survives a longer
run. The user requested continuing until 99%. Both P97 seeds resume their complete
23,000-update task145 states. Ordinary controls first reached 99% validation at
32,000 updates (seed 11) and 37,000 updates (seed 17).

| Case | Parent | Residual coefficient | Stop |
| --- | --- | --- | --- |
| scaled-s11 | mechanism-s11-scaled at 23k | 1/sqrt(8) | First canonical validation >=99% |
| scaled-s17 | mechanism-s17-scaled at 23k | 1/sqrt(8) | First canonical validation >=99% |

The model, Schedule-Free AdamW, learning rate, averaging clock, sampler, split,
FP32 arithmetic, math SDPA and fullgraph max-autotune remain unchanged. The parent
checkpoints, histories and reports remain intact. No optimizer state is reset.
The continuation runner checks restored model/optimizer/sampler tensors exactly
after compilation, before taking a new update.

The runner evaluates every 1,000 updates and stops at the current endpoint on
the first validation observation >=99%; this is not the parent's stricter
three-observation 99.9% stability rule. Its underlying DSL recipe retains the
parent benchmark settings for strict continuation compatibility, with an extended
100,000-update checkpoint horizon. Use the dedicated runner, which overrides
the stopping criterion, rather than the generic `run` command for this cohort.

The original final test has already been reported. It is reused for endpoint
replay only, after both learning runs close, and never for stopping. These are
two continued trajectories, not new independent seeds or a fresh test population.
All endpoint scores are replayed on CPU FP32 and audited in FP64 with FP32 RMS
epsilon. Compare first-99% update counts and cumulative training/wall time, with
resumed compilation cost included and without claiming the cause of any speedup.

Reproduction: `./run.py scaled-continuation freeze`, then
`./run.py scaled-continuation run`, then `./run.py scaled-continuation report`.
Freeze verifies parent checkpoints, source hashes and split fingerprints before
learning. A started phase cannot be overwritten silently. `protocol.json` pins
the declaration; `results.json` will record the measured outcomes.
