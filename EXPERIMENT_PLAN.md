# Project experiment plan: sparsemax generalization and runtime

Updated on 2026-10-03 UTC. **Paused at the user's explicit request.**
Resume the experimental cycle only after a new user instruction to continue.
This is the project-level plan; detailed historical evidence remains in
[the synthetic trainer handoff](experiments/archive/synthetic_trainers/HANDOFF.md).

## Objective and established results

Identify experimentally why sparsemax fails to generalize as well as softmax
in the controlled modular-division setting, then prove the supported mechanism
in Lean. Improve experiment throughput through measured changes, preserving
the distinction between faster execution and a changed learning procedure.

The task is division `x/y mod 193`, with nonzero denominator: 37,056 exhaustive
operand pairs, split into 9,264 train and 27,792 held-out examples. In the
completed 300,000-update normalizer pair, sparsemax reaches 100% train and
9,465/27,792 = 34.056563% held-out accuracy; softmax finishes at 100% on both.
Sparsemax fails all 201 final-window checks; softmax fails two and recovers.
Neither result establishes the required persistent stable benchmark.

Commit `0cf0524` proves that a strict unit score gap makes the actual causal
sparsemax row locally constant with zero Frechet derivative. A concrete wrong
route has positive loss and zero score derivative. Lean also certifies the
accuracy of all 37,056 supplied final CPU prediction records. These results
establish a possible obstruction and exact table counts; they do not establish
the cause of this model's generalization gap or verify its PyTorch trajectory.
See [the proof and certificate report](experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-mechanism-and-final-certificate.md).

## Abandoned schedule pair

The frozen constant/cosine schedule pair is incomplete and will not be
resumed. Trainer PID 168237 (session 11503) and archive worker PID 168332
(session 60201), stopped with `SIGSTOP` at 07:48:53 UTC, were terminated with
`SIGKILL` at 08:03 UTC on 2026-10-03 by the user's decision, before the Python
code moved out of `experiments/`. Neither process ran again after the pause,
so the raw state below is exactly the paused state.

The constant case's last canonical observation is update 288,000 and its last
complete gradient record is 288,250. Its latest canonical train and held-out
accuracies are 100%, but failures at 274,000 and 275,500 already violate the
strict final-window condition. Only 153 of 201 final-window checks exist.
The durable checkpoint is update 285,000, SHA-256
`14706a5662e124119fef1caf882c6adae1ba8b8a8dfd7ed5cf14e6080a371380`,
rechecked after termination. The cosine case never started. Raw state is under
`experiments/archive/synthetic_trainers/runs/adamw_stability_20261002/schedule_mod193_fraction25_lr0003_budget300k/`
(`experiments/runs/` until 2026-10-03).
The incomplete pair supports no conclusion about either schedule. Its frozen
plan pins sources at their `experiments/` paths; replaying it requires a
checkout of commit `698d904`.

## 1. Measure acceleration before choosing a new recipe

The pre-pause read-only sample at 07:43 UTC observed GPU utilization of 65--96%
(median 76%), 221 MiB of 2,048 MiB VRAM used, and approximately one fully busy
CPU core on a 56-core host. Ten samples covered 9.31 seconds; this is a short
snapshot, not a whole-run performance profile. Low total host CPU utilization
does not imply that the training process has spare CPU capacity.

The corpus is already resident on the GPU. The trainer uses one CPU thread,
checks gradient finiteness every update, synchronizes and writes the gradient
trace every update, and performs exhaustive canonical and neighboring
evaluations. Evaluation currently uses batch 1,024. Measure training,
evaluation, diagnostics and checkpoint overhead separately.

Benchmark isolated softmax and sparsemax
copies with training batches 512, 1,024, 2,048 and 4,096, subject to available
memory. Keep LR at 0.0003 initially. Start every benchmark from the same saved
model and optimizer state; use warmup and synchronized timings. Record update
latency, examples per second and peak allocated VRAM. Benchmark runs have no
scientific generalization status and must never overwrite scientific histories.

First prefer execution improvements that preserve the learning recipe:
larger evaluation batches, fewer CPU/GPU synchronization points in evaluation,
and buffered trace writes with identical records and checkpoint recovery.
Keep diagnostic cadence and all accuracy/recovery criteria. Verify per-example
predictions, model gradients, native AdamW updates, sampler state and exact
resumption as applicable; retain explicit numerical checks for loss reductions.

A larger training batch changes gradient noise, examples per update, epoch
counts, and the optimizer's effective time scale per example. Report runtime
and generalization against updates, examples seen and elapsed time separately.
At the same 300,000-update limit it is a changed experiment. Do not automatically
increase AdamW LR with batch size. If a larger batch is selected, first compare
it at the existing LR; freeze any later LR sweep as a separate factor. Preserve
batch 512/LR 0.0003 controls. Use one common recipe for the causal normalizer
study; do not tune the four normalizer cases separately.

## 2. Review fixed-checkpoint diagnostics

Inspect both actual final checkpoints on the entire frozen corpus. Separate
numeric-answer attention from EOS, train from held-out, and correct from wrong
numeric predictions. Record each layer and head's support size, singleton
frequency, strict unit gaps, entropy, and zero numerator/denominator weights.
Zero direct operand weight does not prove absence of operand information:
earlier layers and residual paths can retain it.

Evaluate all four checkpoint-weight/forward-normalizer combinations without
updates. This measures the learned representation's dependence on routing;
it does not identify a training cause. Inspect inactive-score and singleton
gradients on fixed train and held-out batches. Instrumented logits, losses
and every parameter gradient must match ordinary execution exactly.

A preliminary CPU inspection already finished; its observations are under
`experiments/archive/synthetic_trainers/runs/adamw_stability_20261002/bootstrap/sparsemax-generalization-inspection-20261003/`
(not tracked). The inspector that wrote them is in commit `441f47b`, at
`experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/inspect_sparsemax_generalization.py`,
and left the tree with the other archive scripts; its SHA256 is the
`program_sha256` the observations record. It imports the removed synthetic
trainers and reads `experiments/runs/`, so it does not run as it is. Review
the source, independently check the observations, and archive them before
treating them as final. No intervention training has started.

## 3. Separate forward routing from backward sensitivity

Prepare and freeze a two-by-two training intervention:

| Forward weights | Backward score map | Purpose |
| --- | --- | --- |
| Softmax | Softmax Jacobian | Original control |
| Sparsemax | Sparsemax support Jacobian | Original candidate |
| Sparsemax | Softmax Jacobian | Preserve sparse routing; change score gradients |
| Softmax | Sparsemax support Jacobian | Preserve dense routing; change score gradients |

The mixed cases use declared surrogate gradients; these are not the derivative
of the chosen forward loss. CPU controls must establish that diagonal cases
reproduce original initialized parameters, logits, every gradient, AdamW
updates, sampler exposure and resumed execution. Mixed cases must reproduce
the chosen forward weights and independently checked Jacobian-vector products,
including inactive entries. Preparation checks establish no learning result.

The reference recipe is prime 193, train fraction 25%, model/data seeds 0/0,
width 128, two layers, four heads, native AdamW betas (0.9, 0.98), epsilon 1e-8,
decay 0.1 on all parameters, LR 0.0003, warmup 10, batch 512 with short tails,
and no clipping. Execution changes and any prospective batch/LR amendments
must be separately measured, documented and frozen before the first update.
Each scientific case has **at most 300,000 total optimizer updates**, with
the main four-case comparison planned for the complete 300,000-update budget
and no target-based early stop. Freeze source hashes and commit the final plan.

Retain canonical diagnostics every 250 updates and neighboring observations,
dense gradient tracing, the memorization-before-generalization plateau,
twenty-observation long confirmation, and all 201 final-window persistence
checks from 250,000 through 300,000. Measure numeric and EOS losses separately,
support, failure depth, recovery delay, failure fractions and episode onsets
per 10,000 updates. Keep quiet intervals and late relapses in the report.

## 4. Interpret, narrow and confirm the mechanism

Analyze all four complete cases together. If dense backward routing rescues
sparse forward routing and sparse backward routing damages dense forward
routing, the backward map is implicated in this setting. This intervention
changes both support and gradient magnitude; follow with a focused intervention
before attributing the effect specifically to inactive-score zeros. If forward
mixing is implicated, test operand-information retention and value mixing.
Other patterns may indicate interaction or an unsuitable surrogate.

Repeat a supported effect across independently frozen model/data seeds before
claiming repeatability. One seed pair cannot establish general applicability.
The existing stable-benchmark gate still requires all six fresh crossed
confirmations; CPU fixtures and exploratory mechanism studies do not open
the architecture or complementary-attention gates.

## 5. Prove the experimentally supported statement in Lean

Only after the experiments isolate a mechanism, formulate its precise
mathematical statement with explicit assumptions and satisfiable examples.
Possible targets concern support, derivatives, loss geometry, information
retention or finite-precision bounds. Distinguish a real-arithmetic theorem,
a floating-point measurement and a supplied-data certificate. Do not infer a
whole-model training or generalization theorem from a local attention-row proof.

Follow AGENTS.md: check statements against the local papers, document extensions,
inspect external lemmas and their proof dependencies, add no axioms or new
`sorry`, build affected modules and the full tree, run the full axiom audit,
and regenerate INDEX.md. Commit each verified logical change and update the
global plan and detailed handoff. Until the user resumes work, stop here.
