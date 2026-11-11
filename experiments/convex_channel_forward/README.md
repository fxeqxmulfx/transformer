# Forward-only learning versus the usual convex channel on MQAR

Does the finite learner proved in `a7c246fe`,
`ConvexChannel.Forward` (CF1-CF5), improve native training cost while retaining
the accuracy of the ordinary joint convex channel?

## Frozen protocol

Fixed on 2026-11-11 before these runs. The earlier `convex_channels` experiment
already found exact recovery after one Frank-Wolfe update. Therefore **one
update**, rather than its unnecessarily long 1000-update endpoint, is the
primary training-time baseline. The 1000-update endpoint is retained as a
correctness control. All six task/seed combinations must be reported.

| Item | Fixed setting |
| --- | --- |
| Tasks | Ordinary MQAR and recoded MQAR from the unchanged `convex_channels::task::make` |
| Seeds | 0, 1, 2 for each task |
| Inputs | Four keys, eight values, shuffled visible bindings and query order |
| Data | Same 256 training and 256 disjoint test dictionaries for both learners; 1024 query answers per split |
| Classes | Full 20-class vocabulary; recoded answers are absent from the inputs |
| Direct learner | Candidate keys 0,1,2,3; empty records; default class 0 for unobserved values; exact label-conflict elimination |
| Usual channel | Unchanged `Channel::new` and `Channel::step`; uniform tensor; one full-batch Frank-Wolfe update with eta=1 |
| Endpoint control | Same usual channel after its original 1000 updates |
| Inference control | Losslessly compact the one-update tensor to a key and decoder only if it is exactly a deterministic vertex |
| Correctness | Training, held-out and all 4096 possible dictionaries; ordinary CE, accuracy, exact tensor agreement, feasibility, FW gap and spectral error |
| Timing | Five warm-up trials, 31 measured trials; 64 independent fresh fits per training trial; 128 repetitions of the test split per prediction trial |
| Ordering | Alternate training-arm order and rotate the three prediction arms; retain every timing observation |
| Execution | One native Rust process, one pinned CPU, release build; no Python or GPU kernels |

Training time includes model allocation/initialization, input validation and
query grouping for the direct learner, and sample-reference construction for
the usual channel. It excludes dataset generation, diagnostics, serialization
and evaluation. Each timed fit uses the complete training set. The direct
learner implements the Lean search literally: one full left-to-right scan
per attempted key, including visits after a candidate has been eliminated.
Its counter must obey `candidate keys * training answers`.

Prediction timing returns all four class IDs per dictionary, with inputs and
results passed through `black_box`. The usual tensor uses its existing dense
logits and argmax path. Direct and compacted-channel prediction use the same
lookup implementation. Compaction is checked for exact equality, has no access
to labels, and is outside prediction timing. This control separates a cheaper
representation from a benefit unique to the learning algorithm. Parameter
payload bytes exclude container metadata and training scratch; they are not
peak process memory.

The native port uses array records instead of Lean's functional updates and
groups examples by query before scanning. Its training API receives only
dimensions and labeled samples; its prediction API receives only values and
queries. Target-rule tables are used only by the unchanged task generator and
evaluation. Exhaustive partial Boolean datasets provide an independent
brute-force reference for success/failure, alongside the Lean Boolean witness
and compaction tests. No outcome-dependent tuning or extra seeds are allowed.

The runner records source hashes embedded at compilation, data hashes, model
parameters, raw timing observations, CPU/affinity information and diagnostics.
The report replays saved models on regenerated data and recomputes all metrics.
Host activity can affect timings even with CPU affinity; percentile ranges
must accompany the medians. This comparison covers the noiseless realizable
MQAR setting, not a noisy-label optimizer or arbitrary multi-key interaction.

## Reproduce

```sh
cargo test --offline --manifest-path experiments/convex_channel_forward/Cargo.toml
cargo fmt --manifest-path experiments/convex_channel_forward/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path experiments/convex_channel_forward/Cargo.toml -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_channel_forward/Cargo.toml
taskset -c 24 experiments/convex_channel_forward/target/release/convex-channel-forward run experiments/convex_channel_forward/runs/mqar-v1
experiments/convex_channel_forward/target/release/convex-channel-forward report experiments/convex_channel_forward/runs/mqar-v1 > experiments/convex_channel_forward/runs/mqar-v1/summary.json
```

Use a fresh output directory; the runner refuses to overwrite evidence. Outputs
remain ignored and linked to the root checkout while working in a worktree.

## Findings

Protocol fixed; measurements have not started.
