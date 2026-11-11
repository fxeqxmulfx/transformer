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

All six frozen comparisons completed and passed saved-model replay. No runs
failed, no endpoints were extended, and no timing observations were dropped.
The training and test hashes also match the six corresponding channel runs
from the earlier experiment. The measured source is commit `3e0e2cb1`, with
embedded SHA-256
`30eb9035fe5ada7353c9638cb0dfb148535522dba232b98a6abb02c521103453`.

Both methods obtain **100% training, held-out and full-domain accuracy** on
ordinary and recoded MQAR, in every seed. Full-domain evaluation covers all
4096 dictionaries and 16384 query answers. The direct model, one-update
Frank-Wolfe model and 1000-update endpoint have exactly identical joint
tensors. All constraint violations, parameter-witness errors, FW gaps and
full-domain spectral errors are zero. Ordinary CE is 2.0781543864329897 on
each held-out split, the expected optimum under the fixed unit-logit scale.

The direct learner makes 2560 candidate observations, below the proved
4096-observation bound. It checks one, two, three and four candidate keys
for the respective query identities; each attempted candidate consumes its
256 training observations. The selected decoder covers all eight values.

Measurements used one thread pinned to CPU 24 on an Intel Xeon E5-2680 v4,
Rust 1.99.1, on a shared host (starting one-minute load average 51.91 across
56 logical CPUs). Each entry below is the median with its p10-p90 interval
over 31 trials; each training trial averages 64 independent fresh fits.

| Task | Seed | Direct fit, us | One FW update, us | Training speed ratio |
| --- | ---: | ---: | ---: | ---: |
| MQAR | 0 | 17.21 [16.94, 17.49] | 469.85 [468.44, 474.14] | 27.30x |
| MQAR | 1 | 17.59 [17.24, 17.91] | 470.93 [468.28, 474.77] | 26.78x |
| MQAR | 2 | 17.36 [17.04, 17.61] | 468.89 [467.43, 470.72] | 27.02x |
| Recoded | 0 | 17.81 [17.57, 18.22] | 472.42 [470.43, 476.70] | 26.53x |
| Recoded | 1 | 17.78 [17.29, 18.29] | 473.57 [470.13, 481.84] | 26.64x |
| Recoded | 2 | 17.85 [17.54, 18.22] | 470.83 [469.39, 472.86] | 26.38x |

The speed ratio is against the existing implementation's **first** update,
which already reaches the optimum. It does not count the extra 999 updates.
The direct routine avoids dense CE/gradient evaluation and records categorical
consistency instead. These are implementation measurements on this realizable
task; a specialized counting implementation of the initial FW oracle could
reduce the gap.

Prediction below returns all four answers per dictionary. Units are nanoseconds
per dictionary; brackets again give p10-p90 over the 31 trials, each averaging
128 complete test-set traversals.

| Task | Seed | Direct lookup, ns | Usual dense channel, ns | Compacted usual channel, ns |
| --- | ---: | ---: | ---: | ---: |
| MQAR | 0 | 28.90 [28.56, 29.49] | 614.78 [611.62, 631.38] | 28.94 [28.45, 29.38] |
| MQAR | 1 | 28.97 [28.57, 29.51] | 604.50 [601.34, 607.16] | 29.11 [28.63, 29.85] |
| MQAR | 2 | 28.56 [28.30, 29.09] | 607.37 [605.01, 610.83] | 28.63 [28.22, 29.13] |
| Recoded | 0 | 29.34 [28.91, 29.86] | 607.19 [604.96, 609.00] | 29.32 [29.03, 29.89] |
| Recoded | 1 | 28.86 [28.50, 29.44] | 606.66 [604.35, 609.59] | 28.79 [28.44, 29.41] |
| Recoded | 2 | 28.63 [28.23, 29.17] | 609.53 [606.32, 614.25] | 28.83 [28.23, 29.54] |

Lookup prediction is 20.70-21.29x faster than the current dense forward. Exact
compaction of the usual channel gives the same decoder and essentially the
same inference time (within 0.7% of the direct model's median in every case).
Its payload also falls from 2560 f64 entries (**20480 bytes**) to four routing
indices and 32 output classes (**288 bytes** as usize arrays), equal to the
direct model. Thus training is cheaper with direct consistency learning here;
the inference and storage gains are available to either trained model through
the same exact representation. Compaction cost is outside prediction timing.

Validation: six new native tests, including all 81 partial Boolean datasets;
five unchanged baseline tests; formatting and clippy with warnings denied; all
six endpoint, source/data hash, timing-summary and saved-model replay checks.
The generated evidence is preserved under `runs/mqar-v1/`; the exact executable
and validation logs are also preserved in `.kanban/artifacts/149/` at the root.
These outcomes retain CF1-CF5's noiseless single-key realizability scope.
