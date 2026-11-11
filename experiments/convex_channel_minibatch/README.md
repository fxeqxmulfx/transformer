# Persistent minibatches for the forward-only convex channel

Does retaining candidate records across minibatches compute exactly the same
model as the complete-data learner, and how much does batch size affect cost?
The source is `review/flat_spectrum.md`, CF6, formalized in
`ConvexChannel.Forward.Minibatch`. Its exact equality theorem includes failure,
partial coverage, empty batches and defaults. It assumes the same ordered data.

## Frozen protocol

Fixed on 2026-11-11 before measurement. This is a separate native experiment;
the previous `convex_channel_forward` comparison and its source hashes stay intact.

| Item | Fixed setting |
| --- | --- |
| Tasks | Unchanged ordinary and recoded MQAR from `convex_channels::task::make` |
| Seeds | 0, 1, 2 for each task |
| Data | Four keys, eight values, 20 classes; 256 train and 256 disjoint test dictionaries |
| Batch sizes | 1, 4, 7, 16, 32, 64, 256 dictionaries; each dictionary supplies four query labels |
| Persistent learner | Keep every candidate's record and elimination flag; increasing key order; default class 0 |
| Reference | Unchanged `convex_channel_forward::model::Model::fit` on all training examples |
| Correctness | Exact final routes, decoders and tensors; predictions on train, test and all 4096 dictionaries |
| Prefix checks | Snapshot after every batch agrees with reference fitting the entire prefix |
| Extra order control | Reversed training stream has the same final model on this covered, identifying cohort |
| Timing | Five warm-up trials, 31 measured trials, 64 fresh complete fits per trial |
| Ordering | Rotate eight timing arms: reference plus seven minibatch sizes |
| Execution | Release Rust, one pinned CPU, no Python or GPU kernels |

Each timed fit includes allocation, initialization, validation, every batch
update and one final snapshot. It excludes task generation, diagnostics,
serialization and prefix checks. The original reference also includes its
query grouping. No optimizer learning rate or gradient accumulation is used.
All candidate visits are counted, including visits to eliminated candidates.
All raw timing observations and p10/median/p90 summaries are retained.

Persistent storage is measured as live record and flag array payload, excluding
container metadata, capacity, the emitted model, caller-owned data and process
overhead. It is independent of the number of examples. Snapshot defaults must
not be written into the records, and snapshots must not freeze the chosen key.
The API retains no examples. Exhaustive partial Boolean data and every
contiguous partition test these properties, including failure and late conflicts.

The runner embeds source hashes, records data hashes, parameters, CPU/affinity
and timing observations. The report regenerates the data, replays every model
and rechecks prefixes and timing summaries. No failed or slow case is excluded.
This checks batch-boundary invariance for noiseless single-key recall; it does
not turn the learner into a noisy-label optimizer. The reverse-order check is
empirical, separate from the ordered-stream Lean theorem.

## Reproduce

```sh
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexChannelAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
cargo test --offline --manifest-path experiments/convex_channel_minibatch/Cargo.toml
cargo fmt --manifest-path experiments/convex_channel_minibatch/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path experiments/convex_channel_minibatch/Cargo.toml -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_channel_minibatch/Cargo.toml
taskset -c 24 experiments/convex_channel_minibatch/target/release/convex-channel-minibatch run experiments/convex_channel_minibatch/runs/mqar-v1
experiments/convex_channel_minibatch/target/release/convex-channel-minibatch report experiments/convex_channel_minibatch/runs/mqar-v1 > experiments/convex_channel_minibatch/runs/mqar-v1/summary.json
```

Use a fresh output directory. Generated evidence remains ignored and local.

## Findings

Pending the frozen 42 batch-size comparisons and saved-model replay.
