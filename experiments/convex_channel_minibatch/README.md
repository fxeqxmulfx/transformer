# Persistent minibatches for the forward-only convex channel

Does retaining candidate records across minibatches compute exactly the same
model as the complete-data learner, and how much does batch size affect cost?
The source is `review/flat_spectrum.md`, CF6, formalized in
`ConvexChannel.Forward.Minibatch` at `e9b0260c`. Its exact equality theorem includes failure,
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

The native API allows predictions between updates without freezing a candidate:

```rust
let mut learner = Streaming::new(shape, 0);
for batch in train.chunks(16) {
    learner.update_batch(batch);
}
let model = learner.snapshot()?;
```

`Streaming` is in `convex_channel_minibatch::streaming`; `Shape` is the unchanged
`convex_channel_forward::model::Shape`. A snapshot returns `Err(query)` if every
candidate for that query has been eliminated. It may be called after any batch.
The learner can continue updating afterward; snapshots do not change its records.

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

All **42 comparisons passed**: seven batch sizes on ordinary and recoded MQAR,
each with seeds 0, 1 and 2. Every final model has exactly the same routing,
decoder and joint tensor as the unchanged complete-data forward learner.
Training, held-out and full-domain accuracy are **100%** in every case; the
full domain has 4096 dictionaries and 16384 query answers. All **2316 prefix
snapshots** agree with independently refitting the corresponding complete
prefix. Reversing the training stream also preserves the final model in all
42 cases. This empirical order check is distinct from the Lean theorem for
partitioning the same ordered stream.

All six train/test fingerprint pairs match the earlier forward/FW experiment.
Saved-model replay, source/data hashes, raw timing summaries and prefix checks
pass. There were no failed or omitted runs or discarded timing observations.
The measured source is `c1b5f814`, with embedded SHA-256
`4c25e8d960429ea6fbb47c429453c7ff1a43e84426f95d1c9eeabcf78439501b`.

Measurements used Rust 1.99.1 on an Intel Xeon E5-2680 v4, pinned to CPU 24.
The shared host's starting one-minute load average was 43.06 across 56 logical
CPUs. Times below are microseconds to initialize, consume **all 256 training
dictionaries**, and return one final model. They are not per-batch latencies.
Each case has 31 trials of 64 fresh fits. The median column spans the six case
medians; the last column is the lowest case p10 through highest case p90,
not a pooled confidence interval. Individual distributions and observations
remain in the saved JSON.

| Method / batch size | Range of case medians, us | Envelope of case p10-p90, us |
| --- | ---: | ---: |
| Original complete-data forward fit | 17.33-18.22 | 17.01-18.54 |
| Persistent, 1 | 19.53-20.26 | 18.96-20.68 |
| Persistent, 4 | 17.49-18.19 | 17.02-18.63 |
| Persistent, 7 | 17.05-17.86 | 16.90-18.27 |
| Persistent, 16 | 17.12-17.87 | 16.70-18.27 |
| Persistent, 32 | 16.82-17.49 | 16.45-17.98 |
| Persistent, 64 | 16.87-17.50 | 16.39-17.92 |
| Persistent, 256 | 16.80-17.45 | 16.36-18.05 |

Single-dictionary updates take **8.9-13.3% longer** than the original forward
fit. Batch size 4 is within 2.1% of that reference; sizes 7-256 are between
0.35% slower and 5.23% faster. These small differences include changes in
iteration order and allocation: the streaming routine avoids building query
groups but visits every candidate, whereas the original stops searching after
the first consistent key.

Comparing within the persistent implementation separates batch-call overhead
from that implementation change: batch size 1 takes 12.6-17.5% longer than one
batch of 256; batch size 4 takes 2.3-6.0% longer, size 7 takes 0.9-4.2% longer,
and size 16 takes 0.6-3.0% longer. Sizes 32 and 64 are within 1% of size 256
in either direction, with overlapping timing intervals. Thus minibatches work
exactly here, with modest overhead for the smallest batches. No general speed
claim follows for other data sizes or noisy tasks. The previous usual-channel
FW comparison is unchanged; FW was not retimed in this minibatch experiment.

Persistent training payload is **2064 bytes**: 128 optional class entries at
16 bytes each plus 16 one-byte survival flags on this platform. A snapshot
adds the unchanged **288-byte** model payload. These numbers exclude vector
headers, capacities, input data and process overhead. The state retains no
samples and does not grow with the number of batches. It makes **4096 candidate
visits** per complete fit, compared with 2560 for the original early selection
implementation. Snapshots retain every candidate and never store default
predictions as training labels.

Validation includes six native tests, exhaustive coverage of all 81 partial
Boolean datasets and every contiguous partition with both defaults, late
conflicts, persistent failure, empty batches and mixed query order. Formatting
and clippy with warnings denied pass. Lean's full build and channel/global
axiom audits pass: no new `sorry`, no dependence on an unfinished result and
no extra axioms. Generated evidence is retained under `runs/mqar-v1/`; the
exact executable and validation logs are also preserved at the root under
`.kanban/artifacts/151/`.
