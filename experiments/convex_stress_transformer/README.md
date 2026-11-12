# Conventional transformer on the original stress data

Can the usual causal softmax transformer be compared with the convex model on
the original short and long tasks without repeating the memory failure?

The native backend and all 35 data/resource checks pass. **No transformer
training cohort or new stress accuracy is reported here yet.** Freeze training
budgets, seeds, observations and checkpoint replay before starting that cohort.
This is a conventional comparator; it makes no new convexity claim.

## Model and bounded execution

`model.rs` ports `convex_recall_compare/src/transformer_model.rs` at `da0714aa`,
which previously extended `grok_division` at `891e4525` to longer contexts.
It retains learned embeddings, fixed sinusoidal positions, causal softmax
attention, post-LayerNorm residual blocks, ReLU feed-forward layers of width
four times the model width, and an untied full-vocabulary readout. Parameters,
activations and gradients use f32. The independent numerical reference uses
f64; ordinary evaluation CE uses the original benchmark's f64 metric code.

| Component | Existing comparison | Stress adapter |
| --- | --- | --- |
| Layers | Exactly two | Original task's one, two or six |
| Width and heads | Width 32, four heads | Original task metadata |
| Supervision | One common mask and input length | Original variable lengths and masks, ignoring only `-100` |
| Gradient storage | Entire assigned worker batch | Serial chunks of at most 32 examples, accumulated before one AdamW update |
| CE weighting | Mean over targets | Same mean over every supervised target in the complete optimizer batch |
| Evaluation | Recall/division helper | Unchanged stress metrics, including rollout, EOS and residue pooling |

Every nonfinal block retains the complete causal prefix. The final block
computes just the requested query positions. This also works for one layer.
Initialization visits every block with the existing RNG conventions; the
two-layer parameters, logits and gradients are bit-identical to the earlier
implementation on the regression fixture. Iteration syntax changes only to
satisfy current Clippy checks.

The optimizer helper remains the existing AdamW implementation. Its warmup
uses `t-1` for update `t`, so the first learning rate is zero. The historical
stress reference describes a completed-step warmup; a future comparison must
explicitly retain or account for this difference. This backend is not a
reproduction of the historical DiffGPT architecture or GPU timing.

Before creating model vectors, `memory.rs` checks a conservative buffer bound
covering parameters, Adam state, f64 gradient accumulation, transposed weights,
retained activations, temporary matrices, capacity growth and Kernite packing.
The model buffer limit is 2 GiB. The launcher imposes an independent 8 GiB
address-space limit. Dataset reconstruction is measured separately. Full-batch
training can use the same streamed accumulation; its activations do not grow
with the total number of training examples. Streamed sums can differ by f32
roundoff, which is tested against the independent reference.

Kernite CPU dependencies use the existing checkout at
`/home/misha/rust_projects/kgen`, commit
`a1b6bc2c956e1a5f35960e26b209e1b6c213096c`. The checkout was clean during the
checks. The original generators and metrics are imported from
`convex_stress_minibatch`; their frozen data hashes are checked during rebuild.
No held-out labels choose the architecture, masks or resource limits.

## Streamed checkpoint state

`checkpoint::save_new` preserves f32 parameter bits, both Adam moments, the
complete MT sampling RNG, dimensions, optimizer constants and progress counts.
Its caller supplies a SHA-256 binding for the frozen case, source and data
specification. The binary format streams tensors directly with a small bounded
header and checksum, avoiding a complete checkpoint byte buffer or copies of
all three tensors. It does not use the earlier f64 checkpoint representation.

The destination is immutable: a synced partial file is published by a hard link
that cannot replace an existing checkpoint. Interrupted partial files remain
for explicit inspection. Loading checks the metadata and exact payload size
before allocating model vectors, rejects nonfinite parameters or invalid
moments, and verifies the checksum. The caller should drop its previous model
before loading a replacement and compare the returned digest with its run
record. This library does not automatically resume an interrupted experiment.

Full-batch and sampled-minibatch fixture trajectories continue bit for bit
after saving/restoring at update five, for one, two and six layers. Restored
logits also agree with the independent f64 reference. Corrupted tensor bytes,
truncation, oversized metadata/payloads, mismatched provenance bindings and
invalid optimizer/progress state are rejected. Existing checkpoint files are
preserved. This supplies state recovery for the future runner; original-data
training and its observation/stopping protocol remain pending.

## Evidence

Nine native tests pass. Six check one/two/six-layer logits and gradients against the
independent f64 reference; exact two-layer compatibility; streamed whole-batch
gradients and Adam state; causal and loss-mask isolation; invalid-input and
preallocation rejection; and the unchanged original metrics against reference
predictions, including generative and residue-decoded examples. Three check the
checkpoint continuation, integrity and invalid-state cases above. Clippy passes
with warnings denied.

All 35 original tasks reconstruct with their original hashes and target
counts, and all pass the buffer forecast. The largest model bound is
358,506,496 bytes (342 MiB), for the six-layer hard-depth task with context 128
and a 16-example chunk. The largest retained dataset-vector footprint is
30,038,304 bytes. Reconstruction itself peaks at 250,092 KiB RSS (244 MiB);
this includes generator/fingerprint intermediates and is **not a measured
training peak**. These checks provide no prediction accuracy or speedup result.

Generated reports and resources remain in ignored `runs/preflight-v1/`;
the frozen binary and test logs are preserved in `.kanban/artifacts/191/backend-v1/`.
The existing convex cohorts and their serial queue remain unchanged.

## Reproduce

From the repository root, with the pinned Kernite checkout available:

```sh
cargo test --locked --manifest-path experiments/convex_stress_transformer/Cargo.toml -- --test-threads=1
cargo clippy --locked --manifest-path experiments/convex_stress_transformer/Cargo.toml --all-targets -- -D warnings
cargo build --locked --release --manifest-path experiments/convex_stress_transformer/Cargo.toml
bash experiments/convex_stress_transformer/run_preflight.sh experiments/convex_stress_transformer/runs/preflight-fresh
```

The output directory must be fresh. Each dataset runs in its own capped
process; the terminal summary appears only after all 35 complete. The command
performs shape checks and reconstruction, with no training or scoring.
