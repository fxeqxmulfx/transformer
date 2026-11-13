# Conventional transformer on the original stress data

Can the usual causal softmax transformer be compared with the convex model on
the original short and long tasks without repeating the memory failure?

The native backend, checkpoint/runner tests and all 35 data/resource checks
pass. [`protocol.json`](protocol.json) freezes every original task with its
original update cap and original success metrics. The original queue stopped
at its decoded-parent gate because relative and absolute summary paths differed.
Task221 recovered that parent with identical models and scores and restarted
the unchanged frozen transformer producer`414431c7` on all35 tasks. All training,
endpoint replays and complete checkpoint-hash checks are now finished under the
original8GiB cap, below the user's64GiB limit. It passes33/35 primary splits and
32/35 across every held-out split. The original failed gate remains preserved.
This comparator makes no new convexity claim.

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
stress reference describes a completed-step warmup; the frozen comparison
explicitly retains the comparator's existing convention. It takes learning
rate, warmup and weight decay from original task metadata and keeps the
validated helper's betas `(0.9, 0.98)` and epsilon `1e-8`, without clipping.
This is not a reproduction of the historical DiffGPT architecture, sampling
trajectory or GPU timing.

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
preserved. The runner below uses these immutable states and preserves partial
attempts. It does not silently restart an interrupted case.

## Original-cap cohort

| Choice | Frozen value |
| --- | --- |
| Inventory | All 35 original tasks, unchanged order and data hashes, seed 1 |
| Architecture | Original task width, head count and one/two/six layers |
| Training | Original minibatch size, sampled uniformly with replacement; one independent MT stream |
| Update limit | Original `case.max_updates`: 20,000, 30,000 or 100,000 |
| Observation | Initial primary metric as a control; then every 250 completed updates and at the cap |
| Stopping | First positive-update observation at 99% original designated validation sequence accuracy, otherwise original cap |
| Endpoint report | Every original split, including train, with both primary and all-held-out 99% outcomes |
| Replay | Every observed checkpoint's primary metrics and stopping rule; all final metrics; exact final optimizer update from saved pre-update state |
| Resources | Serial execution, 2 GiB forecast guard, 8 GiB process cap and 16 GiB disk reserve |

The historical seed-1 records needed up to 75,000 updates. The fresh comparator
therefore retains the original caps. Its first observed success is quantized to
the new fixed 250-update cadence. Initial validation success does not stop
training. Test splits are evaluated at the selected endpoint and never choose
when to stop. This is one minibatch-trained baseline per task; comparison with
the convex full/mini/forward modes can reference the same saved checkpoint and
its full standalone costs.

Every observation retains both its pre-update and post-update state. The full
update log records ordinary batch CE, examples, supervised targets and learning
rate. A separate process rebuilds the original data, verifies every checkpoint
and primary metric, checks the stopping rule and reconstructs the final AdamW
update with the saved sampling RNG. Every parameter and moment bit must match.
This verifies all observed predictions and one optimizer update; it does not
independently rerun the entire gradient trajectory. All checkpoint files are
checked again before the complete 35-case summary is produced.

Preparation requires the complete unchanged decoded cohort and its parent
summaries. Its purpose is to preserve the serial queue; it does not provide
training labels or model parameters to the transformer. Capped failures stay
in the report. An interrupted process retains its inventory, logs and saved
states; a continuation must be explicit and recorded. The cohort summary does
not establish that the convex model passes all 35 tasks.

## Evidence

Twelve native tests pass. Six check one/two/six-layer logits and gradients against the
independent f64 reference; exact two-layer compatibility; streamed whole-batch
gradients and Adam state; causal and loss-mask isolation; invalid-input and
preallocation rejection; and the unchanged original metrics against reference
predictions, including generative and residue-decoded examples. Three check the
checkpoint continuation, integrity and invalid-state cases above. Three check
the full original inventory/caps, capped failure and successful validation
stopping, and saved-run replay. Forging an optimizer moment is detected even
after updating its checkpoint/log hashes while preserving all predictions.
Clippy passes with warnings denied.

All 35 original tasks reconstruct with their original hashes and target
counts, and all pass the buffer forecast. The largest model bound is
358,506,496 bytes (342 MiB), for the six-layer hard-depth task with context 128
and a 16-example chunk. The largest retained dataset-vector footprint is
30,038,304 bytes. Reconstruction itself peaks at 250,092 KiB RSS (244 MiB);
this includes generator/fingerprint intermediates and is **not a measured
training peak**. These checks provide no prediction accuracy or speedup result.

The cohort's additional two-state replay forecast also admits all 35 tasks:
its maximum is 387,244,032 bytes (370 MiB rounded up), and the new data-only
preflight peaks at 250,656 KiB RSS (245 MiB). Its records are in
`runs/cohort-preflight-v1/`. This adds the second saved model's buffers to the
forecast; it still performs no original-data model initialization or scoring.

Generated reports and resources remain in ignored `runs/preflight-v1/`;
the frozen binary and test logs are preserved in `.kanban/artifacts/191/backend-v1/`.
The existing convex cohorts and their serial queue remain unchanged.

## Completed original comparison

Frozen producer `414431c71806aa632807af697b63cd0b8b97c1f9` completed all 35
seed-1 cases and saved endpoint replays. The summary verifies every observed
checkpoint hash, all original metrics and the exact final optimizer update.
There are 33 primary-split passes and 32 all-held-out-split passes; two models
reach their original update caps. Every unsuccessful endpoint remains included.

| Unsuccessful all-split case | Updates | Limiting split | Exact sequence accuracy |
| --- | ---: | --- | ---: |
| Easy depth | 250 | Test length 128 | 97.0703% |
| Sum fourth power | 100,000 | Validation | 92.4336% |
| Long MQAR | 20,000 | Every held-out split | 0% |

Easy depth already passes its original primary validation split at 250 updates,
so the frozen stopping rule selects that endpoint. The other two failures
exhaust their original caps. The recurrence task passes its primary split at
86,000 updates with 99.0434% sequence accuracy; shifted Boolean AND passes at
250 updates. Both pass every held-out split.

The cohort performs 617,750 optimizer updates. Summed run time including data,
evaluation and checkpoint work is 15,173.915 seconds, of which 9,813.293 seconds
are optimizer time. Replay is additional. Peak prepare/run/replay RSS is
269,964 KiB (about 264 MiB). These CPU measurements are not a matched-compute
comparison with the convex families or the historical GPU reference. All 12
native tests, formatting and clippy pass again at `2bab4b89`; the pinned Kernite
checkout remains clean at `a1b6bc2c956e1a5f35960e26b209e1b6c213096c`.

Complete generated evidence stays in `runs/cohort-v1` and the root's
`.kanban/artifacts/191/` and `.kanban/artifacts/221/transformer/`. The complete
summary SHA256 is
`70f26e21fa756fd54416f8ac14116d63aa67950439a6d4048236dae904080057`.
The source fingerprint is
`08007a3dbbfcb4d30e1e1fcfb0ad35942b6feaa34862ea3e458643783c70dd28`.
The reviewed retained-path model passes all 48 of its depth/binding/MQAR cases,
but integration with the new 23-task decoded-consequence cohort is still
pending. Neither result alone establishes a full35 convex-model pass.

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

Inspect the training/replay forecast for a single original task, without
training, with `cohort-preflight ORIGINAL_TASK`. After the frozen decoded
cohort is complete, launch the full comparison from a fresh destination:

```sh
TRANSFORMER_PRODUCER_COMMIT=$(git rev-parse HEAD) bash experiments/convex_stress_transformer/run_cohort.sh \
  experiments/convex_stress_transformer/runs/cohort-v1 \
  experiments/convex_stress_decode/runs/cohort-v1
```

Use the frozen binary and producer commit preserved with the run. Generated
states, resource records and reports remain ignored and root-linked.
