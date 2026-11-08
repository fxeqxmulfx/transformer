# Hidden local-update direction and step size

Does the hidden block A need a different direction, a smaller step, or both?
At the eight failed local-training endpoints, the **current local-gradient
direction reduces training and validation CE at every tested scale**.
Shrinking the actual AdamW displacement helps two additional validation
cases, but leaves five pointing uphill. This identifies a direction problem
at these states; it does not establish a repaired training trajectory.

The [protocol](PROTOCOL.md) was frozen at `0f233d16` before outcomes. All
40 states from [task #193](../forward_alignment/README.md) are retained:
eight initializations and eight endpoints for each of four training rules.
Seeds are 7/11; arms are real SwiGLU, complex diagonal, complex product
radial and complex product plain. No training was extended.

## Fixed comparison

Let `d` be A's realized original AdamW displacement and `g` its current
local gradient, both in independent real coordinates. Compare `d` with
`-norm(d)*g/norm(g)` at scales **1, 1/4, 1/16, 1/64, 1/256**. This crosses
two directions with five scales at every state: **400 probes**. Each probe
starts from the same saved state, rounds independent parameters to FP32,
rebuilds complex aliases, and holds E and B bitwise fixed.

The second direction omits moment history, adaptive coordinate scaling and
an extra decay term together; this study does not isolate their individual
effects. Its radius still comes from AdamW, so it is not yet a cheaper
standalone optimizer. Scaling realized motion is also not a claim of
equivalence to rerunning AdamW at a different learning rate.

Loss improvement/worsening uses the frozen threshold
`1e-7*(1+abs(before))`; other changes are near-flat. This is a descriptive
dead zone, not a certified rounding bound. Raw losses, first-order predictions,
remainders and exact correct counts remain in the report. Predictions use
the actual rounded displacement. No scale is selected per state.

## Failed local-training endpoints

Each entry below counts improvements out of eight states. Every probe in
this table lowers A's own frozen-boundary local loss. CE and pair columns
evaluate the complete split, not just the update minibatch.

| Direction | Scale | Training CE improves | Validation CE improves | Validation pair improves | Validation correct-count change range |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | 1 | 0/8 | 1/8 | 1/8 | -19 to +5 |
| AdamW | 1/4 | 2/8 | 2/8 | 2/8 | -4 to +2 |
| AdamW | 1/16 | 2/8 | 2/8 | 3/8 | -3 to +1 |
| AdamW | 1/64 | 2/8 | 3/8 | 3/8 | 0 to +1 |
| AdamW | 1/256 | 2/8 | 3/8 | 3/8 | 0 |
| Current gradient | 1 | 8/8 | 8/8 | 6/8 | -4 to +14 |
| Current gradient | 1/4 | 8/8 | 8/8 | 7/8 | -1 to +4 |
| Current gradient | 1/16 | 8/8 | 8/8 | 7/8 | -3 to +1 |
| Current gradient | 1/64 | 8/8 | 8/8 | 7/8 | -1 to +1 |
| Current gradient | 1/256 | 8/8 | 8/8 | 7/8 | 0 to +1 |

At scales 1/64 and 1/256, the actual AdamW validation CE changes have the
same signs as all eight first-order predictions: three improve and five
worsen. Shrinking the step recovers the two descent predictions overturned
at full scale, but cannot fix the five observed ascent directions.

At scale 1 the current-gradient direction changes validation CE by
-0.03311 to -0.01233, versus -0.00735 to +0.02472 for AdamW at the same
requested norm. At scale 1/256 its changes are -2.25e-4 to -7.12e-5.
Training pair loss also improves in all eight cases at every gradient scale;
validation pair loss does not. Lower CE therefore does not imply that every
reported objective improves.

Accuracy is not repaired. At gradient scale 1, three of eight validation
correct counts fall despite lower CE; changes range from -4 to +14 out of
4705 examples. The resulting accuracies remain only 2.19–10.80%.
The smallest gradient scale changes one case by one correct prediction and
leaves the other seven unchanged. These are one-step probes of poorly
trained models, not new endpoint accuracies after training.

## All 40 states

Cells give validation CE **improved / near-flat / worsened** counts, each
out of eight. All rows apply the same local A objective at the named state;
the AdamW direction retains that origin's saved moments. This is not a
ranking of the methods that produced those states.

| State origin | Direction | 1 | 1/4 | 1/16 | 1/64 | 1/256 |
| --- | --- | --- | --- | --- | --- | --- |
| Initial | AdamW | 5/0/3 | 5/0/3 | 5/0/3 | 4/2/2 | 0/8/0 |
| Initial | Current gradient | 7/0/1 | 7/0/1 | 7/0/1 | 4/3/1 | 1/6/1 |
| CE BP endpoint | AdamW | 0/0/8 | 2/0/6 | 2/0/6 | 2/0/6 | 2/0/6 |
| CE BP endpoint | Current gradient | 0/0/8 | 1/0/7 | 1/0/7 | 1/0/7 | 2/0/6 |
| Pair BP endpoint | AdamW | 0/0/8 | 1/0/7 | 1/0/7 | 1/0/7 | 1/0/7 |
| Pair BP endpoint | Current gradient | 0/0/8 | 0/0/8 | 1/0/7 | 2/0/6 | 3/0/5 |
| Auxiliary BP endpoint | AdamW | 1/0/7 | 1/0/7 | 1/0/7 | 1/0/7 | 1/0/7 |
| Auxiliary BP endpoint | Current gradient | 3/0/5 | 4/0/4 | 4/0/4 | 4/0/4 | 4/0/4 |
| Local pair endpoint | AdamW | 1/0/7 | 2/0/6 | 2/0/6 | 3/0/5 | 3/0/5 |
| Local pair endpoint | Current gradient | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 |

All 400 probes lower their own local objective, including those that worsen
terminal CE. Initialization remains mixed; the smallest initial steps are
often near-flat. The favorable local-endpoint result does not transfer
uniformly to the other origins. The fixed hidden target's task alignment
therefore remains an open issue even after changing optimizer direction.

## Next experiment and limits

The [standalone training follow-up](../forward_normalized/README.md) freezes
a 1%-of-A-norm current-gradient rule, retaining the objectives and E/B updates,
and compares 24 from-scratch runs on fresh seeds 17/23. It does not repair
training: validation accuracy is 2.23–11.63%, with worse CE than original
local-pair in all eight cases. This reinforces the limit of transferring
the present same-state, AdamW-radius observation to a new training rule.
The next control keeps the A optimizer while restoring full-gradient credit.
No validation gradient or observed loss change is available as a free
training-time safeguard.

This study does not establish the cause of the entire failed trajectory,
a better loss, a generalization guarantee, or a speed/memory advantage.
It leaves complex multiplication unchanged. The separate 90.38% result is
forecast certificate coverage, not classifier accuracy.

## Verification, cost and reproduction

All **40 states and 400 probes pass** with no zero directions, failures or
caps. All 40 source full-local state hashes replay exactly; the full-scale
A probes reproduce A bits, prior metrics and frozen local objectives.
All baseline metrics replay. Every probe preserves parameter ties and
bitwise E/B identity. All **59 input files** remain unchanged.

All **320** independent FP64 directional checks pass at the frozen tolerance;
maximum absolute error is 4.45e-7. All **9600** group/objective/scope coordinate
pairings pass, maximum error 4.94e-15; E/B pairings are zero for these A-only
probes. Maximum unrounded norm-matching error is 6.14e-15. The largest FP32
rounding-error norm is 1.60e-7, and the largest relative rounding error over
the cohort is 0.1321%. Fifteen Rust tests, formatting and warning-free Clippy
pass. No Lean statement changed.

Cost: 9280 source forward batches, 1680 terminal backward batches, 40 each
of output-local, hidden-local and vocabulary Jacobians, and 40 source
optimizer steps to construct the directions. The 400 parameter probes are
not optimizer steps. CPU verification evaluates 360 backbones on 1525920
rows and 440 frozen local objectives. The cohort takes 44.59 seconds under
the 600-second cap. These charged backward-assisted diagnostics establish
no training efficiency result.

The build reuses the pinned original trainer, local kernels and alignment
coordinate mapping. It requires the same local native dependencies and
immutable input runs as the preceding experiments. No Python is used.

```sh
cargo test --offline --locked --manifest-path experiments/forward_step/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/forward_step/Cargo.toml -- -D warnings
cargo fmt --check --manifest-path experiments/forward_step/Cargo.toml
rustfmt --edition 2024 --check experiments/forward_step/src/instrumentation.rs
cargo build --offline --locked --release --manifest-path experiments/forward_step/Cargo.toml
mkdir -p experiments/forward_step/runs
experiments/forward_step/target/release/rotate90-complex-kernite \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/forward_training/runs/pilot-01 \
  experiments/forward_alignment/runs/diagnostic-01 \
  experiments/forward_step/runs/diagnostic
jq -f experiments/forward_step/summary.jq \
  experiments/forward_step/runs/diagnostic/report.json \
  > experiments/forward_step/runs/diagnostic/summary.json
```

Use a fresh output directory; the executable refuses to overwrite evidence
or write inside any input directory. Executed evidence is retained at
`experiments/forward_step/runs/diagnostic-01/` in the primary checkout, with
check logs and the executable in `.kanban/artifacts/194/`. The compiled
source digest is
`5d131208fb4e8bf0dbc7cb804f40a34f7156fcdc42fa980f2ecf6b1e8982b14a`.
Generated reports, summaries, logs and binaries are ignored and uncommitted.
