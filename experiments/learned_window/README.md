# A learned window and slope in the main model

## Question

ReRoPE and Leaky ReRoPE ([`rerope`](../rerope)) give every head the same
window w = 16 and the same slope. Here each head of each layer learns both. A
query t and a key j, r = t − j apart, are rotated by
p(r) = min(r, w) + s · max(r − w, 0). The slope s = 0 is ReRoPE, s = 1/k Leaky
ReRoPE and s = 1 RoPE (`Transformer.RoPE.LeReRoPE.windowPosition`).

Does each head open its window where the task needs it, and keep it narrow
where it does not? Lean proves that nothing in the loss closes a window. A
head whose window covers every distance of a batch gets no gradient in its
window or slope (`hasDerivAt_loss_of_window_open`). At s = 1 the window gets
none at all (`hasDerivAt_window_loss_of_slope_one`). The runs show what
training does with that freedom.

[`PROTOCOL.md`](PROTOCOL.md) holds the plan, frozen before the first cell.

## Runs

| arm | position of r = t − j | parameters added |
| --- | --- | --- |
| control (gpt `phase-score-lerope`, recorded) | r | none |
| `rerope` ([`rerope`](../rerope), recorded) | min(r, 16) | none |
| `leaky-rerope` ([`rerope`](../rerope), recorded) | r if r < 16, else 16 + (r − 16)/16 | none |
| `learned-window` | min(r, w) + s · max(r − w, 0) | w and s per layer and head |

The window is w = 16 e^ω and the slope s = 1/(1 + 15 e^(−σ)). ω and σ start
at 0, so every head starts as the fixed Leaky arm. They join LeRoPE's
frequencies in their optimizer group: AdamW without weight decay, clipped
on its own.

Of the 35 tasks, 25 never exceed 16 tokens. Their windows get no gradient,
so their cells must equal the control's bit for bit. The other ten reach
beyond the window (see [`rerope`](../rerope/README.md#runs)).

## Design

The crate starts as a copy of [`rerope`](../rerope), whose source stays
frozen while its cohort runs. The fixed arms and their host far tables are
gone from it.

- **Parameters.** `layout.rs` and `optimizer.rs` are local copies. ω and σ
  per layer and head follow LeRoPE's frequencies, in their optimizer group.
- **Tables.** Before each forward, `window_rotary::state` forms w, s and ⌈w⌉
  per layer and head, and `window_rotary::table` every layer's far
  rotations. The far query of row t sits at min(t, w) + s · max(t − w, 0),
  the far key of row j at s·j. Rows t < w have no far pair, so their far
  query is never read. The angles are formed on the GPU, so the initial far
  rotations agree with the fixed Leaky arm's to rounding, not bit for bit.
- **Routing.** A cell whose sequences can be longer than 16 tokens routes
  every batch through the routed kernels, whatever its length: a window may
  narrow below it. A cell within 16 tokens routes none. It has no far pair
  at w = 16, so its windows never move.
- **Attention.** The routed kernels read each head's window as the integer
  ⌈w⌉: t − j < w holds exactly when t − j < ⌈w⌉.
- **Gradients.** `torques` sums each head and band's torques. `finish` gives
  the window's, the slope's and the far part of the frequencies' gradients
  (`hasDerivAt_window_chunk_slope`, `hasDerivAt_window_chunk_window`).
- **Validation.** An f64 oracle for the attention block, whose heads get
  windows and slopes that differ, one window open, and its gradients: of q,
  k and v, the frequencies, the score coordinates, ω and σ. Central
  differences of the loss in the `fp32-affine-reference` build, at windows
  off the integers, away from the kink. Replays of the control's cells on
  the 25 tasks within the window. A CPU test assembles the experiment's
  kernels with `ptxas` for sm_86.

## Commands

```
cargo build --offline --release
cargo build --offline --release --features fp32-affine-reference --target-dir target/fp32
mkdir -p runs/cohort-1-bin
cp target/release/learned-window target/release/main-phase-score-native runs/cohort-1-bin/
CONTROL=<gpt>/runs/main-model-phase-score-01/phase-score-lerope
./target/fp32/release/learned-window validate $CONTROL runs/validation-fp32-1
./runs/cohort-1-bin/learned-window validate $CONTROL runs/validation-1
./runs/cohort-1-bin/learned-window suite runs/cohort-1 short
./runs/cohort-1-bin/learned-window suite runs/cohort-1 long
./runs/cohort-1-bin/learned-window analyze runs/cohort-1 $CONTROL ../rerope/runs/cohort-1
```

As in `rerope`, the cohort runs from a copy of the binaries: the copy that
was validated is the copy that trains. `analyze` prints the six comparisons
of `PROTOCOL.md` § Analysis as JSON. The fourth reads each head's final
window and slope from the cell's last checkpoint, written when it finished.

GPU work waits until the `rerope` cohort finishes.

## Findings

Pending.
