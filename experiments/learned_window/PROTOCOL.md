# Protocol: a learned window and slope in the main model

Draft. It is frozen before the first cell: the runner compiles this file into
the binary, and every cell records its hash. The findings go to `README.md`.

## Question

ReRoPE and Leaky ReRoPE (`../rerope`) give every head the same window
w = 16 and the same slope. Here each head of each layer learns both. A query
t and a key j, r = t − j apart, are rotated by

    p(r) = min(r, w) + s · max(r − w, 0)

(`Transformer.RoPE.LeReRoPE.windowPosition`). The slope s = 0 is ReRoPE,
s = 1/k Leaky ReRoPE and s = 1 RoPE (`windowPosition_zero`,
`windowPosition_inv`, `windowPosition_one`). Does each head open its window
where the task needs it, and keep it narrow where it does not? Does the
model then learn the far tasks as fast as the control, and keep its accuracy
on the length and position-shift splits?

Lean proves that the loss has no force that closes a window. A head whose
window covers every distance of a batch gets no gradient in its window or
its slope (`hasDerivAt_loss_of_window_open`). At s = 1 the window gets none
at all (`hasDerivAt_window_loss_of_slope_one`).

## Control

The control is gpt 5a37c51, architecture `complex-phase-score-lerope`, from
cohort `runs/main-model-phase-score-01/phase-score-lerope` (source
`7968240c…`, PTX `17c689bd…`), as in `../rerope/PROTOCOL.md`. The recorded
`rerope` and `leaky-rerope` arms of `../rerope` are compared too. They share
this experiment's suite, seeds and initial weights.

This experiment does not train the control again. `validate` replays
recorded control cells and requires them to match bit for bit.

## Arm

`learned-window` keeps every parameter of the control and adds two per layer
and head, after LeRoPE's frequencies:

- the window w = 16 · e^ω;
- the slope s = 1 / (1 + 15 e^(−σ)), between 0 and 1.

ω and σ start at 0, so every head starts as Leaky ReRoPE with w = 16 and
k = 16, the fixed arm's values. Its far rotations are formed on the GPU from
w and s, so they agree with the fixed arm's to rounding, not bit for bit.
ω and σ join LeRoPE's frequencies in their optimizer group: AdamW without
weight decay, with its own clip at 1.

Each score is computed once. A key with t − j < w is scored from the query
rotated by t and the key by j, as in the control. A key beyond the window is
scored from the query rotated by w + s(t − w) and the key by s·j. The
difference of the two is p(t − j) (`route_eq_windowPosition`). The rotations
take LeRoPE's learned frequencies θ̂ in both regions.

The gradients of w and s are read off the far rotations, as Lean states them
(`hasDerivAt_window_chunk_slope`, `hasDerivAt_window_chunk_window`). With
τ = Im(conj(x̃) dx̃) the torque of a rotated chunk x̃ and its adjoint dx̃:

- ∂L/∂s = Σ_bands θ̂ (Σ_t (t − w) τ(far query t) + Σ_j j τ(far key j));
- ∂L/∂w = (1 − s) Σ_bands θ̂ Σ_t τ(far query t).

∂L/∂w does not exist when a pair sits at r = w. The kernels' test
t − j < ⌈w⌉, the same as t − j < w for integer distances, then takes the
derivative from below.

A task whose sequences never exceed 16 tokens has no far pair at w = 16. Its
windows and slopes get no gradient and do not move, so it runs the
control's kernels. Its 25 tasks must match the control's cells bit for bit.
The other ten tasks route every batch, since a learned window may narrow
below any length.

## Runs

- **Tasks.** gpt's fixed suite `complex_geometry_suite/suite.json` (sha
  `6d2735e9…`): 35 tasks, 25 in the short tier and 10 in the long tier.
- **Seeds and cells.** Seeds 0, 1 and 2: 105 cells, short tier first.
- **Budgets and stopping.** The suite's own budgets, batches and observation
  intervals. A cell stops at the first observation where the selected split's
  exact accuracy is at least 99%, or when its budget runs out.
- **Final evaluation.** Every split, including the length and position-shift
  splits, is evaluated at the end of each cell.
- **Integrity.** No cell is rerun or dropped. Failures are kept. No source
  changes after the first cell; the runner refuses a changed source.

## Analysis

1. Cells that pass, per tier, against the control's 105 and each ReRoPE
   arm's 105.
2. Epochs to first99 per task, as the median over seeds, relative to the
   control and to Leaky ReRoPE, on the 10 tasks with a sequence longer than
   16.
3. Final exact accuracy on the length and position-shift splits, against the
   control and the ReRoPE arms.
4. The final window and slope of every head in the 30 far cells: how many
   heads opened beyond the longest training distance, how many narrowed
   below 16, and how far the slopes moved toward 0 or 1.
5. On the 25 tasks within the window, every cell against the recorded
   control cell: the same observations, end and final splits.
6. Seconds per update relative to the control and to Leaky ReRoPE on the 10
   far tasks.

Each statement in `README.md` cites the cells it rests on.
