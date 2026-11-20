# Protocol: ReRoPE and Leaky ReRoPE in the main model

Frozen before the first cell. The runner compiles this file into the binary,
and every cell records its hash. The findings go to `README.md`.

## Question

What happens when gpt's main research model trains with ReRoPE's or Leaky
ReRoPE's position map in place of LeRoPE's, in a single attention pass?
Does it learn the fixed suite as fast, and does it keep its accuracy on the
length and position-shift splits? Is the one-pass form as cheap as Lean says?

Su (kexue.fm/archives/9708) clamps RoPE's relative position at a window w
(ReRoPE), or continues it beyond w with slope 1/k (Leaky ReRoPE). He trains
with RoPE and switches to ReRoPE only for inference, and he computes two
attention matrices. Here both arms train with the map from the first step,
and each score is computed once.

## Control

The control is gpt 5a37c51, architecture `complex-phase-score-lerope`, from
cohort `runs/main-model-phase-score-01/phase-score-lerope` (source
`7968240c…`, PTX `17c689bd…`). It has:

- Peri-RMS q and k;
- the score ln(t + 1)/D · (Re Z + β Im(Z)²/D), with β = a/√(1 + a²) and one
  zero-initialized a per layer and head;
- LeRoPE: the angle of band l at position p is
  p · 10000^(−2l/D) · e^(α_l), with the α shared by all layers and heads;
- phase-mHC with four streams;
- the plain complex FFN;
- FP16 forward GEMM, TF32 backward GEMM, FP32 everywhere else;
- AdamW with betas (0.9, 0.95), weight decay 0.1, clip 1 and the WSD
  schedule.

This experiment does not train the control again. `validate` replays four
recorded control cells and requires them to match bit for bit.

## Arms

Each arm replaces the relative position r = t − key of a query t and a key
`key` that LeRoPE rotates by. Nothing else changes: LeRoPE's learned
frequencies rotate both regions, and every parameter, initialization and
optimizer group is the control's.

| arm | relative position | source |
| --- | --- | --- |
| `rerope` | r if r < w, else w | Su §ReRoPE, eq. (rerope) |
| `leaky-rerope` | r if r < w, else w + (r − w)/k | Su §Leaky ReRoPE, eq. (leaky-rerope) |

The arms use w = 16 and k = 16:

- **Window.** Su's ablation puts the best w at 1/4 to 1/2 of the training
  length. Here w is a quarter of 64, the training length of the depth and
  recall tasks.
- **Slope.** k = 16 is Su's best slope for w at a quarter of the training
  length (Leaky ReRoPE-w128-k16 at length 512).

Su scales each query by log n; the control's SSMax gain does this already.

The arms compute each score once, in one pass:

- A key inside the window is scored from the query rotated by t and the key
  rotated by `key`, as in the control.
- A key beyond the window is scored from the query rotated by w and the key
  unrotated (ReRoPE), or from the query rotated by (t − w)/k + w and the key
  by `key`/k (Leaky ReRoPE).

Lean proves that this is attention under the arm's relative position for
this score (`Transformer.RoPE.ReRoPE.rerope_phase_attend_eq_onlineMerge`
and `leaky_phase_attend_eq_onlineMerge`).

The far rotations take LeRoPE's learned frequencies, and their gradient
joins the control's. A batch with no sequence longer than w runs the
control's kernels. 25 tasks never exceed 16 tokens, so their cells must
match the control's bit for bit.

## Runs

- **Tasks.** gpt's fixed suite `complex_geometry_suite/suite.json` (sha
  `6d2735e9…`): 35 tasks, 25 in the short tier and 10 in the long tier.
- **Seeds and cells.** Seeds 0, 1 and 2, with the arms rotated per seed. That
  makes 210 cells, run short tier first.
- **Budgets and stopping.** The suite's own budgets, batches and observation
  intervals. A cell stops at the first observation where the selected split's
  exact accuracy is at least 99%, or when its budget runs out.
- **Final evaluation.** Every split, including the length and position-shift
  splits, is evaluated at the end of each cell.
- **Integrity.** No cell is rerun or dropped. Failures are kept. No source
  changes after the first cell; the runner refuses a changed source.

## Analysis

1. Cells that pass, per arm and tier, against the control's 105.
2. Epochs to first99 per task, as the median over seeds, relative to the
   control, on the 10 tasks with a sequence longer than w.
3. Final exact accuracy on the length and position-shift splits, per arm,
   against the control.
4. On the 25 tasks within the window, every arm cell against the recorded
   control cell: the same observations, end and final splits.
5. Seconds per update relative to the control on the 10 longer tasks, against
   the same ratio on the 25 tasks within the window, where both arms do the
   control's work.

Each statement in `README.md` cites the cells it rests on.
