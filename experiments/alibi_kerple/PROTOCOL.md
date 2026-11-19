# Protocol: ALiBi, KERPLE and −log(1 + r) in the main model

Frozen before the first cell. The runner compiles this file into the binary,
and every cell records its hash. The findings go to `README.md`.

## Question

Can a distance bias added to the logit after SSMax's gain replace LeRoPE in
gpt's main research model? And does KERPLE's learned logarithmic bias move
toward −log(1 + r)? Lean's PB5 shows that this bias reaches the recall
threshold at every distance at once
(`Transformer.Grokking.FlatSpectrum.Recall.LogBias`).

## Control

The control is gpt 5a37c51, architecture `complex-phase-score-lerope`, from
cohort `runs/main-model-phase-score-01/phase-score-lerope` (source
`7968240c…`, PTX `17c689bd…`). It has:

- Peri-RMS q and k;
- the score ln(t + 1)/D · (Re Z + β Im(Z)²/D), with β = a/√(1 + a²) and
  one zero-initialized a per layer and head;
- phase-mHC with four streams;
- LeRoPE;
- the plain complex FFN;
- FP16 forward GEMM, TF32 backward GEMM, FP32 everywhere else;
- AdamW with betas (0.9, 0.95), weight decay 0.1, clip 1 and the WSD
  schedule.

This experiment does not train the control again. `validate` replays four
recorded control cells and requires them to match bit for bit.

## Arms

Each arm makes two changes and leaves the rest of the control intact:

- it replaces LeRoPE's rotation with the identity;
- it adds b_h(t − key) to the logit after the gain, as
  `biasedLogScaleAttention` does
  (`Transformer.Grokking.FlatSpectrum.Recall.PositionBias`).

| arm | b_h(r) | learned | source |
| --- | --- | --- | --- |
| `alibi` | −m_h r with m_h = 2^(−8(h+1)/H) | no | ALiBi slopes, as cited in arXiv:2505.22842v4 §2.2. Lean PB2 (`ALiBiWindow`): a window of radius ∝ log n |
| `kerple-log` | −r1 ln(1 + r2 r) | (r1, r2) per head, shared by all layers | arXiv:2205.09921v2 §4, logarithmic variant |
| `kerple-power` | −r1 r^r2, with 0 < r2 ≤ 2 | (r1, r2) per head, shared by all layers | arXiv:2205.09921v2 §4, power variant |
| `log-bias` | −ln(1 + r) | no | Lean PB5 (`LogBias`), the case α = 1 of PB6 (`PowerBias`) |
| `nope` | 0 | no | no position information besides causality |

KERPLE follows its released code (`megatron/mpu/layers.py` at 25be3d71):

- **Placement.** The bias is added after the scaling, not inside Eq. 1's
  1/√d.
- **Initialization.** Per head, bias_p = 2U, then bias_a = U, drawn here from
  `Random(seed + 30000)`.
- **Clamps.** Every parameter is clamped to at least 0.01, and the power
  exponent to at most 2. The clamps apply at initialization and after every
  update.
- **Optimizer.** The parameters sit in the trailing optimizer group, which
  holds LeRoPE's frequencies in the control: L2 clip 1, no weight decay, the
  same schedule and betas.

Every other parameter has the control's bits for the same seed.

## Runs

- **Tasks.** gpt's fixed suite `complex_geometry_suite/suite.json` (sha
  `6d2735e9…`): 35 tasks, 25 in the short tier and 10 in the long tier.
- **Seeds and cells.** Seeds 0, 1 and 2, with the arms rotated per seed. That
  makes 525 cells, run short tier first.
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
   control.
3. Final exact accuracy on the length and position-shift splits, per arm,
   against the control.
4. KERPLE's final (r1, r2) per head, and its bias over the trained distances,
   compared with −α ln(1 + r). For `kerple-log`, the prior is
   (1 + r2 r)^(−r1): is its tail exponent r1 near 1, and is r2 near 1?

Each statement in `README.md` cites the cells it rests on.
