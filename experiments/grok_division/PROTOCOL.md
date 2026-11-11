# Division mod 97 under the author's Adam and AdamW: protocol

Frozen on 2026-11-11, before any training run. The runner was committed with this file.

## Sources

- The paper: `papers/arXiv-2201.02177v1/main.tex`, Figure 1 (left) with its caption and the
  appendix *Additional experimental details* (binary operations; model and optimization).
- The author's implementation: `https://github.com/openai/grok` at
  `3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3` (MIT), files `grok/data.py`,
  `grok/transformer.py`, `grok/training.py`. It was read; it was never executed.
- Kernite at `a1b6bc2c956e1a5f35960e26b209e1b6c213096c` for the matrix-product kernel.

The paper fixes the task, the sizes and the optimizer's main constants. Everything else below
is fixed by the code. Where the two disagree the runner follows the code, which produced the
paper's figures, and the disagreement is listed.

## The task and its tokens

- Equations `a / b = c` in the field of 97 elements, `0 ≤ a < 97`, `0 < b < 97`: 9312 of them
  (paper, appendix; code, `_make_binary_operation_data("/")`, which lists them with `c` in the
  outer loop, `b` in the inner one and `a = b c mod 97`).
- The vocabulary has 239 tokens (`ArithmeticTokenizer.get_tokens`): `<|eos|>` (0), `=` (1),
  the 20 operator names in sorted order (`/` is 10), the residues `0..96` (22 to 118), the 120
  elements of S5. An equation is the seven tokens `<|eos|> a / b = c <|eos|>`; the model reads
  the first six and is trained on the next token (`ArithmeticIterator.__next__`). The 142
  tokens that division never shows still have embedding rows and readout columns, and the
  softmax runs over all 239.

## The split

- Paper: "we chose a fraction of all available equations at random"; 50% in Figure 1.
- Code: `make_data(operator, shuffle=True, seed=0)` shuffles the list with
  `np.random.RandomState(seed=0)`, and `splits` takes the first `round(9312 · 0.5) = 4656`
  equations for training. The seed is the constant 0: every run of the code has the same split,
  whatever its `--random_seed`.
- Runner: the same shuffle (MT19937 as NumPy's legacy `RandomState` drives it), seeded with the
  run's seed. Seed 0 is the code's split; seeds 1 and 2 are other splits, as the paper's text
  describes. **Deviation D1**: the code never draws them.

## The model (`grok/transformer.py`)

- `embed`: token embedding, 239 rows of 128, plus the sinusoidal encoding
  `sin(pos / 10000^(i/128))` at even channels `i` and `cos(pos / 10000^((i-1)/128))` at odd
  ones, computed in doubles and rounded to `f32`.
- Two `DecoderBlock`s, post-norm: `a1 = LayerNorm(x + Attention(x))`,
  `a2 = LayerNorm(a1 + FFN(a1))`. `LayerNorm` has a gain and a bias and adds `1e-5` to the
  variance.
- `MultiHeadAttention`: 4 heads of width 32. Each has maps `Wq`, `Wk`, `Wv` from 128 to 32
  without bias; the scores are `q · k / sqrt(32)`, positions after the query's are masked, the
  rest go through a softmax; the heads' outputs are concatenated and mapped by `Wo`, 128 to
  128, without bias.
- `FFN`: 128 to 512 without bias, ReLU, 512 to 128 without bias. Dropout is 0.
- Readout: a linear map from 128 to 239 without bias, not tied to the embedding.
- Parameters: 424 832 outside the embedding (the paper's "about 4 · 10^5"), 455 424 in all.
- Initialization (PyTorch's defaults; the paper is silent): embedding standard normal; every
  linear map uniform on `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`; gains 1, biases 0.

## The loss and the accuracy

- Paper: "calculated loss and accuracy only on the answer part of the equation".
- Code (`_step`): the cross-entropy with `reduction="mean"` over the targets to the right of
  `=`, which are the answer `c` **and the closing `<|eos|>`**: a mean over `2 · batch` terms.
  Its accuracy counts an equation only if both tokens are predicted.
- Runner: the code's loss. **Deviation D2** from the paper's wording is the code's own. Every
  evaluation records the answer alone as well as the code's quantities. The flag
  `--answer-only` trains on the answer's term alone, for the sensitivity cohort below.

## The optimizer (`CustomAdamW`, `_scheduler_lr`, `configure_optimizers`)

- One class serves both optimizers: `betas = (0.9, 0.98)`, `eps = 1e-8`, weight decay in the
  form `to_zero`. Update `t = 1, 2, …` at learning rate `lr_t` and gradient `g`:
  `p ← p (1 - lr_t · wd)` if `wd > 0`; `m ← 0.9 m + 0.1 g`; `v ← 0.98 v + 0.02 g²`;
  `p ← p - lr_t / (1 - 0.9^t) · m / (sqrt(v) / sqrt(1 - 0.98^t) + eps)`.
- All parameters are in one group: with `wd > 0` the normalization gains and biases and the
  embedding decay too.
- `lr_t = min(t - 1, 10) / 10 · 10⁻³`: `LambdaLR` has stepped `t - 1` times at update `t`. The
  first update has rate 0 and only starts the moments; update 11 is the first at `10⁻³`. This
  is the paper's "linear learning rate warmup over the first 10 updates".
- The paper's constants are used: rate `10⁻³`; weight decay 1 (AdamW) or none (Adam). The
  code's command-line defaults `--weight_decay 0` and `--train_data_pct 5` are not the paper's
  settings. The paper gives no epsilon; `1e-8` is the code's.
- Batches (`ArithmeticIterator`): size `min(512, ceil(4656 / 2)) = 512`; each epoch draws a
  fresh uniform permutation of the training set and cuts it in order: nine batches of 512 and
  a tenth of 48, which is an update like the others.
- The budget counts optimizer updates, the first one included.

## Randomness

A run with seed `s` has three independent MT19937 streams, and nothing else in it is random:

| Stream | Seed | Draws |
| --- | --- | --- |
| split | `s` | NumPy's legacy shuffle of the 9312 equations |
| initialization | `1000003 (s + 1) mod 2³²` | the parameters, in the order of the flat vector of `src/layout.rs`: normals by Box-Muller from two 53-bit doubles, uniforms from one |
| batches | `2000003 (s + 1) mod 2³²` | a legacy shuffle of `0..4655` for each epoch |

The laws are the code's; the streams are not PyTorch's, so no run reproduces a PyTorch run
bit for bit. Equal seeds give equal splits, initial parameters and batches in both cohorts.

## Arithmetic

`f32` parameters, activations, gradients and moments, as in the code. A batch is cut into
chunks of 32 equations in order; products are chains of fused multiply-adds in the order of
Kernite's kernel; a chunk's gradient sums its rows in order and the batch's sums the chunks in
order, so the result does not depend on the number of threads. Layer-normalization moments are
accumulated in doubles. The loss and its gradient in the logits are PyTorch's in `f32`
(Amendment 1). The optimizer's scalars are computed in doubles and rounded to `f32`.
Evaluations report cross-entropies computed in doubles from the `f32` logits.

## Cohorts

| Label | Paper's setting | Weight decay | Updates | Seeds | Checkpoints every |
| --- | --- | ---: | ---: | --- | ---: |
| `adam` | Figure 1 and the appendix: "Adam optimizer with no weight decay", budget `10⁶` | 0 | 1 000 000 | 0, 1, 2 | 50 000 |
| `adamw` | the appendix's default: "AdamW … weight decay 1", budget `10⁵` | 1 | 100 000 | 0, 1, 2 | 10 000 |
| `adamw --answer-only` | sensitivity to D2; run after the two above if compute allows | 1 | 100 000 | 0, 1, 2 | 10 000 |
| `adam`, loss in doubles | sensitivity to Amendment 1: the trainer of commit `b8ff967f`; if compute allows | 0 | 1 000 000 | 0, 1, 2 | 50 000 |

The two main cohorts are separate protocols. `adamw` is not evidence about the trajectory of
`adam`, and neither is the repository's earlier GPTMini or Rotate90 work, which trains other
models.

## What a run records

After updates 0, 10, 20, 50, every 100 up to 10 000, every 1000 after that, and the last, the
whole training set and the whole held-out set are evaluated:

- `history.jsonl`: the mean cross-entropy of the answer and of the closing token, the code's
  loss (their mean), the answer's accuracy, the code's accuracy, the number of wrong answers,
  and the least, 5% and median margin, for both sets; the parameters' norm; seconds trained.
- `heldout_predictions.bin`, `heldout_margins.bin`: for every held-out equation, in the order
  of `heldout.txt`, the predicted answer and its margin.
- A prediction is the first of the largest logits over all 239 tokens. The margin of an
  equation is the answer's logit less the largest logit of any other token.
- Checkpoints hold the parameters, both moments and the sampler; a run resumed from one is
  the uninterrupted run bit for bit, which the tests check on disk.

## Terminal window and outcomes

The terminal window is the recorded steps of the last tenth of the budget: 11 observations
from 90 000 to 100 000 for `adamw`, 101 from 900 000 to 1 000 000 for `adam`. For each seed
`report` gives:

- the first recorded step with training accuracy at least 99%, and the first from which it
  stays there; the same for held-out accuracy; the first from which every held-out answer is
  right at every later recorded step;
- over the window: the least, mean and largest held-out accuracy, the largest number of wrong
  answers, the least margin, the largest held-out and training cross-entropy of the answer.

A seed's outcome is *exact* if every held-out answer is right at every observation of the
window, *at least 99%* if the least accuracy of the window is, and *not generalized within the
budget* otherwise. Every seed is reported, failures included; a run whose loss stops being
finite is reported as failed, and no seed is replaced.

## Checks before training

`cargo test --release` (31 tests) and `grok-division check`. No expected value comes from the
code it checks: the generator is held to published outputs of MT19937 and NumPy and to split
fingerprints that NumPy itself produced earlier in this repository; the model in doubles to
closed forms and to finite differences of its own forward pass; the `f32` model to the model
in doubles, at the paper's sizes; the optimizer to the recurrence written out in doubles, to
the schedule's values, and to closed forms at a constant and at a zero gradient.

## What the runs cannot show

A run is a finite horizon on one machine in `f32`. Three seeds that reach perfect held-out
accuracy do not prove that held-out accuracy tends to 1, for this optimizer or any other; a
seed that fails within the budget does not prove that it never would.

## Amendment 1 (2026-11-11): the loss gradient in single precision

The trainer of the first launch (commit `b8ff967f`) computed the softmax of the logits and
its gradient in doubles. Within 10⁴ updates the `adam` runs had a training cross-entropy near
`10⁻⁹`. There PyTorch's `f32` arithmetic is not a rounding of the exact gradient:
`log_softmax` is `x - max - ln(Σ exp(x - max))`, and its backward pass is
`grad_output - exp(output) · Σ grad_output`. Once the other classes' probabilities sum to
less than half a unit in the last place of 1, the sum rounds to 1, the loss to 0 and the
target's entry of the gradient to exactly 0, while every other class keeps a positive entry.
The source's runs have this; doubles hide it.

The first launch was stopped before any terminal observation, at 14 000 updates (`adam`) and
8 000 (`adamw`); its outputs are kept aside and belong to no cohort. The trainer now computes
the loss and its gradient in the logits in `f32` by PyTorch's formulas (`loss_gradient` in
`src/model.rs`, with a test of the collapsed regime), and both main cohorts start again from
update 0 with it. Nothing else of the protocol changed.

The arithmetic matters for what the runs are evidence of. The `f32` cohorts reproduce the
source. A trajectory in exact real arithmetic has no collapse, so a cohort `adam`, loss in
doubles, run by the trainer of `b8ff967f`, is recorded separately as the closer proxy of it;
its activations, gradients elsewhere and optimizer state are still `f32`. The two are never
pooled.
