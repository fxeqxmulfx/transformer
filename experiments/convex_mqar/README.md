# Convex content routing versus a RoPE Transformer on MQAR

Standalone Python project managed by **uv**, with a committed `uv.lock`.
It compares a new convex recall construction with an ordinary, end-to-end
trained, two-layer causal Transformer using RoPE.

## Run

From the repository root:

```sh
uv sync --project experiments/convex_mqar --locked
uv run --project experiments/convex_mqar convex-mqar check --device cuda
uv run --project experiments/convex_mqar convex-mqar benchmark --profile full \
  --output experiments/convex_mqar/runs/full \
  --data-root experiments/convex_mqar/data
```

The benchmark requires CUDA and does not silently fall back to CPU. Checkpoints
are written after every epoch. Repeating the same command resumes incomplete
runs and skips completed ones. A directory lock prevents concurrent use of the
same checkpoints. `runs/full/status.json` and each run's `epochs.jsonl` show
progress; `runs/full/comparison.json` contains completed comparisons.

Profiles:

| Profile | Purpose | Vocabulary | Training examples | Epochs | Widths |
| --- | --- | ---: | ---: | ---: | --- |
| `full` | Full training budget and length sweep | 8192 | 100,000 | 64 | 64 |
| `capacity` | Also sweep the paper's larger widths | 8192 | 100,000 | 64 | 64, 128, 256, 512 |
| `sanity` | Check that the ordinary model can learn recall | 16 | 2048 | 128 | 32 |
| `smoke` | Exercise the complete pipeline quickly | 64 | 256 | 3 | 32 |

The full profile runs **16 training jobs**: four lengths and four learning rates.
Training this grid on a laptop GPU is a long-running experiment. Data and
checkpoints are ignored by Git. Final summaries can be copied into `reports/`.

## Data and evaluation

Local source: `papers/arXiv-2312.04927v1/Sections/appendix/mqar_framework.tex`,
Procedure `alg:synthetic` and Training Details.

- Vocabulary 8192: first half keys, second half values.
- Distinct keys per sequence; a fresh random value assignment in every sequence.
- Lengths 64, 128, 256, 512; dictionary size `D = N/4`.
- Adjacent key/value pairs occupy the first `2D` tokens.
- Each key repeats once at a distinct subsequent position. Position probabilities
  are proportional to `p^-0.1`; unused positions contain random value tokens.
- Both models use identical cached train, validation, and test examples. The
  splits have independent random seeds; their identifiers are saved in results.
- 100,000 training sequences, 3000 validation sequences, and 3000 test sequences.
- Recall accuracy counts all `D` labeled queries per sequence.

The manuscript leaves collision handling, filler generation, and the precise
power-law convention unspecified. The choices above define this implementation;
it is a documented MQAR variant, not a claim of identical original data.

The Transformer sees the full raw sequence. Query positions only select the
output logits used for loss and evaluation; they do not restrict attention.
All attention is causal, including the current query token.

## Ordinary Transformer baseline

Two pre-norm blocks, one attention head, width 64, MLP width `4d`, GELU, LayerNorm,
learned token embeddings, and a tied vocabulary readout. Q/K/V projections and
all embeddings are learned. RoPE replaces the paper's learned positional
embeddings; rotary base is 10,000. There is no prescribed dictionary parser or
equality lookup in this model.

AdamW, weight decay 0.1 for matrix parameters, 64 epochs, and learning rates
`np.logspace(-4, -2, 4)`. Biases and normalization parameters have no weight
decay. Warmup is linear for 10% of steps; the learning rate is constant afterward.
Batch sizes are 64, 16, or 8 according to the paper's length/width thresholds.
The GPU run uses BF16 autocast and fused AdamW.

For each length/width, the checkpoint and learning rate are chosen using
validation accuracy, then validation loss as a tie-break. The held-out test is
evaluated only after that selection. This differs from reporting the maximum
test accuracy over a sweep, as described in the paper.

Recorded evaluation times include Transformer cross-entropy and first-use CUDA
overhead. They are diagnostic timings; accuracy is the primary comparison.

## Convex construction and its explicit priors

This extends the convex-training aim of arXiv:2211.11052v1 §3.1–§3.2; it is not
the paper's shared positional simplex matrix and is not equivalent to joint
Q/K/V training.

The token encoder is a **fixed binary code**, and the prefix format supplies a
**fixed local key/value alignment**. Thirteen metric coordinates are trainable.
The metric is calibrated with thirteen one-bit mismatch examples, using

\[
J(w)=\sum_r\left(\max(0,1-w_r)+\frac{w_r^2}{4}\right).
\]

This unconstrained convex problem has the unique global optimum `w_r = 1`.
The implementation solves its hinge-epigraph quadratic program with SLSQP.
A lower-bound repair `max(w_r, 1)` removes numerical feasibility error.
This matching calibration is extra supervision. It does not use test examples
or memorize any key/value assignments. The training sequence set is available
to both models, but the convex mechanism requires only its calibration pairs.

For each query, a weighted Hamming cost and a null slot of cost `1/2` define
another convex program:

\[
\min_{a\geq0,\ \sum_j a_j=1}
\frac14\sum_j a_j^2+\sum_j a_j c_j.
\]

Future slots are excluded. The solver is projection onto the simplex, with
input `-2c`. Value mixing copies the dictionary's one-hot values. At the trained
metric, the unique optimum selects the matching key, or null if none exists.

The Lean theorem `Transformer.ConvexRecall.trained_convexRecall_solves_mqar`
proves exact recall on aligned, distinct-key streams, for every query and
arbitrary value assignments. The raw-prefix adapter is checked separately in
Python. Neither the encoder nor the local alignment is learned in this
construction. Comparing accuracy therefore measures recall capability with
different inductive biases, not equal model classes or equal supervision.

## Checks and reports

`check` verifies raw MQAR labels, causal RoPE behavior, positional-logit
selection, finite gradients, CPU/GPU simplex agreement, exact recall, and
abstention at first occurrences. `sanity` tests whether the baseline learns a
small nontrivial recall problem.

`reports/certified_recall.json` records the construction's earlier numerical
check: 720,000 queries over 12,000 sequences, 100% recall at all four lengths.
Its softmax control is a **constructed dot-product head**, not the trained
RoPE Transformer. It must not be used as the requested empirical comparison.
The trained comparison is produced by `benchmark`.

`reports/rope_sanity.json` records the completed small-vocabulary GPU control:
the trained RoPE Transformer and the convex mechanism both answered all 1024
test queries correctly. This is a sanity check, not the full 8192-token result.
