# Convex content routing versus a RoPE Transformer on MQAR

> The code this README documents is no longer in the tree: read it at commit
> `9416d03` (`git show 9416d03:experiments/convex_mqar/src/convex_mqar/cli.py`).
> The trained Transformers of the comparison, with softmax and sparsemax
> attention, and its learning-rate selection are ported to
> [`python/src/lab`](../../../python/src/lab), as
> [`experiments/mqar_sparsemax`](../../mqar_sparsemax); the convex
> construction and its certificate, the Zoology models and the diagnostics
> are not. The commands below are how the reports kept here were produced,
> at the paths of that commit; the reports have since moved to
> `experiments/archive/convex_mqar/` ([old paths](../README.md#old-paths)).

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

Add `--compile` to compile the training loss with TorchInductor:

```sh
uv run --no-sync --project experiments/convex_mqar convex-mqar benchmark \
  --profile full --compile --output experiments/convex_mqar/runs/full \
  --data-root experiments/convex_mqar/data
```

This resumes ordinary model/AdamW checkpoints. Batch sizes, data order, learning
rates, and the epoch budget stay the same. BF16 fusion can change rounding;
compiled training is not promised to reproduce eager weights bit for bit.
Compilation is lazy and its first-step cost is included in training time.
`execution_history.json` preserves source fingerprints and execution changes;
`execution_segments` in checkpoints/results records the eager and compiled
epochs. Completed test comparisons are preserved when resuming.

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

## Replace only attention normalization

The matched experiment trains the existing RoPE Transformer with **causal
sparsemax in place of softmax**:

```sh
uv run --no-sync --project experiments/convex_mqar python -m convex_mqar.attention_ablation \
  --profile full --device cuda --compile \
  --output experiments/convex_mqar/runs/sparsemax \
  --baseline experiments/convex_mqar/runs/full \
  --data-root experiments/convex_mqar/data
```

`SparsemaxTransformer` preserves all initialized parameter values, their names,
the RNG state, RoPE, Q/K/V projections, MLPs, normalization, and tied vocabulary
readout. Both width-64 models have 623,872 trainable parameters. All layers
are trained with the same cached splits, batch order, optimizer, learning-rate
grid, warmup, and epoch budget. There is no prescribed key/value parser,
fixed token encoder, value-copy decoder, or additional matching supervision.
The Zoology reference modules are not used by this experiment.

Given scaled scores `z = RoPE(Q) RoPE(K)^T / sqrt(head_width)`, each row solves

\[
\min_{a\geq 0,\ \sum_j a_j=1,\ a_j=0\ (j>i)}
\frac12\sum_j a_j^2-\sum_j a_j z_j.
\]

This is simplex projection, with the same minimizer as the row objective in
`Transformer.GPTMini.Convex.Basic`. It is a new attention replacement motivated
by the simplex construction of arXiv:2211.11052v1 §3.1, rather than an
implementation of the paper's full training reformulation. The inference
program is convex for fixed scores; end-to-end Transformer training remains
nonconvex. Unlike softmax, sparsemax can assign exactly zero weight to a token.

Softmax uses fused SDPA; sparsemax forms explicit FP32 scores and computes
projection thresholds in FP32, with a support-based backward pass. Their
numerical kernels differ, so timing is not an isolated comparison of
normalization cost. Initial weights and the training budget match; compiled
BF16 training is not promised to be bitwise identical to earlier eager runs.

The new sweep starts from the common initialization. Completed ordinary test
results are reused after checking the configuration and split identities.
The ordinary process must release its run lock before this command starts;
GPU jobs are not overlapped. If an ordinary length is unfinished, its comparison
is temporarily recorded as `null`; the ordinary saved checkpoints are resumed
after the sparsemax sweep to finish that comparison. Each model selects its
checkpoint and learning rate on validation, then evaluates the held-out test.

`runs/sparsemax/status.json`, `epochs.jsonl`, and `comparison.json` record progress.
`baseline.json` preserves the ordinary report and its execution history.
Source fingerprints include the new attention and experiment modules. A model
class tag prevents accidental reuse of ordinary checkpoints for sparsemax,
despite their compatible parameter names and shapes.

### First epoch reaching 99% validation accuracy

The target is **at least 99% recall accuracy on validation**, measured after
each epoch on 3000 sequences. For each model and length, select the learning
rate with the earliest observed crossing; ties use measured training seconds,
then the smaller LR. Reaching the target in any epoch qualifies, even if
accuracy falls later.

Results for seed 0, width 64, four learning rates, and 64 epochs per run:

| Attention with RoPE | Length | LR | First epoch at 99% | Validation accuracy at that epoch | Training seconds to that epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Softmax | 64 | — | Not reached in 64 epochs | — | — |
| Sparsemax | 64 | 0.01 | 1 | 99.8021% | 3.92 |
| Softmax | 128 | 0.01 | 18 | 99.7073% | 107.66 |
| Sparsemax | 128 | 0.0021544347 | 3 | 99.5375% | 20.72 |
| Softmax | 256 | 0.0021544347 | 5 | 99.9599% | 71.35 |
| Sparsemax | 256 | 0.00046415888 | 6 | 99.9526% | 112.31 |

These are validation milestones. Validation examples never enter gradient
updates; validation is used to select checkpoints and learning rates. The
held-out **test** is evaluated separately, after selecting the checkpoint and
LR by peak validation accuracy, then loss. The first epoch reaching 99% on
test was not measured.

The table comes from the `99` entries of the exporter with `--selection first99`:

```sh
uv run --no-sync --project experiments/convex_mqar python -m convex_mqar.validation_milestones \
  --softmax experiments/convex_mqar/runs/full \
  --sparsemax experiments/convex_mqar/runs/sparsemax \
  --lengths 64 128 256 \
  --selection first99 \
  --output experiments/convex_mqar/reports/validation_first99
```

This reads existing epoch logs on CPU. Epochs are numbered from 1; crossings
are observed after an epoch, with no interpolation within it. Times are
cumulative measured training-loop seconds for the selected LR, including lazy
compilation. They exclude validation, checkpoint writing, setup, pauses,
discarded partial epochs, and the other LR candidates. Compilation warmup
differs across runs, so these times are not an isolated kernel benchmark.

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

Run the unit and integration tests on CPU, without sharing the active GPU:

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync --project experiments/convex_mqar python -m unittest discover \
  -s experiments/convex_mqar/tests -t experiments/convex_mqar -v
```

The tests compare attention with explicit causal softmax and RoPE with rotation
matrices; check simplex optimality against KKT conditions and an independent QP;
and verify raw MQAR labels, gradients, CPU BF16, optimizer coverage, warmup,
partial batches, validation selection, and exact checkpoint resume. A small
CPU benchmark also checks that both models share the test and that test results
do not choose a checkpoint or learning rate. These tests do not exercise CUDA
kernels or prove that no bugs remain.

Audit completed data caches and checkpoint snapshots of a running experiment
without changing that experiment:

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=experiments/convex_mqar \
  uv run --no-sync --project experiments/convex_mqar python -m tests.audit_running \
  --run experiments/convex_mqar/runs/full --data experiments/convex_mqar/data
```

It verifies the original source fingerprint, every cached raw query label,
and finite model/AdamW state and step counts. `reports/running_audit.json` is a
snapshot of the active run at the time of this check.

`check` verifies raw MQAR labels, causal RoPE behavior, positional-logit
selection, finite gradients, CPU/GPU simplex agreement, exact recall, and
abstention at first occurrences. `sanity` tests whether the baseline learns a
small nontrivial recall problem.

`reports/certified_recall.json` records the construction's earlier numerical
check: 720,000 queries over 12,000 sequences, 100% recall at all four lengths.
Its softmax control is a **constructed dot-product head**, not the trained
RoPE Transformer. It must not be used as the requested empirical comparison.
The trained comparison is produced by `benchmark`.

`reports/baseline_diagnostic.json` examines the best width-64 checkpoint on
3000 training sequences and the full validation split. It records dictionary
candidate and reconstructed attention controls. Restricting readout to known
values is an oracle diagnostic, not an alternative model. The observations do
not isolate a causal effect of RoPE. The convex construction uses no RoPE.

`reports/rope_sanity.json` records the completed small-vocabulary GPU control:
the trained RoPE Transformer and the convex mechanism both answered all 1024
test queries correctly. This is a sanity check, not the full 8192-token result.

## Standalone RoPE recall variant

`convex_rope.py` defines `RopeConvexRecall` with fixed binary codes and a
calibrated weight per active rotary pair. It uses the same interleaved
frequencies and rotations as `RotaryAttention`, at the actual token positions.
The default width 64, vocabulary 8192, base 10000, and maximum length 512 meet
the bounded-angle routing margin. Unsupported margins are rejected.

The tests check rotations, an independent quadratic program, causal recall,
and the margin bounds. This RoPE extension is not covered by the existing
Lean proof. `RopeSoftmaxRecall` shares its codes, metric, positions, mask, null
slot, and decoder; it is a control with fixed features. These standalone
variants are not connected to the `benchmark` command yet. Existing benchmark
results use the original unrotated convex construction.

## Zoology reference modules

`zoology_config.py`, `zoology_data.py`, and `zoology_model.py` contain the
reference components ported from Zoology's `iclr24` Figure 2 experiment,
commit `de4e258784224e09909c257ff3ea040f089ed660`. The generator preserves
the source's random draws, zero fillers, distinct values, query positions,
and train/test seeds. A separate validation split at data seed + 20 is added.

The model preserves the two attention blocks, Identity state mixer, LayerNorm,
biases, dropout, initialization, and tied readout. Causal SDPA replaces the
source's explicit softmax. `positions="learned"` selects the original learned
position embeddings; `positions="rope"` is an extension. The configurations
record these choices explicitly. These components are not connected to the
benchmark command, and training results for this model are pending.
