# Synthetic trainers for mini GPT

The suite contains 17 tasks and 37 default comparison variants sharing the
original [`GPTMini`](../gpt_mini.py), reproducible data, and one training protocol.

| Trainer | CLI selection | Task |
| --- | --- | --- |
| MQAR control | `--trainer mqar` | Retrieve values for several distinct repeated keys |
| Composed lookup | `--trainer lookup` | Follow `h` table links for each query |
| Prefix computation | `--trainer prefix` | Dyck-1 status and/or neutral-letter `E_k` membership |
| Histograms | `histogram`, `histogram2` | Symbol frequencies; number of types sharing each frequency |
| Frequency selection | `mode`, `most_freq` | Unique mode; frequency ranking with first-occurrence ties |
| Sequence transformations | `copy`, `reverse`, `sort` | Full ordered output, repeated or unique symbols |
| Typed brackets | `dyck2` | Dyck-2/3 prefix status; configurable up to eight bracket types |
| Counting | `count` | Generate an inclusive integer interval, then EOS |
| Arithmetic | `addition` | Decimal addition, carry chains, output order, index hints |
| Parity | `parity` | Direct answer, running states, or indexed one-bit scratchpad |
| Position transfer | `boolean_and` | Detect a zero at familiar or unseen positions |
| Temporal programs | `crasp` | Prefix classification by a fixed sampled counting formula |

Use each name with `--trainer`. The default `--trainer all --variants all`
runs all 37 variants separately. `--trainer core` retains the original four
modes; `rasp` and `rasp-l` select the corresponding paper task groups.
`--variants core` runs exactly the supplied settings for each selected task,
without expanding controls. The experimental rationale and formal boundaries
are in [TASKS.md](TASKS.md); the research objective is in [PLAN.md](PLAN.md).

The additional `random_lm` control and `--study memorization` /
`--study double_descent` profiles support finite-pool memorization, fixed label
noise, interpolation and delayed transfer, complete epoch curves, and paired
model/sample/noise sweeps. See [STUDIES.md](STUDIES.md) for definitions, source
boundaries, commands, and optional plot exports. The random control is selected
explicitly and does not expand the default 37 algorithmic variants.

## Run checks and training

Run commands from the repository root. The generators and oracle checks need
only Python's standard library. Training and the full test suite use PyTorch
from the existing `.venv`; the environment at
`experiments/convex_mqar/.venv` also works. No additional package is required.

```bash
python3 -m experiments.synthetic_trainers check --examples 128
.venv/bin/python -m unittest discover -s experiments/synthetic_trainers/tests -v
```

A short CPU pipeline run of the complete suite:

```bash
.venv/bin/python -m experiments.synthetic_trainers train \
  --trainer all --output experiments/runs/synthetic/smoke \
  --length 24 --min-length 22 --pairs 4 --queries 2 \
  --symbols 96 --number-limit 128 \
  --width 8 --layers 1 --heads 1 --batch-size 2 \
  --steps 1 --eval-every 1 \
  --train-examples 2 --validation-examples 2 --test-examples 2 \
  --eval-lengths 48 96 --device cpu
```

For a baseline comparison, set difficulty and the validation target before
running variants. For example:

```bash
.venv/bin/python -m experiments.synthetic_trainers train \
  --trainer lookup --output experiments/runs/synthetic/lookup-baseline \
  --length 64 --min-length 48 --pairs 8 --queries 4 --hops 2 \
  --width 64 --layers 2 --heads 4 --steps 200 --eval-every 20 \
  --seeds 0 1 2 --data-seed 0 --target 0.95 --device cpu
```

Select a prefix mode with `--prefix-mode dyck` or `--prefix-mode blocks`;
the default is `both`. `--blocks` sets `k`, and `--max-balance`,
`--neutral-fraction`, and `--max-neutral-gap` control the prefix inputs.
`--query-gap` sets a minimum gap between the table and its queries.
`--symbols`, `--pairs`, and `--queries` control lookup capacity.
The MQAR query-position distribution has weights proportional to `p^-alpha`.
`--eval-lengths` defaults to twice and four times the training maximum;
each evaluation retains the same `h`, `k`, bracket types, or temporal program.
`--symbols` defaults to 256 and must cover the longest unique-symbol test.
`--number-limit` defaults to 512; it bounds atomic integers and position hints
and must cover numeric output at every evaluated length. Model context tables
cover the full serialized input and generated answer, including scratchpads.
`--device cuda` or `cuda:0` requests CUDA
explicitly and fails if unavailable. Use `--help` for the remaining budgets.

The full matrix includes histogram with/without BOS; mode without a scratchpad,
with frequency-sorted counts, or counts in first-occurrence order; copy/reverse/
sort with/without repeats; Dyck-2/3; eight addition combinations (forward/reverse,
plain/indexed, standard/balanced carry sampling); four parity formats;
shifted/random-position AND; and C-RASP formula depths 1, 2, 3.

For one arithmetic format, use `--trainer addition --variants core
--addition-order reverse --index-hints --carry-sampling balanced`.
`--carry-length` fixes the exact longest carry chain independently of operand
length. For a temporal program, use `--trainer crasp --variants core
--formula-depth 4 --formula-seed 7`; supported count nesting is 1 through 5.

## Data and supervision

Inputs and targets have the same length. The model predicts the answer at
the query identity's position, or the inclusive prefix label at each input
position. Targets at table, filler, query marker, BOS, and padding positions
are ignored where appropriate. Only the input token tensor reaches the model.
Prefix labels use reserved output tokens that never appear in the input.
Right padding cannot influence earlier logits under the causal mask.

Generative tasks serialize `prompt SEP answer EOS`. Training inputs contain
the prompt and all preceding answer tokens; targets are shifted by one, with
prompt positions masked. At evaluation, greedy decoding receives only the
prompt and feeds its own predictions back until EOS or a prompt-derived bound.
Early EOS, missing EOS, omitted tokens, and extra tokens fail exact matching.
Original RASP global sequence outputs are placed after the complete prompt
to make their evaluation causal. Histogram and mode scratchpad counts are
atomic integer tokens; addition uses individual decimal digits.

For sequence tasks, `--length` counts data symbols. For addition it counts
operand digits before the extra leading padding zero; answers keep the padded
width, in the selected direction. For count it is the output interval size.
For prefix and lookup modes it remains total input length including BOS.

MQAR follows the existing package's adjacent key/value prefix convention,
with an added BOS and configurable query count. Keys are distinct, values
are resampled, fillers come from the value vocabulary, and each queried key
repeats once after the table. Set `--queries` equal to `--pairs` to query all
keys. This implementation does not change existing MQAR datasets or results.

Composed lookup serializes each record as `KEY identity VALUE identity`.
Values share the key identity domain so they can become subsequent keys.
The initial generator samples randomly relabelled cycles with `h < pairs`;
queried paths have no repeated vertices. Records and queries are shuffled.
Matched examples keep all token frequencies and queries but rewire the table
to change an answer. Two-identity one-hop inputs are a control with only one
possible cycle, so their paired examples coincide.

Prefix datasets mix matched examples with broader random inputs. Dyck pairs
preserve counts and endpoints while introducing a prefix violation;
`--max-balance` bounds absolute balance. Block pairs for `k >= 2` preserve
counts, endpoints, and neutral positions while introducing two extra runs.
For `k = 1`, a negative example necessarily changes symbol counts. Background
examples include wrong starting symbols, other block counts, and varied
parenthesis walks. Gaps and block sizes vary independently. State-change
masks mark the first supervised label and every subsequent label change.

Typed Dyck pairs preserve symbol counts and neutral positions while swapping
closers of different types. Temporal formulas use symbol predicates, inclusive
past counts, integer addition/negation, comparisons, and Boolean combinations.
The sampled program has both true and false witnesses and stays fixed across
splits. Its AST, expression, and syntactic depth are saved in the run config;
syntactic depth does not certify a minimum transformer depth.

Addition reports hard-carry tests at the training maximum and every OOD length.
AND position-shift runs train/validate with negatives in the early three
quarters of positions and test separately in the disjoint final quarter,
at both the training length and OOD lengths. A random-position variant is the
control. Both distributions include all-one positives and one-zero negatives.

Splits use independent deterministic seed streams. Changing a model seed or
an unrelated task's controls leaves the data unchanged. Exact input overlap
between independently sampled splits is possible in small finite task
domains; splits are not claimed to be exhaustive disjoint partitions.
Response-format controls share underlying input problems: mode scratchpads,
histogram BOS, parity scratchpads/hints, and addition order/hints. Format version
2 adds generation fields and reserves new structural tokens; its fingerprints
and token IDs differ from the original four-mode format.

## Reports and architecture comparisons

Each `<output>/<variant>/seed-<seed>` directory contains frozen input/target
JSONL files, split fingerprints, source hashes, configuration, validation
history, `best.pt`, `final.pt`, and `result.json`. Existing run directories
are rejected to preserve their artifacts.
The root `suite.json` indexes completed variants and their separate scores.

Validation selects checkpoints by complete-example accuracy, then
class-balanced accuracy, then loss. Held-out test data are
evaluated after selection; test scores never choose a checkpoint or trigger
early stopping. The report includes the final validation score and tests of
the selected checkpoint, separately for the training length distribution
and each longer length.
For generated answers, primary accuracies measure free rollout; teacher-forced
scores appear separately under `teacher_forced`, and loss is teacher-forced CE.
Mode/parity also report final-answer accuracy independently of scratchpad
correctness. Selecting `--target-metric final_answer_accuracy` ranks checkpoints
by that score first; this option requires a generative task.

Metrics include supervised-token accuracy, complete-example accuracy,
class-balanced accuracy with observed per-class supports, and accuracy at
state changes. Losses and metrics count actual targets, including partial
batches. `loss` weights targets; `example_loss` averages each example's mean
target loss with equal weight. Check class supports when setting a validation target.

Time to target is the first scheduled validation observation meeting
`--target` under `--target-metric`. It records updates, examples, supervised
targets, training seconds, and elapsed wall time including setup and prior
validation/checkpoint work. An unreached target is `null`, not a successful
time measurement. Fixed-step training continues after reaching the target;
`--stop-at-target` optionally stops at that observation. The default 0.95
target is configurable and still requires calibration for a real comparison.

Reports also include parameter counts, padded input positions processed, and
peak CUDA allocation over the run when CUDA is used. CPU memory and exact
FLOP counts are not estimated. CUDA timings synchronize the selected device.
Generation reports its own time, emitted tokens, forward calls, processed
padded positions, and attention-matrix cells (summed batch × length² per call).
The cell count is a workload proxy, not a FLOP estimate. Scratchpad comparisons
must include their extra generation cost and vocabulary/parameter differences.

The Python API `training.train_run(..., model_factory=factory)` accepts a
named factory taking the reference `gpt_mini.Config` and returning a causal
PyTorch module with logits shaped `[batch, length, vocabulary]`. This allows
architecture variants to reuse exactly the same tasks, seeds, budgets, and
selection protocol. Count any additional or reused computation in comparisons.

## Test coverage

Tests exhaust short words for sequence tasks, typed stacks, parity, and AND;
compare temporal formulas with recursive point semantics; check small decimal
sums and every prescribed carry length; verify paired controls, split seeds,
masked gradients, causal padding, feedback of generated mistakes, EOS limits,
metric weighting, checkpoint selection, and time-to-target accounting. Every
variant trains and reloads on CPU; a small copy problem learns correct free
generation. CLI checks cover the complete matrix. Short runs validate the pipeline;
baseline difficulty calibration and architecture ablations remain experiments.
Study tests additionally cover fixed corruption in all formats, causal shifts,
entropy references, whole-sequence clipping and normalized mixtures, prefix
extraction, finite double-descent witnesses, novel-input transfer lags, final
checkpoint reloads, and paired sweeps with complete epochs.

## AMSGradW + softmax baseline

The fixed baseline protocol runs the 37 variants, three-seed delayed
generalization probes, a noisy parity capacity grid, and a random-sequence
memorization control. It reuses `GPTMini` and the raw, unguarded AMSGrad update
from the existing optimizer benchmark with decoupled weight decay. Its moments
have **no bias correction**; `torch.optim.AdamW(amsgrad=True)` is a different
optimizer. Matrices use normal initialization with standard deviation 0.02.
The usual training CLI retains AdamW and constructor initialization as defaults.

```bash
.venv/bin/python -m experiments.synthetic_trainers.baseline \
  --device cuda --output experiments/runs/synthetic/amsgradw-softmax-baseline
.venv/bin/python -m experiments.synthetic_trainers.baseline_report \
  experiments/runs/synthetic/amsgradw-softmax-baseline
MPLCONFIGDIR=/tmp/synthetic-trainer-mpl python3 \
  -m experiments.synthetic_trainers.baseline_plots \
  experiments/runs/synthetic/amsgradw-softmax-baseline
```

Each output must be fresh. `--phase suite|transitions|capacity|control` selects a
phase; the manifest freezes every recipe before the first update. Reports
include both final and validation-selected checkpoints. See
[STUDIES.md](STUDIES.md#amsgradwsoftmax-baseline-protocol) for budgets and
interpretation. The plot command requires Matplotlib in the chosen Python
environment. The standard train/sweep CLI also accepts `--optimizer amsgradw`,
`--init-std 0.02`, and `--no-grad-clip` for individual comparisons.

The measured 2026-10-02 baseline is preserved in
[baselines/amsgradw_softmax_20261002](baselines/amsgradw_softmax_20261002):
[scalar measurements](baselines/amsgradw_softmax_20261002/metrics.csv),
[compact report and curves](baselines/amsgradw_softmax_20261002/measurements.json),
[suite figure](baselines/amsgradw_softmax_20261002/plots/suite-accuracy.png), and
[three-seed learning curves](baselines/amsgradw_softmax_20261002/plots/transition-curves.png).
The original checkpoints, complete corpora, and reports remain under
`experiments/runs/synthetic/amsgradw_softmax_20261002` (gitignored).
The archive retains hardware metadata, every recipe, source hashes, split
fingerprints, and observation curves, with PNG/PDF exports. An independent
short throughput pilot is stored in the separate `_pilot` run directory.

To preserve a completed future baseline outside the ignored run directory,
add `--archive <fresh-directory>` to `baseline_report`. `baseline_plots` accepts
either the original run directory or its compact archive.
