# Memorization, double descent, and algorithmic transfer

The existing 17 algorithmic tasks / 37 variants now support `memorization` and
`double_descent` study profiles. A separate `random_lm` control measures coding
gain on random sequences; it is selected explicitly and is outside `--trainer all`.
The standard task-quality / time-to-target protocol remains available.

## What “remembering to understanding” can mean here

Use an operational distinction. Fitting the finite training pool is observable;
learning a reusable algorithm is supported by solving new inputs and transferring
to greater lengths or changed dependencies/positions. Neither accuracy nor a
finite transfer test establishes human-like understanding or correctness on all
inputs. Random-sequence prediction is a memorization control, not an algorithm
discovery task.

Every study observation records teacher-forced `train` and `train_clean`, clean
validation, and separate validation probes for every requested OOD length and
existing hard-carry/position-shift control. Generated validation answers use
free rollout. `validation_novel` and `validation_ood_novel` exclude complete
inputs appearing in the clean training pool, even under independent sampling.
If a probe has no novel inputs, it cannot confirm a transfer transition.

The reported fit event is the first observation with empirical training risk
strictly below `--fit-epsilon` (default 0.01). `--fit-metric example_error` uses
teacher-forced complete-example error; `token_error` uses token error; `loss`
uses mean per-example CE in nats per supervised target. This last normalization
gives each example equal weight despite different answer lengths. It is recorded
alongside the usual token-weighted `loss`.

`generalization_transition` requires novel in-distribution validation and **all**
novel transfer probes to reach `--target` under `--target-metric` for
`--generalization-patience` consecutive observations (default two). It records the
first observation in that confirmed streak, its confirmation, and its lag after
clean train fit in steps, epochs, and training seconds. Positive lag is called a
`delayed_transfer_candidate`; absent events stay `null`. This criterion detects
delayed generalization, without claiming an abrupt phase transition. Its times
are limited by evaluation frequency and budget. Predefine thresholds and probes
before comparing architectures.

This follows the measurable grokking lag discussed in the local
[convex-transformer paper](../../papers/arXiv-2211.11052v1/arxiv.tex), Sections 1
and 4 (algorithmic datasets), with stronger transfer probes added for this suite.

## Double descent is a separate observation

Double descent is a descent, ascent, and second descent in held-out error as
model size, sample count, or training time changes. It can accompany delayed
algorithmic generalization, but does not establish that mechanism. The local
[double-descent paper](../../papers/arXiv-1912.02292v1/model_dd.tex), Section 5,
also discusses linear-model examples and noise absorption among interpolating
solutions. Section 6 describes an epoch-wise second descent after overfitting.
The mechanisms behind deep-network double descent remain open in that paper.

Both study profiles run the full budget; `--stop-at-target` is rejected. Each
observation records train fit and held-out risk. After training, the clean tests
evaluate both `final.pt` (`test_final`) and the validation-selected `best.pt`
(`test`), keeping them separate. Test metrics never select checkpoints or end
training. Study pools may prepare test inputs before training for consistent
separation; test evaluation still occurs after training and selection.

`epoch_double_descent` finds four strictly ordered measured points satisfying
first descent, ascent, and second descent, with `--curve-tolerance` absolute
margin (default 0.001). It uses both task error and mean per-example loss. A
stronger flag records whether the second descent improves on the first minimum.
The detector matches the finite predicate in
[Section5_Curves.lean](../../src/Transformer/DoubleDescent/Section5_Curves.lean).
A witness describes sampled points, not statistical significance or monotonicity
between observations. `peak_transition_alignment` records whether the confirmed
transfer onset follows that witness's peak; temporal ordering is not a causal test.

The interpolation frontier comes from observed train risk, not parameter count.
Sweep reports list fitted and unfitted pool sizes, the largest observed fitted
pool, and any holes. They do not call this finite grid the population EMC from
[Section 4](../../papers/arXiv-1912.02292v1/general_dd.tex).

## Fixed label noise on algorithmic tasks

`--label-noise p` corrupts training content labels only. With probability `p`, a
label is replaced uniformly by a different legal content label. EOS, BOS,
separators, and index hints stay intact; singleton content domains cannot flip.
The assignment is frozen by `--noise-seed` and causal input / prompt-and-output
position. Repeated problems receive identical noisy labels. Changing the number
of examples or model seed does not resample existing labels; changing the noise
rate uses the same keyed draws. Corrupted generated answers rebuild the shifted
teacher-forcing inputs, so no current answer token appears in its own input.
Shared causal contexts deliberately share noise; this differs from independent
label flips for duplicate IID records and avoids contradictory deterministic
training targets. The report records actual eligible/flipped counts, not only
the requested rate.

The clean oracle corpus and observed corpus are saved separately as `train_clean`
and `train`. Oracle validation applies to the clean corpus. Held-out validation
and tests retain clean labels, and `noise_fit` distinguishes accuracy on corrupted
labels from accuracy on their clean counterparts, conditional on observed inputs.
High observed train accuracy together with high novel transfer accuracy can
therefore mean noise memorization and algorithmic generalization coexist.

This is a training-noise adaptation of the uniform incorrect-label mechanism in
[Section 4](../../papers/arXiv-1912.02292v1/setup.tex), not a reproduction of the
paper's image/translation datasets or its train/test noise settings. The structured
generators include matched pairs, so their rows are not generally IID samples of
the learning-procedure definition.

Frozen pools report duplicate inputs, overlap, and conflicting supervised causal
contexts. `causal_contexts` gives empirical minimum token error / CE for an
unrestricted deterministic predictor on the observed contexts. Shared random
prefixes with different continuations can make zero train risk impossible.
These are data limitations rather than evidence of an optimization failure.

`--split-policy disjoint` rejection-samples unique complete inputs and separates
train/validation/test. Small domains can be exhausted; the command fails clearly.
For example, the one-zero Boolean-AND control has very few possible fixed-length
inputs. Independent sampling remains the default and reports overlap explicitly.

## Uniform sequence control and coding gain

`random_lm` supplies `BOS, requested_length, SEP`, then asks for `S` independent
uniform symbols from an alphabet of size `V`, followed by deterministic EOS.
There is no deterministic raw-prompt answer oracle. Conditional on supplied
lengths, the generator's entropy reference is `sum(S) * log2(V)` bits. Fixed
length gives `N * S * log2(V)`. Independent sampling with replacement is required;
deduplication or added label noise would change that interpretation.

The measured model code length is whole-answer NLL in bits, including the cost
of predicting EOS. Payload and EOS costs are also recorded separately. Reports
include signed `net_gain_bits`, its bits-per-parameter ratio, and two reference
comparisons from [Sections 2.3 and 3.2](../../papers/arXiv-2505.24832/main.tex):

- `clipped_sequence_gain_bits`: sum of positive **whole-sequence** coding gains.
  The underlying pointwise maximum likelihood is not a normalized distribution.
- `mixture_gain_bits`: gain from the normalized equal mixture of model and
  uniform reference sequence probabilities. Its code length is at most one bit
  per sequence above the smaller of the two lengths.

These are finite likelihood proxies, not direct measurements of Shannon mutual
information or exact Kolmogorov complexity. Sweep maxima are observed coding-gain
proxies under the chosen budget; saturation needs a longer convergence study.
Negative net gain is retained when training has not improved on the reference.
No universal 3.6 bits per parameter is assumed. The mini GPT parameters remain
float32; actual parameter storage bits and dtype are reported, counting tied
weights once. Precision comparisons are not implemented by silently changing
this architecture's arithmetic.

For deterministic algorithmic tasks, clean answer entropy conditional on the raw
input is zero; their accuracy is not reported as capacity in random-data bits.
Checkpoint diagnostics also record loss-only membership AUC after excluding
exact input overlap, and random-control greedy suffix extraction from true
partial prefixes on train and novel validation. Extraction always leaves at least one payload token unseen;
these are auxiliary diagnostics inspired by Section 4 of the memorization paper.
The control's length prompt, reserved tokens, EOS, architecture, optimizer, and
training budget differ from the paper's GPT-2 synthetic experiment.

## Commands and comparable budgets

Run from the repository root. A noise/transfer study of the original lookup task:

```bash
.venv/bin/python -m experiments.synthetic_trainers train \
  --trainer lookup --study double_descent --label-noise 0.2 \
  --split-policy disjoint --length 64 --pairs 8 --queries 4 --hops 2 \
  --steps 2000 --eval-every 20 --train-examples 512 \
  --validation-examples 128 --test-examples 128 --eval-lengths 128 256 \
  --output experiments/runs/synthetic/lookup-transfer
```

A model/sample/noise grid; every pool gets the same number of complete epochs:

```bash
.venv/bin/python -m experiments.synthetic_trainers sweep \
  --trainer lookup --study double_descent --split-policy disjoint \
  --widths 32 64 128 --layer-counts 1 2 --sample-sizes 128 512 2048 \
  --noise-rates 0 0.2 --seeds 0 1 2 --data-seeds 0 1 2 \
  --epochs 200 --eval-every 20 --output experiments/runs/synthetic/lookup-dd
```

A capacity-control grid (memorization sweeps default to noise zero):

```bash
.venv/bin/python -m experiments.synthetic_trainers sweep \
  --trainer random_lm --study memorization --length 64 --symbols 2048 \
  --widths 32 64 128 --layer-counts 1 2 --sample-sizes 128 512 2048 \
  --noise-rates 0 --seeds 0 1 2 --data-seeds 0 1 2 \
  --steps 2000 --eval-every 50 --output experiments/runs/synthetic/random-capacity
```

Without `--epochs`, sweeps use equal update counts (`--steps`). Both modes record
actual examples seen and epochs. The largest train pool is generated once per
data seed; smaller pools are nested prefixes. Validation/test pools remain common
even with disjoint sampling. Width/depth/noise/model-seed comparisons share data,
and model seeds are crossed with independent data seeds. Validation probes also
stay fixed. Reports are `sweep.json` and `sweep.csv`, with per-setting means,
sample standard deviations, individual runs, final/selected curves, and observed
fit frontiers. Curve witnesses cover width, depth, and sample count; sample-count
curves also record a measured more-data-hurts pair. Transfer-lag statistics count
only observed events and retain their supports alongside unconfirmed runs.
The manifest records planned/completed runs and the requested grid. These are
descriptive statistics, not confidence intervals.

Export standalone PNG plots with Matplotlib; this command needs no PyTorch:

```bash
python3 -m experiments.synthetic_trainers.plots \
  experiments/runs/synthetic/lookup-dd/lookup
```

The same command accepts an individual run directory to plot train/validation/
transfer trajectories and fit/transfer event markers. Plotting is optional;
training and CSV/JSON reports need only the existing PyTorch environment.
The Python `run_sweep(..., model_factory=factory)` API supports architecture
ablations under the same pools and budgets. Baseline calibration, converged
capacity estimates, and actual double-descent/grokking demonstrations remain
experiments rather than consequences of passing the pipeline tests.
