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

`id_generalization_step` records the separate, weaker event where novel ID
validation reaches the same threshold for the same patience, irrespective of
OOD scores. Its lag after clean train fit is `id_lag_steps` / `id_lag_epochs` /
`id_lag_training_seconds`. Positive lag is a
`delayed_id_generalization_candidate`. This distinguishes ordinary delayed ID
generalization from algorithmic length/position transfer; both events are
disabled for the random-sequence control. A positive sampled lag alone does not
establish an abrupt grokking transition.

This follows the measurable grokking lag discussed in the local
[convex-transformer paper](../../papers/arXiv-2211.11052v1/arxiv.tex), Sections 1
and 4 (algorithmic datasets), with stronger transfer probes added for this suite.

## AMSGradW/softmax baseline protocol

The baseline command in [README.md](README.md#amsgradw--softmax-baseline) measures
the existing causal softmax `GPTMini`: RMSNorm, QK normalization, learned head
temperatures, XSA, RoPE, ReLU-squared FFNs, and tied embeddings. It uses float32,
two layers, four heads, width 64, and FFN width 256. Capacity probes change model
width to 16, 32, 64, or 128 while retaining four heads and FFN multiplier four.
Parameter counts vary with vocabulary and width and are recorded per run.

The optimizer is the raw AMSGradW recurrence defined in
[AMSGradW/Basic.lean](../../src/Transformer/AMSGradW/Basic.lean), implemented by
the existing [`CoordinateOptimizer`](../optimizer_benchmark/coordinate.py):

```text
m = 0.9*m + 0.1*g
v = 0.999*v + 0.001*g^2
maximum = max(maximum, v)
x = (1 - lr*decay)*x - lr*m/(sqrt(maximum) + 1e-8)
```

There is no bias correction, learning-rate schedule, guard/fallback, or gradient
clipping. Constant `lr=0.001` and `decay=0.1` apply to every trainable parameter
with a gradient, including head temperatures; tied weights are updated once.
Decay stays out of the gradient moments. All matrix parameters are initialized
with `Normal(0, 0.02)`, matching the existing optimizer benchmark's initialization;
one-dimensional temperatures retain the `GPTMini` constructor values. This
stochastic training experiment does not satisfy or invoke the formal full-gradient
convergence theorem's sufficient assumptions.

| Phase | Runs | Updates/run | Train / validation / test | Model seeds |
| --- | ---: | ---: | --- | --- |
| Suite calibration | 37 | 1,000 | 128 / 32 / 64 | 0 |
| Copy, direct parity, running parity | 9 | 5,000 | 64 / 64 / 128 | 0, 1, 2 |
| Direct parity, four widths, 20% label noise | 12 | 1,000 | 64 / 64 / 128 | 0, 1, 2 |
| IID random-sequence control | 3 | 1,000 | 8 / 64 / 128 | 0, 1, 2 |

Batch size is 32; observations occur every 100 updates (250 in the long phase),
including step zero and the final step. All runs complete their budgets.
Data seed is 1, and the noise seed is 2. Initialization-seed repetitions share
one frozen data pool; their sample SD does not estimate variation across data
seeds. The capacity phase measures a sampled size curve at fixed update count,
not equal compute or a converged capacity frontier.

Suite input lengths are 8--16, with Dyck lengths 12--16; MQAR/lookup use length
24, four records, two queries, and two lookup hops; addition uses 2--4 decimal
digits. Main OOD lengths are twice and four times the training maximum, with
the existing hard-carry and position-shift probes included. MQAR/lookup OOD
retains association counts, measuring spacing transfer. Other tasks increase
problem size. The symbol alphabet is 64, and atomic-number limit is 128.

Long copy probes use lengths 4--8 and alphabet size 8. Direct/running parity use
exactly eight bits. Both have OOD lengths 16 and 32, with disjoint unique ID
train/validation/test pools. The parity split partitions all 256 eight-bit
inputs into 64/64/128. The suite uses independent sampling and reports duplicate
inputs, overlap, and novel validation support; ID tests can include training
inputs there. Shifted AND has especially small finite support. Empty novel
validation cannot certify either delayed-generalization event.

The random control has eight IID sequences, eight symbols per sequence, and
alphabet size four: a 128-bit uniform-reference entropy. Shared prefixes impose
an empirical causal-conflict floor, reported with measured coding gains. These
gains measure memorization on this finite pool, rather than reusable algorithms.

Both delayed-generalization criteria use complete-example accuracy at least
0.95 for two consecutive observations, with train error strictly below 0.01.
The finite double-descent detector uses a 0.02 absolute margin in the respective
loss/error units. `summary.json` and `metrics.csv` preserve actual measurements;
PNG/PDF plots display initialization-seed means and sample SD. A finite witness
or a positive delay is a candidate effect, without statistical significance or
causal attribution. Test scores never alter the frozen plan.

## Measured AMSGradW baseline (2026-10-02)

The [archived measurements](baselines/amsgradw_softmax_20261002/measurements.json)
contain all 61 runs and source/data hashes; [CSV](baselines/amsgradw_softmax_20261002/metrics.csv)
and [plots](baselines/amsgradw_softmax_20261002/plots/transition-curves.png)
are retained outside the ignored checkpoint directories. There were 97,000
updates, 948.17 seconds of training updates, and 1,235.11 seconds of total run
wall time on an RTX 3050 Laptop GPU. These are measured budgets, not estimates.

| Long phase, final checkpoint | Train accuracy | ID test accuracy, mean ± sample SD | OOD 16 / 32 accuracy |
| --- | ---: | ---: | ---: |
| Copy, alphabet 8 | 100% | 3.39% ± 0.45% | 0% / 0% |
| Direct eight-bit parity | 100% | 40.63% ± 4.75% | 51.56% / 51.04% |
| Running eight-bit parity | 100% | 96.35% ± 4.30% | 0% / 0% |

Running parity has a finite epoch-error descent/ascent/descent witness in all
three initialization seeds. However, its novel ID accuracy already reached the
threshold at the first observed train fit (step 250, confirmed at 500). The
later dip and recovery therefore do not demonstrate a delayed memorization-to-
algorithm transition. No run confirmed delayed length/position transfer.
The suite's only positive ID lag was 100 updates on blocks in one seed; its
OOD scores remained low. Of the 37 short suite runs, 35 reached more than 99%
train accuracy. Shifted/random-position AND reached 100% ID and OOD accuracy
with zero measured lag. MQAR fitted train but had 0% ID test accuracy at this
small finite-pool budget.

The width sweep used a requested noise rate of 20%; its fixed pool actually
contained 16 changed labels out of 64 (25%). Widths 16/32/64/128 gave mean clean
ID CE 1.275/1.621/2.203/1.602 nats per target. The last width fitted train in
only two of three seeds. Neither mean loss nor mean error has a complete
four-point double-descent witness. An individual seed's witness does not
establish a repeated size effect, and an unfitted last point confounds a
capacity interpretation. The random control learned a mean net coding gain of
103.80 bits relative to its 128-bit reference, without algorithmic transfer.

## Theory-guided data and model scaling

The local theorem statements constrain which quantities can be used in an
experiment:

- [EMC](../../src/Transformer/DoubleDescent/Section2_EffectiveComplexity.lean)
  is defined through expected IID train risk for a particular training
  procedure. It is not parameter count. The paper's heuristic fit tolerance is
  0.1; these experiments use a stricter 0.01 complete-example error. A single
  nested pool and finite budget measure an observed frontier, not population EMC.
- [Fixed-feature interpolation](../../src/Transformer/DoubleDescent/AppendixD_Interpolation.lean)
  requires `n ≤ d` when a linear design can fit every real label vector.
  This does not identify GPT parameters or embedding width with `d` for a
  jointly trained nonlinear classifier. The [formal counterexample](../../src/Transformer/DoubleDescent/Section2_Hypothesis.lean)
  also rules out a general test-risk ordering from EMC alone.
- [RASP compilation counts](../../src/Transformer/RASP/Compilation.lean) bound
  program aggregation heads and layers; the paper gives no quantitative
  embedding-width bound. [C-RASP's depth hierarchy](../../src/Transformer/CRASP/Transformers.lean)
  uses future-masked rounded fixed-precision transformers with its specified
  positional restrictions. GPTMini's float32/RoPE model differs, so this is
  motivation for a depth ablation rather than a hard lower bound for GPTMini.
- The RASP-L [experiment table](../../papers/arXiv-2310.16028v1/appendix.tex),
  Table 1, uses width 512, six layers, and eight heads for binary copy, with
  100,000 AdamW updates and fresh online examples. These are empirical
  reference sizes, not necessary/sufficient learning bounds. The corrected
  [convexification width results](../../src/Transformer/Convexifying/Section3_Corrected.lean)
  concern heads in a simplex-attention model (`h ≥ n` / `h ≥ n*c` with the
  stated loss assumptions), not softmax GPTMini embedding width.

For binary copy with uniform lengths 1–32, define the finite coverage property
as observing every four-bit word at every valid position of every length. There
are `M = 16 * sum(length - 3, length=4..32) = 6960` such events. A particular
event has probability at least `q = 1/(32*16) = 1/512` in the first row of each
independent generator pair. Counting only those rows gives the conservative
union bound `P(any event missing) ≤ M*(1-q)^(N/2)`. For failure probability 0.05,
`N ≥ 2*ceil(log(M/0.05)/(-log(1-q))) = 12118`; use 16,384 rows, with a bound
of 0.0007711. This specific local coverage is weaker than the RASP-L Diversity
conjecture and gives no training, double-descent, or generalization guarantee.
The sampled pool actually covers all 6,960 events.

For vocabulary 38 and FFN multiplier four, GPTMini's parameter count is
`38*w + L*(12*w^2 + H)`, counting tied weights once. Holding `H=8`, the paired
copy configurations `(w,L)=(64,2),(64,6),(512,2),(512,6)` have respectively
100,752 / 297,392 / 6,310,928 / 18,893,872 parameters. They share the same
16,384-example pool, batch 64, 5,000 updates, raw AMSGradW, learning rate 0.0001,
and one model/data seed. Width and depth are therefore compared independently;
their single-seed results remain exploratory. This finite-pool budget is shorter
than the RASP-L paper's online training experiment.

Noisy 16-bit direct parity is first calibrated at N=64/256/1024/4096/16384 with
width 64, depth two, eight heads, 3,000 updates, batch 64, and learning rate
0.0003. Only N=64 and 256 have final error below 0.01. The next width sweep uses
N=512, the geometric midpoint of the fitted/unfitted bracket; test scores do not
choose it. Widths 16/32/64/128/256/512 are repeated over three initializations
with the same data and noise assignment. Equal updates do not give equal
epochs across the calibration's different sample counts.

The [completed scaling archive](baselines/amsgradw_softmax_scaling_20261002/measurements.json)
contains 27 runs and 89,000 updates: 2,646.53 seconds of training updates,
3,257.10 seconds of total run wall time, and peak CUDA allocation 1,659,151,872
bytes. All width runs share identical split fingerprints and realized noise
105/512 = 20.51%. At widths 16/32/64/128/256/512, final train fit occurred in
0/3, 3/3, 3/3, 3/3, 3/3, and 2/3 runs. Mean clean ID CE was respectively
0.366/1.163/1.431/1.734/1.766/1.822 nats per target. Mean ID accuracy stayed
between 50.52% and 52.86%. There is no mean size-double-descent witness in
either error or loss at margin 0.02; the two individual error witnesses do
not align across seeds or beat their first minima. The largest model's seed-2
train accuracy was 95.51%, so that run also remains an optimization failure.

| Binary copy, final checkpoint | Parameters | Teacher-forced train | ID test | Novel ID validation | Update seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Width 64, depth 2 | 100,752 | 94.19% | 96.09% | 97.14% | 73.32 |
| Width 64, depth 6 | 297,392 | 100% | 100% | 100% | 172.21 |
| Width 512, depth 2 | 6,310,928 | 100% | 100% | 100% | 383.59 |
| Width 512, depth 6 | 18,893,872 | 100% | 100% | 100% | 1,119.80 |

All four models have 0% complete-answer test accuracy at lengths 64 and 128.
The [copy figure](baselines/amsgradw_softmax_scaling_20261002/plots/copy-scaling.png)
also retains OOD token scores: zero complete answers does not mean zero correct
tokens. Independent sampling gives 46 train-overlapping rows among 128 ID
test rows; the separate novel-ID validation column excludes all train inputs
and has support 35. Enlarging the data and model solves same-length-distribution
copy here, but these 5,000-step, single-seed measurements do not demonstrate
length extrapolation, delayed generalization, or a necessary size threshold.

The selected existing Lean resource/interpolation/capacity/depth theorems were
checked with `#print axioms`; their dependencies use only `propext`,
`Classical.choice`, and `Quot.sound`. No Lean sources are changed by these
experiments.

## Reproduced random-feature double descent (2026-10-02)

The local source is *Deep Double Descent*, arXiv:1912.02292v1,
[Appendix C](../../papers/arXiv-1912.02292v1/rffs.tex), Figures 14–15. The
[new trainer](paper_reproduction/RANDOM_FEATURES.md) uses the specified frozen
Gaussian first layer with variance 1/d, exp(-i*x) activation, Fashion-MNIST,
and zero-initialized complex MSE head. QR computes the minimum-norm
gradient-flow limit. All 123 planned fits completed over three paired
independent data/feature seeds and all 10,000 official test images.

| Slice, mean test classification error ± sample SD | Initial point | First minimum | Peak at n=d=1000 | Last point |
| --- | ---: | ---: | ---: | ---: |
| Samples n, fixed d=1000 | 34.45% ± 1.18% at n=100 | 31.28% ± 0.28% at n=300 | 86.45% ± 2.25% | 21.28% ± 0.32% at n=2000 |
| Width d, fixed n=1000 | 24.07% ± 0.09% at d=100 | 21.51% ± 0.19% at d=300 | 86.45% ± 2.25% | 32.42% ± 0.29% at d=2000 |

Both complete classification-error descent/ascent/descent shapes occur in all
three seeds, at margin 0.02, and each seed's global peak is exactly n=d=1000.
The mean rise from the first minimum is 55.17 / 64.93 percentage points; the
second descent is 65.17 / 54.02 points. The sample-wise last point improves on
its first minimum; the model-wise last point does not. The generic earliest
four-point witness can select a point before the actual interpolation peak,
so [summary.json](baselines/fashion_rff_20261002/summary.json) separately retains
the global peak, both branches, and per-seed interpolation points.

Every n≤d fit has MSE below 4.36e-25; the maximum normal-equation residual
over all fits is 7.62e-10. The campaign took 33.74 seconds wall time, including
19.62 seconds of QR fitting on the RTX 3050 Laptop GPU. The
[archive](baselines/fashion_rff_20261002/measurements.json) preserves all
dataset checksums, source hashes, nested training indices, feature fingerprints,
and numerical residuals. [CSV](baselines/fashion_rff_20261002/metrics.csv) and
[PNG/PDF figures](baselines/fashion_rff_20261002/plots/random-feature-slices.png)
are independent of the ignored fitted heads.

This reproduces the published qualitative interpolation effect. The paper
does not disclose pixel normalization, seeds, finite gradient-flow time, or
the complex classification rule. This run records uint8/255, float64/complex128,
argmax of the real part, and the infinite-time limit; its numerical values
differ from Figure 15. Full complex test MSE has an interpolation spike and
recovery, but no complete first-descent/ascent/second-descent witness at the
chosen margin. Classification error and MSE are therefore reported separately.
Because features are frozen, this control demonstrates double descent without
learning new feature representations; it does not establish a causal link to
grokking in GPTMini.

## Modular-division calibration (2026-10-02)

The first author-reference run used the default 20% train fraction: 1,862
training equations, 7,450 exhaustive held-out equations, 455,424 parameters,
two layers, width 128, four heads, AdamW betas (0.9, 0.98), learning rate 0.001,
weight decay 1, and 150,000 updates. Train first reached 99% at update 500 and
ended at 100%, but final held-out accuracy was 1.785%; its maximum across all
601 observations was 2.148%. No held-out transition reached the fixed target.
The measured training/wall budgets were 1,662.67 / 1,690.07 seconds.

The [complete negative archive](baselines/mod97_fraction20_reference_20261002/measurements.json)
retains the original source/data hashes and every observation, with
[CSV](baselines/mod97_fraction20_reference_20261002/metrics.csv) and
[curves](baselines/mod97_fraction20_reference_20261002/plots/modular-generalization.png).
There is a finite held-out CE double-descent witness, but no error witness
and no algorithmic generalization. Later confidence losses and recovery must
therefore not be interpreted as understanding. One unsuccessful initialization
at an explicitly chosen, undisclosed paper train fraction does not refute the
paper's experiment.

A second calibration changed only the train fraction to 50% (4,656 train /
4,656 held-out equations) and completed the full 150,000-update budget.
Its early held-out target onset is update 4,000, confirmed at 4,250. The first
train target crossing at 1,250 is transient, and later joint train/held-out
collapses occur. The complete trajectory and a sustained memorization-phase
diagnostic are needed before claiming the paper's prolonged grokking pattern.
It ended at 67.20% train / 60.59% held-out accuracy, with training/wall budgets
1,653.86 / 1,681.04 seconds. Its
[complete archive](baselines/mod97_fraction50_wd1_reference_20261002/measurements.json)
retains every successful and failed checkpoint, with
[CSV](baselines/mod97_fraction50_wd1_reference_20261002/metrics.csv) and
[curves](baselines/mod97_fraction50_wd1_reference_20261002/plots/modular-generalization.png).
Initialization/data confirmations and GPTMini optimizer controls remain part
of the active reproduction loop.

Reports now retain a second train-fit event requiring the same two-observation
streak as held-out confirmation. For the 50% calibration, this train event is
3,000/3,250 rather than the transient first crossing at 1,250, giving a lag
of 1,000 updates to the first sustained held-out onset. An additional explicit
diagnostic asks for consecutive train≥99% / held-out≤10% observations before
that onset. The 50% / weight-decay-1 run has no such plateau. Reports also keep
the observed target fraction and worst held-out score after confirmation;
initial success does not imply persistence. The 10% ceiling is a conservative
mod-97 diagnostic, not a parameter recovered from the paper. Only 33.73% of
scheduled observations after held-out confirmation met the target, and the
worst held-out accuracy was 0.859%. This transient success does not reproduce
the paper's prolonged, stable memorization-to-generalization pattern.

A third fixed-budget calibration keeps 50% train and changes weight decay
from 1 to 0.1. It completed 150,000 updates using the same model/data seed.
Sustained train fit starts at update 750, confirmed at 1,000; held-out onset
is 31,500, confirmed at 31,750. The lag is 30,750 updates, and the onset/fit
ratio is 42. The longest consecutive memorization plateau spans updates
8,500–14,750 (26 observations): train is at least 99.10% while held-out is
at most 4.40%. Final train and held-out accuracies are both 100%, with final
held-out CE 5.408e-6. Training/wall budgets are 1,659.15 / 1,686.32 seconds.

The [complete calibration archive](baselines/mod97_fraction50_wd01_reference_20261002/measurements.json)
retains all 601 observations and original source/data hashes, with
[CSV](baselines/mod97_fraction50_wd01_reference_20261002/metrics.csv) and
[curves](baselines/mod97_fraction50_wd01_reference_20261002/plots/modular-generalization.png).
Of 474 observations after held-out confirmation, 97.26% meet the target;
the worst held-out accuracy is 11.51%. This run shows a prolonged
memorization phase followed by generalization to unseen equations, with rare
later collapses. Its 31,500-update onset differs from the paper's >100,000;
the train fraction and regularization were explicitly chosen because the
manuscript does not specify them. This is an exploratory qualitative result,
not an exact numerical reproduction or evidence of repeatability.

Before this calibration completed, a separate nine-run confirmation was
frozen: data seed 1, initialization seeds 1/2/3, and the same full budget for
reference/AdamW, GPTMini/AdamW, and GPTMini/raw AMSGradW. It is now running
after checks of the calibration condition, source hashes, exhaustive split,
and independence from calibration seed 0/data seed 0. Its complete trajectories
and failures are required before drawing conclusions about reproducibility or
the requested mini GPT optimizer/architecture combination.

## Independent modular confirmation (in progress)

The reference/AdamW arm completed all 150,000 updates for initialization
seeds 1/2/3 on independent data seed 1. All three have a consecutive
train≥99% / held-out≤10% plateau followed by generalization and 100% final
train/held-out accuracy. Calibration seed 0/data seed 0 is excluded.
Each linked archive retains all 601 observations, the source/split hashes,
CSV, and standalone PNG/PDF curves.

| Seed / archive | Sustained train fit | Held-out onset | Lag | Confirmation (training / wall seconds) | Later observations at target | Worst later held-out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [1](baselines/mod97_fraction50_wd01_confirmation_reference_seed1_20261002/measurements.json) | 750 | 34,750 | 34,000 | 386.24 / 393.52 | 96.75% (461 observations) | 26.42% |
| [2](baselines/mod97_fraction50_wd01_confirmation_reference_seed2_20261002/measurements.json) | 750 | 38,000 | 37,250 | 419.11 / 425.84 | 97.32% (448 observations) | 24.66% |
| [3](baselines/mod97_fraction50_wd01_confirmation_reference_seed3_20261002/measurements.json) | 1,000 | 62,500 | 61,500 | 690.30 / 701.17 | 97.71% (350 observations) | 44.87% |

The mean lag is 44,250±15,027 updates; confirmation costs
498.55±166.87 training / 506.84±169.07 wall seconds. These are means and
sample SD across three initializations on one confirmation split, rather than
confidence intervals or independent-split variability. Longest memorization
plateaus span 9,250–18,750, 750–6,250, and 29,750–46,500, respectively.
Full-run training time averages 1,648.44±2.24 seconds. All later collapses
remain in the curves; repeated phase sequences do not imply uninterrupted
performance or an identified internal algorithm.

This confirms the qualitative memorization-then-generalization sequence in
3/3 independent reference initializations. Our predefined 99% onsets are
34,750–62,500. For comparison with the manuscript's wording about perfect
generalization, the first observed 100% held-out scores are 36,000, 40,250,
and 64,000 (descriptive observations, not new stopping criteria). Both sets
of timings are earlier than the reported >100,000 updates; train fraction
and regularization were chosen explicitly because their exact values are
undisclosed.

GPTMini/AdamW seed 1 also completed 150,000 updates, with 423,816 parameters
and the explicitly different matrix initialization. Sustained train fit starts
at 10,250 and held-out onset at 49,750, confirmed at 50,000: a 39,500-update
lag. Its 21,750–28,500 memorization plateau keeps train at least 99.89% and
held-out at most 9.66%. Final train/held-out accuracies are 100%, with held-out
CE 3.576e-8. Confirmation costs 504.44 training / 514.11 wall seconds; the full
budget costs 1,514.44 / 1,543.37 seconds. After confirmation, 97.26% of 401
observations meet the target; the worst held-out accuracy is 46.39%.

The [complete GPTMini archive](baselines/mod97_fraction50_wd01_confirmation_gptmini_adamw_seed1_20261002/measurements.json)
and [accuracy/error/CE curves](baselines/mod97_fraction50_wd01_confirmation_gptmini_adamw_seed1_20261002/plots/modular-generalization.png)
also show the restricted epoch-error double-descent shape: held-out error
falls from 100% initially to 23.05% at update 2,000, rises to 93.15% at
28,250 while train is fitted, and ends at 0%. The overfitting rise is 70.10
percentage points and recovery is 93.15. This run shows finite error
double descent together with the memorization-then-generalization diagnostic. The
synthetic adaptation and post hoc selection rule are explicit; coexistence
does not identify causality, and one seed does not establish repeatability.

GPTMini/AdamW seed 2 completed the same full budget with 100% final
train/held-out accuracy. Sustained train fit starts at 1,000 and held-out onset
at 1,250, confirmed at 1,500: a 250-update lag and 15.13 training / 15.44 wall
seconds to confirmation. There is no consecutive low held-out memorization
plateau and no restricted pre-generalization error double-descent shape.
This is rapid generalization, rather than the prolonged sequence of seed 1.
After confirmation, 97.98% of 595 observations meet the target; the worst
held-out accuracy is 27.26%. Full-run costs are 1,515.02 / 1,543.90 seconds.
The [complete second GPTMini archive](baselines/mod97_fraction50_wd01_confirmation_gptmini_adamw_seed2_20261002/measurements.json)
and [curves](baselines/mod97_fraction50_wd01_confirmation_gptmini_adamw_seed2_20261002/plots/modular-generalization.png)
retain all later collapses. The large seed dependence must accompany any
time-to-target comparison: seed 1 takes 504.44 training seconds, while seed 2
takes 15.13.

GPTMini/AdamW seed 3 completed the full budget with 99.87% train / 99.57%
held-out final accuracy (6 and 20 incorrect equations, respectively). Train
fit starts at 500, held-out onset at 1,000, and confirmation at 1,250 costs
12.64 training / 12.90 wall seconds. The 500-update gap has no low held-out
memorization plateau or restricted epoch-error double-descent shape. After
confirmation, 97.32% of 596 observations meet the target; worst held-out
accuracy is 88.25%. Full-run costs are 1,518.54 / 1,547.43 seconds. The
[complete third GPTMini archive](baselines/mod97_fraction50_wd01_confirmation_gptmini_adamw_seed3_20261002/measurements.json)
and [curves](baselines/mod97_fraction50_wd01_confirmation_gptmini_adamw_seed3_20261002/plots/modular-generalization.png)
retain the final errors and all earlier observations.

All three GPTMini/AdamW seeds meet the frozen 99% final target, with mean
held-out accuracy 99.8568±0.2480%. Only 1/3 has the prolonged memorization
sequence and restricted epoch-error shape; 2/3 generalize early. Mean
confirmation time is 177.40±283.23 training / 180.82±288.64 wall seconds,
compared with 498.55±166.87 / 506.84±169.07 for the reference. The observed
ratio of mean training costs is about 2.81, with large per-seed variation;
these are sample SD, not confidence intervals or evidence of universal speedup.
Architecture, parameter count, and initialization differ as documented, so
this comparison does not isolate the cause.

GPTMini/raw AMSGradW seed 1 completed 150,000 updates with 100% final
train/held-out accuracy and held-out CE 0.009266. Sustained train fit starts
at 3,000, held-out onset at 21,000, and confirmation at 21,250: an
18,000-update lag and 229.62 training / 233.71 wall seconds to confirmation.
Full-budget costs are 1,620.21 / 1,649.05 seconds. There is no consecutive
train≥99% / held-out≤10% memorization plateau before generalization; the
held-out split is already partially generalized while train fits. Retain
that distinction from the stronger plateau diagnostic used for the reference.

The [complete first AMSGradW archive](baselines/mod97_fraction50_wd01_confirmation_gptmini_amsgradw_seed1_20261002/measurements.json)
and [accuracy/error/CE curves](baselines/mod97_fraction50_wd01_confirmation_gptmini_amsgradw_seed1_20261002/plots/modular-generalization.png)
show the restricted epoch-error shape: 100% initial error falls to 59.66% at
1,250, rises to 70.19% at 3,000 while train is fitted, and ends at 0%.
The overfitting rise is 10.52 percentage points and recovery is 70.19.
This error shape alone does not satisfy the plateau-then-generalization
diagnostic. After confirmation, 465 of 516 observations (90.12%) retain the
99% target, with a worst held-out accuracy of 0%. At that worst observation
(73,750), numeric-answer accuracy is also 0% while EOS accuracy is 99.72%;
the collapse is not an EOS-only scoring artifact. These later collapses
remain separate from the pre-generalization error diagnostic and their cause
is not identified.

GPTMini/raw AMSGradW seed 2 completed the same full budget with 100% final
train / 97.53% held-out accuracy and held-out CE 0.037016. The 115 incorrect
held-out equations mean that this run fails the frozen 99% final target,
despite reaching it earlier. Sustained train fit starts at 5,000, held-out
onset at 40,500, and confirmation at 40,750: a 35,500-update lag and
440.23 training / 448.09 wall seconds to confirmation. Full-budget costs
are 1,620.38 / 1,649.26 seconds. There is no consecutive low held-out
memorization plateau before generalization.

The [complete second AMSGradW archive](baselines/mod97_fraction50_wd01_confirmation_gptmini_amsgradw_seed2_20261002/measurements.json)
and [curves](baselines/mod97_fraction50_wd01_confirmation_gptmini_amsgradw_seed2_20261002/plots/modular-generalization.png)
retain the failed final score. The restricted error shape goes from 100%
initial error to 75.04% at 1,250, then 89.07% at 13,750 while train is fitted,
and 2.47% finally. The rise is 14.02 percentage points and recovery is 86.60;
finite error double descent does not require attaining the final 99% target.
Only 164 of 438 observations after confirmation (37.44%) retain that target,
with a worst held-out accuracy of 4.32% at 90,000. Numeric-answer accuracy is
also 4.32% there, while EOS is 99.98%. Both completed AMSGradW seeds show
the restricted error shape without the stronger memorization plateau, and
their final target outcomes differ. Eight of nine recipes are complete;
raw AMSGradW seed 3 remains required before judging optimizer repeatability
or declaring the campaign complete.

## Double descent is a separate observation

Double descent is a descent, ascent, and second descent in held-out error as
model size, sample count, or training time changes. It can accompany delayed
algorithmic generalization, but does not establish that mechanism. The local
[double-descent paper](../../papers/arXiv-1912.02292v1/model_dd.tex), Section 5,
also discusses linear-model examples and noise absorption among interpolating
solutions. Section 6 describes an epoch-wise second descent after overfitting.
The mechanisms behind deep-network double descent remain open in that paper.

Modular reports now also describe early held-out improvement before sustained
train fit, worsening while train is fitted before the first held-out target
onset, and final recovery. Each error difference must exceed the existing
0.02 margin at four ordered observations. This restricted, post hoc diagnostic
was added after the first GPTMini/AdamW transition; the nine training recipes
remain frozen. It reports error double descent and coexistence with the
memorization-then-generalization diagnostic separately, and excludes later
collapses from its overfitting peak. It is a synthetic analogue of Section 6,
whose source experiments use label noise on CIFAR and CNN/ResNet models.
Coexistence does not identify a causal mechanism; a negative diagnostic leaves
other epoch-error shapes unclassified.

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
