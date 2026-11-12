# Division mod 97 under the author's Adam and AdamW

Does the transformer of arXiv:2201.02177 (Power et al.), trained on half of the division table
mod 97 exactly as the author's code trains it, come to answer every held-out equation: under
Adam without weight decay within the `10⁶` updates of the paper's Figure 1, and under AdamW
with weight decay 1 within the `10⁵` updates of its default setting?

[PROTOCOL.md](PROTOCOL.md) pins the task, the model, the loss, the optimizer and the sampler
to the paper and to openai/grok at `3d64b1d8`, lists where the two disagree, and fixes the
recorded steps, the terminal window and the outcome classes. It was frozen before training.

## Runs

| Cohort | Optimizer | Weight decay | Updates | Seeds | Supervised tokens |
| --- | --- | ---: | ---: | --- | --- |
| `adam` | Adam | 0 | 1 000 000 | 0, 1, 2 | answer and closing `<\|eos\|>` (the code's loss) |
| `adamw` | AdamW | 1 | 100 000 | 0, 1, 2 | answer and closing `<\|eos\|>` |
| `adamw --answer-only` | AdamW | 1 | 100 000 | 0, 1, 2 | answer alone (the paper's wording) |
| `adam`, loss in doubles | Adam | 0 | 1 000 000 | 0, 1, 2 | answer and closing `<\|eos\|>`; the trainer of commit `b8ff967f` |
| `adam --flush-from 100000` | Adam | 0 | 1 000 000 | 0, 1, 2 | answer and closing `<\|eos\|>`; subnormals flushed to zero after update 100 000 (Amendment 2) |

Equal seeds have equal splits, initial parameters and batches in every cohort. Seed 0 has the
split of the author's code.

## Commands

The trainer is native Rust; its matrix products run Kernite's blocked x86 kernel (AVX2 and
FMA). Kernite is expected at `/home/misha/rust_projects/kgen`, clean at
`a1b6bc2c956e1a5f35960e26b209e1b6c213096c`; `check` refuses any other state.

```sh
cd experiments/grok_division
cargo test --offline --release           # 38 tests: 36 of the trainer, 2 of `numerator`
cargo build --offline --release
target/release/grok-division check       # pins, split fingerprints, f32 model against the model in doubles
target/release/grok-division bench 8     # the time of an update
for seed in 0 1 2; do
  target/release/grok-division run adamw $seed runs/adamw-seed$seed --threads 8
  target/release/grok-division run adam $seed runs/adam-seed$seed --threads 8
  target/release/grok-division run adamw $seed runs/adamw-answer-seed$seed --threads 8 \
    --answer-only
done
# Amendment 2: the Adam runs go on from their update 100000 with subnormals flushed to zero
for seed in 0 1 2; do
  target/release/grok-division run adam $seed runs/adam-flush-seed$seed --threads 8 \
    --flush-from 100000 --adopt runs/adam-seed$seed
done
target/release/grok-division report runs/*-seed*
target/release/diagnose runs/adamw-seed1 1 10000 50000   # what checkpoints do at the answer
target/release/numerator runs/*-seed*    # the held-out equations with numerator zero
target/release/numerator --list runs/adamw-seed1   # its evaluations with one of them wrong
```

`diagnose RUN SEED STEP...` reads checkpoints with the model in doubles: accuracy on 512
training equations, the hidden units that fire, the entropy of the attention, what the
normalizations divide by, and the norms of the parameter groups. It is a second binary and
does not change the trainer or its source fingerprint.

`numerator RUN...` reads the held-out predictions that a run recorded at every evaluation and
counts the wrong answers among the equations `0 / y = 0`; with `--list` it prints the
evaluations that have one of them wrong. It is a third binary, shares only the constants of
the data with the trainer, and refuses a run where its count over all held-out equations
differs from the trainer's at any evaluation.

One test of the trainer was added after the runs: `hidden_units_at_zero_pass_no_gradient`
holds both models to the source's derivative 0 of the ReLU at 0, where the loss has a kink.
It is in `src/tests.rs`, which the trainer's source fingerprint does not cover.

A run resumes from its last checkpoint when the command is repeated, on any number of
threads, and is then bit for bit the run that was not stopped. `runs/` is ignored by git.

Hardware and precision. The runs were made on two Xeon E5-2680 v4 (28 cores, 56 threads),
several at a time on 4 to 16 threads each; a run of 10⁵ updates took 4750 to 6360 s, and the
9 · 10⁵ flushed updates of an `adam` run 20 040 to 20 140 s, the three runs at a time, on 8
threads to update 150 000 and on 16 after it. All training arithmetic is `f32` on the CPU,
in the order that PROTOCOL.md fixes, with gradual underflow unless `--flush-from` is given.
The machine's GPU, a GeForce GTX 1050 with 2 GiB, was not used.

## Findings

State on 2026-11-12: the cohorts `adamw`, `adamw --answer-only` and
`adam --flush-from 100000` are complete. The `adam` runs were stopped at updates 114 000,
131 000 and 100 000 (seeds 0, 1, 2) and were continued as `adam --flush-from 100000`
(Amendment 2) to update 1 000 000. The numbers are those of `report`, of `numerator` and of
`history.jsonl`. Accuracy is the answer's, on the 4656 training or the 4656 held-out equations;
a guess among the 97 residues has 1.0%.

### AdamW with weight decay 1, 10⁵ updates: perfect held-out accuracy is reached and lost

| Cohort | Seed | Training ≥ 99%, first | Held-out ≥ 99%, first | Every held-out answer right: evaluations (first to last) | Held-out ≥ 99%, last | Held-out in the window: least / mean / largest | Final training / held-out |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| `adamw` | 0 | 1500 | 3100 | 57 (3300 to 40 000) | 59 000 | 0.102 / 0.134 / 0.150 | 0.134 / 0.102 |
| `adamw` | 1 | 1100 | 3200 | 62 (3600 to 47 000) | 47 000 | 0.025 / 0.094 / 0.116 | 0.150 / 0.106 |
| `adamw` | 2 | 3600 | 4300 | 37 (5300 to 30 000) | 44 000 | 0.068 / 0.221 / 0.338 | 0.330 / 0.265 |
| `adamw --answer-only` | 0 | 1200 | 3800 | 41 (4400 to 26 000) | 26 000 | 0.243 / 0.412 / 0.444 | 0.578 / 0.431 |
| `adamw --answer-only` | 1 | 1000 | 3100 | 78 (3700 to 61 000) | 80 000 | 0.263 / 0.482 / 0.624 | 0.672 / 0.624 |
| `adamw --answer-only` | 2 | 3000 | 9200 | none (best 0.9981) | 99 000 | 0.505 / 0.891 / 0.997 | 0.983 / 0.947 |

Each run has 194 evaluations; the window is the 11 of updates 90 000 to 100 000.

- By the protocol's classes all six runs are *not generalized within the budget*: no seed has
  99% in its whole window, and none has every held-out answer right from some step on.
- By the paper's two measures every run generalizes. The first step with 99% validation
  accuracy (the paper's Figure 1, center) is between 3100 and 9200, and the best validation
  accuracy within the budget (its Figure 2, right) is 1 in five runs and 0.9981 in the sixth.
  The paper's measures do not ask that the accuracy is kept, and here it is not.
- What is lost is the fit, not the generalization alone: training accuracy falls with the
  held-out accuracy, to between 0.13 and 0.98 at the last update. The loss is not one event.
  Between its first and its last evaluation with at least 99%, each run already has
  evaluations below 90%, 8 to 31 of them among 79 to 140, and it recovers from these. After
  the last such evaluation the largest held-out accuracy of the six runs is 0.153, 0.120,
  0.981, 0.460, 0.974 and 0.947.
- Training on the answer alone (the paper's wording, deviation D2) does not remove the
  instability. Its seeds lose less: the window's mean is 0.41, 0.48 and 0.89 against 0.13,
  0.09 and 0.22.

What the checkpoints of `adamw`, seed 1, show on 512 training equations (`diagnose`):

| Update | Accuracy | Readout norm | Gains of block 0: mean of the first, of the second | Divisors of the normalizations of block 0, of block 1 | Attention entropy of block 1 (largest 1.609) |
| ---: | ---: | ---: | --- | --- | ---: |
| 10 000 | 1.000 | 20.3 | 0.009, 0.036 | 0.200 and 0.022, 0.021 and 0.019 | 1.066 |
| 40 000 | 0.930 | 21.4 | 0.005, 0.016 | 0.190 and 0.017, 0.026 and 0.012 | 1.041 |
| 50 000 | 0.152 | 43.9 | 0.001, 0.001 | 0.356 and 0.009, 0.017 and 0.015 | 1.609 |
| 100 000 | 0.148 | 39.9 | 0.002, −0.001 | 0.430 and 0.009, 0.017 and 0.010 | 1.609 |

The code has one parameter group, so weight decay 1 also acts on the gains of the
normalizations and on the embedding. Already at update 10 000, with every answer right, the
gains of the first block are near 0.01, and three of the four normalizations divide by a
standard deviation near 0.02, six times the floor `sqrt(1e-5) = 0.003` that their epsilon
sets. After the loss of accuracy the readout's norm has doubled, the second gain of the first
block is 0.001, and the attention of the second block is uniform over the five positions it
sees. This describes the state before and after; it does not prove what causes the loss.

### Adam without weight decay: the first 10⁵ updates

- Training accuracy first reaches 99% at updates 800, 700 and 900 (seeds 0, 1, 2): the
  paper's "close to perfect at < 10³ optimization steps".
- Held-out accuracy at update 100 000 is 0.031, 0.112 and 0.030, with training accuracy at
  least 0.999: the paper's "very little evidence of any generalization until 10⁵ steps". It
  rises slowly: 0.024, 0.044 and 0.024 at update 50 000.
- Whether it reaches the level of Figure 1 "close to 10⁶ steps" is what the cohort
  `adam --flush-from 100000` measures, in the next section.

### Adam without weight decay, 10⁶ updates: one seed of three reaches 99%

The cohort `adam --flush-from 100000`: the `adam` runs from their update 100 000, with
subnormals flushed to zero (Amendment 2). Held-out accuracy:

| Update | Seed 0 | Seed 1 | Seed 2 |
| ---: | ---: | ---: | ---: |
| 100 000 | 0.031 | 0.112 | 0.030 |
| 200 000 | 0.045 | 0.387 | 0.056 |
| 300 000 | 0.064 | 0.657 | 0.090 |
| 400 000 | 0.093 | 0.866 | 0.164 |
| 500 000 | 0.117 | 0.944 | 0.235 |
| 600 000 | 0.152 | 0.985 | 0.339 |
| 700 000 | 0.159 | 0.993 | 0.436 |
| 800 000 | 0.199 | 0.992 | 0.566 |
| 900 000 | 0.240 | 0.994 | 0.671 |
| 1 000 000 | 0.261 | 0.998 | 0.814 |

| Seed | Held-out ≥ 99%, first | Held-out ≥ 99% from | Every held-out answer right | Held-out in the window: least / mean / largest | Most wrong answers in the window | Final training / held-out | Outcome |
| ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 0 | never | never | never (best 0.272) | 0.224 / 0.247 / 0.272 | 3612 | 1.000 / 0.261 | not generalized within the budget |
| 1 | 623 000 | 785 000 | never (best 0.9991) | 0.9918 / 0.9964 / 0.9991 | 38 | 1.000 / 0.998 | at least 99% |
| 2 | never | never | never (best 0.814) | 0.659 / 0.734 / 0.814 | 1589 | 1.000 / 0.814 | not generalized within the budget |

Each run has 1094 evaluations; the window is the 101 of updates 900 000 to 1 000 000.

- By the protocol's classes seed 1 is *at least 99%*, and seeds 0 and 2 are *not generalized
  within the budget*. No seed is *exact*: no evaluation of any seed has every held-out answer
  right, and the best one, of seed 1 at update 966 000, has 4 wrong answers of 4656.
- Seed 1 is the course of the paper's Figure 1 (left): little held-out accuracy at 10⁵
  updates, 0.112, and more than 99% before 10⁶. It first has 99% at update 623 000 and keeps
  it from update 785 000; between the two, 46 of its 162 evaluations are below 99%, the least
  0.980.
- Seeds 0 and 2 rise through the whole budget and end at 0.261 and 0.814; seed 2 has its best
  value at the last update. Whether they reach 99% later is not measured.
- Training accuracy does not stay at 1. After update 100 000 it is below 1 at 78, 62 and 112
  of the 901 evaluations (seeds 0, 1, 2), the least 0.9963, 0.9942 and 0.9936, and in the
  window the training cross-entropy of the answer is as large as 0.005, 0.075 and 0.009.
  These are the loss spikes of Amendment 2. Unlike under AdamW the fit comes back: the final
  training accuracy is 1 in the three runs.
- After update 100 000 the cohort does not have the source's arithmetic (next section). With
  gradual underflow the runs exist to updates 114 000, 131 000 and 100 000 only.

### The held-out equations with numerator zero

Of the 96 equations `0 / y = 0`, 51, 54 and 53 are held out (seeds 0, 1, 2). The table has
the nine runs that reached their budget; "later" is after the first evaluation with every one
of these equations right.

| Cohort | Seed | All right first at | All right from | Evaluations with all right | Later with one wrong: all, with the training set fitted | Window: all right, most wrong | Last update: wrong of them, of all |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `adamw` | 0 | 50 | 25 000 | 186 of 194 | 5, 0 | 11 of 11, 0 | 0, 4179 |
| `adamw` | 1 | 200 | 99 000 | 182 of 194 | 7, 0 | 9 of 11, 54 | 0, 4162 |
| `adamw` | 2 | 100 | 90 000 | 181 of 194 | 9, 0 | 11 of 11, 0 | 0, 3420 |
| `adamw --answer-only` | 0 | 50 | 38 000 | 189 of 194 | 2, 0 | 11 of 11, 0 | 0, 2651 |
| `adamw --answer-only` | 1 | 100 | 63 000 | 187 of 194 | 3, 0 | 11 of 11, 0 | 0, 1750 |
| `adamw --answer-only` | 2 | 100 | 63 000 | 187 of 194 | 3, 0 | 11 of 11, 0 | 0, 248 |
| `adam --flush-from 100000` | 0 | 50 | 998 000 | 877 of 1094 | 214, 190 | 77 of 101, 2 | 0, 3439 |
| `adam --flush-from 100000` | 1 | 200 | never | 787 of 1094 | 302, 285 | 13 of 101, 7 | 2, 9 |
| `adam --flush-from 100000` | 2 | 100 | 961 000 | 985 of 1094 | 105, 87 | 81 of 101, 3 | 0, 864 |

- These equations are answered first. Every one is right at update 50 to 200, where 67 to 96
  of the 4656 held-out answers are right and they are more than half of those.
- Under AdamW they are wrong only where training equations are. The later evaluations with
  one of them wrong are 5, 7 and 9 for `adamw` and 2, 3 and 3 for `adamw --answer-only`. Each
  has training equations wrong, with training accuracy at most 0.953, and they are few of the
  95 to 130 later evaluations that have. At the last update none of them is wrong in the six
  runs, while 248 to 4179 of all held-out answers are. One window has such evaluations:
  `adamw`, seed 1, at updates 91 000 (all 54 wrong, training accuracy 0.040) and 98 000 (23
  wrong, 0.070).
- Under Adam they are wrong with the training set fitted: 190, 285 and 87 later evaluations
  have every training equation right and one of them wrong, at most 3, 7 and 3 of them at
  once. No seed keeps them to the end of the budget: in the window all are right at 77, 13
  and 81 of the 101 evaluations.
- Seed 1 of Adam, the seed with 99% held-out accuracy, has them wrong more often as it
  generalizes: at 10, 10, 6, 2, 16, 31, 64, 68 and 88 of the 100 evaluations of the second to
  the last tenth of the run. In its window they are 265 of its 1676 wrong answers, 16%, and
  1.2% of its held-out equations, and 2 of the 9 wrong answers of its last update are of this
  kind. In the windows of seeds 0 and 2 they are 29 of 353 997 and 29 of 124 913.

### The arithmetic decides the late trajectory

- Single precision changes the loss (Amendment 1). Once the probabilities of the other
  classes sum to less than half a unit in the last place of 1, PyTorch's `f32` loss is 0 and
  the target's entry of its gradient is exactly 0. The first launch computed the softmax in
  doubles and was stopped at 14 000 updates (`adam`: training accuracy 1, held-out 0.017 to
  0.018) and 8000 (`adamw`); it belongs to no cohort, and the cohort `adam`, loss in doubles,
  was not run.
- Subnormal numbers slow the `adam` runs sevenfold (Amendment 2): 1000 updates take 43 s
  before update 70 000 and more than 300 s at update 100 000, when 18 048 entries of each
  Adam moment are subnormal. Flushing them to zero restores 31 s.
- The flush is not neutral. From the same checkpoint of update 100 000, the run with gradual
  underflow and the run with the flush differ at the first update in 31 entries of the
  gradient of the readout, by at most one unit in the last place, through a rounding tie,
  and 1000 updates later in their record. Side by side, held-out errors of 4656:

  | Update | Seed 0, gradual | Seed 0, flushed | Seed 1, gradual | Seed 1, flushed |
  | ---: | ---: | ---: | ---: | ---: |
  | 100 000 | 4510 | 4510 | 4134 | 4134 |
  | 101 000 | 4490 | 4482 | 4173 | 4128 |
  | 110 000 | 4494 | 4486 | 4098 | 4037 |
  | 114 000 | 4518 | 4483 | 4060 | 3995 |
  | 131 000 | | 4468 | 3915 | 3861 |

  The two arithmetics give different runs of the same slow rise. A seed's late trajectory
  under `adam` is a property of the last bit of the arithmetic, not of the recipe and the
  seed alone.

### What the runs do not show

- The trainer was not compared with a PyTorch run. The author's code was read, never
  executed, and the random streams are not PyTorch's. The loss of accuracy under AdamW may be
  the recipe's or an error of this transcription; the tests hold the optimizer to its
  recurrence written out in doubles and the model to finite differences, not to the source's
  outputs.
- Three seeds, one machine, `f32`, a finite horizon. Seeds 1 and 2 have splits that the
  source never draws (deviation D1).
- A first crossing of 99% is not convergence: under AdamW every run crosses it and none
  keeps it. Under Adam one seed of three keeps 99% for its last 215 000 updates, and it never
  has every held-out answer right. No run here supports a claim that held-out accuracy tends
  to 1.
