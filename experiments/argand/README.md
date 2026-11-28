# Argand on the GPT stress tiers

Can a complex decoder with one residual stream and a tied vocabulary
readout learn the frozen GPT stress tiers from random embeddings? The
runs vary its residual, the starting value $s_0$ of its score
multiplier, its optimizer and its batch.
[Architecture](ARGAND.md) · [Lean forward pass](../../src/Transformer/Grokking/FiniteRecovery/Argand.lean)

## Training and replay

Argand has trainable skip and write gates around every sublayer and
starts its score multiplier at $s_0 = 1 + 0i$. The default `run` command
uses the clipped momentum steps of Lean's `fullGradientMomentumRun`.
Each momentum update takes the gradient of the full
training loss plus $\frac{\lambda}{2}\lVert\theta\rVert^2$, with
$\lambda = 10^{-4}$. The model has no configuration flags and runs on
Kernite 0.8.0. The data generators, scorer, WSD schedule and shared
kernels are modules of the pinned GPT checkout, compiled unchanged.
The build checks the checkout's commit. [Pinned checkout](../phasor_value_transport/README.md#reproducing-the-pair-address-baseline)

```sh
cargo +stable build --locked --release
target/release/argand check
out=runs/argand-short
mkdir -p "$out"
./run_argand.sh short 1 "$out" 1000 | tee "$out/short-1000.jsonl"
```

`check` tests on the GPU the tied classifier, prefix causality,
finite-difference directional derivatives, the microbatch accumulation
and the clipped momentum step. `run CASE SEED REPEAT RUN_DIR
[LAST_OBSERVATION]` trains one cell of the 25 short and 10 long cases.
It stops at the first scheduled observation with at least 99% exact
sequences on the primary held-out split, or at the requested
observation. Rerunning it on the same directory resumes the saved
weights and velocity. `replay CASE SEED REPEAT RUN_DIR` reloads the
checkpoint in a new process and scores every held-out and transfer
split. `run_argand.sh TIER REPEAT RUN_DIR [MAX_UPDATES]` runs the three
seeds of every case in a tier and replays each pass.

`peak` measures the FP32 multiply-add and DRAM copy rates of the GPU.
`bench CASE UPDATES` times full-batch updates of seed 0 of a case. It
counts the real operations of their complex matrix products and
reports the time those take at the measured peak, the compute bound, as
a fraction of the measured time.

Each update visits the training split in order, in microbatches of the
case's batch size. Scoring uses the same batch size and scores each
example independently. One checkpoint replayed with batches of 16, 32
and 128 gave identical scores and losses on all 13 splits. Checkpoints
from commit e38c01a3 resume here. So do those from commit 7399c3bb with
`ARGAND_SCORE_INIT=1 ARGAND_BATCH=full ARGAND_PENALTY=1e-4`.

## Adaptive PAGE from native initialization

The adaptive runner trains every coordinate of the same decoder in FP32.
Its only random inputs are the ordinary initializer, the shuffled training
order where applicable, and the PAGE sampler. Training uses observed
cross-entropy. Hidden labels are used only for evaluation.

For a reference batch with $N$ supervised token positions and mean loss
$l_0$ at initialization, the runner fixes
$\lambda=2l_0/(N\max(1,\lVert w_0\rVert^2))$ for the entire run.
It adds $\lambda w$ to each mean gradient, including the gate gradients.
The first trial uses $\eta=F(w_0)/\lVert g_0\rVert^2$ for
$F=l+\lambda\lVert w\rVert^2/2$. Subsequent trials start at twice the
last accepted rate. A trial must reduce the current reference objective
by at least $\eta\lVert g\rVert^2/2$. A rejected estimated direction
triggers a gradient refresh at the saved weights. Exact-direction retries
halve the rate. Each correction evaluates the same randomly selected
positions at both accepted endpoints. With $b$ distinct positions the
scheduled refresh probability is $b/(N+b)$, with a full refresh for $b=N$.
The baseline uses $b=1$. The variance-reduction experiment uses
$b=\min(\lceil\sqrt N\rceil,\text{batch capacity})$. Lean's constant-rate
PAGE results have their own step and reference-objective hypotheses.

The eight-bit sum uses a width of 64 complex channels, four heads,
an FFN width of 256, six layers and one pass. The observed 31-word
staircase is the entire reference batch throughout training. Its
complement contains 225 words. Each long run starts again from its
saved pair of seeds and stores checkpoints from one uninterrupted
20,480-update trajectory. FP32 atomic gradient sums can vary between
repetitions. The accuracy curves use checkpoints of the long run.
The binary phase table is $X_{ij}=P(0\mid i,j)-P(1\mid i,j)$.
Its 16 singular values, nuclear norm and sign certificate are computed
on the frozen weights. [Lean certificate](../../src/Transformer/Grokking/FiniteRecovery/CheckpointCertificate.lean)

The division experiment predicts only $q$ from the five tokens through
the equals sign. It uses the original 239-token vocabulary and the fixed
seed-zero 50/50 split of all 9,312 equations. It uses the width and depth
of the sum experiment, a reference batch of 512, and 8,192 updates.
The stress cohort uses each case's own shape, label mask, batch size,
shuffled epochs, exact-sequence scorer and first 99% observation rule.
It stops by the last scheduled observation through 1,000 updates. For
these larger splits the line search compares both weight vectors on the
same current minibatch. The reference objective is re-evaluated when
moving to the next minibatch. Every terminal checkpoint is independently
reloaded and scored. [Division data](../grok_division/PROTOCOL.md)

```sh
cargo +stable build --release --locked --offline
cargo +stable test --release --locked --offline -- --ignored --test-threads=1
bin=target/release/argand
sum=runs/sum8-page-20480
mkdir -p "$sum"
cp sum8-page-protocol.json "$sum/protocol.json"
"$bin" sum8-page "$sum" 20480
"$bin" sum8-replay "$sum"
bash plot_sum8_accuracy.sh "$sum"
"$bin" sum8-spectrum "$sum"
"$bin" sum8-adamw runs/sum8-adamw-20480 20480
"$bin" sum8-adamw-replay runs/sum8-adamw-20480 20480
"$bin" sum8-spectrum runs/sum8-adamw-20480
bash plot_sum8_accuracy.sh runs/sum8-adamw-20480
"$bin" page-run grok-division-p97 0 runs/division-page 8192
bash run_page_screen.sh all runs/gpt-page
bash run_page_screen.sh all runs/gpt-adamw adamw
jq -n --slurpfile control runs/gpt-page/all-summary.json \
  --slurpfile candidate runs/gpt-adamw/all-summary.json -f compare_optimizer_screens.jq
repair=runs/sum8-page-sqrt-2048
mkdir -p "$repair"
jq '.updates=2048 | .protocol="argand-sum8-adaptive-page-sqrt-v1"' sum8-page-protocol.json > "$repair/protocol.json"
"$bin" sum8-page-sqrt "$repair" 2048
retry=runs/sum8-page-retry-2048
mkdir -p "$retry"
jq '.updates=2048 | .protocol="argand-sum8-adaptive-page-retry-v1"' sum8-page-protocol.json > "$retry/protocol.json"
"$bin" sum8-page-retry "$retry" 2048
for rule in sqrt retry; do
    out="runs/page-paired-$rule"
    "$bin" "page-run-$rule" grok-division-p97 0 "$out" 2048
    for case in suite-boolean-and-shift arithmetic-div arithmetic-abs-diff-blind; do
        for seed in 0 1 2; do "$bin" "page-run-$rule" "$case" "$seed" "$out" 1000; done
    done
done
```

The paired AdamW sum experiment reads the same eight initialization seeds
from `sum8-page-protocol.json`. Each update uses all 31 observed words.
The AdamW parameters and WSD schedule match the division trainer at the
chosen budget. The 225 hidden words are scored from saved weights.

On the 25 short GPT tasks, the single-position PAGE correction reached
the 99% screen threshold in 14 of 75 runs. Seven tasks passed with at
least one seed and three passed with all three. Six runs were strictly
correct on their entire primary split, the Boolean AND shift and random
cases on all seeds. These primary splits contain seven and eight examples,
respectively. The separate exhaustive Boolean audit covers unseen inputs.
All 30 long runs finished below 99%. Across both tiers PAGE forced a
fresh gradient on the current minibatch 91,841 times in 96,275 accepted
updates (95.4%). Independent checkpoint replays verified all 105 outcomes.

At the same 1,000-update cutoff, AdamW with WSD passed 34/75 short runs
and 0/30 long runs. Fifteen short endpoints were strictly correct on all
primary examples. The two optimizers shared 11 short passes. AdamW gained
23 and lost three, all three seeds on `arithmetic-cube-plus`. The shapes,
initial scores and losses matched for all 105 pairs. With the standard
per-task schedule, 21 of 25 short and nine of ten long cases reached
their cutoff during warmup. On long tasks, the best score for each
optimizer came from `associative-divisor-n2-l1-w0`. AdamW scored
3,807/4,656 and PAGE scored 2,525/4,656.

The confirmation protocol selects the first trained task in the agreed
order with an exact primary score and strictly positive margins, then
draws 12 fresh pairs of initialization and sampler seeds. All 12 train
to the original fixed cap. The discovery seeds are excluded from the
estimate. The null is a single-run success probability of at most
three quarters. Under independent sampling, 12 successes have
probability at most $(3/4)^{12}$ under that null. For Boolean AND the
secondary audit enumerates all 130,816 binary strings of lengths 8 to 16,
deduplicates the training prompts and scores both the answer and EOS.
It records generator support and unseen multi-zero strings separately.
`page-confirm CASE RUN_DIR CAP` records the seed pairs before training.

On Boolean AND with a position shift, PAGE scored 7/7 on the primary
split at update 1,000 in all twelve fresh runs. All twelve also solved
the 40 unseen prompts on the training generator's support. Eleven
solved every one of the 130,763 unseen strings. Seed 3433824017 missed nine
strings with at least two zero bits. The one-sided 95% binomial lower
limit for exhaustive success is 0.6613. For this endpoint the upper-tail
probabilities under single-run success at most one half and three
quarters are 0.003174 and 0.158382. The 0.7791 lower limit applies to
the twelve-for-twelve endpoints. Each reported outcome uses the seed
draw as its trial. [Fixed-horizon binomial law](../../src/Transformer/Grokking/FiniteRecovery/TerminalBinomial.lean)

The training-only graph audit is an independent control for division. Among
the 4,656 observed equations, 4,611 have a nonzero numerator. Their
96-by-96 bipartite mask is connected and contains a perfect matching. In
the output coordinates of a faithful multiplicative character the observed
phases are consistent, and propagation recovers every nonzero hidden
answer. The other 45 observations identify the constant zero-numerator
row. Together these checks recover 4,656/4,656 hidden answers. The
`division-certificate CELL terminal` command reconstructs these factors
from the training rows, reloads the checkpoint by SHA-256 and scores every
legal prompt against the recovered answers. It masks all labels during the
forward pass and compares the winning token with every vocabulary competitor.
The same verifier accepts unknown row and column factors and an arbitrary
constant-row token. `division-mask-lean` exports the exact mask, a spanning
witness and a perfect matching. Lean checks the literals with `decide`.
[Ideal character recovery](../../src/Transformer/Grokking/FlatSpectrum/FeatureCodes/PoweredDivision.lean) · [Identifiability including zero](../../src/Transformer/Grokking/FiniteRecovery/DivisionMask.lean)

The exact minimum for identifying this class with a freely chosen mask
is 192 equations. A connected graph on 96 row vertices and 96 column
vertices requires 191 unit cells, and one further cell determines the
constant. The unit diagonal and one full unit row attain 191 and include
a perfect matching, so the same unit mask also gives unique nuclear-norm
completion of the character table. The optimizer runs use the original
4,656-observation split. [Proof of the exact minimum](../../src/Transformer/Grokking/FiniteRecovery/FactorObservations.lean)

### Eight-bit sum on the 31-word staircase

| Accepted updates | Runs fitting 31/31 | Mean hidden accuracy | Hidden correct per run (of 225) | Exact hidden recoveries |
|---:|---:|---:|---|---:|
| 2,048, original run | 5/8 | 43.61% | 100, 103, 99, 98, 94, 97, 99, 95 | 0/8 |
| 2,048, six-position PAGE correction | 7/8 | 43.22% | 92, 102, 99, 98, 94, 96, 98, 99 | 0/8 |
| 2,048, retry at the last accepted rate | 5/8 | 43.44% | 97, 103, 98, 98, 93, 98, 99, 96 | 0/8 |
| 2,048, checkpoint in the long run | 5/8 | 43.44% | 100, 102, 98, 98, 93, 97, 99, 95 | 0/8 |
| 20,480 | 8/8 | 43.28% | 97, 100, 96, 100, 94, 96, 97, 99 | 0/8 |

The 789,788-parameter decoder began with 107/225 strict hidden predictions
on all eight seeds. The long runs averaged 263 seconds each. Mean observed
cross-entropy at the endpoint was 0.00638 and mean hidden cross-entropy was
3.844. The measured nuclear norm of the learned phase table ranged from
28.78 to 35.38, compared with 16 for the rank-one target. The initial
calibration gave $\lambda$ between 0.0001243 and 0.0001303. On average,
55.37% of accepted steps required a forced full-gradient refresh. The two
horizons reuse the same seed pairs. Zero exact successes in eight
independent initializations at a fixed horizon gives a one-sided 95%
upper confidence limit of $1-0.05^{1/8}=0.3123$.

With the same eight seed pairs and initial scores, increasing the
correction sample from one to six positions reduced forced gradient
refreshes from 9,506 to 7,081 across 16,384 accepted updates. The
six-position runs produced 778 correct hidden answers across 1,800
predictions, compared with 785 in the original short runs. Their
terminal phase-table nuclear norms ranged from 23.10 to 30.41.

Retrying a rejected estimated direction once at the previous accepted
rate succeeded on 1,065 of 10,098 retries. This variant forced 9,033
gradient refreshes and solved 782/1,800 hidden sum predictions. Its
phase-table nuclear norms ranged from 22.46 to 30.21.

The two repairs also used matched initial scores and losses on three
stress tasks and modular division. At update 988 the baseline and both
repairs all scored 96, 172 and 81 out of 4,656 on `arithmetic-div` and
317, 271 and 519 out of 4,705 on `arithmetic-abs-diff-blind`. Across
these six runs the retry rule rejected all 5,922 attempts at the
previous rate and recomputed the current minibatch gradient each time.
All three rules reached 7/7 on the Boolean AND shift test for every seed.
On `grok-division-p97` at update 2,048, the original rule, square-root
sample and retry scored 88, 90 and 85 out of 4,656. The first count is
the recorded intermediate evaluation of the 8,192-update baseline.

### Modular division with AdamW and WSD

The AdamW control uses the coordinate kernel and settings of Argand at
`7399c3bb`. It clips the labeled minibatch gradient to Euclidean norm one,
uses bias-corrected moments with betas 0.9 and 0.95 and epsilon $10^{-8}$,
and applies decoupled decay 0.1 to every trainable real coordinate. The peak
rate is 0.001. WSD warms up over the first 2% of the declared budget and
cools from 80% to a final rate of $10^{-8}$. Both runs start from the same
native seed-zero weights and use the same input, order seed 10000, loss and
820,124-parameter model as the PAGE control. Each learns only from the
4,656 observed equations. Hidden accuracy uses the other 4,656 and all
239 tokens compete at readout.

| Optimizer | Updates | Training correct | Hidden correct | Hidden minimum logit gap |
| --- | ---: | ---: | ---: | ---: |
| Adaptive PAGE | 8,192 | 106 | 85 | −0.563 |
| AdamW, WSD budget 8,192 | 8,192 | 4,656 | 2,604 | −14.361 |
| AdamW, WSD budget 20,000 | 20,000 | 4,656 | 4,656 | 17.453 |

In the 8,192-step run hidden accuracy climbed from 1,388/4,656 at step
7,168 to 2,514 at 7,936 and 2,604 at the endpoint. Changing the WSD
budget also changed its warmup and cooldown. At step 8,192 the longer
run scored 422/4,656. On the stable rate it reached 4,616 at 12,288,
4,653 at 12,544, and 4,656 at 12,800. The first perfect checkpoint
replayed with minimum gap 10.273. All 30 evaluations from 12,800 through
20,000 were perfect, including the final independently reloaded weights.
The observation spacing was 256 updates, with a separate evaluation at
the declared endpoint. The discovery run took 719 seconds on the RTX 3050.

The first fresh seed, 889832341, fit every training equation by the
20,000-step endpoint and scored 4,287/4,656 on hidden prompts (92.07%).
With a new 100,000-step WSD schedule and identical initialization and
batch order, it reached its first strict 4,656/4,656 at 18,432. The
training-only certificate independently recovered all labels at that
checkpoint, with minimum hidden margin 2.079. At 18,944 the same trajectory
scored 4,595/4,656. At update 59,904 it scored only 56/4,656 on hidden
equations and 775/4,656 on training equations. Its first subsequent perfect
hidden evaluation was at 63,232. Both measurements used a learning rate
of 0.001. At update 100,000 this seed again scored 4,656/4,656 on each
split. Independent replay gave a minimum hidden gap of 19.092. The
observed-label certificate accepted all 9,312 predictions and gave a
minimum training gap of 19.225. Terminal success for the replication is determined
solely by all 4,656 strict hidden answers at update 100,000. All twelve
preregistered seeds use this horizon, including early successes. The
horizon was selected after seeing the first seed's 20,000-step result.
If any terminal replay fails, all twelve repeat from the same seeds with
a 200,000-step WSD schedule. The empirical fractions use separate
denominators of twelve for the two budgets.

The certificate also measures the learned character table on the 9,216
unit pairs. It averages the faithful character of generator 5 over the
softmax of all 239 logits, assigning code zero to nonunit tokens.
The target $Y$ is the character table recovered from the training labels
and has nuclear norm 96. The two measured norms below come from separate
FP64 singular-value decompositions of $X$ and $X-Y$ on the saved weights.

| Optimizer | Seed | Updates | Hidden correct | $\lVert X\rVert_*$ | $\lVert X-Y\rVert_*$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAGE | 0 | 8,192 | 85/4,656 | 0.4563 | 96.4550 |
| AdamW | 0 | 8,192 | 2,604/4,656 | 608.4275 | 581.8333 |
| AdamW | 0 | 20,000 | 4,656/4,656 | 95.9999995 | 0.00002668 |
| AdamW | 889832341 | 18,432 | 4,656/4,656 | 96.496897 | 0.526854 |
| AdamW | 889832341 | 100,000 | 4,656/4,656 | 95.9999972 | 0.00002388 |
| AdamW | 2849388409 | 100,000 | 4,656/4,656 | 95.9999970 | 0.00002399 |

For any $m$ by $s$ table with $n$ candidate tokens, codes in the closed
unit disk, and minimum correct-token gap $\gamma$, the nuclear distance
is at most $\sqrt{ms\min(m,s)}\,2(n-1)e^{-\gamma}$. At the first
100,000-update endpoint $\gamma=19.0915989876$ gives an upper bound of
0.00229. Lean proves both $\lVert X-Y\rVert_*<3/1000$ and
$|\lVert X\rVert_*-96|<3/1000$ whenever the unit-row gap is at least 19.
It applies to every choice of the unknown row and column factors on
these dimensions. [Checkpoint margin and nuclear norm](../../src/Transformer/Grokking/FiniteRecovery/CheckpointMarginNorm.lean)

The 20,000-update pilot had records for the first two seeds before the
100,000 cohort was registered. Both passed the strict 100,000-update
test. The second seed, 2849388409, first reached 4,656/4,656 at 23,808.
Its final minimum gaps were 20.249 on hidden and 20.262 on observed
equations. The other ten seeds had no training
results at registration. For a fixed budget with single-run success at
most $3/4$, the probability of ten independent successes is at most
$(3/4)^{10}=0.05631$. If either of two prespecified budgets can be
reported, the corresponding bound is $2(3/4)^{10}=0.11263$ even when
the same seeds are used for both. The twelve-seed fractions describe
the registered cohort. The generic confirmation command's binomial
fields assume one fixed budget and twelve fresh trials.
For the original one-half question, nine of the ten fresh draws suffice
at the 5% level. Across two candidate budgets the upper bound under the
null is $2\binom{10}{9}2^{-9}=5/128$. The verified report includes the
corresponding bounds for the number of fresh successes on uninterrupted
first attempts. Even if interruption depends on the trajectory, removing
a success can only decrease the count. The bound keeps all ten registered
draws in its sample size. A restarted seed still contributes its terminal
result to the requested twelve-seed fraction. The first attempt at seed
2434973590 reached a strict 4,656/4,656 at update 25,600 and stopped after
the recorded evaluation at 36,864. The rerun uses the same seed and schedule.
[Terminal validation theorem](../../src/Transformer/Grokking/FiniteRecovery/TerminalValidation.lean)

Two independent ordinary AdamW runs of seed 113 on the same 256-update
schedule started with identical loss and scored 57 and 58 hidden prompts
at the endpoint. Both checkpoints replayed exactly. The backward pass
uses FP32 atomic additions for the shared LeRoPE offsets and complex
score multipliers. The order of additions across GPU blocks can vary.
For this backend the cohort measures success across seed draws and GPU
executions. Its finite collection of stored checkpoints can be rescored
without retraining.

For reproducibility controls, append `--ordered-gradients` to an AdamW
or PAGE single-run command. Each shared gradient is summed by a fixed
warp tree and written once per coordinate. On the RTX 3050, two separate
256-update runs with seed 113 and a third run using restart trial zero
produced byte-identical weights, loss 4.604454520360096 and 60/4,656
correct hidden answers. All three saved checkpoints replayed these scores.
The ordered backend records `gradient_reduction: ordered-f32-v1` in its
checkpoint identity. The registered 100,000- and 200,000-update cohorts
use the original atomic backend.

```sh
case=grok-division-p97
"$bin" adamw-run "$case" 0 runs/division-adamw-short 8192
"$bin" adamw-run "$case" 0 runs/division-adamw-long 20000
bash plot_division_accuracy.sh "runs/division-page/$case/seed0" \
  "runs/division-adamw-short/$case/seed0" "runs/division-adamw-long/$case/seed0" runs/division-comparison
"$bin" adamw-confirm-plan "$case" runs/division-registered 20000
bash run_division_horizon.sh runs/division-registered runs/division-confirm-100k 100000
bash double_division_horizon.sh runs/division-confirm-100k runs/division-confirm-200k
bash plot_division_cohort.sh runs/division-confirm-100k runs/division-comparison
bash report_division_horizon.sh runs/division-confirm-100k
"$bin" division-certificate "runs/division-adamw-long/$case/seed0" first-strict
"$bin" division-mask-lean > runs/OriginalDivisionMask.lean
(cd ../.. && LAKE_ARTIFACT_CACHE=false lake env lean experiments/argand/runs/OriginalDivisionMask.lean)
```

The cohort plot shows the measured hidden accuracy of each seed and
the remaining error count throughout training. The dashed marker is
the start of the WSD cooldown. The right panel distinguishes the latest measurement of an active run from
the strict score after reloading terminal weights. An empty row marks a
seed awaiting training. Set `ARGAND_PLOT_SNAPSHOT=1` to keep a PNG and PDF
named for the latest draw, update count and number of completed replays.
After a process interruption the runner archives its recorded trajectory
and first-strict checkpoint under the draw's `interrupted/` directory.
The optimizer keeps its moments and batch order in memory, so the same
registered seed starts again at step zero. Dashed gray curves show
interrupted attempts. The right panel uses one endpoint per seed.
The adjacent `*-cohort-verified.json` counts a seed when independent
replay and the training-label certificate agree on its strict hidden
score. It gives the twelve-seed fraction once all twelve have been
verified, together with the ten-seed subset first trained after the pilot.

The Lean construction in `DivisionNative.lean` represents all 9,312
quotients with the same architecture as these runs. Two attention heads
separate the operands, and the product FFN computes the quotient in a
faithful character coordinate. All 239 tied scores compete at the output.
For a sufficiently small total learning rate, this gives a positive
Gaussian probability of strict recovery at a finite AdamW horizon. The
numerical cohorts measure the separately specified WSD peak of 0.001.

The same capacity suffices for every table in the unknown-factor class.
The row and column factors vary independently, and the constant row may
return any of the 239 tokens, including the equals sign. A single set
of attention and FFN weights suffices in the construction as the embedding
varies across targets. For any required confidence below one, a common
bound on the total learning rate and a common finite number of independent
certified trials work uniformly for the class on the optimal 192-cell
mask. [Uniform native realization and AdamW certificate](../../src/Transformer/Grokking/FiniteRecovery/FactorNative.lean)

The `division-mask-lean` export also checks the actual 4,656-cell mask.
Its observed permutation and connected unit graph imply unique
nuclear-norm completion for every unit-modulus rank-one table on the
96 by 96 block. The constant-row observation extends identification to
all 9,312 inputs. The generated `original_factor_class_stabilizes`
specializes certified AdamW restarts to this mask for every table in the
class. [Mask witnesses](../../src/Transformer/Grokking/FiniteRecovery/DivisionMask.lean)

An optional restart schedule applies to any task. For zero-indexed trial
$j$ and horizon $K_j$, the first $M$ trials use ordinary WSD. Subsequent
trials multiply every rate by
$\frac{M+1}{j+1}\min(1,\frac{K_0+1}{K_j+1})$. The total rate tends to
zero for arbitrary finite budgets. In real arithmetic, the first terminal
checkpoint accepted by the observed-label verifier is eventually retained
and is strictly correct on all inputs almost surely. The theorem assumes
independent Gaussian native starts, positive variances and native strict
representability of an identifiable finite class. It gives no numerical
single-run success rate at the stock peak. [General restart theorem](../../src/Transformer/Grokking/FiniteRecovery/AdamWRestartRates.lean)

`adamw-restart-run CASE SEED OUT CAP STOCK_RUNS TRIAL FIRST_BUDGET`
performs one fixed-horizon trial, records its rate scale, and replays the
endpoint. Each trial has its own `OUT/CASE/draw-TRIAL` directory.
`FIRST_BUDGET` is the first trial's effective update count. For example,
with 12 stock trials of 100,000 and a constant budget, trial 13 uses
13/14 of each scheduled rate. The 100,000- and conditional 200,000-step
confirmation cohorts use ordinary unscaled WSD for every seed.

The optional `adamw-tail-run CASE SEED OUT WSD_BUDGET CUTOFF` continues
after the entire WSD prefix with rates $10^{-8} 2^{-k}$. In exact real
arithmetic every weight then converges for arbitrary minibatches. On an
identifying mask, a sufficiently small geometric continuation preserves a
strict prediction in the admissible class that fits the observed labels.
For the ordinary positive WSD floor, a common clipped-gradient limit gives
an explicit weight limit. The finite experiments above use ordinary WSD.
[AdamW trajectory](../../src/Transformer/Grokking/FiniteRecovery/AdamWTrajectory.lean)
· [Identifiable checkpoints](../../src/Transformer/Grokking/FiniteRecovery/ArgandAdamW.lean)
· [Positive WSD floor](../../src/Transformer/Grokking/FiniteRecovery/AdamWFloor.lean)

## Runs

The first ten rows ran in `experiments/phasor_value_transport` at commit
7399c3bb or earlier, as described under [Earlier runs](#earlier-runs).
All ran on the RTX 3050 Laptop GPU with one pass through the layers.
"Minibatch" means the suite's minibatches in the frozen order. Times
and update counts are in the linked sections.

| Runs | Residual | $s_0$ | Optimizer and batch | Cells | Result |
|---|---|---|---|---|---|
| [GPT screens](#gpt-screens-and-delayed-generalization) | Ordinary addition | $1+i$ | AdamW, minibatch | 35 cases × 3 seeds, 1,000 updates | 19/75 short and 1/30 long runs reach 99% |
| [Score start](#score-initialization-on-absolute-difference) | Ordinary addition | $1+i$, $1$ | AdamW, minibatch | Blind absolute difference, seed 1 | 99% at updates 12,996 and 14,706 |
| [Norm phase](#one-stream-residual-with-a-norm-dependent-phase) | Ordinary, norm-dependent phase | $1$ | AdamW, minibatch | 35 cases × 3 seeds, 1,000 updates | Short 21/75 and 22/75, long 2/30 and 4/30 |
| [Residual gates](#token-dependent-residual-gates-on-arithmetic-abs-diff-blind) | Five single-stream residuals and complex mHC | $1$ | AdamW, minibatch | Blind absolute difference, 3 seeds, 988 updates | Skip and write gates 66.5%, mHC 70.2% held-out |
| [Gates with AMSGradW](#skip-and-write-gates-with-amsgradw) | Skip and write gates | $1$ | AMSGradW $\beta_2=0.98$, minibatch | Blind absolute difference, 3 seeds, 988 updates | 64.9% held-out |
| [mHC on one task](#complex-mhc-in-argand) | Real, conjugated and edge-phase mHC | $1+i$ | AdamW, minibatch | Blind absolute difference, seed 1 | 99% at updates 9,158, 4,446 and 3,990 |
| [mHC short screen](#argand-short-gpt-stress-screen) | Edge-phase mHC | $1+i$ | AdamW, minibatch | 25 short × 3 seeds, 1,000 updates | 26/75 |
| [mHC score start](#argand-score-initialization-at-1) | Edge-phase mHC | $1$ | AdamW, minibatch | 25 short × 3 seeds, 1,000 updates | 32/75 |
| [AMSGradW screens](#argand-with-amsgradw) | Edge-phase mHC | $1$ | AMSGradW: $\beta_2=0.95$; $\beta_2=0.98$; $a=0.9$, minibatch | 25 short × 3 seeds, 1,000 updates | 25/75, 26/75, 24/75 |
| [Full-batch momentum](#argand-full-batch-momentum) | Skip and write gates | $1+i$ | Clipped momentum, full batch, $\lambda=10^{-4}$ | `suite-boolean-and-shift`, seed 0, 200 updates | 3.43 s, against 0.97 s with minibatches |
| [This model](#this-model) | Skip and write gates | $1$ | Clipped momentum, full batch, $\lambda=10^{-4}$ | `suite-boolean-and-shift`, seed 0 | 99% at update 224 |

## Findings

With AdamW minibatches and 1,000 updates, the four-stream complex mHC
with independent edge phases and $s_0=1$ passed the most short runs,
32/75. Among one-stream residuals, skip and write gates came closest to
it on blind absolute difference: 66.5% against 70.2% held-out at update
988. They reached 50% in less elapsed time than mHC on all three seeds. The
three AMSGradW screens passed fewer short runs than AdamW.

### This model

On `suite-boolean-and-shift` seed 0, this model reached 7/7 on the
primary held-out split at update 224. Training and scheduled scoring
took 3.9 s. An independent replay reproduced the score and loss, and
scored 27/64 on the `test` split. The model has not yet been screened
on the tiers.

```sh
target/release/argand run suite-boolean-and-shift 0 1 runs/argand-boolean-shift
target/release/argand replay suite-boolean-and-shift 0 1 runs/argand-boolean-shift
```

With 40 updates on the same cell, this crate reached the loss
4.572885853903634 of the runner it replaces, commit e38c01a3. The final
states differed by at most `2e-8`. Repeated runs of either runner differ
by up to `9e-8`, because the LeRoPE $\alpha$ and score adjoints are
summed by atomic additions.

### Speed on this GPU

The RTX 3050 Laptop GPU reached 7.5 to 7.9 TFLOPS of FP32 multiply-adds
once its clock had risen under load, and copied 182 GB/s. The compute
bound counts only the matrix products of an update:

| Case | Width, layers | Microbatches × batch | Positions | Product GFLOP | Compute bound ms |
|---|---|---|---|---|---|
| `suite-boolean-and-shift` | 64, 2 | 4 × 32 | 2,432 | 8.37 | 1.07 |
| `arithmetic-abs-diff-blind` | 128, 1 | 19 × 256 | 9,408 | 62.3 | 8.0 |
| `recurrence-fibonacci-k1-c2-lgt` | 128, 1 | 10 × 512 | 9,408 | 66.9 | 8.6 |
| `basis-depth-hard` | 128, 6 | 1,250 × 16 | 1,217,200 | 49,001 | 6,497 |

Milliseconds per update after each change, and the compute bound as a
fraction of them, each against the peak measured in the same run:

| Change | Boolean and shift | Absolute difference | Fibonacci | Basis depth |
|---|---|---|---|---|
| Baseline | 17.3 (6.2%) | 83.3 (9.6%) | 90.0 (9.9%) | 61,840 (10.5%) |
| Register-tiled products | 16.4 (6.5%) | 59.6 (13.4%) | 57.3 (15.0%) | not measured |

The tiled product keeps a 4 × 4 block of complex outputs per thread in
registers, after the FP32 SGEMM of Kernite's `examples/matmul`. Each
output is the chain of fused multiply-adds of the 16 × 16 kernel, in
the same order, so with `k` whole the two agree bit for bit; `check`
compares them on all 128 views of transposition, conjugation and
addition. When the output has too few tiles to occupy the GPU, `k` is
split, and the partial products are added in order. With 40 updates
on `suite-boolean-and-shift`, the loss was 4.572885819843838 and the
weights differed from the 16 × 16 kernel's by at most `6e-8`; the cell
again passed at update 224, in 3.7 s.

```sh
target/release/argand peak
target/release/argand bench arithmetic-abs-diff-blind 30
```

## Earlier runs

These runs trained in `experiments/phasor_value_transport` at commit
7399c3bb or earlier, and their run directories remain in that
directory's `runs/`. Run their commands there at commit 7399c3bb. At
that commit, the `v2-*` operations and the variables `PHASOR_V2_MHC`,
`ARGAND_RESIDUAL`, `ARGAND_SCORE_INIT`, `ARGAND_OPTIMIZER`,
`ARGAND_AMSGRADW_SQRT_SCALE`, `ARGAND_AMSGRADW_BETA2`, `ARGAND_BATCH` and
`ARGAND_PENALTY` select the variants, and `run_v2_stress.sh` is the run
script. Source paths in these sections refer to that commit.

The single-stream configuration of Argand, originally named Phasor v2,
uses a complex residual stream and a tied vocabulary classifier. For
each position it computes
`(Re(x) Re(E)^T + Im(x) Im(E)^T) / 2` against all rows of the learned
embedding table. The outer cross-entropy uses the frozen GPT supervision
mask and the original scorer, including free generation and EOS.

The implementation uses FP32 real coordinates throughout the complex
forward and reverse passes. RMS divides by the Euclidean norm plus
`1e-6` and multiplies by the square root of the complex width. Each
layer uses Q/K RMS, a shared learned LeRoPE table and two causal
softmaxes for the complex attention weights. Score row `n` is multiplied
by `s log(n)/sqrt(d)`, where the complex scalar `s` is shared by the
heads of its layer. It starts at `1 + i` unless a section says
otherwise. The FFN uses a width-`4c` complex product. Both residual
branches use PeriNorm and `1/sqrt(2 * layers * repeat)`. The same layer
parameters are applied on every repeat. The experiments use the suite's
widths, head counts and depths, with `repeat=1`.

Use the pinned checkout described under
[reproduction](../phasor_value_transport/README.md#reproducing-the-pair-address-baseline).
From `experiments/phasor_value_transport`, the following commands verify
the tied classifier and directional derivatives, then screen all three
seeds of every canonical case.

```sh
cargo +stable build --locked --release
target/release/phasor-value-transport v2-check
out=runs/phasor-v2-kernite
mkdir -p "$out"
export PHASOR_EVAL_BATCH_CAP=16
./run_v2_stress.sh 1 "$out" 1000 | tee "$out/stress-1000.jsonl"
```

The optional cap stops at the last scheduled observation within the
specified number of updates. With the cap omitted, the runner uses
the complete frozen budgets. Reusing the same directory resumes the
stored weights, optimizer state and minibatch order. Passing checkpoints
are reloaded and scored on the held-out and available transfer splits.

### GPT screens and delayed generalization

On the RTX 3050 Laptop GPU, the first 1,000 updates gave 19/75 passing
short runs and 1/30 passing long runs. Each passing checkpoint reproduced
its primary score in an independent reload. The long pass was easy
recall, seed 2, with 511/512 at update 750.

On `arithmetic-abs-diff-blind`, two input tokens encode operands and the
target is `floor(abs(a-b)/2)`. Seed 1 had the highest held-out score at the
screen and was continued with the same optimizer and schedule. The gap
between train and validation narrowed substantially after the model had
fit nearly all the training pairs.

| Update | Training pairs correct /4704 | Held-out pairs correct /4705 |
|---|---:|---:|
| 988 | 4,594 | 1,432 |
| 4,997 | 4,697 | 4,015 |
| 12,996 | 4,704 | 4,658 |

```sh
bin=target/release/phasor-value-transport
"$bin" v2-run arithmetic-abs-diff-blind 1 1 "$out" 4997
"$bin" v2-run arithmetic-abs-diff-blind 1 1 "$out" 19988
"$bin" v2-replay arithmetic-abs-diff-blind 1 1 "$out"
```

The train column uses the same replay with `train` included in the split
predicate in `src/stress/argand.rs`. The checkpoint digest, weights and scorer
are identical for both columns.

The matched control is the original fixed-mean Phasor. It uses the
learned complex backbone, four residual streams with a fixed mean
and the tied classifier. Its held-out score on the same seed was
4,662/4,705 at update 4,978. The single-stream model needed 2.61 times
as many updates to reach 99%. Both runs used the case's configured
width of 128, which counts real coordinates in the original model and
complex numbers in Argand.

```sh
CARGO_TARGET_DIR=target/plain-phasor cargo +stable build --locked --release \
    --features fp32-affine-reference,original-stream-mean
plain=target/plain-phasor/release/phasor-value-transport
"$plain" screen arithmetic-abs-diff-blind 1 "$out/plain-long" 12996
"$plain" replay arithmetic-abs-diff-blind 1 "$out/plain-long"
```

### Score initialization on absolute difference

Both coordinates of the score multiplier train from either starting
value, `1 + i` or `1 + 0i`. The two values differ in scale and in
how they mix the real and imaginary logits before their respective
softmaxes. With two input tokens, every supervised row has the same
prefix length. The comparison uses seed 1 and the same schedule and
stopping threshold as the earlier single-stream Argand run.

| Initial multiplier | Held-out correct at 988 /4705 | Held-out correct at 4,997 /4705 | First 99% update |
|---|---:|---:|---:|
| `1 + i` | 1,432 | 4,015 | 12,996 |
| `1 + 0i` | 1,520 | 4,261 | 14,706 |

Both passing checkpoints replayed at 4,658/4,705. Real initialization
improved early held-out accuracy and took 13.2% more updates to reach
99%. The delayed improvement on novel pairs persists under both
initializations.

To reproduce the real initialization in a separate experiment worktree,
set the score entry in `Parameters::new` in `src/argand/weights.rs`
to `values.extend([1.0, 0.0]);` and append `-s-init-real` to the model
identifier for `Routing::Single` in `src/argand/complex_mhc.rs`.
Build the same release binary and use a fresh run directory with the
following commands.

```sh
cargo +stable build --locked --release
bin=target/release/phasor-value-transport
real=runs/phasor-v2-s-init-real
for stop in 988 4997 19988; do
    "$bin" v2-run arithmetic-abs-diff-blind 1 1 "$real" "$stop"
done
"$bin" v2-replay arithmetic-abs-diff-blind 1 1 "$real"
```

### One-stream residual with a norm-dependent phase

Does rotating the skip help a single-stream Argand reach 99% held-out exact sequences
within 1,000 updates? Each attention and FFN sublayer gets one real coefficient
$\theta$, initialized to zero and shared across tokens. It updates the incoming
complex vector by $x' = e^{i\theta\lVert x\rVert_2} x + u(x)$, where $u$ is the
usual scaled, PeriNorm-normalized write. The rotation uses the incoming norm before
RMS. A repeated pass reuses the same coefficients. [Lean definition and proof](../../src/Transformer/PhasorV2/PhaseResidual.lean)

| `ARGAND_RESIDUAL` | Skip | Additional real parameters per layer |
|---|---|---:|
| `single` | $x$ | 0 |
| `norm-phase` | $e^{i\theta\lVert x\rVert_2} x$ | 2 |

Both arms use the tied vocabulary, the score multiplier initialized to `1 + 0i`,
AdamW, one pass, and the same frozen schedule and minibatch order for each seed.
Both coordinates of the multiplier train. The shared weights start at identical
values. The runner checks the threshold at the scheduled observations through
update 1,000 and reloads every passing checkpoint.

```sh
cargo +stable build --locked --release --features fp32-affine-reference
export ARGAND_SCORE_INIT=1 ARGAND_OPTIMIZER=adamw PHASOR_EVAL_BATCH_CAP=16
for residual in norm-phase single; do
    export ARGAND_RESIDUAL=$residual
    out="runs/argand-norm-phase-s1/$residual"
    mkdir -p "$out"
    target/release/phasor-value-transport argand-check
    ./run_v2_stress.sh --argand all 1 "$out" 1000 > "$out/stress-1000.jsonl"
done
unset ARGAND_RESIDUAL ARGAND_SCORE_INIT ARGAND_OPTIMIZER PHASOR_EVAL_BATCH_CAP
jq -n --argjson cap 1000 --slurpfile suite ../convex_stress/suite.json \
    --slurpfile control runs/argand-norm-phase-s1/single/stress-1000.jsonl \
    --slurpfile candidate runs/argand-norm-phase-s1/norm-phase/stress-1000.jsonl \
    -f compare_v2_screens.jq
```

| Tier | Tasks × seeds | Ordinary addition reaching 99% | Norm-dependent phase reaching 99% |
|---|---:|---:|---:|
| Short | 25 × 3 | 21/75 | 22/75 |
| Long | 10 × 3 | 2/30 | 4/30 |

On the short tier, the pass status changed only on `basis-depth-hard`.
The phase residual passed seeds 0 and 2. Ordinary addition passed seed 1.
For seed 0 at update 1,000, the phase residual scored 512/512 and ordinary
addition scored 315/512. Of the 20 short runs where both crossed 99%, the
phase residual crossed earlier in three, at the same observation in 15, and
later in two. The earlier four-stream complex mHC screen with the same
`s=1`, AdamW, cases, seeds and cutoff passed 32/75.

The phase residual added two long passes. On `basis-recall-easy` seed 0
it scored 509/512 at update 1,000, where ordinary addition scored 0/512.
On `associative-divisor-n2-l1-w0` seed 2 it crossed at update 912, while
ordinary addition finished at 4,597/4,656 at update 988. Both models passed
`basis-recall-easy` seed 2 at update 450. On the other shared long pass,
`associative-divisor-n2-l1-w0` seed 1, ordinary addition crossed at 817
and the phase residual at 912.

On `arithmetic-abs-diff-blind`, the task from the original mHC comparison,
seed 1 scored 1,524/4,705 with the phase residual and 1,520/4,705 with
ordinary addition at update 988, the last observation within the cap.

The step-zero held-out scores and losses matched in all 105 pairs. All 210
terminal state digests matched their records, and independent checkpoint
reloads reproduced the held-out scores and losses of all 49 passes. All
105 phase runs learned at least one nonzero rotation coefficient.

#### Token-dependent residual gates on `arithmetic-abs-diff-blind`

The embedding RMS gives the first attention sublayer inputs of almost
constant norm. A gate that reads the token coordinates can distinguish
directions of equal norm. Let $n=\mathrm{RMS}(x)$ and let $u$ be the usual
scaled, normalized branch write. The single-stream `write-gate` updates
$x'=x+(1+n^\mathsf{T}w)u$. Each sublayer learns one zero-initialized
complex vector $w$. A single complex scalar controls the magnitude and
phase of the write at each token. Zero initialization reproduces ordinary
addition, including all initial logits.

Two variants add one independent complex vector per sublayer. Write
$g_r=1+n^\mathsf{T}w_r$, $g_s=1+n^\mathsf{T}w_s$, and $g_w=1+n^\mathsf{T}w_w$.
The read-and-write variant passes $g_r n$ into attention or the FFN and
multiplies the normalized write by $g_w$. The skip-and-write variant keeps
the usual block input $n$ and computes $x'=g_s x+g_w u$. Both gates see
the same incoming token. Each variant uses one stream, four width-128
complex vectors per layer, and ordinary addition at initialization.

The blind absolute-difference task has 4,704 training pairs and 4,705
held-out pairs. All runs use width 128, one layer, one pass, the tied
vocabulary, AdamW, and a score multiplier initialized to `1 + 0i`.
The schedule checks every 19 updates. Update 988 is the last observation
within the 1,000-update cap. The timing run used one executable for all
six models, with the order varied by seed. Times include training and
scheduled scoring on the RTX 3050 Laptop GPU and exclude setup, checkpoint
writes and replay.

| Residual | Extra real parameters per layer | Held-out correct, seeds 0 / 1 / 2 | Mean accuracy | Mean seconds to update 988 |
|---|---:|---|---:|---:|
| Ordinary addition | 0 | 1,430 / 1,520 / 1,357 | 30.5% | 7.64 |
| Norm-dependent phase | 2 | 1,437 / 1,524 / 1,346 | 30.5% | 7.67 |
| Write gate | 512 | 2,394 / 2,347 / 2,359 | 50.3% | 8.12 |
| Read and write gates | 1,024 | 2,336 / 2,346 / 2,387 | 50.1% | 9.60 |
| Skip and write gates | 1,024 | 3,115 / 3,005 / 3,270 | 66.5% | 9.39 |
| Four-stream complex mHC, independent edge phases | 49,152 | 3,285 / 3,220 / 3,399 | 70.2% | 11.80 |

The write gate gained 19.8 percentage points over the norm-dependent phase.
Adding a skip gate gained a further 16.2 points. Adding a read gate changed
the mean by −0.2 points. Complex mHC led the skip-and-write model on every
seed at update 988, by 3.6 points on average.

At 50% accuracy, the skip-and-write model took 646 / 627 / 646 updates,
exactly 38 more than mHC on each seed. In elapsed time it reached that
threshold sooner on all three, averaging 6.14 seconds versus 7.24 seconds
for mHC. All runs stayed below 99% through the cap. Independent checkpoint
reloads reproduced the held-out scores and losses of all 18 runs.

```sh
cargo +stable build --locked --release --features fp32-affine-reference
bin=target/release/phasor-value-transport
task=arithmetic-abs-diff-blind
out=runs/argand-abs-diff-s1
export ARGAND_SCORE_INIT=1 ARGAND_OPTIMIZER=adamw PHASOR_EVAL_BATCH_CAP=16
for variant in single norm-phase write-gate read-write-gate skip-write-gate entry-phase; do
    ARGAND_RESIDUAL=$variant "$bin" argand-check
done
for seed in 0 1 2; do
    case "$seed" in
        0) variants='single write-gate skip-write-gate entry-phase read-write-gate norm-phase' ;;
        1) variants='norm-phase read-write-gate entry-phase skip-write-gate write-gate single' ;;
        2) variants='entry-phase skip-write-gate single norm-phase write-gate read-write-gate' ;;
    esac
    for variant in $variants; do
        export ARGAND_RESIDUAL=$variant
        cell="$out/$variant/$task/seed$seed"
        mkdir -p "$cell"
        if [ ! -s "$cell/result.json" ]; then
            "$bin" argand-run "$task" "$seed" 1 "$out/$variant" 988 2> "$cell/observations.jsonl"
        fi
        "$bin" argand-replay "$task" "$seed" 1 "$out/$variant" > "$cell/replay.json"
    done
done
unset ARGAND_RESIDUAL ARGAND_SCORE_INIT ARGAND_OPTIMIZER PHASOR_EVAL_BATCH_CAP
jq -s 'sort_by(.identity.model,.identity.seed)' "$out"/*/"$task"/seed*/result.json > "$out/comparison.json"
./plot_argand_abs_diff.sh "$out"
```

#### Skip and write gates with AMSGradW

At commit 7399c3bb the Argand commands selected `skip-write-gate` by
default. These gates are the residual of this crate. Each attention
and FFN sublayer computes both complex gates from the normalized incoming
token vector. The skip gate multiplies the incoming residual and the write
gate multiplies the normalized block output before scaling the write.
The two vectors in each sublayer start at zero, giving both gates an
initial value of `1 + 0i`.
[Architecture](ARGAND.md)

At that commit Argand defaulted to minibatch SGD with momentum `0.9`, zero
weight decay and the WSD schedule on Kernite 0.8.0.
`ARGAND_OPTIMIZER=adamw` or `ARGAND_OPTIMIZER=amsgradw` selected the
earlier optimizer controls, and `ARGAND_RESIDUAL=entry-phase` the
four-stream mHC model.
The historical checkpoints record their optimizer and Kernite version.
The command below checks the skip-and-write residual on
`arithmetic-abs-diff-blind` with AMSGradW at `beta2=0.98`, the original
WSD horizon, and a cap of 1,000 updates. Update 988 is the final scheduled
observation within that cap.

```sh
cargo +stable build --locked --release --features fp32-affine-reference
bin=target/release/phasor-value-transport
task=arithmetic-abs-diff-blind
out=runs/argand-skip-write-amsgradw-abs-diff
unset ARGAND_RESIDUAL ARGAND_AMSGRADW_SQRT_SCALE
export ARGAND_SCORE_INIT=1 ARGAND_OPTIMIZER=amsgradw ARGAND_AMSGRADW_BETA2=0.98
export PHASOR_EVAL_BATCH_CAP=16
"$bin" argand-check
for seed in 0 1 2; do
    cell="$out/$task/seed$seed"
    mkdir -p "$cell"
    "$bin" argand-run "$task" "$seed" 1 "$out" 988 2> "$cell/observations.jsonl"
    "$bin" argand-replay "$task" "$seed" 1 "$out" > "$cell/replay.json"
done
unset ARGAND_SCORE_INIT ARGAND_OPTIMIZER ARGAND_AMSGRADW_BETA2 PHASOR_EVAL_BATCH_CAP
```

The default gate run used width 128, four heads, one layer and one pass.
All runs reached update 988. The two AdamW rows use the recorded results
from the gate comparison above. The skip-and-write screens matched the
case, seeds, splits and step-zero metrics. Times include training and
scheduled scoring on the RTX 3050 Laptop GPU.

| Residual | Optimizer | Held-out correct /4705, seeds 0 / 1 / 2 | Mean accuracy | Mean time (s) |
|---|---|---|---:|---:|
| Skip and write gates | AMSGradW, `beta2=0.98` | 2,987 / 3,029 / 3,144 | 64.9% | 8.28 |
| Skip and write gates | AdamW | 3,115 / 3,005 / 3,270 | 66.5% | 9.39 |
| Four-stream complex mHC | AdamW | 3,285 / 3,220 / 3,399 | 70.2% | 11.80 |

The GPU check verified identity at initialization, the gate adjoints and
AMSGradW updates. Independent reloads reproduced the held-out scores
and losses of all three new checkpoints. Selecting `entry-phase` also
reproduced the saved mHC checkpoint for seed 1.

### Complex mHC in Argand

Each sublayer has a separate complex router projection from the
RMS-normalized stack of four streams. The pinned mHC kernels apply
sigmoid to read amplitudes and twice sigmoid to write amplitudes.
Twenty column/row Sinkhorn cycles produce the residual amplitudes $P$
from logits with a diagonal offset of 2. The three routers start
with identical amplitude projections and zero phase projections.

The real arm uses these amplitudes directly. The conjugated and
entry-phase arms learn a phase for each read and write gate. The
conjugated residual uses $DPD^*$ with one angle per stream at each
position. The entry-phase residual uses $P_{ij}e^{i\theta_{ij}}$
with one angle per edge. All three pass the complex read through the
same Argand PeriNorm attention and FFN. The vocabulary projection reads
the RMS-normalized mean of the four final streams. Until commit
3ead6c25, [ARGAND.md](ARGAND.md) also described the independent-phase
variant.

The comparison uses seed 1, 128 complex numbers, four heads, one
layer and one pass on `arithmetic-abs-diff-blind`. Each run uses the
frozen WSD schedule and stops at the first 99% observation or update
19,988. The single-stream row uses the saved checkpoint from the
same experimental setup. All runs use Kernite 0.7.0 on the RTX 3050.

| Residual | Held-out correct at 988 /4705 | First 99% update | Held-out correct at first pass /4705 |
|---|---:|---:|---:|
| Ordinary addition | 1,432 | 12,996 | 4,658 |
| Real mHC | 2,047 | 9,158 | 4,660 |
| Complex gates and conjugated residual | 2,995 | 4,446 | 4,659 |
| Complex gates and independent edge phases | 2,879 | 3,990 | 4,661 |

Independent checkpoint reloads reproduced the scores and losses.
Among the tested configurations, independent edge phases reached 99%
in the fewest updates, 10.3% fewer than conjugated mixing. Both complex
arms change read and write phases as well as residual mixing relative
to real mHC. These results cover one case and one seed.

```sh
cargo +stable build --locked --release --features fp32-affine-reference
bin=target/release/phasor-value-transport
export PHASOR_EVAL_BATCH_CAP=16
for mode in real conjugated entry-phase; do
    export PHASOR_V2_MHC=$mode
    trial="runs/v2-mhc-comparison/$mode"
    "$bin" v2-check
    for stop in 988 4997 19988; do
        "$bin" v2-run arithmetic-abs-diff-blind 1 1 "$trial" "$stop"
    done
    "$bin" v2-replay arithmetic-abs-diff-blind 1 1 "$trial"
done
unset PHASOR_V2_MHC
```

#### Argand short GPT stress screen

Set `ARGAND_RESIDUAL=entry-phase` to select the independent-phase model
for this screen. The screen uses the 25 short cases with three
seeds each and one pass through the layers. It stops each run
at its first held-out observation at or above 99% or at the last scheduled
observation at or below 1,000 updates. The learning-rate schedule uses the
original case budget. The runner replays each passing checkpoint.

```sh
cargo +stable build --locked --release --features fp32-affine-reference
export ARGAND_RESIDUAL=entry-phase
target/release/phasor-value-transport argand-check
out=runs/argand-short-1000
mkdir -p "$out"
PHASOR_EVAL_BATCH_CAP=16 ./run_v2_stress.sh --argand short 1 "$out" 1000 | tee "$out/short-1000.jsonl"
unset ARGAND_RESIDUAL
```

The runs used Kernite 0.7.0 on an RTX 3050 Laptop GPU. The single-stream
screen above used the same cases, seeds and observation limit.

| Model | Runs reaching 99% /75 | Cases with a pass /25 | Cases passing all three seeds /25 |
|---|---:|---:|---:|
| Single-stream control | 19 | 7 | 6 |
| Argand with independent phases | 26 | 10 | 8 |

Argand passed all 19 cells that passed in the single-stream screen.
It also passed all three seeds of `arithmetic-square-sum` at updates
893–950, all three seeds of `power-division-p97-k3` at updates 741–798,
and seed 1 of `power-division-p193-k3` at update 949. All 75 terminal
state digests matched the saved records. Independent reloads reproduced
the held-out scores and losses of the 26 passing checkpoints.

#### Argand score initialization at 1

`ARGAND_SCORE_INIT=1` initializes each layer's score multiplier to
`1 + 0i`. Both coordinates of the multiplier train. The checkpoint
identifier records this initialization, so use a separate run directory.

```sh
cargo +stable build --locked --release --features fp32-affine-reference
export ARGAND_RESIDUAL=entry-phase ARGAND_SCORE_INIT=1 PHASOR_EVAL_BATCH_CAP=16
target/release/phasor-value-transport argand-check
out=runs/argand-short-score-one
mkdir -p "$out"
./run_v2_stress.sh --argand short 1 "$out" 1000 | tee "$out/short-1000.jsonl"
unset ARGAND_RESIDUAL ARGAND_SCORE_INIT
```

The paired screens used Kernite 0.7.0 on an RTX 3050 Laptop GPU with the
same cases, seeds, splits, learning-rate schedule and observation limit.

| Initial score multiplier | Runs reaching 99% /75 | Cases with a pass /25 | Cases passing all three seeds /25 |
|---|---:|---:|---:|
| `1 + i` | 26 | 10 | 8 |
| `1` | 32 | 13 | 8 |

All 26 cells that passed with `1 + i` also passed with `1`. The six
additional passes came from `arithmetic-cube-plus` (seeds 0 and 2),
`arithmetic-mul` (seeds 0 and 1), `arithmetic-add` (seed 0) and
`basis-depth-hard` (seed 1). Among the 26 shared passes, `1` reached
99% earlier in 10, `1 + i` reached it earlier in 7, and 9 tied.
All 75 terminal state digests matched the saved records. Independent
reloads reproduced the held-out scores and losses of the 32 passing
checkpoints.

#### Argand with AMSGradW

`ARGAND_OPTIMIZER=amsgradw` tracks the running maximum of each coordinate's
second moment before bias correction, following AMSGrad in Reddi et al.,
Algorithm 2. The parameter update uses that maximum in the denominator
and applies decoupled weight decay. Both optimizers use the case's WSD
schedule, `beta1=0.9`, `beta2=0.95`, `epsilon=1e-8`, weight decay `0.1`,
bias correction and gradient norm clipping at `1`. The AMSGradW
checkpoint includes the extra moment buffer and its own model identifier.

```sh
cargo +stable build --locked --release --features fp32-affine-reference
export ARGAND_RESIDUAL=entry-phase ARGAND_SCORE_INIT=1 ARGAND_OPTIMIZER=amsgradw
export PHASOR_EVAL_BATCH_CAP=16
export ARGAND_AMSGRADW_SQRT_SCALE=1 ARGAND_AMSGRADW_BETA2=0.95
target/release/phasor-value-transport argand-check
out=runs/argand-short-score-one-amsgradw
mkdir -p "$out"
./run_v2_stress.sh --argand short 1 "$out" 1000 | tee "$out/short-1000.jsonl"
unset ARGAND_AMSGRADW_SQRT_SCALE ARGAND_AMSGRADW_BETA2 ARGAND_OPTIMIZER ARGAND_RESIDUAL
```

The paired screen used `s=1`, Kernite 0.7.0 and an RTX 3050 Laptop GPU.
The two runs matched the case, seed, split and step-zero score and loss
for each of the 75 pairs. Time sums the training and evaluation segments.

| Optimizer | Runs reaching 99% /75 | Cases with a pass /25 | Cases passing all three seeds /25 | Updates | Time (s) |
|---|---:|---:|---:|---:|---:|
| AdamW | 32 | 13 | 8 | 57,762 | 870 |
| AMSGradW | 25 | 9 | 8 | 59,341 | 902 |

AdamW passed each of the 25 cells that passed with AMSGradW. Among these
shared passes, AdamW crossed earlier in 11, AMSGradW crossed earlier in
one, and 13 tied. The seven additional AdamW passes were two seeds each
of `arithmetic-cube-plus` and `arithmetic-mul`, and one seed each of
`arithmetic-add`, `power-division-p193-k3` and `basis-depth-hard`.
The five modulo-97 arithmetic misses with AMSGradW reached 86.8–91.8%
at update 988 and improved over their last four observations.
`power-division-p193-k3` seed 1 reached 25.0% at update 949.

A GPU update check verified clipping, both moments, the running maximum
and decoupled decay. At update 8, the resumed and uninterrupted probe
had the same exact score and a maximum absolute state difference of
`2.8e-8`. All 75 terminal state digests matched the saved records, and
all 25 passing checkpoints reproduced their held-out scores and losses
on independent reload.

#### Argand AMSGradW with a larger second-moment coefficient

`ARGAND_AMSGRADW_BETA2` sets the second-moment coefficient in $[0,1)$
and defaults to `0.95`. AMSGradW uses it in both the moment recurrence
and bias correction. A non-default coefficient is recorded in the
checkpoint identity. The `0.98` screen tests stronger smoothing before
the running maximum, with a unit denominator scale and the full
second-moment history.

```sh
cargo +stable build --locked --release --features fp32-affine-reference
export ARGAND_RESIDUAL=entry-phase ARGAND_SCORE_INIT=1 ARGAND_OPTIMIZER=amsgradw
export ARGAND_AMSGRADW_SQRT_SCALE=1 ARGAND_AMSGRADW_BETA2=0.98 PHASOR_EVAL_BATCH_CAP=16
target/release/phasor-value-transport argand-check
out=runs/argand-short-score-one-amsgradw-beta2-0.98
mkdir -p "$out"
./run_v2_stress.sh --argand short 1 "$out" 1000 | tee "$out/short-1000.jsonl"
unset ARGAND_AMSGRADW_SQRT_SCALE ARGAND_AMSGRADW_BETA2 ARGAND_OPTIMIZER ARGAND_RESIDUAL
```

The 75 runs used the same cases, seeds, held-out splits and WSD schedules
as the unit-scale AMSGradW baseline. The scores and losses at update
zero matched exactly. Time sums the training and evaluation segments
recorded for each screen on the same RTX 3050 Laptop GPU.

| Optimizer | Runs reaching 99% /75 | Cases with a pass /25 | Cases passing all three seeds /25 | Updates | Time (s) |
|---|---:|---:|---:|---:|---:|
| AMSGradW, $\beta_2=0.98$ | 26 | 9 | 8 | 58,815 | 1,721 |
| AMSGradW, $\beta_2=0.95$ | 25 | 9 | 8 | 59,341 | 902 |
| AdamW, $\beta_2=0.95$ | 32 | 13 | 8 | 57,762 | 870 |

All 25 baseline passes also passed at $\beta_2=0.98$. The additional pass
was `basis-depth-hard` seed 0, which rose from 310/512 to 510/512 at
update 1,000. Among the 25 shared passes, `0.98` reached 99% earlier
in nine, `0.95` reached it earlier in one, and 15 tied.

Among the 49 pairs finishing below 99% at the same cutoff, `0.98`
scored higher in 47, tied in one, and scored lower in one. The median
change was +2.2 percentage points. On `power-division-p193-k3` seed 1,
accuracy rose from 25.0% to 50.2% at update 949. On `basis-depth-hard`
seed 2, it fell from 92.2% to 63.3% at update 1,000.

The GPU update matched the scalar reference at both coefficients. At
`0.95` the eight-update probe had a maximum absolute state difference
of `1.6e-8` from the prior build, and the saved baseline checkpoint
reproduced its full replay. All 75 terminal state digests matched their
records. Independent reloads reproduced the scores and losses of all
26 passing checkpoints.

#### Argand AMSGradW with a scaled denominator

The adaptive denominator is $a\sqrt{v_{\max}/(1-\beta_2^t)}+\epsilon$.
`ARGAND_AMSGRADW_SQRT_SCALE` sets $a$ in $(0,1]$. The checkpoint
identity records each non-unit scale. The $a=0.9$ screen uses the
same Argand initialization, WSD schedule and decoupled decay as the
unit-scale AMSGradW screen.

In the constant-step full-gradient update, dividing by
$a\sqrt{v_{\max}}+\epsilon$ is equivalent to using standard AMSGradW
with $\eta/a$, $\epsilon/a$ and $a\lambda$. The products
$\eta\lambda$ and $\lambda\epsilon$ in the convergence conditions
have the same values under this substitution.
[AMSGradW convergence theorem](../../src/Transformer/AMSGradW/Convergence.lean)

```sh
cargo +stable build --locked --release --features fp32-affine-reference
export ARGAND_RESIDUAL=entry-phase ARGAND_SCORE_INIT=1 ARGAND_OPTIMIZER=amsgradw
export PHASOR_EVAL_BATCH_CAP=16
export ARGAND_AMSGRADW_SQRT_SCALE=0.9 ARGAND_AMSGRADW_BETA2=0.95
target/release/phasor-value-transport argand-check
out=runs/argand-short-score-one-amsgradw-scale-0.9
mkdir -p "$out"
./run_v2_stress.sh --argand short 1 "$out" 1000 | tee "$out/short-1000.jsonl"
unset ARGAND_AMSGRADW_SQRT_SCALE ARGAND_AMSGRADW_BETA2 ARGAND_OPTIMIZER ARGAND_RESIDUAL
```

The 75 short runs matched the cases, seeds, splits, and step-zero scores
and losses of the unit-scale AMSGradW and AdamW screens. AMSGradW kept
its running maximum throughout every run. Time sums the training and
evaluation segments.

| Optimizer | Runs reaching 99% /75 | Cases with a pass /25 | Cases passing all three seeds /25 | Updates | Time (s) |
|---|---:|---:|---:|---:|---:|
| AMSGradW, $a=0.9$ | 24 | 8 | 8 | 59,364 | 961 |
| AMSGradW, $a=1$ | 25 | 9 | 8 | 59,341 | 902 |
| AdamW | 32 | 13 | 8 | 57,762 | 870 |

Among the 24 shared passes, $a=0.9$ reached 99% earlier in 10 runs,
$a=1$ reached it earlier in one, and 13 tied. For `basis-depth-hard`
seed 1, $a=1$ reached 511/512 at update 750 and $a=0.9$ finished at
308/512 at update 1,000. The five modulo-97 arithmetic misses improved
from 86.8–91.8% with $a=1$ to 92.5–95.6% with $a=0.9$ at update 988.
`power-division-p193-k3` seed 1 improved from 25.0% to 42.1% at update
949.

All 75 terminal state digests matched the saved records. Independent
checkpoint reloads reproduced the held-out scores and losses of all
24 passes.

#### Argand full-batch momentum

`ARGAND_BATCH=full` trains the Lean `fullGradientMomentumRun`. Each
update takes the gradient of the mean loss over every labeled training
position, adds $\lambda\theta$ and then takes the clipped momentum step.

$$
g_k = \nabla L(\theta_k) + \lambda\theta_k, \qquad
v_{k+1} = 0.9 v_k + \frac{g_k}{\max(1,\lVert g_k\rVert_2)}, \qquad
\theta_{k+1} = \theta_k - \eta_k v_{k+1}.
$$

Microbatches of the suite's batch size visit the training split in
order. Each divides its loss gradient by the split's label count, so the
microbatch gradients sum to $\nabla L$, as in
`accumulated_batchGradient`. The penalty enters once, before clipping,
as in `fullGradientObjective_gradient`. `ARGAND_PENALTY` sets
$\lambda$. The convergence theorem needs $\lambda>0$, and the runner
refuses other values. The mode also refuses residuals other than
`skip-write-gate` and optimizers other than `sgd-momentum`. The model
name gains `-full-batch` and the checkpoint identity records the
penalty.

The rate keeps the case's WSD schedule, so a run of $B$ updates is the
first $B$ states of the Lean run with WSD as its rate prefix.
`fullGradientMomentumRun_limits` concerns a constant tail rate below a
threshold that it only shows to exist.
[Full-gradient run](../../src/Transformer/Grokking/FiniteRecovery/ArgandFullGradient.lean) · [Accumulation](../../src/Transformer/Grokking/FiniteRecovery/ArgandAccumulation.lean)

```sh
cargo +stable build --locked --release --features fp32-affine-reference
bin=target/release/phasor-value-transport
export ARGAND_BATCH=full ARGAND_PENALTY=1e-4
"$bin" argand-check
out=runs/argand-full-batch-smoke
"$bin" argand-run suite-boolean-and-shift 0 1 "$out" 200
unset ARGAND_BATCH ARGAND_PENALTY
```

The check splits a batch of two examples, with three and two labels, into
two microbatches. Under every residual and optimizer, the accumulated
gradient matched the batch gradient within `3.0e-8`, with a largest entry
of `0.287`. A zero-rate step with $\lambda=0.5$ matched the clipped
penalized gradient, whose norm was `2.4533`.

On `suite-boolean-and-shift` seed 0, 200 full-batch updates and their
scheduled scoring took 3.43 s, against 0.97 s for 200 minibatch updates.
A run stopped at update 20 and resumed to 40 reached the same loss as an
uninterrupted run. Two repeated runs of either mode differed in 33 of the
551,976 state coordinates, by at most `5e-6` relative. Most were LeRoPE
$\alpha$ coordinates and their velocities, whose gradients are summed by
atomic floating-point additions.
