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
bash plot_sum8_pair.sh "$sum" runs/sum8-adamw-20480 runs/sum8-optimizer-comparison
"$bin" page-run grok-division-p97 0 runs/division-page 8192
bash run_page_screen.sh all runs/gpt-page
bash run_page_screen.sh all runs/gpt-adamw adamw
bash run_page_screen.sh all runs/gpt-equal-decay equal-moment
bash run_page_screen.sh all runs/gpt-uncorrected-adam uncorrected-adam
bash run_page_screen.sh all runs/gpt-normalized-adamw normalized-adamw
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
| 20,480, PAGE | 8/8 | 43.28% | 97, 100, 96, 100, 94, 96, 97, 99 | 0/8 |
| 20,480, AdamW + WSD | 8/8 | 43.89% | 95, 102, 96, 99, 98, 101, 99, 100 | 0/8 |

The 789,788-parameter decoder began with 107/225 strict hidden predictions
on all eight seeds. The long PAGE runs averaged 263 seconds each. Mean observed
cross-entropy at the endpoint was 0.00638 and mean hidden cross-entropy was
3.844. The measured nuclear norm of the learned phase table ranged from
28.78 to 35.38, compared with 16 for the rank-one target. The initial
calibration gave $\lambda$ between 0.0001243 and 0.0001303. On average,
55.37% of accepted steps required a forced full-gradient refresh. The two
horizons reuse the same seed pairs. Zero exact successes in eight
independent initializations at a fixed horizon gives a one-sided 95%
upper confidence limit of $1-0.05^{1/8}=0.3123$.

The 20,480-step AdamW control used decoupled weight decay 0.1, a WSD
peak of 0.001, the same eight initial states and all 31 observed words
per update. It scored 790/1,800 hidden predictions across the seeds,
compared with 779/1,800 for PAGE. Both fitted all observed words in
all eight runs. Mean hidden cross-entropy was 11.277 for AdamW and
3.844 for PAGE. The AdamW phase-table nuclear norms ranged from 29.76
to 40.66. The rank-one target has nuclear norm 16. All checkpoints
were independently replayed.

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

#### Optimizer comparison on the same division inputs

Seed 0, order seed 10000, a WSD budget of 20,000 and decay 0.1 were held
fixed. Multiplying the entire AdamW rate schedule by two, including its floor,
moved the first independently verified strict hit from 12,800 to 4,096 updates.
The minimum hidden margins at these first hits were 10.273 and 8.347. Timing
across the two runs includes different amounts of GPU contention.

The one-moment variant of Lion uses the globally clipped gradient $g$ and
the previous coordinate momentum $m$. Its direction is $c/(|c|+10^{-6})$
for $c=0.9m+0.1g$, and the new buffer is $0.99m+0.01g$. The update applies
decoupled decay to the entire weight vector. At a peak of 0.0003 the first
4,096 updates produced 111 correct hidden answers. The checkpoint-proposal
command tests the ordinary weights first. On division it derives the expected
tokens only from the observed equations. From the first eligible update it
also tests a Gaussian-perturbed copy after ordinary misses. The optional
WSD budget can be fixed when extending the observation cutoff. A further
argument sets decoupled weight decay. At a peak of 0.0003, a decay of
$2/3$ matches the per-update shrink factor of the AdamW control at peak
0.002 and decay 0.1 throughout WSD.
The native sampler is seeded and its weights are FP32. [Lion, Algorithm 2](https://arxiv.org/abs/2302.06675)

On seed 0 the 0.0003 peak reached all 4,656 strict hidden answers at
update 17,920, with minimum gap 7.094 on replay. This took 4.375 times
as many updates as the 0.002 AdamW run on the same seed. Across the
4,992 complex matrix output rows the median Euclidean norms at first
success were 1.934 for softsign Lion and 0.870 for the faster AdamW.
The observed-label certificate records these row statistics. A projected
pilot caps each hidden projection's output-row norm at 1.75 after every
update, with the same unrestricted stock initialization, rate 0.0003
and decay 0.1. Both components of a complex coefficient count in the
same row. The tied embedding, scores, learned position frequencies and
residual gates receive the ordinary update. [Per-row clipping experiments in a different architecture](https://github.com/NiftyliuS/cliptogrok)

At peak 0.001, decay 0.1, the final checkpoint tied the correct target with
a competitor on every one of the 9,312 legal equations. At peak 0.0003,
raising decay from 0.1 to 2/3 reduced the first strict hit from 17,920 to
7,168 updates. The reloaded minimum hidden gap was 4.411. That hit preceded
the first eligible noise proposal at 8,192. Increasing decay to 2 left
4,645 strict hidden answers at the 4,096-update cutoff. Decay 4 reached
4,656 at step 3,072 with minimum gap 6.272 on the same seed and peak.

Projecting the three previously saved solutions once gives the following
independent check on feasibility. Scores use all 239 candidate tokens and
labels reconstructed from the 4,656 observed equations.

| Source checkpoint | Hidden strict before | Hidden strict after radius 1.75 | Minimum hidden gap after |
|---|---:|---:|---:|
| AdamW, peak 0.001, step 12,800 | 4,656 | 4,656 | 10.273 |
| AdamW, peak 0.002, step 4,096 | 4,656 | 4,656 | 8.276 |
| Softsign Lion, peak 0.0003, step 17,920 | 4,656 | 4,474 | −13.270 |

Projection throughout training is a separate experiment. For any feasible
representative, Lean proves that radial row projection can only decrease
squared distance to it relative to the unprojected update. The FP32
post-projection row maxima in this check were at most 1.7500003.
[Row projection and squared error](../../src/Transformer/Grokking/FiniteRecovery/ProjectionRows.lean)

An independent one-buffer pilot uses uncentered bias-corrected RMSPropW.
It retains $v'=0.95v+0.05g^2$ and updates with the current clipped gradient
divided by $\sqrt{v'}/\sqrt{1-0.95^{t+1}}+10^{-8}$. It uses peak 0.001 and
decay 0.2 so the per-update shrinkage matches AdamW with peak 0.002 and
decay 0.1. The GPU regression compares mixed-sign and zero gradients with
a host reference. Its coordinate direction has magnitude at most five in
the formal real recurrence. The exact division observation graph gives the
same almost-sure recovery result for independent Gaussian checkpoint copies.
[RMSProp update and bounds](../../src/Transformer/Grokking/FiniteRecovery/RmsProp.lean)

For any positive decay $\lambda$ and $0\leq\eta_t\lambda\leq 1$, the
softsign direction has magnitude at most one. Every ordinary softsign checkpoint
therefore satisfies $\|w_t\|_\infty\leq\max(\|w_0\|_\infty,1/\lambda)$.
For an identifiable, strictly representable finite task this gives almost-sure
hidden recovery when fresh ideal Gaussian copies are checked indefinitely.
The finite miss bound $(1-p)^m$ has an unspecified $p>0$. On the actual
division mask the generator proves this for the quotient-factor class, all
native initial draws and the experimental six-layer model.
[Certified softsign training in Lean](../../src/Transformer/Grokking/FiniteRecovery/ArgandSoftsign.lean)

For ordinary updates write $h_t=\lambda w_t+c_t/(|c_t|+10^{-6})$ coordinatewise
and let $u$ be a strictly realizing representative. If on a constant-rate
tail $\langle w_t-u,h_t\rangle\geq\mu\|w_t-u\|_2^2$ and $\|h_t\|_2^2\leq M$,
then for $0<2\eta\mu\leq 1$ the squared error is bounded by a term decaying
as $(1-2\eta\mu)^k$ plus $\eta M/(2\mu)$. When that residual fits strictly
inside the representative's margin neighborhood, every sufficiently late
ordinary iterate predicts every hidden label correctly. Establishing the
alignment and the required radius for the trained division trajectory remains
open. [One-point alignment and finite-step bound](../../src/Transformer/Grokking/FiniteRecovery/SoftsignAlignment.lean)

Eleven independent 32-bit model seeds were drawn before running comparisons
on them. Seed 0 is reserved for choosing the candidate. The comparison
uses paired first hits and carries forward every miss and interruption.
[Paired division protocol and seeds](division-optimizer-paired-registration.json)

Complex Muon applies Nesterov momentum 0.95 and five quintic Newton–Schulz
steps to each complex Q, K, V, O, gate, up and down projection. It normalizes
the starting matrix by its Frobenius norm with floor $10^{-8}$, uses conjugate
transposes, and scales each matrix direction by $0.2\sqrt{\max(m,n)}$.
The tied embedding, residual gates, scores and frequencies use the control's
bias-corrected AdamW. [Muon reference implementation](https://github.com/KellerJordan/Muon)

Complex Muon reached all 4,656 strict hidden answers at update 15,616,
with minimum gap 1.817 after independent replay.

The two uncorrected Adam GPT screens transfer their division hyperparameters to
all 35 canonical short and long cases and their three seeds. Each case
keeps its own vocabulary, model shape and WSD budget. The run stops on
strict 100% accuracy or at the 1,000-update screening cap. Each terminal
checkpoint is reloaded, and the summary reports 99% and strict 100%.
The recorded coefficients distinguish the equal-decay run from the
0.9/0.95 repair.

Another comparison uses Adam with a common decay of 0.9 for both moments,
no bias corrections, epsilon $10^{-8}$, and decoupled weight decay 0.1.
For each clipped gradient $g$, the new moments satisfy
$v'-(m')^2=0.9(v-m^2)+0.09(m-g)^2$. Hence the direction
$m'/(\sqrt{v'}+10^{-8})$ has magnitude at most one in each coordinate.
At peak 0.002 the 4,096-update prefix reached 4,651 strict hidden answers,
with minimum gap −5.649. At peak 0.003 ordinary training reached 4,656
strict hidden answers at update 2,816, with minimum reloaded gap 0.742.
This is 31.25% fewer updates than the frozen 0.002 AdamW control on
discovery seed 0. WSD was fixed at 20,000 throughout. When stock AdamW
also used 0.003, it reached the same first perfect checkpoint at update
2,816 with a larger minimum gap of 6.406.
[Exact update and moment invariant](../../src/Transformer/Grokking/FiniteRecovery/EqualMoment.lean)

A second pilot changes the fraction of each new squared gradient from
0.1 to 0.05, keeping the first-moment gain 0.1. This matches the stock
AdamW moment decays. Both runs use peak 0.003, weight decay 0.1 and the
20,000-step WSD. Removing the two bias corrections gave a first strict
hit at update 2,304 on discovery seed zero, with minimum hidden gap
7.999755 on replay. Stock AdamW and the equal-decay pilot first hit at
2,816. The historical validation seeds that motivated the repair are
excluded from its registered eleven-seed comparison.
Six of the controls are shared with the predeclared 0.003 comparison.
Each reused result is checked against the original weights, replay and
observation certificate. The remaining five controls run on fresh seeds.
The paired time is measured in updates through the first strict hidden
success, with the penalty for missing the cutoff specified in the protocol.
[Repair validation protocol](division-uncorrected-paired-registration.json)

For eleven complete independent pairs, let $D$ be candidate time minus
control time. If $m$ is a population median, $P(D<m)\leq 1/2$. For the
ninth-smallest observed difference to fall below $m$, at least nine of
the eleven values must be strictly smaller. The binomial upper tail is
at most $67/2048$, giving one-sided coverage of at least $1981/2048$
even when a grid produces ties. Both arms on a given seed may be dependent.
If an arm is interrupted, the conservative calculation uses zero for its
unavailable control time or the maximum penalty for its unavailable
candidate time. After all eleven pairs are accounted for, the ninth
ordered upper difference has the same coverage, including when GPU
interruptions depend on the outcomes. Retried hits remain visible
separately from complete first attempts.
[Paired median bound](../../src/Transformer/Grokking/FiniteRecovery/PairedFirstHitValidation.lean)

For moment decays $a,b$ with $a^2<b<1$, the two uncorrected buffers
satisfy $m_t^2\leq K v_t$ for all clipped gradient histories, where
$K=b(1-a)^2/[(1-b)(b-a^2)]$. Taking the second gain to be half the
first gives $b=(1+a)/2$ and $K=2(1+a)/(1+2a)$. The pilot therefore has
$K=19/14$. The corresponding update direction has magnitude at most
$\sqrt{19/14}$ per coordinate. The ordinary-restart theorem and the
common-gradient limit both allow these two different decays.
[Moment bound for unequal decays](../../src/Transformer/Grokking/FiniteRecovery/UncorrectedAdam.lean)

Stock AdamW admits another implementation that retains its bias correction.
For each decay $\beta$ start a scalar gain at $h_0=1$ and update it by
$h_{t+1}=h_t/(\beta+h_t)$. The normalized moment uses
$(1-h_t)m_t+h_tg_t$ and the second moment uses the same rule on $g_t^2$
with its own decay. Lean proves $h_t=(1-\beta)/(1-\beta^{t+1})$ and equality
of every weight with stock AdamW for arbitrary starting weights, rates and
state-dependent gradient sequences. Thus every finite-grid first-hit time
and its probability agree in real arithmetic. On an identifying mask,
ordinary independent restarts retain a strictly correct checkpoint almost
surely if one real run reaches one. The whole quotient-factor family
satisfies the witness condition when the grid includes initialization.
The FP32 implementation rounds at different intermediate operations.
[Trajectory equivalence](../../src/Transformer/Grokking/FiniteRecovery/NormalizedAdam.lean) · [Hidden recovery](../../src/Transformer/Grokking/FiniteRecovery/NormalizedRecovery.lean)

For any hidden set of more than 288 division inputs, stock Gaussian
initialization strictly solves the whole set with probability at most
$1/234$. There are at most 288 equations on which the answer occurs
in the five-token prompt. On any other equation at least 234 absent
tokens, including the correct answer, are exchangeable before training.
Consequently, if a training procedure succeeds with probability $p$,
the probability of success after starting with an imperfect hidden set is
at least $p-1/234$. This is a joint probability over all draws.
[Initialization and learned division success](../../src/Transformer/Grokking/FiniteRecovery/DivisionInitialAccuracy.lean)

On discovery seed zero, the recursive-gain kernel first solved all 4,656
hidden equations at update 1,536 with minimum reloaded gap 0.620017.
It solved 4,653 of the 4,656 observed equations at that checkpoint.
Stock AdamW first solved the hidden split at 2,816 on the same schedule.
The native fast reductions and the different FP32 operations can affect
first-hit times. The paired replication reuses each stock control and
checks its recorded weights digest, step, replay and observed-data
certificate. All eleven seeds were fixed before their first recursive-gain
run. The same implementation also runs on the original GPT short and long cases.
[Paired replication](division-normalized-validation.json)

The step-2,816 equal-decay and step-2,304 unequal-decay checkpoints
were also checked from their recorded FP32 logits.
The exporter reloads the weights, reconstructs the labels from the 4,656
observations, and saves all 239 scores for each of the 9,312 legal prompts.
Every finite FP32 score is an integer multiple of $2^{-149}$. The exporter
round-trips each value to its original bits, including the sign of zero.
The Lean kernel checks 97 blocks of 96 inputs and proves that the token
for division in the field with 97 elements strictly beats every rival.
The numerical forward pass and the training trajectory are supplied by Kernite.
At the normalized AdamW first hit, three observed predictions are still
wrong. The hidden-only checker retains all 2,225,568 recorded FP32
logits and checks all 238 rival inequalities on each complementary
input. Lean checks the original training mask and proves that its
complement contains exactly 4,656 equations.
[Proof for recorded logits](../../src/Transformer/Grokking/FiniteRecovery/Binary32Scores.lean)

For either ordinary optimizer, a common limit $G$ of the actual clipped
minibatch gradients on the positive WSD floor gives the limiting weight
$-G_i/[\lambda(|G_i|+\epsilon)]$ in each coordinate. A strict winner table
at these limiting weights generalizes if it belongs to an identifiable class
and fits the observed labels. Both the common gradient and the class test
are hypotheses for the ordinary division dynamics.
[One-moment limit](../../src/Transformer/Grokking/FiniteRecovery/SoftsignLimit.lean) · [Equal-moment limit](../../src/Transformer/Grokking/FiniteRecovery/EqualMomentLimit.lean)

Independent stock initializations give a separate guarantee for ordinary
equal-decay training. If its finite evaluation grid admits one strictly
successful real trajectory, the open set of successful initializations has
positive mass $p$. The observation-only certificate retains the first
correct checkpoint almost surely and the exact failure probability after
$m$ independent attempts with fixed batch order is $(1-p)^m$. Every table
in the quotient-factor class has a strict representative at the stock
initial gate settings. Including step zero gives a uniform finite restart
count for any required confidence, even with seed-dependent minibatches.
The theorem leaves the numerical size of that count open.
[Ordinary restarts and uniform confidence](../../src/Transformer/Grokking/FiniteRecovery/EqualMomentConfidence.lean)

For a fixed horizon, identifiability and strict representability alone
permit single-run success probabilities at or below one half. Consider
the two rows $(1,0)$ and $(2,1)$ with the lower left label hidden, using
input tokens 8 through 11. The character table on these labels has rank
one. The three observed cells uniquely determine the minimum nuclear-norm
completion, and six-layer Argand of width 64 can represent all four labels.
Swapping unused vocabulary tokens 2 and 3 preserves the training data,
the hidden prompt, and the Gaussian initialization law. At any fixed
update the chance of strictly predicting 2 is at most one half. The
same bound holds for all sufficiently late updates being strictly correct.
Across different token encodings, transporting the data and initialization
also preserves the full distribution of the first strict hit.
At a single update all 58 unused tokens are exchangeable and at most one
can strictly win. On a grid of $k$ evaluations, the probability of hitting
the hidden answer anywhere on that grid is at most $k/58$. Checking every
256 updates from zero through 4,096 gives an upper bound of $17/58$ for
either choice of the second-moment decay. The bound allows dependence
between checks and uses the same early stopping rule as the division pilots.
[Hidden-corner example](../../src/Transformer/Grokking/FiniteRecovery/HiddenCorner.lean) · [First-hit distribution](../../src/Transformer/Grokking/FiniteRecovery/EqualMomentRelabeling.lean) · [Finite-grid bound](../../src/Transformer/Grokking/FiniteRecovery/FirstHitSymmetry.lean)

The same limitation reaches the 200,000-update budget for ordinary AdamW
and the normalized recurrence. On the three-by-three table
$F(i,j)=(1+i-j)\bmod 64$, the diagonal and first superdiagonal give five
observations with labels 0 and 1. They identify the rank-one character
table by nuclear-norm minimization. Nine distinct input tokens from 8 to
16 let a six-layer, width-64 Argand model strictly realize every label.
Two hidden cells require the ordered answers $(2,3)$. For each fixed
minibatch stream and any evaluation, all $53\cdot52=2756$ ordered pairs
of distinct unused tokens have equal probability under stock Gaussian
initialization. At most one pair strictly wins. Including initialization,
every 256th step and the final step, there are 783 evaluations. The
chance of any checkpoint answering the entire hidden set correctly is
therefore at most $783/2756$ (28.41%).
[Two hidden answers](../../src/Transformer/Grokking/FiniteRecovery/TwoHiddenAnswers.lean)

Short qualification runs keep the 20,000-step WSD but stop at 4,096 updates
or the first perfect checkpoint. The equal-moment rate multipliers are 2
then 3 after a miss. For softsign at peak 0.0003 the stronger decay values
are 2 then 4 after a miss. Among qualifiers from these two screens, use the
earliest hit for the fresh paired comparison, breaking ties in favor of
one-buffer softsign. The protocol records the two discovery digests, the
validation binary digest, and alternate arm order before the first new run.
The equal-decay candidate at 0.003 qualified at update 2,816 and was
locked for the eleven fresh seeds. A secondary 0.003 stock AdamW arm was
registered before validating the first seed. It uses the same trained
candidate checkpoints as the primary comparison.
[Secondary control protocol](division-optimizer-higher-rate-registration.json)

```sh
task=grok-division-p97
bin=target/release/argand
$bin adamw-scaled-until-perfect "$task" 0 runs/division-x2 20000 2
$bin softsign-until-perfect "$task" 0 runs/division-lion 20000 0.3
$bin softsign-checkpoint-proposals "$task" 0 719 runs/division-lion-checked 20000 8192 1 20000
$bin muon-until-perfect "$task" 0 runs/division-muon 20000 1
$bin rmsprop-checkpoint-proposals "$task" 0 719 runs/division-rms 20000 8192 1 20000 0.2
$bin softsign-checkpoint-proposals "$task" 0 719 runs/division-row-clip 20000 8192 0.3 20000 0.1 1.75
$bin division-project-rows "runs/division-lion/$task/seed0" terminal 1.75
$bin equal-moment-until-perfect "$task" 0 runs/division-equal 20000 3 20000
$bin uncorrected-adam-until-perfect "$task" 0 runs/division-uncorrected 20000 3 20000
$bin normalized-adamw-until-perfect "$task" 0 runs/division-normalized 20000 3 20000
bash run_division_optimizer_pairs.sh CONTROL_DISCOVERY_CELL QUALIFIED_DISCOVERY_CELL runs/division-fresh-pairs
bash run_division_higher_rate.sh runs/division-fresh-pairs ADAMW_003_DISCOVERY_CELL runs/division-fresh-high
bash report_division_optimizer_pairs.sh runs/division-fresh-pairs runs/division-fresh-high
bash plot_division_optimizer_pairs.sh runs/division-fresh-pairs runs/division-fresh-high
ARGAND_BIN="$REPAIR_BINARY" ARGAND_CONTROL_BIN="$CONTROL_BINARY" \
    bash run_division_uncorrected_pairs.sh runs/division-fresh-high "runs/division-uncorrected/$task/seed0" runs/division-repair-pairs
bash report_division_optimizer_pairs.sh runs/division-repair-pairs
ARGAND_BIN="$NORMALIZED_BINARY" bash run_division_normalized_pairs.sh runs/division-repair-pairs \
    runs/division-fresh-high "runs/division-normalized/$task/seed0" runs/division-normalized-pairs
$bin division-certificate "runs/division-lion/$task/seed0" terminal
$bin division-recorded-lean "runs/division-equal/$task/seed0" terminal
bash check_recorded_division.sh "runs/division-equal/$task/seed0" terminal
$bin division-recorded-lean "runs/division-normalized/$task/seed0" terminal --hidden
bash check_recorded_division.sh "runs/division-normalized/$task/seed0" terminal --hidden
bash plot_division_optimizers.sh runs/division-comparison "runs/division-x2/$task/seed0" "runs/division-lion/$task/seed0"
```

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
minimum training gap of 19.225. The original replication specified all
4,656 strict hidden answers at exactly update 100,000 and registered
twelve seeds. Three seeds completed that protocol before the stopping
rule changed. All three succeeded. The horizon was selected after seeing
the first seed's 20,000-step result.

#### Stop at the first perfect evaluation

The active protocol ends a seed immediately when its primary hidden
score is 4,656/4,656 and every correct logit gap is positive. It checks
every 256 updates and at the budget. The saved weights are reloaded and
independently checked against the quotient table reconstructed from the
observed labels. The WSD schedule still spans 100,000 updates. If any
seed exhausts that budget, the twelve seeds start a separate 200,000-step
schedule with the same stopping rule.

The first four registered seeds already had verified successes at 18,432,
23,808, 25,600 and 23,808 on their original attempts. Seed 2434973590
reached 100% before its process interruption. The first hit of the
original attempt counts once. The remaining eight seeds were unrun when
the new rule was chosen. All twelve are included in the descriptive
fraction at each cap. A prospective analysis of this new event uses
the eight. Its probability is the chance of reaching 100% at a recorded
evaluation by the cap. Under independent runs, even eight successes in
eight have probability $0.75^8=0.10011$ at the three-quarters null
boundary for one budget. A population probability above three quarters
therefore cannot be established at the 5% level from these eight alone.
The first-hit event contains the fixed endpoint event on the same path
when the cap is evaluated. The observations within a run can be arbitrarily
dependent. [Stopping time and certified recovery](../../src/Transformer/Grokking/FiniteRecovery/FirstPerfect.lean)

```sh
bash run_division_to_perfect.sh runs/division-confirm-100k runs/division-first100-100k 100000
bash plot_division_to_perfect.sh runs/division-first100-100k runs/division-comparison
```

Each completed seed has a draw record giving its first successful update
or the exhausted cap, the replay digest and the certificate's minimum
gap. The plot truncates the historical curves at the selected checkpoint.
The adjacent `division-first100-<budget>-inference.json` counts the
updates omitted after actual early stops. Once the eight prospective
draws finish, it gives exact binomial upper tails for single-run hit
probabilities of at most one half and three quarters, along with bounds
allowing either registered budget. An interrupted prospective attempt
stays in the denominator and a successful rerun contributes only to
the descriptive count. The seed list and the prospective cutoff are
recorded in [the first-hit analysis plan](division-first-hit-registration.json).

At the fixed 100,000-update schedule the eight prospective draws can
be followed by three additional seeds. They were registered after three
of the eight had finished. The extension runs only after eight credited
first-attempt hits and ends at its first miss or interrupted attempt.
Each seed stops at its first strictly perfect evaluation. Rejection
requires all eleven first attempts to succeed in this one series. For
independent runs, the unconditional probability of that event under
$p\leq3/4$ is at most $(3/4)^{11}=0.04224$. Conditional on the three
successes already seen when the extension was chosen, the probability
of eight more at $p=3/4$ is $0.10011$. A 200,000-update diagnostic
does not contribute to this test. [All-success series in Lean](../../src/Transformer/Grokking/FiniteRecovery/TerminalValidation.lean)

```sh
bash run_division_probability_extension.sh runs/division-first100-100k runs/division-first100-extension
```

For ordinary AdamW under ideal Gaussian initialization, the probability
of hitting a strict hidden score from an open set of starts is positive
exactly when one real trajectory from that set hits. The set can require
that a rival initially beats the correct token on some hidden input.
If initialization itself is evaluated, native strict representability
already gives positive hit probability for every WSD budget. This
applies to every division-factor table and the 8-, 16- and 32-variable
binary sums. For a fixed number of trained updates, a sufficiently long
ordinary WSD warmup also preserves an open set of initially correct
draws, uniformly in minibatch order. Through update 256 the sums of the
actual rates are 0.01632 at budget 100,000 and 0.00816 at 200,000.
The probability of learning from an initially wrong draw and its size
require additional evidence. [Ordinary AdamW first-hit probability](../../src/Transformer/Grokking/FiniteRecovery/NativeFirstPerfect.lean)

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
| AdamW | 2434973590 | 100,000 | 4,656/4,656 | 95.9999971 | 0.00002360 |

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
first attempts. It also evaluates the exact single-budget binomial tail
at the null boundary and doubles it for the two budgets. With nine
credited successes in ten fresh trials the latter bounds would be
$11/512=0.02148$ under $p\le1/2$ and $0.48805$ under $p\le3/4$.
The report releases these numbers after all ten fresh endpoints have been
verified. [Binomial comparison and budget selection](../../src/Transformer/Grokking/FiniteRecovery/TerminalTail.lean)

Even if interruption depends on the trajectory, removing
a success can only decrease the count. The bound keeps all ten registered
draws in its sample size. A restarted seed still contributes its terminal
result to the requested twelve-seed fraction. The first attempt at seed
2434973590 reached a strict 4,656/4,656 at update 25,600 and stopped after
the recorded evaluation at 36,864. The rerun uses the same seed and schedule.
It first solved all hidden equations at update 24,064, fell to 4,319 at
52,992 and returned to 4,656 at 55,040. At update 100,000 its replay
verified all 4,656 hidden equations with a minimum logit gap of 20.379.
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

A bijection of the token vocabulary preserves the probability that ordinary
AdamW gets every hidden answer strictly right at a fixed step in real
arithmetic. Rename the tokens in each prompt and label, and apply the same
permutation to the native embedding rows. The tied logits, batch gradients
and both moment buffers then match at every update after reindexing. All
sampled real and imaginary entries have the same standard deviation, 0.02,
so the ideal Gaussian initializer has the same law in each encoding. This
holds on the 100,000- and 200,000-step schedules for division with the
correspondingly renamed minibatches. For independent trials with a fixed
batch order and an identifying mask, one strictly successful real training
trajectory gives the same positive single-run mass $p$ and the same
miss probability $(1-p)^m$ after $m$ trials in every encoding.
[Full training dynamics](../../src/Transformer/Grokking/FiniteRecovery/AdamWRelabeling.lean) · [Division and other factor tables](../../src/Transformer/Grokking/FiniteRecovery/DivisionRelabeling.lean)

`division-relabel CELL` checks the finite-precision forward pass of the
saved weights under the vocabulary reversal $e(t)=238-t$. It compares all
239 logits at all five positions of all 9,312 prompts. Labels for the strict
answer check are reconstructed from the training split. At update 100,000
the first three completed checkpoints matched bit for bit on all 33,383,520
logit comparisons combined. Each retained 4,656 strict hidden answers.

For a fixed sequence of minibatches and any finite stock WSD horizon $K$,
write $d_K=\prod_{t<K}(1-\eta_t/10)$. The real AdamW endpoint in each
coordinate satisfies $|w_{K,i}-d_K w_{0,i}|\le120(1-d_K)$. This bound is
independent of the initialization. Since $d_K>0$ and the gradients are
continuous, Brouwer's fixed-point theorem implies that, as the native
draw varies, the final embeddings and all attention and FFN projections
jointly attain every prescribed vector. The scores, gates and frequency
coordinates train as usual and obey the same bound around their
deterministic initial values times $d_K$. If a choice of the free
coordinates gives strict answers throughout that box, an open set of
native Gaussian draws strictly generalizes at the original WSD rates.
[Endpoint reachability](../../src/Transformer/Grokking/FiniteRecovery/ArgandReachability.lean)

#### AdamW with one Gaussian perturbation

`adamw-first-noise-run CASE MODEL_SEED NOISE_SEED OUT CAP` performs the
ordinary first AdamW update, adds a centered Gaussian draw with standard
deviation $10^{-6}$ to each real weight, and continues with the same WSD
schedule. The model seed determines the native initialization and batch
order. The separate noise seed drives a single host draw and upload. The
two moment buffers retain the values from the first, unperturbed gradient.
The rule has the same parameters for every task. The twelve-seed division
confirmation uses ordinary AdamW.

In real arithmetic, fix any initialization, minibatch stream and positive
finite WSD horizon. Varying the single perturbation reaches every terminal
weight vector, including the trained gates and frequencies. A strictly
representable target therefore has a positive one-trial success probability
$p$ by openness and full Gaussian support. When the observation mask
identifies the target in its class, the probability that all $m$ independent
endpoints fail its observation-only certificate is exactly $(1-p)^m$.
The first accepted model is almost surely eventually retained. The result
also covers fresh native seeds that determine both initialization and batch
order. The argument supplies no practical lower bound on $p$ for a given
noise scale. For noise amplitudes tending to zero, every threshold strictly
below the stock real-arithmetic success probability is eventually exceeded.
The FP32 trainer samples a finite pseudorandom stream. [Finite-horizon dynamics](../../src/Transformer/Grokking/FiniteRecovery/AdamWFirstNoise.lean) · [Random seeds and batch orders](../../src/Transformer/Grokking/FiniteRecovery/FirstNoiseRandomStart.lean) · [Small-noise probability](../../src/Transformer/Grokking/FiniteRecovery/FirstNoiseConfidence.lean)

The theorem gives strict generalization with certified restarts for every
unknown group-factor table on the 192-observation mask and the actual
4,656-observation division mask. For binary sums of 8, 16 and 32 variables
it applies on their exact Flat Spectrum masks of 31, 511 and 131,071
observations. These instances use four heads of width 16, six layers and
one pass at the original rates. [Factor tables](../../src/Transformer/Grokking/FiniteRecovery/FactorFirstNoise.lean) · [Flat Spectrum and modular sums](../../src/Transformer/Grokking/FiniteRecovery/FlatFirstNoise.lean)

For series with any shared deterministic next-state law on 64 states, the
same architecture realizes every transition table at the native RMS
offset. The minimum number of labeled histories is the number of distinct
final states occurring across all sites. A covering mask suffices for
certified recovery at the stock WSD rates. For example, the 64 constant
histories cover all $64^{32}$ histories of length 32. [Multi-series transition recovery](../../src/Transformer/Grokking/FiniteRecovery/TransitionFirstNoise.lean)

On Boolean AND, model/noise seed pairs 0/371 and 1/372 each gave strict
correct answers on all 130,763 hidden binary strings of lengths 8 through
16 at update 1,000. A 256-update division control with model seed 113 and
ordered shared-gradient reductions scored 60/4,656 for stock AdamW and
59/4,656 for each of noise seeds 371 and 372.

At a matched 20,000-update WSD horizon, model seed 0 and order seed
10000, stock AdamW first reached 4,656 strict hidden answers at update
12,800. The perturbed run with noise seed 371 first reached them at 13,312.
The independently reloaded endpoints both got all 4,656 equations right.
Their minimum hidden logit gaps were 17.4530 and 18.4390 respectively.
The training-only quotient reconstruction accepted both checkpoints.
This is one matched pair with the native atomic gradient reductions.
Run `bash plot_division_first_noise.sh STOCK_CELL NOISE_CELL OUT` to
compare the two trajectories, after checking their initial scores, data
identities, schedules and terminal replays.

`bash run_page_screen.sh all runs/first-noise-screen first-noise` uses the
ordinary 99% stopping rule and at most 1,000 updates for each of the 35
canonical stress cases and its three model seeds. Noise seeds 371, 372
and 373 accompany model seeds 0, 1 and 2 respectively across all cases.
The terminal scorer reloads the saved weights, and the summary verifies
all 105 identities, endpoint scores and replay digests.

The current `first-noise-perfect` screen uses the same 35 cases, seeds
and caps. Each cell stops at its first strictly perfect primary evaluation
or exhausts its cap. Reloaded weights verified all 105 outcomes. The
initial scores, losses, model shapes, WSD schedules and data orders
matched the ordinary AdamW controls in all 105 pairs.

| Tier | Runs | Reached 99% | Reached strict 100% | Updates used / available |
| --- | ---: | ---: | ---: | ---: |
| Short | 75 | 34 | 23 | 59,457 / 73,716 |
| Long | 30 | 0 | 0 | 29,820 / 29,820 |

Both AdamW variants reached 99% on exactly the same 34 case/seed pairs.
Their first 99% steps agreed in 32 pairs. On `basis-depth-hard` the
perturbation reached 99% one evaluation earlier for seed 1 and one
evaluation later for seed 2. The best long endpoint was 3,800/4,656
on `associative-divisor-n2-l1-w0`, compared with 3,807/4,656 for stock
AdamW. The historical control reached strict 100% in 15 short runs.
It stopped another 17 short runs below 100% before the cap under its
99% rule. Their possible later successes leave the control's by-cap
strict count between 15 and 32 on those 75 original trajectories.

```sh
stock=runs/gpt-adamw-wsd-20261128
candidate=runs/gpt-adamw-first100-20261128
jq -es --slurpfile control "$stock/all-summary.json" --slurpfile candidate "$candidate/all-summary.json" \
  -f compare_screen_first_hits.jq "$stock"/*/seed*/observations.jsonl "$candidate"/*/seed*/observations.jsonl
```

`sum8-adamw-first-noise OUT 20480` uses the eight registered pairs of the
31-observation modular-sum experiment. In each pair the first seed sets the
native weights and the second sets the one-time noise. The stock AdamW
control uses the same starting weights, complete observed gradient and WSD
schedule. `sum8-adamw-first-noise-replay OUT` verifies the saved terminal
weights and intermediate snapshots. Run `bash plot_sum8_accuracy.sh OUT`
to render their strict accuracies on all 31 observed and 225 hidden words.
For the current early-stopping protocol, use
`sum8-adamw-first-noise-until-perfect OUT 20480`. It checks all 225 hidden
words at initialization, powers of two, multiples of 256 and the budget.
Only the 31 observed words contribute to gradients. The first perfect
checkpoint ends its seed. The plotted mean retains that checkpoint for
the remaining comparison times.

All eight first-update-noise runs reached 20,480 updates. Every final
checkpoint fit 31/31 observed words. The strict hidden counts were 95,
103, 98, 98, 96, 99, 102 and 97 out of 225, or 788/1,800 in total.
None reached 225/225 at any scheduled check. The paired ordinary
AdamW endpoints had 790/1,800. All noisy endpoints and saved
intermediate measurements were independently replayed.

#### Rank-one loss on unlabeled inputs

The binary factor class admits a rank-one table of signs. On the same
31 labeled words, this experiment adds a penalty to the observed mean
cross-entropy. Each AdamW update draws five rows and five columns
independently without replacement from the 16-by-16 cut. The resulting
25 **unlabeled** words supply a matrix $P$ of last-position gaps
$z_0-z_1$. For its leading singular component $P_1$, the penalty is
$\|P-P_1\|_F^2/50$. Its selected logit adjoint is $(P-P_1)/25$ on
token zero and its negative on token one. Both losses contribute before
the ordinary clipping, moment update and 0.1 decoupled decay. All model
parameters train from their paired native initialization. Strict
stopping uses the sign table reconstructed from the observed staircase.
An independent held-out replay checks the saved weights.

For any additive binary row and column labels, correct observed signs,
minimum absolute logit gap $m>0$, and expected patch loss below
$m^2/7200$ imply strict recovery everywhere. Each sign-inconsistent
two-by-two rectangle appears in exactly $1/144$ of the sampled
patches. Every such patch has squared residual at least $m^2$.
This gives a sufficient condition on the trajectory of the full model.
A single sampled loss value does not estimate its population mean
with a guaranteed error. [Rectangle loss and strict recovery](../../src/Transformer/Grokking/FiniteRecovery/RectangleLoss.lean)

The certificate command checks all 14,400 four-corner sign identities
at the saved weights. The eight ordinary 20,480-update controls fit
all 31 observations and violated 4,864 to 5,589 identities. The first
paired seed violated 5,212. These counts use logits alone.

On that first seed, the rank-one penalty reduced the full table's
half-mean-square residual from 87.8043 to 0.05553 and violations from
5,212 to 3,328. The smallest absolute gap also fell from 0.18234 to
0.0004533. Both runs fit 31/31 observed words at 20,480 updates.
Hidden accuracy moved from 95/225 to 99/225. The sampled penalty can
therefore become small while many signs are still incorrect.

The margin variant adds $\sum_{ij}\max(0,1-|P_{ij}|)^2/50$ on the
same sampled patch. It uses the same native seed, rectangle stream,
architecture, WSD and stopped-checkpoint rule. For any binary factor
class, identifying mask and rectangular design, Lean bounds recovery
from the sum of the two penalties and the observed cross-entropies.
For these 31 observations a total observed CE plus **expected** patch
penalty below $1/28800$ guarantees correct strict gaps on all 256 words.
The bound requires no separate margin assumption or hidden labels.
Whether the optimizer reaches the bound is an empirical question here.
[Objective threshold and general mask theorem](../../src/Transformer/Grokking/FiniteRecovery/RectangleMargin.lean)

```
target/release/argand sum8-adamw-rank-one runs/sum8-rank-one 20480 0
target/release/argand sum8-adamw-rank-one-replay runs/sum8-rank-one 20480 0
target/release/argand sum8-rectangle-certificate runs/sum8-rank-one/draw-0
bash plot_sum8_rank_one.sh runs/sum8-adamw/draw-0 runs/sum8-rank-one/draw-0 runs/sum8-comparison
target/release/argand sum8-adamw-rank-one-margin runs/sum8-margin 20480 0
```

#### Checked proposals on a single training trajectory

`adamw-checkpoint-proposals CASE SEED PROPOSAL_SEED OUT CAP FIRST_PROPOSAL_UPDATE`
runs ordinary AdamW and stops at the first strictly perfect primary score.
At eligible measurements, a failed ordinary checkpoint gets one forward
evaluation of a copy with independent Gaussian coordinate perturbations
at standard deviation `1e-6`. A rejection restores the exact weight
buffer. The ordinary optimizer retains its moments and minibatch position.
A perfect ordinary checkpoint ends training immediately. Setting the first
eligible update to the effective cap reserves the extra check for failures
at that cap. Division obtains its verification labels from the observed
factor graph. Other GPT tasks use their specified primary validation split.

With ordered gradient reductions, the checked and ordinary Boolean AND shift
runs on seed 0 both selected byte-identical weights at update 48 (7/7).
On division, seed 113 at update 256 retained byte-identical ordinary weights
and replayed 60/4,656 after rejected proposals. In particular the two
forward checks did not change the next AdamW update.

`sum8-checkpoint-proposals STOCK_CELL PROPOSAL_SEED OUT FIRST_PROPOSAL_UPDATE`
evaluates the saved checkpoints from one ordinary eight-bit sum training
trajectory. It reconstructs all labels from the 31 observed staircase signs
and independently replays the selected weights on the 225 hidden words.
With proposal seeds 371 through 378 on the eight paired stock trajectories,
none of 248 proposed copies passed. The original weights were preserved
on all eight. Only one proposal changed the number of correct hidden
answers, from 99 to 98 at update 13,312 for model seed 4195447240.

For a finite native task with an identifying mask and a strict full-model
realization, Lean proves almost-sure finite success on one continuing
real-arithmetic AdamW trajectory, conditional on any initialization. The
0.1 weight decay bounds every checkpoint in the same compact ball.
Full-support independent Gaussian copies then give a common positive
conditional hit probability, so the chance of missing all of the first
$m$ checks tends to zero. The native FP32 pseudorandom generator has
finite support. Its actual success probability and the number of checks
needed to exceed 75% on hardware remain unknown. [Single-trajectory proof and the modular tasks](../../src/Transformer/Grokking/FiniteRecovery/ArgandCheckpointProposals.lean)

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
