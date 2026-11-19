# Schedule-Free training of the actual network

Moved unchanged from [`../flat_spectrum.md`](../flat_spectrum.md) on 2026-11-19: the sections
*Alias symmetry and training* through *Gradient forcing*. They follow the actual normalized,
tied Rotate90 network through Schedule-Free AdamW updates: exact aliases kept by full-table
training, their escape under sampled updates (S1–S7), the effect of one update on unseen
equations (T1–T6), the stored averaging history (M1–M6) and the two-buffer update as a
recurrence forced by normalized gradients (F1–F4). The Lean is in
`Transformer.Rotate90.Training`, `Learning`, `Memory` and `Forcing`.

These labels are this file's, not the S1–S2, T1–T4, M1–M5 and F1–F6 of
[`../flat_spectrum.md`](../flat_spectrum.md). That note holds Theorem A and *Actual network
expressivity for every exponent*, the result these sections start from; its *General
positional models* identifies the Schedule-Free recurrence with the run of F1–F4.

## Alias symmetry and training

The expressivity result above removes insufficient representational capacity as
an explanation for the width-16 failures. It does not say which weights training
will find. The next result concerns the actual normalized, tied network and the
actual Schedule-Free AdamW update, at arbitrary finite width and depth.

### Full-table symmetry of the actual loss

Fix a prime p, natural exponent k, positive RMSNorm epsilon and a finite list of
branch matrices B. Write F(A,Bin,C) for the mean cross-entropy on all
`a in ZMod p`, `b in (ZMod p)^units`, with independent names for the first input
embedding, second input embedding and output embedding. The actual tied loss is
`L(E) = F(E,E,E)`. This auxiliary notation changes no layer: it retains the fixed
quarter-turn, every RMSNorm, every residual SwiGLU block and the full softmax.

For units r,s, define row pullback by `T_r E(q)=E(r*q)`. The arithmetic identity

```
((r*a)/(s*b))^k = (r/s)^k * (a/b)^k
```

gives the global identity

```
F(T_r A, T_s Bin, T_((r/s)^k) C) = F(A,Bin,C).
```

Indeed the output transformation reindexes every competing softmax class and
its target. Reindexing a by r and b by s then permutes the complete dataset.
This also holds at a=0 and k=0. It generally fails for a sampled half-table or
one minibatch, because the transformed pair need not belong to that sample.
`Training.fullLoss_permute` proves this identity for the existing arbitrary-depth
forward, rather than postulating symmetry of a surrogate loss.

### The embedding gradient preserves exact aliases

Let S be a multiplicative subgroup of nonzero residues. An embedding is
S-aliased when `E(h*q)=E(q)` for every h in S and every q. Zero is a fixed row;
it need not equal any nonzero row. Assume `gcd(k, |S|)=1`.

At the tied point (E,E,E), each of the three table transformations above fixes
the parameters whenever r,s belong to S. Differentiate the global identity,
first in a direction supported only in A, then only in Bin, then only in C.
The A and Bin partial gradients are constant on S-orbits. The C partial is
constant under multiplication by h^k; coprimality makes these kth powers cover
S. The actual tied derivative is the sum of all three partial derivatives.
Consequently,

```
E(h*q)=E(q) for all h in S  ==>  grad L(E)(h*q)=grad L(E)(q).
```

Every derivative used here exists: the finite network and its cross-entropy are
real analytic in the embedding when RMSNorm epsilon is positive. No generic
position assumption excludes repeated rows or zero activations. The proof does
not assume fixed branch matrices throughout training: the result holds for each
possible matrix choice at each step. It covers simultaneous learning of all
branches. `gradient_sharedLoss_aliased` proves the gradient equality;
`alias_difference_derivative_zero` proves the corresponding zero first-order
signal in a direction that separates two aliased rows.

This is an invariant subspace, not a stationary point theorem. Common embedding
rows and branch matrices can keep moving. It also is not a Hessian, attraction,
or random-initialization probability theorem. Coprimality is a real hypothesis:
without it, the output gradient need not preserve the whole subgroup.

### The source Schedule-Free update preserves the invariant

Use the revised optimizer recurrence in `ScheduleFree.AdamW.Defs`, corresponding
to the lab's `lean_optim/schedule_free_adamw.py`: query y, base z, raw second
moment v, group averaging coefficient c, scalar rate eta and query decay lambda.
In each embedding coordinate the update is

```
v_new = beta2*v + (1-beta2)*g^2
d     = g / (sqrt(v_new/(1-beta2^(t+1))) + eps_opt) + lambda*y
z_new = z - eta*d
y_new = y + c*(z-y) + eta*(beta1*(1-c)-1)*d.
```

The evaluation embedding is `x=y+(1-1/beta1)*(z-y)`. The source's warmup,
maximum-rate history, bias correction and weighted averaging determine scalar
eta and c equally for all coordinates. If two rows of y,z,v and g agree, these
formulas preserve their equality. Initialization sets y=z=E and v=0. Combining
this coordinate fact with the actual-gradient theorem proves that every query,
base, second moment and evaluation embedding stays S-aliased forever.

`trainingRun_aliased` proves this by induction on the independently defined
optimizer recurrence. `trajectory_aliased` permits arbitrary external scalar
parameter schedules satisfying the same source update, including the lab's
external warmup. It also permits the lab's initial maximum-rate history of 0,
where the upstream recurrence starts at -1: no scalar-history condition enters
`StateAliased`. The branch trajectory remains arbitrary. The statements are
real-arithmetic results; implementation roundoff and missing/sparse gradients
are not silently included in the recurrence. The experiment uses ordinary valid
parameters beta1=.9, beta2=.98 and positive epsilons.

### An incorrect model can remain incorrect despite enough capacity

Tied readout gives `logit(h*q)=logit(q)` whenever E is S-aliased, for every input
and hidden state. If S contains h unequal to 1, the query a=b=1 has target 1
for every k, yet class h ties its score. Thus the model does not have the unique
correct winner on the complete task. `trainingRun_not_solves` proves this for
every finite update count, at every finite width/depth, regardless of the branch
trajectory or the chosen evaluation branch matrices.

There is also a quantitative obstruction. For a nonzero target q, its |S|
distinct orbit classes all have the target's score, so

```
sum_c exp(logit(c)) >= |S| * exp(logit(q))
cross_entropy(logit,q) >= log(|S|).
```

`crossEntropy_ge_log_subgroup` and `trainingRun_loss_floor` prove this bound.
`poweredLoss_floor` applies it directly to every nonzero numerator and unit
denominator, including exponent zero. The zero target is excluded from the
orbit-cardinality claim because its orbit is a singleton.

Nonzero witnesses satisfy all hypotheses: at p=3, k=1, take S to be the entire
unit group, put 2 in every coordinate of the zero embedding row and 1 in every
coordinate of each nonzero row. The examples use positive rate and epsilon,
width 16, hidden size 43 and actual Schedule-Free states. Branch matrices are
unrestricted by the theorem. These witnesses refute unconditional convergence
from every initialization; they do not establish failure for almost every
initialization or for the measured random starts.

`aliased_limit` proves that any pointwise limit of aliased embeddings remains
aliased, and `limit_not_solves` retains the readout obstruction at that limit.
Together with the trajectory theorem, even unlimited training time cannot split
these exact aliases if a limit exists. No convergence or infinite-width/depth
limit is assumed. Wider finite networks still admit these bad starts, while the
preceding expressivity theorem supplies good weights at width 16 already.

### Tests of the mechanism and its limits

The [50-run intervention study](../../experiments/rotate90_alias_escape/README.md)
records its hypotheses and source checkpoint hashes before training. All runs
use Schedule-Free AdamW, Python 3.15.2, Torch 2.15.0+cu132 and CUDA compile
max-autotune. Each restart resets optimizer histories and batch order from a
pinned original training query; unchanged restarts provide matched controls.
The original averaged checkpoints and original 175-run study remain intact.

The portable [gradient checks](../../experiments/rotate90_alias_escape/gradient_checks.json)
use CPU float64 while explicitly retaining the original FP32 RMSNorm epsilon.
At seed 11's embeddings projected onto the observed subgroup, the transverse
fraction of the actual embedding gradient is:

| Task | Subgroup order | Complete table | Original half-table | Noncoprime exponent, complete table |
| --- | ---: | ---: | ---: | ---: |
| P97/K2 | 3 | 2.41e-16 | .22285 | .39131 with K3 |
| P193/K3 | 8 | 3.15e-16 | .22040 | .63062 with K8 |

The complete-table controls satisfy the exact theorem to float64 roundoff.
The other columns demonstrate why its data and exponent hypotheses matter.
After 200 compiled FP32 full-batch steps from exactly projected embeddings,
relative embedding asymmetry is about 7.3e-8 and 7.6e-8 in the evaluation averages.
Minibatches on the same complete table produce about .00888 and .01074 instead.
All eight symmetry-control runs have 100% quotient accuracy, which merges the
aliased classes; their actual task accuracy is only about 34% or 13%. The
nonzero-target entropy floors are log 3 and log 8. FP32 tie-breaking also changes
argmax accuracy between compiled CUDA and eager CPU replay by up to .226
percentage points in the projected controls; this drift is retained separately.

On the half-table interventions, no run crosses the prespecified 99% final
heldout threshold within 5000 new updates. At P193, seeds 17 and 23 improve from
58.14% and 45.09% originally to 72.62% and 74.31% under unchanged restarts, and to
95.81% and 98.39% at zero decay. These two embeddings are far from the seed 11
order-eight alias subspace: their initial relative projection residuals are .94
and .91. The exact symmetry theorem is not an explanation for their remaining
errors. At P193 seed 11, zero decay instead reduces heldout accuracy from 11.79%
to 8.01%; at P97 it also initially improves fitting without reliable generalization.
Freezing either parameter family or injecting the tested noise does not produce
a 99% solution. Every seed and failed intervention remains in the report.

The user's follow-up asks whether more updates or width repair the remaining
errors. [The adaptive budget study](../../experiments/rotate90_alias_budget/README.md)
compares 20000-update restarts at decay 1/.1/0 and fresh width 64 controls, with its
hypotheses recorded before those additional runs. This is a finite empirical
test of duration and architecture, not a proof that any failed random start is
trapped forever. All 24 additional runs finished. At zero decay, the two P193
runs above reach 99.74% and 99.989% after 20000 updates; even seed 11 rises to 92.12%.
All six fresh width 64 controls reach 99.96--100% at 5000 updates, with their first
measured 99% points at 750. The P97 width 16 restarts remain below 56% in these
conditions after 20000 updates. The recorded specifications and split hashes
confirm that the duration controls change only the budget, and the fresh width
controls change only width (with hidden size following it). These outcomes
refute a permanent-trap interpretation of the earlier half-table plateaus.
They support an optimization effect of width and decay in this finite study,
while leaving a theorem about typical random-start learning open.

The corrected theory now separates an actual capacity construction from an
actual optimizer obstruction. Predicting which approximate quotient codes
random partial-table training selects, the stability of their neighborhoods,
and the separate P17 memorization failures remain unresolved. Increasing
width or changing decay can help measured training without falsifying the exact
invariant theorem; its hypotheses must be checked before applying it.

### Lean inventory for training

All declarations are in `Transformer.Rotate90.Training`, reachable from the
project root. Every theorem has hypothesis witnesses; no new unfinished claim
or extra axiom is introduced.

| Module | Main declarations |
| --- | --- |
| `Defs` | `untiedLogit_diagonal`, `crossEntropy_permute`, `aliased_iff`, `aliased_zero_nonzero` |
| `Basic` | Analyticity and differentiability of the actual untied and tied full-table loss |
| `Symmetry` | `fullLoss_permute`, `differential_fixed`, `differential_fullLoss_fixed` |
| `Gradient` | Three partial-gradient equalities, `gradient_sharedLoss_aliased`, `alias_difference_derivative_zero` |
| `Optimizer` | `trainingStep_equal_coordinates`, `trainingRun_aliased`, `trajectory_aliased`, `evaluation_trainingRun_aliased` |
| `Readout` | `crossEntropy_ge_log_subgroup`, `poweredLoss_floor`, `not_solves_of_aliased` |
| `Failure` | `aliased_limit`, `limit_not_solves`, `trainingRun_not_solves`, `trainingRun_loss_floor`, `trajectory_not_solves` |

## Sampled updates can break aliases

The next gap is demonstrated by the existing symmetry controls: minibatches on
the complete table produce unequal embedding rows even where a full-table update
preserves their equality. The theorem above concerns the loss used in each step,
not merely the dataset from which that step was sampled. A symmetric mean gradient
does not make each sampled gradient symmetric. The following local calculation
gives an exact mechanism in the actual model, with no change to the architecture.
It treats a completely collapsed embedding; the measured controls merge smaller
subgroup orbits. Thus this is a rigorous escape mechanism and counterexample to
extending the full-table invariant to every batch, not a quantitative prediction
for those particular checkpoints.

### The tied derivative and empirical label counts

Let all embedding rows initially equal a vector v, which need not be zero. For
arbitrary finite branch weights B and positive RMSNorm epsilon, every pair then
has the same actual final hidden vector h = hidden(eps,B,v,v). A perturbation D
of the tied embedding changes each logit by

```
(S1) d logit_q[D] = <D_q,h> + <v, d hidden[D]>.
```

The second term includes both operand lookups, all residual blocks, SiLU and
every normalization. It is common to every output class. Equal initial logits
give uniform softmax, so this indirect derivative cancels in cross-entropy:

```
(S2) d CE_t[D] = (1/p) sum_q <D_q,h> - <D_t,h>.
(S3) grad_E CE_t(q,i) = (1/p - indicator[t=q]) * h_i.
```

This is a derivative at the collapsed point in the full parameter space. It is
not a derivative restricted to moving all rows together or a frozen-feature
approximation. The branch weights are arbitrary, and their input-dependent
derivative is retained until its exact cancellation.

For any nonempty finite indexed batch J, let f_J(q) be its empirical label
frequency, counting repetitions. Ordinary mean cross-entropy therefore satisfies

```
(S4) grad_E L_J(q,i) = (1/p - f_J(q)) * h_i.
(S5) grad_E L_J(q,i) - grad_E L_J(r,i) = (f_J(r)-f_J(q)) * h_i.
```

Different label counts and a nonzero hidden coordinate give a transverse signal.
A label-balanced batch has zero embedding gradient at this point. Sampling with
replacement and fixed subsets are both covered by the indexed-family definition.
The formula does not claim a whole dataset's labels are uniform for every K:
powering can merge distinct quotients, and K=0 has only target 1.

For K=1, fixing a unit denominator makes division a permutation of all p
residues. The full table therefore has frequency exactly 1/p for every label,
including zero. Its embedding gradient at the collapsed point is zero. This
cancellation coexists with a nonzero sampled signal: for two distinct classes
q,r and a uniformly sampled target t, equation (S3) gives

```
(S7) mean_t (grad CE_t(q,i) - grad CE_t(r,i))^2 = (2/p) * h_i^2.
```

Only t=q and t=r contribute, each with squared magnitude h_i^2. The formal
statement is a finite average over labels with arbitrary chosen operand pairs;
the division witness uses a=t, b=1. It also describes uniform sampling from
the complete division table because every label has the same multiplicity and
the collapsed hidden vector is input independent. No independence assumption
about successive optimizer steps or stochastic convergence theorem is used.

### An actual Schedule-Free escape step

Start the source optimizer with y=z=E and zero second moment. At a positive first
rate eta, the averaging coefficient is one, for all real averaging powers. With
beta2<1, the first bias correction cancels 1-beta2, giving

```
(S6) y_new = z_new = E - eta * (g / (|g| + eps_opt) + decay*E).
```

The evaluation embedding agrees with these buffers. If E(q,i)=E(r,i) and
g(q,i)<0<g(r,i), positive eta and eps_opt imply
`y_new(q,i)>y_new(r,i)`. Equal weight-decay terms cancel. This uses the source
AdamW normalization and averaging, not a substitute SGD step. The legal numerical
witness uses beta1=.9, beta2=.98, rate=.003, decay=1, eps_opt=1e-8, with no warmup.
The theorem concerns a fresh first step with positive rate; it does not silently
identify this with the lab's initial zero-rate external warmup step.

For every prime p and natural K, the valid example a=b=1 has target 1. At a
nonzero collapsed hidden coordinate, its gradient has negative target and
positive competitor components. Consequently the optimizer can separate rows
that full-table training preserves under its symmetry hypotheses. Breaking an
alias is necessary progress away from that particular equality constraint; it
does not imply correct predictions on unseen examples, loss convergence, or
reliable escape from every partially collapsed state.

The nonzero witness is explicit: v is the all-ones vector, width is 16, hidden
size is 43, and there are d+1 zero-matrix residual blocks for any natural d.
The rotated input has even coordinates zero and odd coordinates two. Positive
epsilon makes both outer normalization factors positive, so every odd hidden
coordinate is positive. At p=3, classes 1 and 2 share the initial row but separate
after the singleton update. `full_zero_sample_escapes` proves in one statement
that the full-table embedding gradient is zero, the initial table is aliased,
and both the next query and evaluation tables are no longer aliased. The general
`rowStep_not_aliased` covers any prime, exponent and subgroup containing a
nonidentity unit, with the stated positive hidden-coordinate hypothesis.

### Lean inventory for sampled training

The 25 theorems in `Transformer.Rotate90.Training.Sampling` add no unfinished
claims. Every hypothesis has a satisfying example; all modules are reachable
through `Transformer.Rotate90.Training`. The proof uses the existing forward
and optimizer definitions, with no new numerical experiment or optimizer change.

| Module | Main declarations |
| --- | --- |
| `Sampling.Basic` | Actual row-loss analyticity, `fderiv_tiedLoss_constant`, `hidden_ones_pos` |
| `Sampling.Gradient` | `gradient_batchLoss_constant`, `gradient_row_difference`, `singleton_transverse_energy`, `row_gradient_signs` |
| `Sampling.Optimizer` | Source initial-step identities, `initial_step_separates`, `rowStep_separates` |
| `Sampling.Escape` | `frequency_full_division`, `gradient_sharedLoss_division_constant`, `rowStep_not_aliased`, `full_zero_sample_escapes` |

The next unresolved gap is the behavior after this local escape: whether a
sampled trajectory develops a separating answer code and generalizes, with a
prediction of success probability or training time. The new theorem does not
derive those outcomes from breaking one equality. The measured width and decay
effects, partially collapsed checkpoints and P17 memorization remain separate
questions for that next comparison cycle.

Validation on 2026-10-31: full `lake build` passed (13070 jobs); all 25 new
`#print axioms` checks contain only `propext`, `Classical.choice`, and `Quot.sound`.
`./run.py audit` reports 36 existing unfinished statements, zero results resting
on them, and zero extra axioms. Regenerated `INDEX.md` has zero vacuous or
placeholder entries; `./run.py forbidden` passed. No Python code changed.

## Learning transfer

The next comparison replaces the static rank condition with the actual effect
of a sampled optimizer update on unseen equations. Static character capacity
and fixed-code matrix completion do not explain why width 16 and width 64 train
differently, or why P17 fits its training split without generalizing. This is a
conditional dynamics extension, not a proof that a neural optimizer solves the
nuclear-norm interpolation problem of Theorem A.

Let theta contain every embedding, gate, value and output entry, with its usual
Euclidean metric. For any finite ordered stack, let F(theta) be the mean full
softmax cross-entropy on a fixed evaluation split. All normalizations, both
input lookups and the tied output embedding remain in the forward pass.

```
(T1) epsilon > 0 implies F is analytic, hence C^2, jointly in all parameters.
```

At update t, Schedule-Free stores query y, base z, raw variance v and scalar
averaging history. It evaluates at x = y + (1 - 1/beta1)(z-y). The training
gradient g is the gradient of the actual minibatch loss at y. Set

```
v_next = beta2*v + (1-beta2)*g^2,
P_i = 1 / (sqrt(v_next_i / (1-beta2^(t+1))) + epsilon_optimizer),
c = weight_next / (weight_total + weight_next),
eta = current warmup rate.

(T2) delta = x_next-x = c*(z-x) - c*eta*P*g - c*eta*lambda*y.
(T3) <grad F(x),delta>
     = c*<grad F(x),z-x> - c*eta*<grad F(x),P*g>
       - c*eta*lambda*<grad F(x),y>.
```

T2 follows by expanding the independently defined source `trainingStep` and
`evaluation`, with beta1 nonzero. The gradient at y and evaluation derivative
at x generally differ. The preconditioner depends on the actual sampled
gradient and stored second moment; it is not a freely chosen positive matrix.
The two extra terms show why training/heldout gradient alignment alone is an
incomplete test. T3 includes the change caused by averaging and weight decay.

For a bound L_t on the operator norm of D(grad F) throughout the entire segment
from x_t to x_(t+1), the existing mean-value descent lemma proves

```
(T4) F(x_(t+1)) <= F(x_t) + <grad F(x_t),delta_t> + L_t*||delta_t||^2.
(T5) failed_strict_winner_fraction <= F(x_t) / log(2).
(T6) F(x_N) <= F(x_0) + sum_(t<N) [<grad F(x_t),delta_t> + L_t*||delta_t||^2].
```

T5 includes ties as failures: any competitor at least as large as the correct
logit contributes enough softmax mass to force row loss at least log2. It counts
all answer classes. The conservative Taylor coefficient in T4 is L, not L/2.
Joint C^2 regularity and compactness prove that a finite L_t exists for every
finite segment. They do not compute a useful certified value.

If for every t<N the bracket in T6 is at most -mu*F(x_t), with 0<mu<=1,
`geometric_loss_le` proves F(x_N)<=(1-mu)^N F(x_0). Combined with T5, a value
below .01*log2 certifies at least 99% strict correct winners. The Lean rate lemma
only needs mu<=1 algebraically; positive mu is what makes the rate useful.
Its hypotheses have a nonzero, strictly improving witness at every finite
horizon. The actual-model additive theorem has a positive-depth, nonempty-data,
positive-rate five-update witness, with curvature existence proved rather than
assumed. Neither witness asserts positive heldout transfer from random starts.

### Formal scope and prospective test

`Transformer.Rotate90.Learning.runFrom` uses the real revised Schedule-Free
update and the full gradient of the actual minibatch network. Its initial
history is explicit, including the lab's zero initial maximum. Its samples may
change at every step and may have short epoch tails. The model and optimizer
are not replaced with a linearized kernel or infinite-width limit.

| Module | Result |
| --- | --- |
| `Learning.Defs` | Full finite-stack parameterization, actual parameter count, and T1 |
| `Learning.Basic` | Exact evaluation displacement and directional effect, T2--T3; segment-curvature existence and T4; full-softmax calibration T5 |
| `Learning.Trajectory` | Additive and geometric T6; actual minibatch path and finite-time failure bound using source transfer terms |

`experiments/rotate90_transfer/hypotheses.json` declares fifteen fresh runs
before training: P97K2 and P193K3, widths16/64; P17K1 width 64; depth1 and model
seeds 31/37/41. Data and batch seeds stay fixed at 11. Only Schedule-Free AdamW is
used, with Python 3.15/Torch 2.15 and compiled CUDA max-autotune. Every 50-update
checkpoint retains all model, optimizer and sampler state.

Independent CPU float64 copies preserve the trained float32 RMS epsilon and
replay the next actual batch. H1 checks T2 against the source optimizer. H2
tests whether T3 predicts the next-step heldout-loss sign in at least 90% of
eligible probes. H3 freezes a step 400 forecast using relative first-order
progress at 100/200/400; every false positive and false negative is retained.
H4 compares matched widths. Heldout gradients never enter training.

A measured finite-step remainder is not a certified segment-curvature bound.
The H3 forecast extrapolates future progress without proving it. The unresolved
claim is whether typical random starts maintain enough favorable transfer,
with controlled curvature, to obtain a success probability or training-time
prediction. This study tests that missing premise instead of assuming it.

Validation before training (2026-10-31): full `lake build` passed (13074 jobs).
All 17 new theorem axiom sets contain only `propext`, `Classical.choice` and
`Quot.sound`. `./run.py audit` retains the 36 existing unfinished statements,
with zero results resting on them and zero extra axioms. The source check passes.
`./run.py test` passed 410 lab tests (13 skipped) and 104 optimizer tests
(14 skipped), including bit-exact retained-checkpoint/resume comparisons,
independent source-update replays and a full-network finite-difference derivative
check. All 15 experiment recipes pass `./run.py check`.

### Outcome of the frozen transfer test

All fifteen runs completed their declared 1500 updates at setup commit
`d47112d5`, with unchanged definitions and hypotheses. Every P97/P193 width 64
run reaches 99.87--99.97% heldout accuracy, crossing 99% at updates 600--650.
The six width 16 runs remain at 1.96--17.93% within this budget. All P17 controls
fit training perfectly but reach only 7.35--9.56% heldout accuracy. These results
are scoped to the declared update cap and fixed split, not a capacity bound.

H1 and H2 pass all 120 probes. Maximum evaluation-displacement coordinate error
is 2.042e-16 in the independent float64 source replay. The complete directional
derivative predicts 98 improving and 22 worsening next-step heldout losses, with
no sign errors. Initial-state and checkpoint provenance, identical matched
sample fingerprints and all failed forecasts are retained in the study artifacts.
Before-update replay accuracy exactly matches the recorded CUDA evaluation;
the largest mean-loss difference is 4.040e-7. These are local numerical checks,
not a certified supremum of curvature over each segment.

H3 predicts one of the six final successes, with five misses and no false
positives. This has two causes which should not be conflated. The log2
calibration is sufficient, not necessary: every successful run's final CE is
above 0.01*log2. P97 seed31's predicted loss 0.009553 nearly matches its actual
0.009550, yet neither certifies its observed 99.8711% accuracy. The dynamics
forecast is also inaccurate for P193: width 64 seed31 is forecast at loss 1.977,
but reaches 0.01189 after its relative progress increases later in training.

H4 is rejected: width 64 improves the early statistic in only three of six pairs,
below the preset four-pair threshold. All three P193 early comparisons favor
width 16 even though all three wider models later generalize. At P193 width 64
seed31, relative first-order progress rises from about 0.000275 at update 200 to
0.00440 at 400 and 0.00990 at 750. The measured early statistic is not an invariant
ordering of model sizes or a uniform lower bound on future progress.

Thus the finite-time Lean bound must retain the evolving full update, and its
geometric specialization must quantify over every step in the horizon. T1--T6
do exactly that. They close the algebraic connection from this actual model
and optimizer to evaluation loss; they do not explain why the late acceleration
arises, prove a random-start success probability, or turn a few early derivative
probes into future guarantees. That remains the next gap, alongside obtaining
useful accuracy bounds when mean cross-entropy calibration is conservative.

## Optimizer memory

The next comparison separates new base optimization from the effect of stored
Schedule-Free averaging history. This is an extension of the actual-network
learning-transfer account, not a change to the flat-spectrum matrix theorem.
The source is `optimizers/src/lean_optim/schedule_free_adamw.py`, `step`, `train`
and `eval`, with default averaging powers 0 and 2. Write x for the evaluation
weights, y for the query, z for the base, W for the stored averaging total and
M for the stored maximum learning rate. The source relation is
y = beta1*x + (1-beta1)*z, with beta1 positive in the executable API.

**M1, zero-rate continuation.** Setting the current rate to zero, while keeping
the saved state, fixes z and M but advances W by M² per update. Raw second
moments still use each supplied gradient. With M nonnegative, the exact source
evaluation update is x' = (1-c)*x + c*z, where c = M²/(W+M²). Thus a zero
learning rate does not freeze the evaluated model when historical weights
remain positive. This is a continuation, not reinitialization of the optimizer.

**M2, exact clock.** For W > 0, induction through the source recurrence gives

```
z_n = z_0,  M_n = M_0,  W_n = W_0 + n*M_0²,
x_n = z_0 + W_0/(W_0 + n*M_0²) * (x_0-z_0).
```

The Euclidean distance to z_0 is multiplied by that exact scalar. If M_0 > 0,
x_n tends to z_0 regardless of future gradients, decay and changing second
moments. The time scale is W_0/M_0². The real-arithmetic theorem is checked
independently against actual finite-precision source steps; it does not assert
bitwise equality of floating-point repeated averaging and the closed form.

**M3, complete-parameter sign symmetry.** At every finite ordered depth,
`score(-w, a, b, q) = score(w, a, b, q)`. All trainable entries change sign.
The input and normalized hidden states change sign; simultaneously negating
gate and value matrices keeps their preactivations equal; negating the output
matrix reverses the branch; the negative tied embedding reverses the final
readout again. No oddness of SiLU is assumed. Full-softmax losses agree too.

**M4, nonconvex solver region.** Zero weights have zero logits on every row,
and per-row cross-entropy log(p). The existing powered-division construction
supplies a w solving the entire valid table for every prime p and natural k.
By M3, -w solves it too, while their midpoint is zero. Consequently the set
of parameters strictly solving the entire task is not convex, already at
width 16 and one block, with arbitrary additional width/hidden padding allowed.
The witness a=b=1, competing answer 0 works even at k=0. M3 itself covers all
finite depths. This is about parameter averaging; it does not assert a loss
convexity theorem or describe a typical training trajectory.

**M5, an actual source averaging counterexample.** An admissible saved state
with evaluation w, base -w, W=M²=1 and source query y=(2*beta1-1)*w has c=1/2.
One zero-rate source update has evaluation zero for every supplied gradient.
Combining M4 gives perfect endpoints and complete loss of strict winners at
the evaluation midpoint. Admissibility does not prove that random-start
training reaches this particular state.

**M6, classification and loss limit.** For positive RMS epsilon, the actual
all-parameter finite network is analytic. M2 and continuity give convergence
of every logit and every fixed split's full-softmax loss to its base value.
Every strict base logit comparison eventually holds along the stopped path;
finiteness gives one eventual time for all strict comparisons on the whole
table. This includes incorrect base winners. Ties at the base are excluded,
and continuity alone supplies no useful numerical time bound.

The declaration `experiments/rotate90_transfer/memory_hypotheses.json` pins
45 existing checkpoints: all fifteen runs at updates 200, 400 and 750. It
retains narrow models and P17 controls alongside all successful wide models.
The prior seed31 inspection is disclosed in `memory_exploration.json`: strong
evaluation predictions coexisted with poor base predictions, so the local
averaging benefit was not simply delayed arrival at an already solved base.
The counterfactual horizons, source-replay tolerance and transient-benefit
prediction are fixed before their evaluation. The 50000-update point uses
M2 directly; 32 actual source updates independently check the formula using
real gradients and the saved sampler. Compiled CUDA evaluation retains Python
3.15, Torch 2.15 and max-autotune. No new optimizer or fitting recipe is used.

These results distinguish historical averaging from continuing base learning.
They do not yet explain the favorable parameter-path geometry that emerges
under positive-rate Schedule-Free training, predict its late acceleration,
or prove a success probability from random starts.

Setup validation (2026-10-31): all 27 `Rotate90.Memory` theorems pass the full
Lean build (13080 jobs) and individual `#print axioms`; only `propext`,
`Classical.choice` and `Quot.sound` occur. Audit retains 36 existing unfinished
statements, zero results resting on them and zero extra axioms. The regenerated
index has zero vacuous/placeholder declarations, and the forbidden-source check
passes. The full Python suite passes 412 lab tests (13 skipped) and 104 optimizer
tests (14 skipped), including independent real-gradient continuation and a
depth-three sign-symmetry check. The compiled CUDA smoke check has exactly zero
negation logit error and no strict winner at the zero midpoint. All fifteen
source experiment recipes pass their definition check.

### Outcome of the declared memory controls

The analysis at setup `daab04d6` completed every pinned checkpoint and all ten
declared horizons. H1's maximum coordinate error against 32 actual source
updates is 6.661e-16. All 45 bases remain bitwise unchanged, while every raw
second moment changes and every evaluation point moves. H4's maximum negation
logit error is exactly zero; every zero midpoint has zero strict-winner fraction.
All 450 curve points and all controls are retained in `memory_results.json`.

All six wide P97/P193 update-750 checkpoints satisfy every H3 condition. Their
first stopped-base update improves loss, but their 50000-update closed-form
points drop from initial accuracies 99.76--99.98% to 4.06--7.28%, approaching
the poor base classifiers as M2 and M6 describe. Their lowest loss among the
declared horizons occurs at 100 or 200 additional updates, and every final
loss is above its initial value. Thus locally favorable averaging is not a
uniform benefit over the future path. The genuine trained models exhibit the
behavioral failure; the antipodal theorem independently excludes a general
convex solver-region argument, without claiming those trained paths are
antipodal or cross the zero model.

The intervention removes both gradient and decay movement of the base, while
retaining the actual historical clock and second moments. It therefore tests
stopped base optimization, not the separate causal roles of gradient and decay.
Narrow/P17 controls remain unsuccessful, and the earlier transfer forecast
failures remain recorded. These observations narrow the learning gap to how
ongoing base optimization shapes a favorable nonlinear averaging path; the
present theorems do not derive that geometry or its probability from random
initialization. Exact eventual comparison transfer also excludes ties at the
base and supplies no calibrated finite stopping time.

## Gradient forcing

The memory control changed the learning rate to zero, which removes both
normalized-gradient movement and query decay. The next test varies decay
alone in matched positive-rate continuations and uses a zero-gradient decay
control. It preserves each source's query, base, raw second moment, averaging
total, maximum rate, absolute optimizer time and next minibatch. Only the
declared rate/decay may change; the beta, epsilon and warmup settings agree.

**F1, actual two-buffer source update.** Let c be the current averaging
coefficient, eta the current rate, beta the first beta, d the query decay,
h = beta*(1-c)-1, and v the normalized gradient using the updated raw second
moment and its actual bias correction. The source update is exactly

```
[ y' ]   [ 1-c+eta*d*h    c ] [ y ]   [ eta*h*v ]
[ z' ] = [ -eta*d         1 ] [ z ] + [ -eta*v  ].
```

This is an identity for the existing `AdamW.trainingStep`, with the old query
in decay. The gradient input and its changing second moment are retained.
The scalar averaging history depends on rates and time, not the gradient
values. Therefore the same coefficient matrices govern every parameter and
both the original and zero-gradient trajectories from one saved history.

**F2, homogeneous evolution and forcing.** Denote those matrices by A_t and
the normalized-gradient terms by b_t. Independently define the matrix product
Phi from ordered source transitions and R_0=0, R_(t+1)=A_t R_t+b_t. Induction
through the actual source proves buffers_t = Phi_(0,t)*buffers_0 + R_t.
The response also equals the finite sum of past b_j transported by all
subsequent matrices. The b_j remain dependent on the actual trajectory and
variance; this is not a linear model of the neural network or a forecast
that replaces the unknown future gradients.

**F3, the gradient-free parameter plane.** With zero gradient tensors, b_t=0
even while the raw second moments decay from their saved values. Query and
base, and hence evaluation parameters, stay in the linear span of the initial
query and base. More generally any subspace containing the initial buffers
and all normalized-gradient inputs contains the entire finite trajectory.
Escaping it therefore requires at least one normalized-gradient input outside
it. Coordinatewise Adam normalization matters: this statement concerns v,
not merely the unnormalized minibatch gradient.

**F4, actual finite network.** The source identities permit the oracle to be
the gradient of the full finite Rotate90 sampled loss. The evaluation response
uses the source linear recovery x=y+(1-1/beta)*(z-y), and logits are evaluated
by the actual nonlinear tied network. An escape from the initial parameter
plane supplies a new direction; it does not prove correct labels, heldout
generalization, or favorable future transfer from random initialization.

`experiments/rotate90_forcing/hypotheses.json` pins fifteen step-400 sources
and thirty continuations, plus fixed zero-gradient diagnostic horizons. It
declares source replay, gradient-removal and decay-removal predictions before
their outcomes are observed. The initial new observation and compilation
preserve the loaded query exactly; later evaluations use the ordinary engine.
Full and no-decay arms continue to absolute update 1500 with the same samples.
The zero-gradient long-horizon evaluation uses F1's scalar recurrence, checked
against 32 actual source steps with zero tensors. All arithmetic diagnostics
and controls use the original finite architecture and Schedule-Free AdamW.

Setup validation (2026-10-31): all 27 `Rotate90.Forcing` theorems pass the
full build (13085 jobs) and individual axiom checks, using only `propext`,
`Classical.choice` and `Quot.sound`. The audit retains 36 existing unfinished
statements, zero dependencies on them and zero extra axioms; index and source
checks report zero vacuous/placeholder declarations and no forbidden constructs.
All 418 lab tests (13 skipped) and 104 optimizer tests (14 skipped) pass. The
new tests verify preserved source buffers, moments and sampler, exact CPU
continuation, changed decay behavior and independent forced/homogeneous source
identities. An independent P7 CUDA max-autotune smoke test also preserves the
initial query and reproduces the final query bit for bit. All thirty declared
recipes pass their definition check before the scientific continuations run.

The declared study at setup `9f769888` completed all thirty continuations and
all 150 zero-gradient diagnostic points. H2's maximum coordinate error over
32 actual source updates is 1.11e-15, and its maximum zero-gradient relative
span residual is 2.00e-14. All saved raw moments change under zero gradients;
they were not reset. All six wide P97/P193 full and no-decay continuations
exceed 99% heldout accuracy at update 1500. Their zero-gradient decay controls
finish at 2.95--5.15%; their best accuracies among the fixed horizons are
48.50--94.63%, all below 99%. H3 and H4 pass. This separates continued
gradient information from subsequent decay for those step-400 states; it
does not test no-decay training from random initialization.

Every full trajectory leaves its initial parameter span, with relative
distance 0.395--0.764, so H5 passes even for the failed narrow/P17 controls.
New directions alone do not explain generalization. The remaining theory
gap is the task alignment and accumulation of the normalized gradients that
drive F2's response. Its exact finite history still depends on those actual
gradients; it cannot be used as a learning forecast without controlling them.

H1 fails for all three P17 controls: CE differs from the original endpoint
by 0.34223, 0.01828 and 0.14770, and seed 37 also differs in accuracy by
0.7353 percentage points. All twelve P97/P193 cases pass the declared
tolerances. Initial state equality and matched samples hold in every case.
Post-outcome original/continuation comparisons at four saved steps confirm
identical samplers and scalar optimizer states; P17 parameter discrepancies
grow from about 1e-8 at update 450 to order one by update 1500. Their cause
is not established by that check, and H1 remains failed. One narrow P193
CUDA capture failure completed on an unchanged retry and passed replay.
All results, controls, hashes and attempt records are retained in
`experiments/rotate90_forcing`; no threshold or recipe was adjusted.
