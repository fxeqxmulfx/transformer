# Mod 97 with a scalar perceptron

A perceptron that reads x and y as two real numbers and answers with one
can represent x / y mod 97 exactly. Two hidden layers of width
2p + 1 = 195 suffice: `lookup` in
[`python/tests/test_regression.py`](../../python/tests/test_regression.py)
builds such weights under ReLU and under GELU, and the test checks that
every row is answered. Yet gradient descent from a random start does not
find such weights. Why not?

A perceptron's loss has a Lipschitz gradient on bounded sets only, so the
landscape theorems that ask for a global Lipschitz constant do not apply to
it. M8 of [`landscape_plan.md`](../../landscape_plan.md) asks instead for a
condition that a run of gradient descent with step η can check along its
own orbit x_k:

> (S) ‖∇²f(y)‖ ≤ L at every y of every segment [x_k, x_{k+1}], with η L < 1.

What is proved: under (S) step k lowers the loss by at least
η (1 - η L) ‖∇f(x_k)‖²
(`Transformer.Landscape.SaddleAvoidance.iterate_succ_le_sub_of_segment`,
M8.1). An analytic loss satisfies the Łojasiewicz inequality at every
point
(`Transformer.Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticOnNhd`),
and for it M8.3 proves the scheme along the orbit
(`Transformer.Landscape.SaddleAvoidance.ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_segment_of_analyticOnNhd`):
under any distribution of starts with a density, almost every start x_0
whose orbit satisfies (S) has an orbit that converges to a critical point
w with f(w) ≤ f(x_0) that is not a strict saddle, or ‖x_k‖ → ∞. A GELU
perceptron's loss is analytic in its parameters, as GELU is, and PyTorch's
uniform initialization has a density; Lean proves neither for this network.

A run that does not fit the table therefore ends in one of four ways:

- **H1, a spurious minimum or a degenerate saddle:** it converges with
  positive loss. The gradient norm falls to near 0, and λ_min ≥ 0 up to the
  Lanczos error. For gradient descent under (S) this is the first
  alternative of M8.3, with f(w) > 0.
- **H2, slowness:** it still descends when the budget ends: the slopes of
  the loss and of the gradient norm over the final decade are negative.
  M8.3 leaves this open, for it bounds no time. Under (S), M8.4 bounds
  f(x_k) - f(w) by C b^k, b < 1, when the Łojasiewicz exponent a at the
  limit w is at most 1/2
  (`Transformer.Landscape.SaddleAvoidance.exists_le_mul_pow_of_isLojasiewiczAt_of_segment`),
  and by C (k + 1)^(-1/(2a - 1)) when 1/2 < a < 1
  (`Transformer.Landscape.SaddleAvoidance.exists_le_mul_rpow_of_isLojasiewiczAt_of_segment`),
  but gives neither C, b nor a, so no theorem bounds the time a run needs.
  The residual's share of energy at high frequencies of the transform over
  Z/97 × Z/97 shows where the fit lags: the table of x / y has 98% of its
  energy above frequency 5, that of x + y 12%.
- **H3, divergence:** the parameter norm grows without bound. For gradient
  descent under (S) this is the second alternative of M8.3, ‖x_k‖ → ∞.
- **H4, a step past the curvature:** (S), the hypothesis of M8.3, fails.
  η·max(|λ_min|, λ_max) ≥ 1 at a recorded update of a gradient-descent run,
  and M8 says nothing there. The runs sample it every 1,000 updates, not on
  every segment. Ritz values lie inside the spectrum, so a recorded value
  of at least 1 shows that (S) fails, and values below 1 do not show that
  it holds.

The AdamW runs are controls: M8.3 concerns gradient descent, so for them
H1 to H3 name only what a run does, and H4 does not apply.

## Runs

Every run fits the whole table, 9,312 rows with x in [0, 97) and y in
[1, 97). The target is the answer c rescaled to [-1, 1], 2c / 96 - 1, and
the loss is the mean squared error. A prediction is right when it rounds
back to c, that is, when its error is below 1/96.

The perceptron reads its two inputs either as scalars, 2v / 96 - 1, or as
one-hot vectors, which give every operand value weights of its own. It has
two hidden layers and PyTorch's default initialization with model seed 0.

Each update is a full-batch step, 200,000 of them per run. Gradient descent
runs at a constant step without decay, at about half the inverse of the
largest Hessian eigenvalue at the start. That eigenvalue is 8.02 for the
scalar perceptron of width 256, 2.38 for the one-hot one, 3.50 at width 64
and 21.2 at width 1024, so at the start η·max(|λ_min|, λ_max) lies between
0.47 and 0.51. AdamW runs at rate 3e-4, with betas (0.9, 0.999) and no
weight decay, so that every run minimizes the same loss.

Every perceptron uses GELU, which is analytic. Under ReLU the loss is only
piecewise polynomial: its gradient jumps where a unit switches, and a
Hessian exists only off those kinks. The theorems ask for two continuous
derivatives, so they say nothing about ReLU. The Hessian-vector products of
autograd would also miss the kinks, and the tests for H1 and H4 would mean
nothing there.

| Run | Operands | Operation | Width | Optimizer | Step |
| --- | --- | --- | ---: | --- | ---: |
| gd-scalar-div | scalar | x / y | 256 | gradient descent | 1/16 |
| gd-scalar-add | scalar | x + y | 256 | gradient descent | 1/16 |
| gd-onehot-div | one-hot | x / y | 256 | gradient descent | 0.2 |
| gd-onehot-add | one-hot | x + y | 256 | gradient descent | 0.2 |
| adam-scalar-div | scalar | x / y | 256 | AdamW | 3e-4 |
| adam-scalar-add | scalar | x + y | 256 | AdamW | 3e-4 |
| adam-onehot-div | one-hot | x / y | 256 | AdamW | 3e-4 |
| adam-onehot-add | one-hot | x + y | 256 | AdamW | 3e-4 |
| gd-scalar-div-w64 | scalar | x / y | 64 | gradient descent | 0.14 |
| gd-scalar-div-w1024 | scalar | x / y | 1024 | gradient descent | 0.024 |

Width 64 is below the width of the construction, so whether that perceptron
can represent the table is open.

Every 1,000 updates, and at the start, a run records:

- the loss and accuracy on the table;
- the norm of the full gradient;
- the norm of the parameters;
- the smallest and largest Hessian eigenvalue, estimated by 16 Lanczos
  steps on exact Hessian-vector products;
- the share of the residual's energy at frequencies above 5, 10 and 20.

The last model gets 64 Lanczos steps. The result also records:

- the first update at which every row is right;
- the best accuracy;
- the log-log slopes of the loss and of the gradient norm over the last
  decade of updates. A slope near zero means the curve has settled, and a
  negative slope means it is still falling.

For every gradient-descent run the report adds `step_curvature`:
η·max(|λ_min|, λ_max) at every observation, its maximum, and the first
update at which it reaches 1, null if none. For the AdamW runs it is not
applicable.

```sh
./run.py check experiments/mod97_scalar_mlp
./run.py run experiments/mod97_scalar_mlp [labels]
./run.py report experiments/mod97_scalar_mlp [labels]
```

## Findings

All ten runs trained their 200,000 updates on 2026-10-18 (task #56). The
numbers are those of `./run.py report experiments/mod97_scalar_mlp`, to
three significant digits, unless they name `history.jsonl`: the final
observation, the extreme Ritz values of the last model from 64 Lanczos
steps, and for gradient descent `step_curvature`. A slope is the log-log
slope over the final decade. Every c occurs 96 times in either table, so
the targets have mean 0 and variance 784/2304 ≈ 0.340, the loss of the best
constant answer.

| Run | First fitted | Best accuracy | Final loss | Final accuracy | Slope of the loss | Slope of ‖∇f‖ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gd-scalar-div | — | 0.0190 | 0.320 | 0.0188 | -0.0185 | 0.0397 |
| gd-scalar-add | — | 0.674 | 0.00199 | 0.671 | -1.17 | -0.366 |
| gd-onehot-div | — | 0.105 | 0.00157 | 0.105 | -2.31 | -0.689 |
| gd-onehot-add | — | 0.214 | 0.00197 | 0.214 | -2.02 | 0.715 |
| adam-scalar-div | — | 0.0388 | 0.241 | 0.0180 | -0.111 | 0.655 |
| adam-scalar-add | 53,000 | 1 | 1.03e-4 | 0.657 | -0.899 | -0.802 |
| adam-onehot-div | 4,000 | 1 | 1.03e-9 | 1 | -1.18 | -0.587 |
| adam-onehot-add | 5,000 | 1 | 3.46e-8 | 1 | -1.75 | -1.54 |
| gd-scalar-div-w64 | — | 0.0230 | 0.320 | 0.0227 | -0.0177 | -0.577 |
| gd-scalar-div-w1024 | — | 0.0158 | 0.323 | 0.0140 | -0.0142 | 2.29 |

In the second table ‖∇f‖, ‖θ‖ and the residual's shares of energy above
frequencies 5, 10 and 20 are those of the final observation; λ_min and
λ_max are those of the last model.

| Run | ‖∇f‖ | λ_min | λ_max | Peak of η·max(\|λ_min\|, λ_max) | Reaches 1 at | ‖θ‖ | Residual above 5 / 10 / 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gd-scalar-div | 0.0374 | -0.0448 | 32.0 | 2.04 | 12,000 | 19.3 | 0.993 / 0.964 / 0.837 |
| gd-scalar-add | 0.278 | -0.0679 | 32.0 | 2.04 | 1,000 | 26.5 | 0.408 / 0.387 / 0.366 |
| gd-onehot-div | 0.152 | -0.0656 | 9.97 | 2.40 | 9,000 | 37.5 | 0.277 / 0.269 / 0.233 |
| gd-onehot-add | 0.114 | -0.0478 | 10.4 | 2.11 | 28,000 | 33.3 | 0.691 / 0.655 / 0.557 |
| adam-scalar-div | 0.0684 | 6.13e-5 | 1.96e4 | n/a | n/a | 160 | 1.00 / 0.998 / 0.945 |
| adam-scalar-add | 1.66 | -0.217 | 1.50e4 | n/a | n/a | 45.3 | 0.0759 / 0.0422 / 0.0230 |
| adam-onehot-div | 2.85e-4 | -6.57e-8 | 223 | n/a | n/a | 36.0 | 0.817 / 0.787 / 0.670 |
| adam-onehot-add | 7.42e-6 | 1.23e-9 | 170 | n/a | n/a | 34.9 | 0.974 / 0.946 / 0.882 |
| gd-scalar-div-w64 | 0.0196 | -0.0318 | 14.3 | 2.02 | 9,000 | 17.0 | 0.993 / 0.964 / 0.837 |
| gd-scalar-div-w1024 | 0.157 | -0.205 | 83.4 | 2.02 | 17,000 | 30.4 | 0.992 / 0.963 / 0.835 |

| Run | Outcome |
| --- | --- |
| gd-scalar-div | H4 from update 12,000 |
| gd-scalar-add | H4 from update 1,000 |
| gd-onehot-div | H4 from update 9,000 |
| gd-onehot-add | H4 from update 28,000 |
| adam-scalar-div | H3: the parameter norm grows, and the loss falls slowly |
| adam-scalar-add | fits, first at update 53,000; at the end 0.657 of the rows are right |
| adam-onehot-div | fits, first at update 4,000, and at the end |
| adam-onehot-add | fits, first at update 5,000, and at the end |
| gd-scalar-div-w64 | H4 from update 9,000 |
| gd-scalar-div-w1024 | H4 from update 17,000 |

**Every gradient-descent run ends in H4.** η·max(|λ_min|, λ_max) starts
between 0.475 and 0.510, first reaches 1 between update 1,000 (scalar
addition) and 28,000 (one-hot addition), and peaks between 2.02 and 2.40.
It never falls below 1 again, and from update 50,000 to the end it lies
between 1.98 and 2.11 in every run: λ_max climbs to about 2/η, the
curvature beyond which gradient descent on a quadratic diverges, and stays
there (32.0 for η = 1/16, 9.97 and 10.4 for η = 0.2, 14.3 for η = 0.14,
83.4 for η = 0.024). So (S) fails at every recorded update from the first
at 1 on, and M8 says nothing about these runs. No gradient-descent run fits
its table.

**Division against addition.** With scalar operands gradient descent
lowers the loss of addition to 0.00199, with 0.671 of the rows right, and
that of division only to 0.320, barely below the 0.340 of a constant
answer, with 0.0188 of the rows right. The residual of division has 0.993
of its energy above frequency 5, where the table has 98% of its own; that
of addition has 0.408, where the table has 12%. AdamW fits scalar
addition, first at update 53,000, though at the end only 0.657 of the rows
are right. It does not fit scalar division, and that run ends in H3: its
parameter norm grows from 14.8 at the start to 75.6 at update 50,000, 110
at 100,000 and 160 at the end (`runs/adam-scalar-div/history.jsonl`),
while its loss, 0.241 at the end, falls slowly (slope -0.111). Its gradient
norm grows too (slope 0.655), so it is neither H1 nor H2. Its residual has
1.00 of its energy above frequency 5.

**Scalar against one-hot operands.** One-hot operands give every operand
value weights of its own. With them gradient descent lowers the loss of
division to 0.00157, below the 0.00199 of scalar addition, though only
0.105 of the rows are right: a row is right only when its error is below
1/96, and the loss is a mean. One-hot addition ends at loss 0.00197, with
0.214 of the rows right. AdamW fits both one-hot tables, division first at
update 4,000 and addition at 5,000, and every row is right at the end, at
losses 1.03e-9 and 3.46e-8.

**Gradient descent against AdamW.** AdamW fits three of the four tables,
gradient descent none. With λ_max at about 2/η, the loss of gradient
descent still falls fast on scalar addition and on both one-hot tables
(slopes -1.17 to -2.31), and slowly on scalar division (-0.0185). AdamW
ends far sharper, with λ_max between 170 and 1.96e4 against 9.97 to 83.4
for gradient descent. Nor does it keep a fit at its constant rate: from the
first fitted update on, `history.jsonl` records some row wrong at 66 of the
148 observations of scalar addition, 6 of 197 of one-hot division and 9 of
196 of one-hot addition. The one-hot runs recover each time.

**Widths.** Scalar division by gradient descent ends alike at widths 64,
256 and 1024: loss 0.320, 0.320 and 0.323, 0.0227, 0.0188 and 0.0140 of
the rows right, loss slopes -0.0177, -0.0185 and -0.0142, and 0.993, 0.993
and 0.992 of the residual's energy above frequency 5. (S) fails from update
9,000, 12,000 and 17,000 on, and λ_max ends at 14.3, 32.0 and 83.4, about
2/η each time. Width 64 is below the construction's 195 and width 1024 far
above it; in 200,000 updates the width changes nothing.

**What this says.** Gradient descent at a constant step of about half the
inverse initial sharpness fits none of the four tables in 200,000 updates,
and M8 does not say why: in every run λ_max climbs to about 2/η, so (S)
fails. Past that point gradient descent still lowers the loss of three
tables fast, and that of scalar division hardly at all, at each of the
three widths. Scalar division is also the one table that AdamW, to which
(S) does not apply, does not fit: its parameters grow, H3, and its residual
has 1.00 of its energy above frequency 5.
