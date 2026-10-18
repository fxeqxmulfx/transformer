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

What is proved (M8.1): under (S) step k lowers the loss by at least
η (1 - η L) ‖∇f(x_k)‖²
(`Transformer.Landscape.SaddleAvoidance.iterate_succ_le_sub_of_segment`),
and the starts whose orbit satisfies (S) and converges to a strict saddle
form a null set
(`Transformer.Landscape.SaddleAvoidance.ae_not_tendsto_strictSaddle_of_segment`).
What is planned, and not proved (M8.3, task #52): if the loss is C² and
satisfies the Łojasiewicz inequality at its critical points, as an analytic
loss such as a GELU perceptron's does
(`Transformer.Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticOnNhd`),
then almost every orbit that satisfies (S) converges to a critical point
that is not a strict saddle, or leaves every bounded set.

A run that does not fit the table therefore ends in one of four ways:

- **H1, a spurious minimum or a degenerate saddle:** it converges with
  positive loss. The gradient norm falls to near 0, and λ_min ≥ 0 up to the
  Lanczos error.
- **H2, slowness:** it still descends when the budget ends: the slopes of
  the loss and of the gradient norm over the final decade are negative. No
  theorem bounds the time it needs. The residual's share of energy at high
  frequencies of the transform over Z/97 × Z/97 shows where the fit lags:
  the table of x / y has 98% of its energy above frequency 5, that of
  x + y 12%.
- **H3, divergence:** the parameter norm grows without bound.
- **H4, a step past the curvature:** (S) fails. η·max(|λ_min|, λ_max) ≥ 1
  at a recorded update of a gradient-descent run, and M8 says nothing
  there. The runs sample it every 1,000 updates, not on every segment.
  Ritz values lie inside the spectrum, so a recorded value of at least 1
  shows that (S) fails, and values below 1 do not show that it holds.

The AdamW runs are controls: (S) concerns gradient descent, and H4 does
not apply to them.

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
./make.py check experiments/mod97_scalar_mlp
./make.py run experiments/mod97_scalar_mlp [labels]
./make.py report experiments/mod97_scalar_mlp [labels]
```

## Findings

Pending: the runs are task #56 of the board.
