# The ordinary MLP in Landscape

M9.2c instantiates the same two-hidden-layer SiLU MLP used by the matched
M9.3b numerical controls. It does not assume an analytic model, fixed
features, attainable minima or gradient accessibility. The executable
sources are `python/src/lab/infrastructure/nn/mlp.py` and
`python/src/lab/infrastructure/benchmarks/paired.py`, committed in `f178edd8`.

## Statement and source comparison

`Neural.MLP.Model` retains three distinct affine maps: input to hidden,
hidden to hidden, and hidden to output. All three matrices and all three
bias vectors are separate coordinates of one Euclidean space. SiLU follows
the first two maps and does not follow the final map. Matrix indices are
input-first, transposing PyTorch's storage; this only permutes parameter
coordinates and does not scale their Euclidean metric. There is no tying,
residual path or normalization hidden in the definitions.

The exact dimension is `H^2 + (D + O + 2)H + O`. At width 195, scalar and
one-hot operands have dimensions 2 and 194; scalar and vector outputs have
dimensions 1 and 97. The four dimension examples are respectively 39,001,
76,441, 57,817 and 95,257, matching the executable tests. The input encoders
include both operand positions; scalar encoding is `2v/(p-1)-1` and one-hot
encoding concatenates the two operand basis vectors.

`Neural.MLP.Losses` sums squared errors across outputs and divides by twice
the number of equations. It does not divide by output width. The singleton
output theorem identifies this loss with scalar MSE divided by two. At
modulus 97, the continuous numeric MSE before rounding is exactly 4,608
times this normalized half-MSE in real arithmetic. The Python integration
tests independently check the units and factor on a small example; FP32
observations need not satisfy that algebraic identity bit for bit.

The classification loss keeps all output classes in the softmax, uses the
same free final matrix and bias, and includes no EOS term. Fixed input
codes, arbitrary row sets and labels are accepted by both objectives.
Empty splits follow real division by zero, as in the existing loss
definitions; the witnesses and experiments use nonempty splits.

Both actual objectives are analytic at every actual parameter vector.
The existing finite-dimensional analytic Lojasiewicz theorem then supplies
a local exponent at every vector. No generalization, asymptotic optimizer
convergence, positive attraction basin or typical random-start conclusion
follows from those facts. In particular, analytic regularity and the local
inequality hold for the scalar MLP as well as the analytic Rotate90
models: an explanation of their training difference needs additional
architecture-sensitive gradient geometry.

## Validation

The new modules and the full tree build without warnings from these files.
The full build completes 12,776 jobs. All twelve new theorems have only
`propext`, `Classical.choice` and `Quot.sound` in `#print axioms`.
The global audit remains `39 sorry · 0 resting on a sorry · 0 extra axioms`.
The regenerated index has 3,679 modules and 18,901 theorems, with 39 `sorry`,
zero vacuous statements and zero placeholders. The forbidden and whitespace
checks pass.

Strict lint is optional following the user's later instruction, implemented
in `5abf1563`; it is not reported as a passing check. Its earlier failed
attempts remain documented in [dependency_update.md](dependency_update.md).
