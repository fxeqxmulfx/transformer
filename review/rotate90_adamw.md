# Rotate90 AdamW parameter container

`Transformer.Landscape.Neural.AdamW` connects the experiments' inclusive
warmup to the existing initialized native-moment bound. The source is
`domain.training.rate` and the Rotate90 experiment at `c59d4426`, with the
native recurrence in `Grokking/AdamW/ScalarRecurrence.lean` and the proved
bound in `ScheduledParameterBox.lean`.

For completed update count n and base rate a, the applied rate is
`a * min(1, (n + 1) / 10)`. The first rate is a/10 and the tenth is a.
Betas are 0.9 and 0.98, the optimizer epsilon is 1e-8, decay is 1, and the
two moments and their bias-correction clock remain in the state. This is
the same warmup indexing checked by the independent source replay.

For any base rate between zero and one, all decay factors are nonnegative.
Starting with zero buffers and clock, the existing moment-history theorem
bounds the actual corrected adaptive direction by 1.7, independently of
gradient magnitudes. Therefore every coordinate stays in
`[-max(abs(initial), 1.7), max(abs(initial), 1.7)]`. Gradients can depend on
the full learned state and on the entire data and batch history. No bound
on a future state, gradient, variance or direction is a hypothesis.

The product of these intervals is compact in coordinate topology. The
proof transports that set through the actual PiLp coordinate map; at the
finite indices of the 28,800-parameter SiLU and 28,928-parameter SwiGLU
models this is their usual Euclidean parameter topology. The theorem does
not assert compactness in an infinite-dimensional norm topology.

The compact-container result takes the exact native recurrence and zero
initial buffers as inputs and derives containment for the entire path.
Its witness is a nonconstant pure-decay path: initial coordinate 1 becomes
0.999 after the first Fast update. Two distinct nonzero coordinates satisfy
all recurrence and initialization hypotheses. This witness supplies zero
gradients to the recurrence; it does not claim that a neural loss has zero
gradient along that path.

The proof covers an exact-real AdamW update applied to any supplied
gradient sequence, hence also a specified neural gradient sequence if its
update is that recurrence. It does not certify the PyTorch FP32 kernels or
establish convergence, stochastic stability, a basin of attraction or
held-out generalization. Compactness and analytic loss alone do not turn
AdamW into gradient descent or discharge the existing M8 orbit conditions.

The new module builds without warnings. `#print axioms` on all six new
theorems reports only `propext`, `Classical.choice` and `Quot.sound`.
The full build passes all 12,773 jobs. The audit reports 39 direct `sorry`,
zero indirect rests and zero extra axioms. The regenerated index has
3,676 modules and 18,889 theorems, with zero vacuous statements and zero
placeholders; the forbidden and whitespace checks pass.

The unchanged mandatory strict all-linters command was attempted under a
60-second bound and returned timeout status 124. Its existing configuration
blocker remains in [dependency_update.md](dependency_update.md). No linter
was suppressed and no successful strict-lint result is claimed. The normal
build was restored after that attempt before the audit and index ran.
