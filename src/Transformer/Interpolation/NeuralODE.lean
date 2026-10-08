/-
# Measure-to-measure interpolation — Neural dynamics and the norm obstruction

Formalization of arXiv:2411.04551v3, §4:

* `Section4_NeuralSpeed` contains the unchanged integrated neural dynamics
  and their speed and displacement estimates.
* `Section4_NeuralBoundFalse` refutes the uniform parameter estimates in
  `prop: interpolation.neural.ode` and `lem: induction.neural.ode`. For one
  antipodal pair, bounds on all of `(W,U,b)` proportional to `1/T` force
  displacement to vanish as `T` grows. Matching without those norm bounds
  is not settled by these counterexamples.
* The exponential settling estimate `eq: Hartman.Grobman` is in `Settling`
  and `HartmanGrobman`, for the actual field of Step 2.

Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere` and the two propositions.
-/

import Transformer.Interpolation.Section4_NeuralBoundFalse
