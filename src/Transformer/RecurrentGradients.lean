/-
Formalization of:
  Pascanu, Mikolov, Bengio,
  "On the difficulty of training Recurrent Neural Networks",
  arXiv:1211.5063.

Deviations from the source, each recorded in the docstring of the file that
makes it:
* eq. (5) writes the Jacobian of eq. (2) as `W_recᵀ diag(σ'(x_{i-1}))`; it is
  `W_rec diag(σ'(x_{i-1}))` (`Section1_Gradients`), and the supplementary's
  `(W_recᵀ)^l` is `W_rec^l` (`Section2_Linear`);
* §2.1's conditions for nonlinear `σ` with `|σ'| ≤ γ`, `λ₁ < 1/γ` sufficient
  for vanishing and `λ₁ > 1/γ` necessary for exploding, are false
  (`Section2_Counterexample`); their proof proves them with the 2-norm
  `‖W_rec‖` in place of `λ₁` (`Section2_Mechanics`);
* "explode" is read as factors of norm at least `C α^l`, `α > 1`, for
  infinitely many `l` in the linear model, and as unbounded factors for the
  corrected nonlinear condition.

Not transcribed, deliberately: the dynamical-systems discussion of §2.2, the
error surface of Fig. 6 and the hypothesis of §2.3 that "in general when
gradients explode so does the curvature along `v`", which the paper states
without a precise form.
-/

import Transformer.RecurrentGradients.Section1_Recurrence
import Transformer.RecurrentGradients.Section1_Gradients
import Transformer.RecurrentGradients.Section2_Mechanics
import Transformer.RecurrentGradients.Section2_Counterexample
import Transformer.RecurrentGradients.Section2_Linear
import Transformer.RecurrentGradients.Section2_Geometric
