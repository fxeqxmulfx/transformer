/-
Formalization of:
  Pascanu, Mikolov, Bengio,
  "On the difficulty of training Recurrent Neural Networks",
  arXiv:1211.5063.

Deviations from the source, each recorded in the docstring of the file that
makes it:
* eq. (5) writes the Jacobian of eq. (2) as `W_recᵀ diag(σ'(x_{i-1}))`; it is
  `W_rec diag(σ'(x_{i-1}))` (`Section1_Gradients`), the supplementary's
  `(W_recᵀ)^l` is `W_rec^l` (`Section2_Linear`), so the eigenvectors of its
  power iteration are left eigenvectors (`SectionA_PowerIteration`), and
  eq. (`dir_deriv`)'s `W_recᵀ diag(σ'(x_k))` is `W_rec diag(σ'(x_k))`
  (`Section3_Regularizer`);
* §2.1's conditions for nonlinear `σ` with `|σ'| ≤ γ`, `λ₁ < 1/γ` sufficient
  for vanishing and `λ₁ > 1/γ` necessary for exploding, are false
  (`Section2_Counterexample`); their proof proves them with the 2-norm
  `‖W_rec‖` in place of `λ₁` (`Section2_Mechanics`);
* "explode" is read as factors of norm at least `C α^l`, `α > 1`, for
  infinitely many `l` in the linear model, and as unbounded factors for the
  corrected nonlinear condition;
* §3.1's claims that with `λ₁ < 1` the gradient can not explode, and that the
  information inserted in the model, in the regime of the penalty or in an
  Echo State Network, dies out exponentially fast, are false for tanh
  (`Section3_Previous`); the information an input inserts is read as the
  derivative of the later states in that input (`Section3_Inputs`);
* §3.3's "we are not ensured the norm of the error signal is preserved" is
  read as a small `Ω` leaving the error norm unbounded (`Section3_ErrorSignal`);
* the supplementary's eq. (`approx_comp`) divides by `λ_j^l`, so `λ_j ≠ 0` is
  assumed (`SectionA_PowerIteration`); its expansion of the entire gradient
  sums over `i` a term in `k`, and is proved with the sum over `k`; its claim
  that the dominant temporal component makes the entire gradient explode is
  false (`SectionA_Growth`).

Not transcribed, deliberately: the dynamical-systems discussion of §2.2, the
error surface of Fig. 6 and the hypothesis of §2.3 that "in general when
gradients explode so does the curvature along `v`", which the paper states
without a precise form, and the heuristics of §3.3: that increasing
`‖∂x_t/∂x_k‖` "can not be always done while following a descent direction",
and that preventing vanishing gradients makes exploding ones "more probable";
and the supplementary's extension of the power iteration through the Jordan
normal form, sketched in one sentence.
-/

import Transformer.RecurrentGradients.Section1_Recurrence
import Transformer.RecurrentGradients.Section1_Gradients
import Transformer.RecurrentGradients.Section2_Mechanics
import Transformer.RecurrentGradients.Section2_Counterexample
import Transformer.RecurrentGradients.Section2_Linear
import Transformer.RecurrentGradients.Section2_Geometric
import Transformer.RecurrentGradients.Section3_Inputs
import Transformer.RecurrentGradients.Section3_Previous
import Transformer.RecurrentGradients.Section3_Clipping
import Transformer.RecurrentGradients.Section3_Regularizer
import Transformer.RecurrentGradients.Section3_ErrorSignal
import Transformer.RecurrentGradients.SectionA_PowerIteration
import Transformer.RecurrentGradients.SectionA_Growth
