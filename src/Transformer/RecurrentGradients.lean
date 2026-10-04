/-
Formalization of:
  Pascanu, Mikolov, Bengio,
  "On the difficulty of training Recurrent Neural Networks",
  arXiv:1211.5063.

Deviations from the source, each recorded in the docstring of the file that
makes it:
* eq. (5) writes the Jacobian of eq. (2) as `W_recᵀ diag(σ'(x_{i-1}))`; it is
  `W_rec diag(σ'(x_{i-1}))` (`Section1_Gradients`).
-/

import Transformer.RecurrentGradients.Section1_Recurrence
import Transformer.RecurrentGradients.Section1_Gradients
