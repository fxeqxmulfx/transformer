/-
Formalization of:
  McCandlish, Kaplan, Amodei, OpenAI Dota Team,
  "An Empirical Model of Large-Batch Training",
  arXiv:1812.06162.

Deviations from the source, each recorded in the docstring of the file that
makes it:
* the per-example gradients of a batch are pairwise independent random vectors
  of a common mean and covariance, weaker hypotheses than the paper's
  independent samples (`Section2_Batches`);
* the `≈` of eq. (2.4) is read as an `o(ε²)` error for a twice continuously
  differentiable loss, and eq. (2.5) is the mean of that quadratic model, as the
  paper evaluates it (`Section2_Quadratic`);
* `ε_max` needs the curvature `GᵀHG > 0`, which the paper leaves implicit
  (`Section2_Quadratic`).
-/

import Transformer.NoiseScale.Section2_Batches
import Transformer.NoiseScale.Section2_Quadratic
