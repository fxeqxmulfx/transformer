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
* `ε_max` needs the curvature `GᵀHG > 0`, and eqs. (2.6)–(2.7) also
  `tr(HΣ) ≥ 0`, which the paper leaves implicit (`Section2_Quadratic`,
  `Section2_NoiseScale`);
* "the loss may increase" beyond `2ε_opt` is proved as "increases", in the
  model (`Section2_NoiseScale`);
* `B ≪ B_noise` and `B ≫ B_noise` are read as bounds within a factor 2 and a
  limit (`Section2_Implications`);
* §2.3 is derived, as the paper derives it, from eq. (D.1), the trajectory's
  `ds` a finite measure; `B_crit = E_min/S_min` is the model's, not a fit, and
  `S_min`, `E_min` are proved least and the limits of constant batches
  (`Section2_Tradeoff`).
-/

import Transformer.NoiseScale.Section2_Batches
import Transformer.NoiseScale.Section2_Quadratic
import Transformer.NoiseScale.Section2_NoiseScale
import Transformer.NoiseScale.Section2_Implications
import Transformer.NoiseScale.Section2_Tradeoff
