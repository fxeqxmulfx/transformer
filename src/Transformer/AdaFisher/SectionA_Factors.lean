/-
# AdaFisher: linear, convolutional and fallback Kronecker factors

arXiv:2405.16397v3, Appendix A.3, “Computation of KFs”.
Convolutional patch extraction is specified by its finite index map,
including the augmented bias coordinate supplied by the caller.
-/

import Transformer.AdaFisher.Section3_Normalization
import Transformer.AdaFisher.Section3_MinMax

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.AdaFisher

variable {a b t : ℕ}

/-- Diagonal empirical fully connected factor, Appendix A.3. The batch
denominator is absent in the printed FC formula, so none is inserted here. -/
def fullyConnectedFactor (h : Fin t → Fin a → ℝ) : Fin a → ℝ :=
  fun i => ∑ x, h x i ^ 2

/-- Extracting convolutional patches by an explicit spatial index map,
Appendix A.3, the expansion operator `⟦h⟧`. -/
def expandPatches {v : ℕ} (input : Fin v → ℝ) (patch : Fin t → Fin a → Fin v) :
    Fin t → Fin a → ℝ := fun x i => input (patch x i)

/-- Convolutional diagonal factor divided by the number of spatial sites,
Appendix A.3. The positive-site domain is explicit when a mean is required. -/
def convolutionalFactor (patches : Fin t → Fin a → ℝ) : Fin a → ℝ :=
  fun i => (∑ x, patches x i ^ 2) / t

/-- Unsupported-layer fallback is the identity KF, Appendix A.3. -/
def fallbackFactor (a : ℕ) : Fin a → ℝ := fun _ => 1

/-- The FC factor is exactly the diagonal of its unnormalized Gram
matrix, Appendix A.3. It is not silently replaced by a batch average. -/
theorem fullyConnectedFactor_gram (h : Fin t → Fin a → ℝ) (i : Fin a) :
    fullyConnectedFactor h i = (∑ x, outer (h x) (h x)) i i := by
  simp [fullyConnectedFactor, outer, pow_two, Matrix.sum_apply]

/-- The convolutional factor is the normalization factor's diagonal,
Appendix A.3; both compute the mean of squared patch coordinates. -/
theorem convolutionalFactor_eq (patches : Fin t → Fin a → ℝ) (i : Fin a) :
    convolutionalFactor patches i = normalizationH patches i i := by
  rw [normalizationH_diagonal]
  rfl

/-- The spatial KF is nonnegative, Appendix A.3. -/
theorem convolutionalFactor_nonneg (patches : Fin t → Fin a → ℝ) (i : Fin a) :
    0 ≤ convolutionalFactor patches i := by
  rw [convolutionalFactor_eq]
  exact normalizationH_diagonal_nonneg patches i

/-- After min-max scaling the source's identity fallback has zero range,
Appendix A.3 and Proposition 3.2. Under the documented zero extension its
normalized factor is zero, so a pair of fallback factors leaves damping. -/
theorem fallback_normalized [NeZero a] (i : Fin a) :
    minMaxDiagonal (fallbackFactor a) i = 0 :=
  minMaxDiagonal_constant 1 i

end Transformer.AdaFisher
